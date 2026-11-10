"""LangGraph agent: an LLM node plus a tool node, looping until the LLM stops calling tools.

The tools are built in main.py from MCP servers: log_food / check_goals / set_goals /
delete_food_entry (filesystem, food_log.py), suggest_recipes (Tavily, recipes.py), search_web (Tavily,
research.py) and lookup_nutrition (our nutrition server). A final render_results node
swaps the model's [[R#]] recipe picks for the exact search results and removes links
the searches didn't return.
"""

import logging
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt.tool_node import ToolInvocationError

from food_log import LOCAL_TZ
from llm import build_llm
from mcp_utils import failure_reason
from recipes import parse_labelled_results, render_answer
from research import result_urls, strip_unknown_urls

load_dotenv()
log = logging.getLogger("macro-agent")

MAX_HISTORY_MESSAGES = 20  # past user messages + answers the LLM sees (about 10 turns)

SYSTEM_PROMPT = """You are a macro-tracking assistant. You help the user log food, \
check their remaining daily calorie/macro budget, and find recipes that fit it.

Rules:
- Any request for meal, snack or recipe ideas (e.g. "what should I eat?", "breakfast \
ideas with eggs", "a good post-workout snack") must go through the tools: first call \
check_goals, then call suggest_recipes with the remaining calories and protein from \
that result, unchanged. Never suggest recipes from your own knowledge.
- Cooking how-to questions and general nutrition facts (e.g. "how do I poach an egg?", \
"how much protein is in an egg?") are answered directly from your own knowledge, with \
no tool call. So is adapting a dish the user already has in mind: swapping or leaving \
out an ingredient (e.g. "I don't have shrimp for that salad") or adjusting a recipe.
- When the user asks you to search or look something up, or wants current \
information, recent research, news or sources (e.g. "search the web for...", "what \
does the latest research say about..."), call search_web and base your answer on its \
results, citing the sources you used as markdown links: [title](URL). Never say you \
can't browse or search. Call search_web once per question, with one well-chosen query, \
and answer from its results; don't search again to refine.
- When the user states a daily calorie or macro goal (e.g. "my goal is 1800 calories", \
"I want 120 g of protein a day"), call set_goals with only the goals they gave, and \
confirm the new goals. Stating a goal is not eating: don't log it as food.
- When the user tells you what they ate, call log_food. Its result already includes \
that day's totals, so don't call check_goals afterwards unless the user also asks for \
meal ideas. If they ate it yesterday, pass date "yesterday" to log_food (for an \
older day, YYYY-MM-DD); the tool works out the calendar date.
- To correct or remove something already logged, call check_goals to find the entry's \
id, then delete_food_entry; for a correction, then log_food the right values. Never \
say an entry was corrected or deleted unless delete_food_entry succeeded.
- If one message both reports food and asks for ideas, log the food first, then \
check_goals, then suggest_recipes, so the ideas fit the updated budget.
- Call check_goals on its own only when the user asks about their budget, progress, \
goals or what they've logged.
- suggest_recipes returns results labelled [R1], [R2], ... grouped into tiers. Pick \
up to 2 per tier that best fit the request, preferring a page for one specific recipe \
over a list of many when both fit. Under a heading per tier that keeps what \
it searched for (e.g. **Light** (searched: under 400 kcal)), put each pick on its own \
line as: [[R1]] — <reason in under 12 words, based only on its title and \
snippet>. Never type a recipe name or URL yourself; the [[R#]] reference is replaced \
with the real title, link and snippet. Don't state calories or protein for \
individual recipes. If no result fits the request, pick none: say so briefly and \
suggest what to try instead.
- When logging food: for each single food (e.g. egg, banana, rice), estimate the \
weight in grams and call lookup_nutrition, then call log_food with the \
for_requested_grams values of the match whose description fits how it was eaten \
(raw vs cooked, fresh vs dried), with source "USDA <fdc_id>". If no match fits, \
retry the lookup once with different wording (e.g. "oats cooked" instead of \
"oatmeal"). For mixed dishes or restaurant items, or when still nothing fits, \
estimate the values yourself with source "estimate". Whenever you log an \
estimate, say so briefly in your reply and why (e.g. no good USDA match, or the \
nutrition lookup is unavailable). If the user gives the numbers, use theirs. When \
several foods are mentioned, request all their lookups together in one step, then \
all their log_food calls together in the next step.
- lookup_nutrition is only for logging. General nutrition questions are still \
answered from your own knowledge.
- You remember earlier messages in this conversation. If the user asks about \
earlier messages and there are none, say this is the start of the conversation; \
don't claim you have no memory.
- If a tool returns an error, explain the problem to the user plainly; don't \
invent data.
- Keep answers short."""


def system_message(unavailable: list[str]) -> SystemMessage:
    # The model has no clock; without the date it invents one for check_goals.
    today = datetime.now(LOCAL_TZ).strftime("%A, %Y-%m-%d")
    text = f"{SYSTEM_PROMPT}\n\nToday is {today}."
    if unavailable:
        text += (f"\n\nCurrently unavailable (its server can't be reached): "
                 f"{', '.join(unavailable)}. If the user needs it, say so plainly and help "
                 "with what you can.")
    return SystemMessage(text)


def _last_user_index(messages: list[AnyMessage]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return -1


def current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Messages since the latest user message."""
    return messages[_last_user_index(messages) + 1:]


def llm_view(messages: list[AnyMessage]) -> list[AnyMessage]:
    """What the LLM sees from the checkpointed history. Past turns are reduced to the
    user's messages and the final answers (tool calls/results dropped: the answers
    already summarise them, and old tool payloads would bloat every prompt). The
    current turn is kept whole so the tool-call loop works. The checkpointer itself
    still stores everything."""
    start = max(_last_user_index(messages), 0)
    past = [
        m for m in messages[:start]
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    return past[-MAX_HISTORY_MESSAGES:] + messages[start:]


def tool_failure_message(error: Exception) -> str:
    """ToolNode error handler: a tool that raises becomes an error result the LLM can
    explain, instead of failing the whole request. Our wrapper tools already report
    their own errors; this catches the unwrapped MCP tool (lookup_nutrition) when its
    server crashes, times out or breaks protocol, plus any unexpected bug (logged).
    Invalid arguments from the LLM keep ToolNode's default message so it can retry."""
    if isinstance(error, ToolInvocationError):
        return error.message
    log.error("tool call failed", exc_info=error)
    return f"Error: the tool's server {failure_reason(error)}. This feature is unavailable right now."


def build_graph(tools, checkpointer=None, unavailable: list[str] | None = None):
    llm = build_llm(tools)  # primary model with automatic fallback (llm.py)

    async def call_model(state: MessagesState):
        start = time.perf_counter()
        response = await llm.ainvoke([system_message(unavailable or [])] + llm_view(state["messages"]))
        log.info("LLM call %.1fs (%s, %d output tokens)", time.perf_counter() - start,
                 response.response_metadata.get("model_name", "?"),
                 (response.usage_metadata or {}).get("output_tokens", 0))
        return {"messages": [response]}

    async def render_results(state: MessagesState):
        """Final step for turns that used search results. Recipe turns: the model's
        [[R#]] picks become the exact title, link and snippet (recipes.render_answer).
        Web-search turns: links to URLs the search didn't return are removed
        (research.strip_unknown_urls). The rendered answer replaces the original
        message (same id), so only the final text enters the conversation history."""
        turn = current_turn(state["messages"])

        def outputs(tool_name: str) -> list[str]:
            return [m.content for m in turn if isinstance(m, ToolMessage)
                    and m.name == tool_name and m.status != "error"]

        recipe_outputs, search_outputs = outputs("suggest_recipes"), outputs("search_web")
        if not recipe_outputs and not search_outputs:
            return {}
        answer = state["messages"][-1]
        allowed = result_urls(search_outputs)
        if recipe_outputs:
            text = render_answer(answer.text, parse_labelled_results(recipe_outputs), allowed)
        else:
            text, removed = strip_unknown_urls(answer.text, allowed)
            if removed:
                log.warning("removed %d link(s) not in the search results", removed)
        return {"messages": [AIMessage(content=text, id=answer.id)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=tool_failure_message))
    graph.add_node("render_results", render_results)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "render_results"})
    graph.add_edge("tools", "agent")
    graph.add_edge("render_results", END)
    return graph.compile(checkpointer=checkpointer)
