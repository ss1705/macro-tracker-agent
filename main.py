"""FastAPI entrypoint exposing POST /chat."""

import asyncio
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Annotated

import openai
import uvicorn
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, StringConstraints

from agent import build_graph, current_turn
from food_log import make_food_tools
from mcp_utils import PersistentSession, load_server_config
from recipes import make_recipe_tool
from research import make_search_tool

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("macro-agent")

MCP_CONFIG_PATH = os.environ.get("MCP_SERVERS_CONFIG", "mcp_config.json")
# What the user loses when a server is down, as told to the LLM.
FEATURES = {
    "filesystem": "food logging, goals and daily budget (food log storage)",
    "tavily": "recipe search and web search",
    "nutrition": "USDA nutrition lookup (log with your own estimates instead)",
}
DISCOVERY_RETRY_SECONDS = 60
# Hard cap per /chat request. Normal requests take 1-11 s; this only bites if both the
# primary and the fallback model are failing.
REQUEST_DEADLINE_SECONDS = 30


class Agent:
    """The compiled graph plus the MCP servers it was built from. Each server runs as
    one long-lived process with one MCP session (mcp_utils.PersistentSession). Servers
    that fail tools/list at startup are left out and retried at most once a minute; the
    graph is rebuilt when one comes back. The checkpointer is shared across rebuilds,
    so conversation memory survives."""

    def __init__(self, servers: dict):
        self.servers = servers
        client = MultiServerMCPClient(servers)
        self.sessions = {name: PersistentSession(client, name) for name in servers}
        self.data_dir = os.path.abspath(servers["filesystem"]["args"][-1])
        os.makedirs(self.data_dir, exist_ok=True)
        # Conversation memory: LangGraph's in-memory checkpointer (a.k.a. MemorySaver),
        # keyed by thread_id = the request's session_id. Lives as long as the instance.
        self.checkpointer = InMemorySaver()
        self.server_tools: dict[str, dict] = {}
        self.last_discovery = 0.0
        self.graph = None
        self.session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @property
    def missing(self) -> list[str]:
        return [name for name in self.servers if name not in self.server_tools]

    async def discover(self) -> None:
        self.last_discovery = time.monotonic()
        found = False
        for name in self.missing:
            try:
                self.server_tools[name] = {t.name: t for t in await self.sessions[name].load_tools()}
                log.info("MCP tools/list %s -> %s", name, sorted(self.server_tools[name]))
                found = True
            except Exception:
                log.exception("MCP server %s unavailable; its features are disabled for now", name)
                await self.sessions[name].close()
        if found or self.graph is None:
            self.graph = build_graph(self.tools(), self.checkpointer,
                                     unavailable=[FEATURES[name] for name in self.missing])

    async def close(self) -> None:
        for session in self.sessions.values():
            await session.close()

    async def refresh_if_degraded(self) -> None:
        if self.missing and time.monotonic() - self.last_discovery > DISCOVERY_RETRY_SECONDS:
            await self.discover()

    def tools(self) -> list:
        tools = []
        if "filesystem" in self.server_tools:
            tools += make_food_tools(self.server_tools["filesystem"], self.data_dir)
        if "tavily" in self.server_tools:
            tools.append(make_recipe_tool(self.server_tools["tavily"]))
            tools.append(make_search_tool(self.server_tools["tavily"]))
        if "nutrition" in self.server_tools:
            # Our own server's tool goes to the LLM as-is: schema straight from tools/list.
            tools.append(self.server_tools["nutrition"]["lookup_nutrition"])
        return tools


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = Agent(load_server_config(MCP_CONFIG_PATH))
    await app.state.agent.discover()
    yield
    await app.state.agent.close()


app = FastAPI(lifespan=lifespan)


# Empty or whitespace-only values get a 422 like a missing field: an empty query would
# otherwise reach the LLM (which stalled on it), and every "" session would share one
# conversation and food log.
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChatRequest(BaseModel):
    query: NonEmptyStr
    session_id: NonEmptyStr


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    agent: Agent = app.state.agent
    config = {"configurable": {"thread_id": req.session_id}}  # selects this session's memory
    try:
        await agent.refresh_if_degraded()
        # One run at a time per session: two overlapping runs would each see the other's
        # unanswered message and act on it twice (e.g. log the same food twice).
        async with agent.session_locks[req.session_id]:
            result = await asyncio.wait_for(
                agent.graph.ainvoke({"messages": [HumanMessage(req.query)]}, config),
                REQUEST_DEADLINE_SECONDS)
    except asyncio.TimeoutError:  # hard cap, so a response can never take minutes
        log.warning("request deadline reached for session %s", req.session_id)
        return ChatResponse(response="That took too long on my end. Please try again in a moment.")
    except openai.APITimeoutError:  # subclass of APIConnectionError, so checked first
        log.exception("LLM timed out for session %s", req.session_id)
        return ChatResponse(response="The AI model didn't respond in time. Please try again in a moment.")
    except openai.APIConnectionError:
        log.exception("LLM unreachable for session %s", req.session_id)
        return ChatResponse(response="I couldn't reach the AI model right now. Please try again in a moment.")
    except openai.RateLimitError:
        log.exception("LLM rate limit hit for session %s", req.session_id)
        return ChatResponse(response="The AI model is busy right now (rate limit). Please try again shortly.")
    except openai.InternalServerError:  # 5xx from the model provider (fallback included)
        log.exception("LLM provider error for session %s", req.session_id)
        return ChatResponse(response="The AI model service had an error. Please try again in a moment.")
    except Exception:
        log.exception("agent run failed for session %s", req.session_id)
        return ChatResponse(response="Sorry, something went wrong on my end. Please try again.")

    for m in current_turn(result["messages"]):
        if getattr(m, "tool_calls", None):
            log.info("tool calls: %s", [(c["name"], c["args"]) for c in m.tool_calls])
    # .text, not .content: some providers return the reply as a list of blocks
    # (e.g. reasoning + text); .text is the plain answer either way.
    return ChatResponse(response=result["messages"][-1].text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
