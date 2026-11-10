"""suggest_recipes: agent-facing recipe search backed by the Tavily MCP server, plus
rendering of the agent's picks.

Tavily is used for recipe *discovery*, not nutrition data: its results are scraped
page text (often roundup pages listing many recipes), so per-recipe macros can't be
attributed reliably. Instead, code splits the user's remaining budget into tiers and
runs one real MCP tools/call to tavily_search per tier, with the tier's calorie cap
in the query. Results are filtered to prefer single recipe pages (keep_recipe_pages):
meal plans and PDFs are dropped and roundup lists move behind single recipes.

Each result is labelled [R1], [R2], ... The LLM picks results by writing [[R1]] and a
short reason; render_answer() then swaps each reference for the exact title, URL and
snippet from the search results. The LLM never types a recipe name or URL itself, so
it can't invent one, and its answer stays short.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool, ToolException, tool

from mcp_utils import call_mcp_tool

log = logging.getLogger("macro-agent")

# One meal or snack rarely uses a whole day's budget; tiers scale from at most this much.
CALORIE_CEILING = {"meal": 1000, "snack": 400}
MIN_TIER_CALORIES = 150
TIER_SHARES = (("Light", 0.4), ("Medium", 0.7), ("Hearty", 1.0))
SEARCH_RESULTS = 10  # fetched per tier, so a full tier survives keep_recipe_pages
RESULTS_PER_TIER = 3  # shown to the model per tier
SNIPPET_CHARS = 220
# Sites that return feeds, answers or listings rather than recipe pages.
EXCLUDED_DOMAINS = [
    "tiktok.com", "pinterest.com", "pinterest.ca", "pinterest.co.uk",
    "instagram.com", "facebook.com", "youtube.com", "reddit.com",
    "quora.com", "etsy.com", "lemon8-app.com",  # Q&A, marketplace, social: not recipes
]

LABELLED_RESULT_RE = re.compile(r"(?m)^\[(R\d+)\] Title: (.*)\nURL: (\S+)\nSnippet: (.*)$")
TIER_HEADING_RE = re.compile(r"(?m)^## (\w+) tier")
PICK_LINE_RE = re.compile(r"^\s*(?:[-*]\s*)?\[\[(R\d+)\]\]\s*(?:[—–:-]\s*)?(.*)$")
PICK_RE = re.compile(r"\[\[(R\d+)\]\]")
URL_RE = re.compile(r"https?://\S+")


@dataclass
class Tier:
    name: str
    max_calories: int
    min_protein_g: int


@dataclass
class Result:
    tier: str
    title: str
    url: str
    snippet: str


def _round_to(value: float, step: int) -> int:
    return int(step * round(value / step))


def compute_tiers(
    remaining_calories: float, remaining_protein_g: float, meal_type: str = "meal"
) -> list[Tier]:
    """Light / Medium / Hearty caps from the remaining budget, capped by the meal type's
    ceiling. Protein targets scale with each tier's share of the remaining calories.
    Returns fewer tiers when little is left."""
    if remaining_calories < MIN_TIER_CALORIES:
        return []
    base = min(remaining_calories, CALORIE_CEILING[meal_type])
    tiers = []
    for name, share in TIER_SHARES:
        calories = _round_to(base * share, 50)
        if calories < MIN_TIER_CALORIES or any(t.max_calories == calories for t in tiers):
            continue
        protein = _round_to(max(remaining_protein_g, 0) * calories / remaining_calories, 5)
        tiers.append(Tier(name, calories, protein))
    return tiers


def parse_results(text: str) -> list[tuple[str, str, str]]:
    """Split tavily_search text output into (title, url, content) triples."""
    results = []
    for block in re.split(r"(?m)^Title: ", text)[1:]:
        title = block.splitlines()[0].strip()
        url = re.search(r"(?m)^URL: (\S+)", block)
        content = re.search(r"(?ms)^Content: (.*)", block)
        if url:
            results.append((title, url.group(1), content.group(1).strip() if content else ""))
    return results


NOT_A_RECIPE_RE = re.compile(r"meal plan|diet plan|grocery list|\b\d+[- ]day\b|\bpdf\b", re.I)
ROUNDUP_TITLE_RE = re.compile(
    r"^\s*(?:the\s+)?\d+\b"                                   # "25 High-Protein Dinners"
    r"|\b\d+\s+(?:[\w-]+\s+){0,5}(?:recipes|ideas|meals|dinners|lunches|breakfasts|snacks|desserts)\b"
    r"|\b(?:recipes|ideas|meals) (?:under|for|with|to)\b"     # "Meals Under 400 Calories"
    r"|collection|round-?up",
    re.I,
)
ROUNDUP_URL_RE = re.compile(r"/(?:collections?|galler(?:y|ies)|g\d+|roundups?)/", re.I)


def keep_recipe_pages(results: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Prefer single recipe pages: drop meal plans, diet plans and PDFs, and move
    roundup/listicle pages after single recipes (kept as a fallback so a tier isn't
    left empty). Order is otherwise preserved."""
    kept = [r for r in results if not NOT_A_RECIPE_RE.search(f"{r[0]} {r[1]}")]

    def is_roundup(result: tuple[str, str, str]) -> bool:
        return bool(ROUNDUP_TITLE_RE.search(result[0]) or ROUNDUP_URL_RE.search(result[1]))

    return sorted(kept, key=is_roundup)


def clean_snippet(content: str, limit: int = SNIPPET_CHARS) -> str:
    """One readable line from scraped page text: drop markdown/elision noise and cut
    at a word boundary."""
    text = re.sub(r"\[\.\.\.\]|[#*|>]+", " ", content)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def normalize_url(url: str) -> str:
    return url.rstrip(".,;:!?)").rstrip("/")


def parse_labelled_results(tool_outputs: list[str]) -> dict[str, Result]:
    """Rebuild the R-id -> Result table from suggest_recipes' own output."""
    results = {}
    for output in tool_outputs:
        headings = [(m.start(), m.group(1)) for m in TIER_HEADING_RE.finditer(output)]
        for m in LABELLED_RESULT_RE.finditer(output):
            tier = next((name for pos, name in reversed(headings) if pos < m.start()), "")
            results[m.group(1)] = Result(tier, m.group(2).strip(), m.group(3), m.group(4).strip())
    return results


def _render(result: Result, reason: str) -> str:
    block = f"- **{result.title}** — {result.url}"
    if result.snippet:
        block += f"\n  {result.snippet}"
    if reason:
        block += f"\n  Why: {reason}"
    return block


def render_answer(answer: str, results: dict[str, Result],
                  allowed_urls: frozenset[str] | set[str] = frozenset()) -> str:
    """Swap [[R#]] picks for the real search results. Unknown ids are dropped, and so
    is any line with a URL the model typed itself, unless every URL on it came from a
    web search in the same turn (allowed_urls). If the model picked nothing, its own
    reply is kept (e.g. "none of these fit"); only an empty reply falls back to the top
    result of each tier."""
    out, rendered, dropped = [], 0, 0
    for line in answer.splitlines():
        pick = PICK_LINE_RE.match(line)
        if pick:
            if pick.group(1) in results:
                out.append(_render(results[pick.group(1)], pick.group(2).strip()))
                rendered += 1
            else:
                dropped += 1
            continue
        typed_urls = URL_RE.findall(line)
        if typed_urls and not all(normalize_url(u) in allowed_urls for u in typed_urls):
            dropped += 1  # only code-rendered lines, or cited search results, may carry URLs
            continue

        def inline(m: re.Match) -> str:
            nonlocal rendered, dropped
            if m.group(1) in results:
                rendered += 1
                return f"**{results[m.group(1)].title}** ({results[m.group(1)].url})"
            dropped += 1
            return ""

        out.append(PICK_RE.sub(inline, line))

    log.info("rendered %d recipe picks, dropped %d unverifiable lines/ids", rendered, dropped)
    text = "\n".join(out).strip()
    # Replacing a reply that deliberately picked nothing showed the user results the
    # model had rejected (a grocery store and shrimp recipes for "without shrimp").
    if rendered or not results or text:
        return text

    log.warning("no valid recipe picks; falling back to top result per tier")
    lines, seen_tiers = ["Here are a few options from the recipe search:"], set()
    for result in results.values():
        if result.tier not in seen_tiers:
            seen_tiers.add(result.tier)
            lines += ["", f"**{result.tier}**", _render(result, "")]
    return "\n".join(lines)


def make_recipe_tool(tavily_tools: dict[str, BaseTool]) -> BaseTool:
    search = tavily_tools["tavily_search"]

    async def search_tier(query: str, tier: Tier) -> str:
        search_query = f"{query} recipe under {tier.max_calories} calories per serving"
        if tier.min_protein_g > 0:
            search_query += f" at least {tier.min_protein_g}g protein"
        return await call_mcp_tool(
            "tavily", search, query=search_query, search_depth="basic",
            max_results=SEARCH_RESULTS, exclude_domains=EXCLUDED_DOMAINS,
        )

    @tool
    async def suggest_recipes(
        query: str,
        meal_type: Literal["meal", "snack"],
        remaining_calories: float,
        remaining_protein_g: float,
    ) -> str:
        """Search the web for recipes that fit the user's remaining budget, in
        Light / Medium / Hearty tiers. Call check_goals first and pass its remaining
        calories and protein unchanged.

        query describes the food (e.g. "vegetarian dinner", "sweet snack").
        meal_type is "snack" for snacks, otherwise "meal".
        Returns labelled results ([R1], [R2], ...) grouped by tier.
        """
        tiers = compute_tiers(remaining_calories, remaining_protein_g, meal_type)
        if not tiers:
            return (f"Only {round(remaining_calories)} kcal left today, too little for a "
                    "recipe search. Suggest a very light snack or none.")

        outputs = await asyncio.gather(
            *(search_tier(query, t) for t in tiers), return_exceptions=True
        )
        if all(isinstance(o, Exception) for o in outputs):
            raise ToolException("Recipe search is unavailable right now.")

        sections, seen_urls, next_id = [], set(), 1
        for tier, output in zip(tiers, outputs):
            heading = f"## {tier.name} tier (searched: under {tier.max_calories} kcal"
            heading += f", at least {tier.min_protein_g} g protein)" if tier.min_protein_g else ")"
            if isinstance(output, Exception):
                log.warning("tavily search failed for %s tier: %s", tier.name, output)
                sections.append(f"{heading}\nSearch failed for this tier.")
                continue
            candidates = keep_recipe_pages(parse_results(output))
            results = [r for r in candidates if r[1] not in seen_urls][:RESULTS_PER_TIER]
            seen_urls.update(url for _, url, _ in results)
            lines = [heading]
            for title, url, content in results:
                lines.append(f"[R{next_id}] Title: {title}\nURL: {url}\nSnippet: {clean_snippet(content)}")
                next_id += 1
            sections.append("\n".join(lines) if results else f"{heading}\nNo results.")
        return "\n\n".join(sections)

    suggest_recipes.handle_tool_error = True
    return suggest_recipes
