"""search_web: web search for research questions, backed by the Tavily MCP server.

Used when the user explicitly asks to search or look something up, or wants current
information, recent research or sources; general facts are still answered from the
model's own knowledge (see the system prompt). Every search is a real MCP tools/call to
tavily_search. The final answer may only link to URLs these results returned
(strip_unknown_urls), so the model can't cite a source it never saw.
"""

import re

from langchain_core.tools import BaseTool, tool

from mcp_utils import call_mcp_tool
from recipes import EXCLUDED_DOMAINS, clean_snippet, normalize_url, parse_results

MAX_RESULTS = 5
SEARCH_DEPTH = "advanced"  # better-targeted results; 2 Tavily credits per search
SNIPPET_CHARS = 600  # longer than recipe snippets: the answer is written from these
URL_RE = re.compile(r"https?://[^\s)\]>†【】]+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
# Nemotron sometimes cites in its own style, e.g. 【https://…†L3-L9】 or 【2†source】.
BRACKET_CITATION_RE = re.compile(r"【([^】]*)】")


def result_urls(tool_outputs: list[str]) -> set[str]:
    """Normalized URLs of every result in search_web's output."""
    return {normalize_url(u) for out in tool_outputs for u in re.findall(r"(?m)^URL: (\S+)", out)}


def strip_unknown_urls(answer: str, allowed: set[str]) -> tuple[str, int]:
    """Remove links to URLs that weren't in the search results. A markdown link keeps
    its text; a bare URL is dropped; 【…】 citations become plain (URL) links first.
    Returns the text and how many links were removed."""
    removed = 0

    def markdown_link(m: re.Match) -> str:
        nonlocal removed
        if normalize_url(m.group(2)) in allowed:
            return m.group(0)
        removed += 1
        return m.group(1)

    def bare_url(m: re.Match) -> str:
        nonlocal removed
        if normalize_url(m.group(0)) in allowed:
            return m.group(0)
        removed += 1
        return ""

    def bracket_citation(m: re.Match) -> str:
        # A cited result becomes a plain "(URL)"; anything else in the brackets goes.
        nonlocal removed
        url = URL_RE.search(m.group(1))
        if url and normalize_url(url.group(0)) in allowed:
            return f" ({url.group(0)})"
        removed += bool(url)
        return ""

    text = BRACKET_CITATION_RE.sub(bracket_citation, answer)
    text = MARKDOWN_LINK_RE.sub(markdown_link, text)
    return URL_RE.sub(bare_url, text), removed


def make_search_tool(tavily_tools: dict[str, BaseTool]) -> BaseTool:
    search = tavily_tools["tavily_search"]

    @tool
    async def search_web(query: str) -> str:
        """Search the web. Use it only when the user asks you to search or look
        something up, or wants current information, recent research or sources, not
        for general facts you already know. Cite the sources you use by their URL.
        """
        output = await call_mcp_tool(
            "tavily", search, query=query, search_depth=SEARCH_DEPTH,
            max_results=MAX_RESULTS, exclude_domains=EXCLUDED_DOMAINS,
        )
        results = parse_results(output)
        if not results:
            return "No results found."
        return "\n\n".join(
            f"Title: {title}\nURL: {url}\nSnippet: {clean_snippet(content, SNIPPET_CHARS)}"
            for title, url, content in results
        )

    search_web.handle_tool_error = True
    return search_web
