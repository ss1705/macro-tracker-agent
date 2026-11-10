"""Failure modes against a deliberately broken MCP server (tests/fixtures/faulty_mcp_server.py).

Each mode must end as an error result the LLM can explain, quickly, never as an
exception that escapes the agent or a request that hangs.
"""

import asyncio
import os
import sys
import time
from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

os.environ.setdefault("MODEL_NAME", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")

from agent import tool_failure_message  # noqa: E402
from mcp_utils import PersistentSession, call_mcp_tool, load_server_config  # noqa: E402

MODES = ["crash", "garbage", "malformed", "hang"]
TEST_TIMEOUT = timedelta(seconds=2)  # the app uses mcp_utils.MCP_CALL_TIMEOUT
ARGS = {"food_name": "apple", "grams": 100}


async def faulty_tool(mode: str):
    client = MultiServerMCPClient({"faulty": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["tests/fixtures/faulty_mcp_server.py", mode],
        "session_kwargs": {"read_timeout_seconds": TEST_TIMEOUT},
    }})
    return (await PersistentSession(client, "faulty").load_tools())[0]


@pytest.mark.parametrize("mode", MODES)
def test_wrapper_path_turns_every_failure_into_a_tool_exception(mode):
    async def run():
        tool = await faulty_tool(mode)
        with pytest.raises(ToolException) as excinfo:
            await call_mcp_tool("nutrition", tool, **ARGS)
        return str(excinfo.value)

    start = time.perf_counter()
    message = asyncio.run(run())
    assert time.perf_counter() - start < 10
    assert message.startswith("The nutrition server ")
    expected = "sent an invalid response" if mode == "malformed" else "couldn't be reached"
    assert expected in message


@pytest.mark.parametrize("mode", MODES)
def test_unwrapped_tool_failure_becomes_an_error_result_in_the_graph(mode):
    async def run():
        tool = await faulty_tool(mode)
        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode([tool], handle_tool_errors=tool_failure_message))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        call = {"name": tool.name, "args": ARGS, "id": "1", "type": "tool_call"}
        out = await graph.compile().ainvoke({"messages": [AIMessage("", tool_calls=[call])]})
        return out["messages"][-1]

    start = time.perf_counter()
    result = asyncio.run(run())
    assert time.perf_counter() - start < 10
    assert result.status == "error"


def test_server_config_sets_a_read_timeout_on_every_server():
    servers = load_server_config("mcp_config.json")
    assert all(s["session_kwargs"]["read_timeout_seconds"].total_seconds() > 0 for s in servers.values())


@pytest.mark.parametrize("mode", MODES)
def test_a_failed_server_is_replaced_by_a_fresh_one_on_the_next_call(mode, caplog):
    # A long-running server that crashed, hung or sent junk must not poison later
    # calls: each call after a failure gets a newly started server.
    caplog.set_level("INFO", logger="macro-agent")

    async def run():
        tool = await faulty_tool(mode)
        for _ in range(2):
            with pytest.raises(ToolException):
                await call_mcp_tool("nutrition", tool, **ARGS)

    start = time.perf_counter()
    asyncio.run(run())
    assert time.perf_counter() - start < 15
    messages = [r.getMessage() for r in caplog.records]
    # Started for tools/list (and reused by the first call), then again for the second.
    assert messages.count("MCP server faulty started") == 2
    assert messages.count(
        "MCP server faulty stopped after a failed call; next call restarts it") == 2
