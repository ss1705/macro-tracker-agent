"""Shared helpers for talking to MCP servers through langchain-mcp-adapters."""

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import timedelta

from langchain_core.tools import BaseTool, ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

log = logging.getLogger("macro-agent")

MCP_CALL_TIMEOUT = timedelta(seconds=20)


class PersistentSession:
    """One long-running MCP server process and session, reused for every call.

    Starting a server per call (the adapter's default) costs a process launch and an
    MCP handshake each time: on Cloud Run's single vCPU, four parallel USDA lookups
    spent ~12 s just starting Python. The server is started on first use and kept
    running. If a call fails at the transport level (the server crashed, timed out or
    sent something invalid), that process is stopped and the next call starts a fresh
    one. Tool execution errors are normal results and don't restart anything.

    Passed to load_mcp_tools in place of a ClientSession; the adapter only calls
    list_tools() and call_tool() on it, so the tools stay valid across restarts.
    """

    def __init__(self, client: MultiServerMCPClient, name: str):
        self.client, self.name = client, name
        self.session = None
        self.stop = None
        self.task = None
        self.lock = asyncio.Lock()  # one server start at a time
        self.stopping: set[asyncio.Task] = set()  # failed servers still shutting down

    async def connect(self):
        async with self.lock:
            if self.session is not None and self.task.done():  # server exited on its own
                await self._shutdown()
            if self.session is None:
                ready = asyncio.get_running_loop().create_future()
                self.stop = asyncio.Event()
                self.task = asyncio.create_task(self._run(ready, self.stop))
                self.session = await ready  # raises if the server couldn't start
                log.info("MCP server %s started", self.name)
            return self.session

    async def _run(self, ready: asyncio.Future, stop: asyncio.Event) -> None:
        # The session is opened and closed in this one task: the MCP SDK's stdio
        # transport must be exited from the task that entered it.
        try:
            async with self.client.session(self.name) as session:
                ready.set_result(session)
                await stop.wait()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)

    async def restart_after(self, session) -> None:
        """Stop the server behind a failed session, unless it was already replaced."""
        async with self.lock:
            if self.session is session:
                log.warning("MCP server %s stopped after a failed call; next call restarts it",
                            self.name)
                # Don't make the failed call wait for a stuck process to exit (the SDK
                # gives it a couple of seconds before killing it).
                await self._shutdown(wait=False)

    async def _shutdown(self, wait: bool = True) -> None:
        self.session = None
        if self.task:
            self.stop.set()
            if wait:
                try:
                    await asyncio.wait_for(self.task, 5)
                except BaseException:  # a stuck server: give up on a clean exit
                    self.task.cancel()
            else:
                self.stopping.add(self.task)
                self.task.add_done_callback(self.stopping.discard)
            self.task = None

    async def close(self) -> None:
        async with self.lock:
            await self._shutdown()
        if self.stopping:
            await asyncio.wait(self.stopping, timeout=5)

    async def list_tools(self, **kwargs):
        session = await self.connect()
        try:
            return await session.list_tools(**kwargs)
        except Exception:
            await self.restart_after(session)
            raise

    async def call_tool(self, *args, **kwargs):
        session = await self.connect()
        try:
            return await session.call_tool(*args, **kwargs)
        except Exception:
            await self.restart_after(session)
            raise

    async def load_tools(self) -> list[BaseTool]:
        """tools/list on this server, as LangChain tools bound to this session."""
        return await load_mcp_tools(self, server_name=self.name)


def load_server_config(path: str) -> dict:
    """Read the MCP server manifest. ${VAR} references are filled from the environment
    (unset ones become empty) so API keys live in .env / Cloud Run env vars, never in
    the committed file. "python" means this app's own interpreter (venv or container)."""
    with open(path) as f:
        servers = json.load(f)
    for server in servers.values():
        # Without a read timeout, a server that hangs or answers with garbage leaves
        # the request waiting forever.
        server["session_kwargs"] = {"read_timeout_seconds": MCP_CALL_TIMEOUT}
        if server.get("command") == "python":
            server["command"] = sys.executable
        if "env" in server:
            server["env"] = {
                k: re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), v)
                for k, v in server["env"].items()
            }
    return servers


async def call_mcp_tool(server: str, tool: BaseTool, **args) -> str:
    """Run one MCP tools/call and return its text; raise ToolException if the
    server reports the call as failed."""
    shown = {k: v for k, v in args.items() if k != "content"}  # skip full file bodies
    log.info("MCP tools/call %s.%s %s", server, tool.name, shown)
    start = time.perf_counter()
    try:
        msg = await tool.ainvoke(
            {"type": "tool_call", "id": str(uuid.uuid4()), "name": tool.name, "args": args}
        )
    except Exception as e:  # transport closed, timeout, protocol violation, ...
        log.exception("MCP tools/call %s.%s failed", server, tool.name)
        raise ToolException(f"The {server} server {failure_reason(e)}.") from e
    log.info("MCP tools/call %s.%s done in %.1fs", server, tool.name, time.perf_counter() - start)
    content = msg.content
    text = (
        "".join(block.get("text", "") for block in content if isinstance(block, dict))
        if isinstance(content, list)
        else str(content)
    )
    if msg.status == "error":
        # INFO, not WARNING: callers decide whether it matters (a missing session
        # file on first use is expected; a rejected Tavily key is not).
        log.info("MCP tools/call %s.%s returned an error: %s", server, tool.name, text[:300])
        raise ToolException(text)
    return text


def failure_reason(error: Exception) -> str:
    """Why an MCP call failed, in words the LLM can pass on to the user."""
    if isinstance(error, ValueError):  # includes pydantic ValidationError
        return "sent an invalid response"
    return "couldn't be reached (the connection failed or timed out)"
