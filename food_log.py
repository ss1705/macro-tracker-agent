"""log_food / check_goals / set_goals / delete_food_entry: agent-facing tools backed by the
filesystem MCP server.

Each session's food log is one JSON file, <data dir>/<session>.json. These tools
never touch the disk themselves: every read and write is a real MCP tools/call to
the filesystem server (read_text_file / write_file). The file is picked from the
request's session_id (injected via RunnableConfig), never by the LLM, so one
session cannot read or write another's log.
"""

import asyncio
import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, ToolException, tool

from mcp_utils import call_mcp_tool

LOCAL_TZ = ZoneInfo("America/Los_Angeles")
DEFAULT_GOALS = {
    "daily_calories": 2000,
    "daily_protein_g": 150,
    "daily_carbs_g": 200,
    "daily_fat_g": 65,
}
MACROS = ("calories", "protein_g", "carbs_g", "fat_g")
# Sanity bounds for user-set goals: catches typos like 180 kcal or 1800 g protein.
GOAL_RANGES = {
    "daily_calories": (800, 6000),
    "daily_protein_g": (0, 400),
    "daily_carbs_g": (0, 1000),
    "daily_fat_g": (0, 400),
}
SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def parse_day(value: str) -> date:
    """The day the LLM named: empty or "today", "yesterday", or YYYY-MM-DD, with
    today in local time. The words exist because gpt-6-sol does its own date
    arithmetic from the UTC date: in the evening (Pacific), when UTC is already on the
    next day, it passed today's date for "yesterday" even with the date in the prompt."""
    today = datetime.now(LOCAL_TZ).date()
    word = value.strip().lower()
    if word in ("", "today"):
        return today
    if word == "yesterday":
        return today - timedelta(days=1)
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ToolException(f"Invalid date '{value}'; use YYYY-MM-DD.")


def make_food_tools(fs_tools: dict[str, BaseTool], data_dir: str) -> list[BaseTool]:
    """Build log_food, check_goals, set_goals and delete_food_entry on top of the
    filesystem server's MCP tools."""

    # log_food, set_goals and delete_food_entry are read-modify-write on one file per
    # session. Parallel calls (several foods in one message) would otherwise overwrite
    # each other.
    session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def call_fs(name: str, **args) -> str:
        return await call_mcp_tool("filesystem", fs_tools[name], **args)

    def session_file(config: RunnableConfig) -> tuple[str, str]:
        session_id = config["configurable"]["thread_id"]
        # Keep readable ids as-is; hash anything that isn't a safe filename.
        name = session_id if SAFE_SESSION_ID.fullmatch(session_id) else (
            hashlib.sha256(session_id.encode()).hexdigest()[:32]
        )
        return session_id, os.path.join(data_dir, f"{name}.json")

    async def load(config: RunnableConfig) -> tuple[str, dict]:
        session_id, path = session_file(config)
        try:
            raw = await call_fs("read_text_file", path=path)
        except ToolException as e:
            if "ENOENT" in str(e):  # first log for this session
                return path, {"session_id": session_id, "goals": dict(DEFAULT_GOALS), "log": []}
            raise
        try:
            data = json.loads(raw)
            valid = (isinstance(data.get("goals"), dict)
                     and all(f"daily_{m}" in data["goals"] for m in MACROS)
                     and isinstance(data.get("log"), list)
                     and all(isinstance(e, dict) and "logged_at" in e for e in data["log"]))
        except (json.JSONDecodeError, AttributeError):  # not JSON, or not a JSON object
            valid = False
        if not valid:
            raise ToolException("The food log file is corrupted and couldn't be read.")
        return path, data

    def daily_summary(data: dict, day: date) -> dict:
        entries = [
            e for e in data["log"]
            if datetime.fromisoformat(e["logged_at"]).astimezone(LOCAL_TZ).date() == day
        ]
        consumed = {m: round(sum(e.get(m, 0) for e in entries), 1) for m in MACROS}
        goals = data["goals"]
        return {
            "date": day.isoformat(),
            "goals": goals,
            "consumed": consumed,
            "remaining": {m: round(goals[f"daily_{m}"] - consumed[m], 1) for m in MACROS},
            "entries": [
                {"id": e.get("id", ""),
                 **{k: e[k] for k in ("meal", "item", "quantity", *MACROS)},
                 "source": e.get("source", "estimate")}
                for e in entries
            ],
        }

    async def save(path: str, data: dict, day: date | None = None) -> dict:
        """Write the session file and return that day's summary (today by default)
        without the entry list: the model knows what it just changed."""
        await call_fs("write_file", path=path, content=json.dumps(data, indent=2))
        summary = daily_summary(data, day or datetime.now(LOCAL_TZ).date())
        del summary["entries"]
        return summary

    @tool
    async def log_food(
        item: str,
        quantity: str,
        config: RunnableConfig,
        meal: Literal["breakfast", "lunch", "dinner", "snack"] = "snack",
        calories: float = 0,
        protein_g: float = 0,
        carbs_g: float = 0,
        fat_g: float = 0,
        source: str = "estimate",
        date: str = "",
    ) -> str:
        """Log a food item the user ate, with its calories and macros.

        Log each distinct food as its own call. source is "USDA <fdc_id>" when the
        values come from lookup_nutrition, otherwise "estimate". date is only for food
        eaten on an earlier day: "yesterday", or YYYY-MM-DD for an older day; leave it
        empty for today. The result includes that day's updated totals and remaining budget, so
        there's no need to call check_goals afterwards.
        """
        day, now = parse_day(date), datetime.now(LOCAL_TZ)
        if day > now.date():
            raise ToolException("Can't log food for a future date.")
        # Earlier days get a midday timestamp: the day is what matters for totals.
        logged_at = now if day == now.date() else datetime.combine(day, time(12), LOCAL_TZ)
        entry = {
            "id": uuid.uuid4().hex[:8],
            "logged_at": logged_at.isoformat(timespec="seconds"),
            "meal": meal,
            "item": item,
            "quantity": quantity,
            "calories": calories,
            "protein_g": protein_g,
            "carbs_g": carbs_g,
            "fat_g": fat_g,
            "source": source,
        }
        _, path = session_file(config)
        async with session_locks[path]:
            _, data = await load(config)
            data["log"].append(entry)
            return json.dumps({"logged": entry, "day": await save(path, data, day)})

    @tool
    async def set_goals(
        config: RunnableConfig,
        daily_calories: float | None = None,
        daily_protein_g: float | None = None,
        daily_carbs_g: float | None = None,
        daily_fat_g: float | None = None,
    ) -> str:
        """Set the user's daily calorie and/or macro goals. Pass only the goals the
        user states; the others stay as they are. The result includes today's
        remaining budget under the new goals.
        """
        updates = {name: value for name, value in {
            "daily_calories": daily_calories,
            "daily_protein_g": daily_protein_g,
            "daily_carbs_g": daily_carbs_g,
            "daily_fat_g": daily_fat_g,
        }.items() if value is not None}
        if not updates:
            raise ToolException("No goal given; pass at least one of calories, protein, carbs or fat.")
        out_of_range = [
            f"{name} must be between {low} and {high}"
            for name, value in updates.items()
            for low, high in [GOAL_RANGES[name]]
            if not low <= value <= high
        ]
        if out_of_range:
            raise ToolException("Goal not saved: " + "; ".join(out_of_range) + ".")
        _, path = session_file(config)
        async with session_locks[path]:
            _, data = await load(config)
            data["goals"].update(updates)
            return json.dumps({"updated": updates, "day": await save(path, data)})

    @tool
    async def check_goals(config: RunnableConfig, date: str = "") -> str:
        """Return the user's daily goals, what they've eaten (each entry with its id),
        and their remaining calorie/macro budget.

        Leave date empty for today. Only pass date when the user asks about another
        day: "yesterday", or YYYY-MM-DD for an older day.
        """
        _, data = await load(config)
        return json.dumps(daily_summary(data, parse_day(date)))

    @tool
    async def delete_food_entry(entry_id: str, config: RunnableConfig) -> str:
        """Delete one logged food entry by its id (ids come from check_goals). Use it
        when the user wants an entry removed, and to correct an entry: delete the
        wrong one, then log_food the right one.
        """
        _, path = session_file(config)
        async with session_locks[path]:
            _, data = await load(config)
            match = next((e for e in data["log"] if e.get("id") == entry_id), None)
            if match is None:
                raise ToolException(f"No logged entry with id '{entry_id}'. Call check_goals "
                                    "to see the entries and their ids.")
            data["log"].remove(match)
            day = datetime.fromisoformat(match["logged_at"]).astimezone(LOCAL_TZ).date()
            deleted = {k: match[k] for k in ("id", "meal", "item", "quantity", "calories")}
            return json.dumps({"deleted": deleted, "day": await save(path, data, day)})

    tools = [log_food, check_goals, set_goals, delete_food_entry]
    for t in tools:
        t.handle_tool_error = True  # ToolException -> error message the LLM can explain
    return tools
