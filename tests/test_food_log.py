"""Integration tests against the real filesystem MCP server (needs node_modules)."""

import asyncio
import json
from datetime import datetime, timedelta

import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

from food_log import LOCAL_TZ, make_food_tools
from mcp_utils import PersistentSession


async def food_tools(tmp_path):
    """log_food, check_goals, set_goals, delete_food_entry over a filesystem server
    sandboxed to tmp_path."""
    client = MultiServerMCPClient({"filesystem": {
        "transport": "stdio",
        "command": "node_modules/.bin/mcp-server-filesystem",
        "args": [str(tmp_path)],
    }})
    # The app's code path: one long-running server for every call.
    fs_tools = {t.name: t for t in await PersistentSession(client, "filesystem").load_tools()}
    return make_food_tools(fs_tools, str(tmp_path))


def session(sid: str) -> dict:
    return {"configurable": {"thread_id": sid}}


def test_concurrent_log_food_calls_keep_every_entry_with_unique_ids(tmp_path):
    # Several foods in one message become parallel log_food calls; without the
    # per-session lock, each read-modify-write overwrote the others (4 in -> 1 saved).
    async def run():
        log_food, check_goals, _, _ = await food_tools(tmp_path)
        items = ["apple", "almonds", "yogurt", "granola"]
        await asyncio.gather(*(
            log_food.ainvoke({"item": i, "quantity": "1", "calories": 100}, config=session("race"))
            for i in items
        ))
        return items, json.loads(await check_goals.ainvoke({}, config=session("race")))

    items, summary = asyncio.run(run())
    assert sorted(e["item"] for e in summary["entries"]) == sorted(items)
    assert summary["consumed"]["calories"] == 400
    assert len({e["id"] for e in summary["entries"]}) == 4


@pytest.mark.parametrize("content", [
    '{"log": [ {"item": "toast",',                          # not JSON
    '["not", "an", "object"]',                              # JSON, wrong type
    '{"goals": {"daily_calories": 2000}, "log": []}',       # missing goal keys
    '{"goals": {"daily_calories": 1, "daily_protein_g": 1, "daily_carbs_g": 1, '
    '"daily_fat_g": 1}, "log": [{"item": "toast"}]}',       # entry without a timestamp
])
def test_corrupted_session_file_gives_a_clear_error(tmp_path, content):
    (tmp_path / "broken.json").write_text(content)

    async def run():
        _, check_goals, _, _ = await food_tools(tmp_path)
        return await check_goals.ainvoke({}, config=session("broken"))

    assert "corrupted" in asyncio.run(run())


def test_set_goals_updates_only_the_given_goals_and_drives_the_budget(tmp_path):
    async def run():
        log_food, check_goals, set_goals, _ = await food_tools(tmp_path)
        await set_goals.ainvoke({"daily_calories": 1800}, config=session("amy"))
        await log_food.ainvoke({"item": "lunch", "quantity": "1", "calories": 600}, config=session("amy"))
        amy = json.loads(await check_goals.ainvoke({}, config=session("amy")))
        bob = json.loads(await check_goals.ainvoke({}, config=session("bob")))
        return amy, bob

    amy, bob = asyncio.run(run())
    assert amy["goals"]["daily_calories"] == 1800
    assert amy["goals"]["daily_protein_g"] == 150       # untouched default
    assert amy["remaining"]["calories"] == 1200         # 1800 - 600, not 2000 - 600
    assert bob["goals"]["daily_calories"] == 2000       # another session is unaffected


@pytest.mark.parametrize("args, expected", [
    ({"daily_calories": 180}, "daily_calories must be between"),   # typo-sized goal
    ({"daily_protein_g": 1800}, "daily_protein_g must be between"),
    ({}, "No goal given"),
])
def test_set_goals_rejects_bad_values_without_saving(tmp_path, args, expected):
    async def run():
        _, check_goals, set_goals, _ = await food_tools(tmp_path)
        message = await set_goals.ainvoke(args, config=session("amy"))
        goals = json.loads(await check_goals.ainvoke({}, config=session("amy")))["goals"]
        return message, goals

    message, goals = asyncio.run(run())
    assert expected in message
    assert goals["daily_calories"] == 2000 and goals["daily_protein_g"] == 150


def test_correction_deletes_the_wrong_entry_and_logs_the_right_one(tmp_path):
    # The live probe that prompted this: "3 eggs" then "actually 2" left both logged.
    async def run():
        log_food, check_goals, _, delete_food_entry = await food_tools(tmp_path)
        await log_food.ainvoke({"item": "scrambled eggs", "quantity": "3", "calories": 253},
                               config=session("eggs"))
        wrong = json.loads(await check_goals.ainvoke({}, config=session("eggs")))["entries"][0]
        await delete_food_entry.ainvoke({"entry_id": wrong["id"]}, config=session("eggs"))
        await log_food.ainvoke({"item": "scrambled eggs", "quantity": "2", "calories": 168},
                               config=session("eggs"))
        return json.loads(await check_goals.ainvoke({}, config=session("eggs")))

    summary = asyncio.run(run())
    assert [(e["quantity"], e["calories"]) for e in summary["entries"]] == [("2", 168)]
    assert summary["consumed"]["calories"] == 168


def test_deleting_an_unknown_id_is_a_clear_error_and_removes_nothing(tmp_path):
    async def run():
        log_food, check_goals, _, delete_food_entry = await food_tools(tmp_path)
        await log_food.ainvoke({"item": "apple", "quantity": "1", "calories": 95}, config=session("s"))
        message = await delete_food_entry.ainvoke({"entry_id": "nope1234"}, config=session("s"))
        return message, json.loads(await check_goals.ainvoke({}, config=session("s")))

    message, summary = asyncio.run(run())
    assert "No logged entry with id 'nope1234'" in message
    assert len(summary["entries"]) == 1


@pytest.mark.parametrize("named_as", ["yesterday", "date"])
def test_food_logged_for_yesterday_counts_on_yesterday_not_today(tmp_path, named_as):
    yesterday = (datetime.now(LOCAL_TZ).date() - timedelta(days=1)).isoformat()
    day_arg = "yesterday" if named_as == "yesterday" else yesterday

    async def run():
        log_food, check_goals, _, _ = await food_tools(tmp_path)
        await log_food.ainvoke({"item": "pizza", "quantity": "1 large", "calories": 2000,
                                "date": day_arg}, config=session("p"))
        today = json.loads(await check_goals.ainvoke({"date": "today"}, config=session("p")))
        past = json.loads(await check_goals.ainvoke({"date": day_arg}, config=session("p")))
        return today, past

    today, past = asyncio.run(run())
    assert today["consumed"]["calories"] == 0
    assert past["consumed"]["calories"] == 2000 and past["date"] == yesterday


def test_logging_for_a_future_date_is_rejected(tmp_path):
    tomorrow = (datetime.now(LOCAL_TZ).date() + timedelta(days=1)).isoformat()

    async def run():
        log_food, check_goals, _, _ = await food_tools(tmp_path)
        message = await log_food.ainvoke({"item": "cake", "quantity": "1", "calories": 400,
                                          "date": tomorrow}, config=session("f"))
        return message, json.loads(await check_goals.ainvoke({"date": tomorrow}, config=session("f")))

    message, future = asyncio.run(run())
    assert "future" in message
    assert future["entries"] == []


def test_one_server_process_serves_every_call_including_parallel_ones(tmp_path, caplog):
    caplog.set_level("INFO", logger="macro-agent")

    async def run():
        log_food, check_goals, _, _ = await food_tools(tmp_path)
        await asyncio.gather(*(
            log_food.ainvoke({"item": i, "quantity": "1", "calories": 50}, config=session("reuse"))
            for i in ["apple", "pear", "plum"]
        ))
        return json.loads(await check_goals.ainvoke({}, config=session("reuse")))

    summary = asyncio.run(run())
    assert summary["consumed"]["calories"] == 150
    starts = [r for r in caplog.records if r.getMessage() == "MCP server filesystem started"]
    assert len(starts) == 1  # tools/list, 3 parallel log_food (6 calls) and check_goals
