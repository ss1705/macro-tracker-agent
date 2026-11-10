import os

os.environ.setdefault("MODEL_NAME", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from agent import MAX_HISTORY_MESSAGES, current_turn, llm_view  # noqa: E402


def tool_turn(i: int) -> list:
    return [
        HumanMessage(f"q{i}"),
        AIMessage("", tool_calls=[{"name": "check_goals", "args": {}, "id": f"c{i}"}]),
        ToolMessage('{"remaining": {}}', tool_call_id=f"c{i}", name="check_goals"),
        AIMessage(f"a{i}"),
    ]


def test_llm_view_keeps_past_questions_and_answers_but_drops_old_tool_traffic():
    history = tool_turn(1) + tool_turn(2)
    current = [HumanMessage("q3"), AIMessage("", tool_calls=[{"name": "log_food", "args": {}, "id": "c3"}]),
               ToolMessage("logged", tool_call_id="c3", name="log_food")]
    view = llm_view(history + current)
    assert [m.content for m in view[:4]] == ["q1", "a1", "q2", "a2"]
    assert view[4:] == current  # current turn kept whole, tool call + result paired


def test_llm_view_caps_past_history():
    history = [m for i in range(30) for m in tool_turn(i)]
    view = llm_view(history + [HumanMessage("now")])
    assert len(view) == MAX_HISTORY_MESSAGES + 1
    assert view[-1].content == "now"


def test_current_turn_is_everything_after_the_last_user_message():
    msgs = tool_turn(1) + tool_turn(2)
    assert current_turn(msgs) == msgs[-3:]
