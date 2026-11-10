import os

import pytest
from pydantic import ValidationError

os.environ.setdefault("MODEL_NAME", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")

from main import ChatRequest  # noqa: E402


@pytest.mark.parametrize("body", [
    {"query": "", "session_id": "s1"},       # reached the LLM and stalled for 80 s
    {"query": "   ", "session_id": "s1"},
    {"query": "hi", "session_id": ""},       # every "" caller would share one session
    {"query": "hi", "session_id": "  "},
    {"query": "hi"},
])
def test_empty_or_missing_fields_are_rejected(body):
    with pytest.raises(ValidationError):
        ChatRequest(**body)


def test_valid_request_is_accepted_and_trimmed():
    req = ChatRequest(query="  How much protein is in an egg? ", session_id=" test-1 ")
    assert req.query == "How much protein is in an egg?"
    assert req.session_id == "test-1"
