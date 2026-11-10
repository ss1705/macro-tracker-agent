import asyncio

import httpx
import openai
import pytest

import llm
from llm import FailoverModel, make_chat_model

REQUEST = httpx.Request("POST", "https://example.invalid/v1/chat/completions")


class FakeModel:
    """Stands in for a tool-bound chat model: returns a reply or raises."""

    def __init__(self, name, error=None):
        self.name, self.error, self.calls = name, error, 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return f"reply from {self.name}"


def run(model):
    return asyncio.run(model.ainvoke(["hi"]))


def test_primary_answers_when_healthy():
    primary, fallback = FakeModel("primary"), FakeModel("fallback")
    assert run(FailoverModel(primary, "p", fallback, "f")) == "reply from primary"
    assert fallback.calls == 0


@pytest.mark.parametrize("error", [
    openai.APITimeoutError(request=REQUEST),
    openai.APIConnectionError(request=REQUEST),
    openai.RateLimitError("429", response=httpx.Response(429, request=REQUEST), body=None),
    openai.InternalServerError("500", response=httpx.Response(500, request=REQUEST), body=None),
])
def test_provider_failure_switches_to_the_fallback(error):
    primary, fallback = FakeModel("primary", error), FakeModel("fallback")
    assert run(FailoverModel(primary, "p", fallback, "f")) == "reply from fallback"


def test_fallback_stays_in_use_during_the_cooldown_then_primary_is_retried():
    primary = FakeModel("primary", openai.APITimeoutError(request=REQUEST))
    fallback = FakeModel("fallback")
    model = FailoverModel(primary, "p", fallback, "f")
    run(model)
    run(model)                      # within the cooldown: primary isn't tried again
    assert (primary.calls, fallback.calls) == (1, 2)

    model.primary_down_until = 0.0  # cooldown over
    primary.error = None
    assert run(model) == "reply from primary"


def test_non_provider_errors_are_not_hidden():
    bad_request = openai.BadRequestError("400", response=httpx.Response(400, request=REQUEST), body=None)
    primary, fallback = FakeModel("primary", bad_request), FakeModel("fallback")
    with pytest.raises(openai.BadRequestError):
        run(FailoverModel(primary, "p", fallback, "f"))
    assert fallback.calls == 0


def test_without_a_fallback_the_provider_error_is_raised():
    primary = FakeModel("primary", openai.APITimeoutError(request=REQUEST))
    with pytest.raises(openai.APITimeoutError):
        run(FailoverModel(primary, "p"))


def test_model_settings_per_provider():
    openai_model = make_chat_model("gpt-6-sol", "k", None, llm.PRIMARY_TIMEOUT_SECONDS)
    assert openai_model.reasoning_effort == "none" and openai_model.temperature is None
    assert openai_model.max_retries == 0 and openai_model.use_responses_api is False

    nim_model = make_chat_model("nvidia/x", "k", "https://integrate.api.nvidia.com/v1",
                                llm.FALLBACK_TIMEOUT_SECONDS)
    assert nim_model.temperature == 0 and nim_model.reasoning_effort is None
    assert nim_model.max_retries == 0
