"""The agent's LLM: a primary model with an automatic fallback on another provider.

Primary:  MODEL_NAME on OPENAI_API_KEY (OpenAI, or any OpenAI-compatible endpoint set in
          OPENAI_BASE_URL).
Fallback: FALLBACK_MODEL_NAME on NIM_API_KEY / NIM_BASE_URL (NVIDIA NIM), optional.

If a primary call times out or the provider fails (connection error, rate limit, 5xx),
the same call is sent to the fallback, and the fallback stays in use for
FALLBACK_COOLDOWN_SECONDS before the primary is tried again, so one outage doesn't
make every later call wait for the primary to time out. Neither client retries on its
own (max_retries=0): the fallback is the retry, so waits can't stack. Errors that
aren't provider failures (e.g. a rejected request) are raised, not hidden.
"""

import logging
import os
import time

import openai
from langchain_openai import ChatOpenAI

log = logging.getLogger("macro-agent")

PRIMARY_TIMEOUT_SECONDS = 10   # primary calls normally take 1-4 s
FALLBACK_TIMEOUT_SECONDS = 15
FALLBACK_COOLDOWN_SECONDS = 600
PROVIDER_ERRORS = (
    openai.APITimeoutError, openai.APIConnectionError,  # timeout is a subclass; listed for clarity
    openai.RateLimitError, openai.InternalServerError,
)


def make_chat_model(model: str, api_key: str, base_url: str | None, timeout: float) -> ChatOpenAI:
    settings = {
        "timeout": timeout,
        "max_retries": 0,
        # Chat Completions everywhere, so replies are plain text (the Responses API
        # returns a list of reasoning/text blocks).
        "use_responses_api": False,
    }
    if base_url:
        # An OpenAI-compatible endpoint such as NIM: deterministic sampling.
        settings["temperature"] = 0
    else:
        # OpenAI's reasoning models accept only the default temperature, and on Chat
        # Completions they take tools only with reasoning off. Measured: reasoning off
        # was faster on every flow, with the same tool choices (12/12).
        settings["reasoning_effort"] = "none"
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, **settings)


class FailoverModel:
    """Calls the primary model, switching to the fallback when the primary's provider
    fails. Wraps tool-bound models, so it is used exactly like one."""

    def __init__(self, primary, primary_name: str, fallback=None, fallback_name: str = ""):
        self.primary, self.primary_name = primary, primary_name
        self.fallback, self.fallback_name = fallback, fallback_name
        self.primary_down_until = 0.0

    async def ainvoke(self, messages):
        if self.fallback is None or time.monotonic() >= self.primary_down_until:
            try:
                return await self.primary.ainvoke(messages)
            except PROVIDER_ERRORS as e:
                if self.fallback is None:
                    raise
                log.warning("primary model %s failed (%s); using fallback %s for %d s",
                            self.primary_name, type(e).__name__, self.fallback_name,
                            FALLBACK_COOLDOWN_SECONDS)
                self.primary_down_until = time.monotonic() + FALLBACK_COOLDOWN_SECONDS
        return await self.fallback.ainvoke(messages)


def build_llm(tools) -> FailoverModel:
    """The primary model (and fallback, if FALLBACK_MODEL_NAME is set), bound to tools."""
    primary_name = os.environ["MODEL_NAME"]
    primary = make_chat_model(primary_name, os.environ["OPENAI_API_KEY"],
                              os.environ.get("OPENAI_BASE_URL"), PRIMARY_TIMEOUT_SECONDS)
    fallback_name = os.environ.get("FALLBACK_MODEL_NAME", "")
    fallback = None
    if fallback_name:
        fallback = make_chat_model(fallback_name, os.environ["NIM_API_KEY"],
                                   os.environ["NIM_BASE_URL"], FALLBACK_TIMEOUT_SECONDS).bind_tools(tools)
    return FailoverModel(primary.bind_tools(tools), primary_name, fallback, fallback_name)
