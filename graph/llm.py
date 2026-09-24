import re

from langchain_groq import ChatGroq

from config import OUTPUT_RESERVE, MissingKeyError, get_api_key
from graph.budget import count_tokens


def get_llm(model: str, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=get_api_key(),
        timeout=60,
        max_retries=1,
    )


TOO_LARGE_MSG = "The data slice is too large for the model. Try a narrower question."
DAILY_MSG = "Groq daily token limit reached for today. Try again after 00:00 UTC."


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if isinstance(exc, MissingKeyError):
        return msg
    # Exception type / status code first, substring matching only as a fallback.
    if name in ("PayloadTooLarge", "RequestEntityTooLargeError") or status == 413:
        return TOO_LARGE_MSG
    if isinstance(exc, PromptTooLarge):
        return TOO_LARGE_MSG
    if isinstance(exc, RateLimitExhausted) and exc.daily:
        return DAILY_MSG
    if name in ("RateLimitError", "RateLimitExhausted") or status == 429:
        return "Groq rate limit reached. Wait a minute and try again."
    if name == "AuthenticationError" or status == 401:
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    if name in ("APITimeoutError", "APIConnectionError"):
        return "Could not reach Groq (timeout or network). Try again."
    if "request too large" in low or "reduce your message size" in low or "413" in low:
        return TOO_LARGE_MSG
    if "rate limit" in low or "429" in low:
        return "Groq rate limit reached. Wait a minute and try again."
    if "401" in low or "invalid api key" in low:
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    if "timeout" in low or "timed out" in low:
        return "Could not reach Groq (timeout or network). Try again."
    return f"Something went wrong: {re.sub(r'gsk_[A-Za-z0-9_-]+', '[redacted]', msg)}"


class RateLimitExhausted(RuntimeError):
    """Every model in the chain is at (or near) its rate limit."""

    def __init__(self, message: str = "", daily: bool = False):
        super().__init__(message)
        self.daily = daily


class PromptTooLarge(RateLimitExhausted):
    """The prompt exceeds every model's per-request headroom; waiting will never help."""


def is_rate_limit(exc: Exception) -> bool:
    return (
        type(exc).__name__ == "RateLimitError"
        or getattr(exc, "status_code", None) == 429
        or "rate limit" in str(exc).lower()
    )


def _reported_tokens(response) -> int | None:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage.get("total_tokens"):
        return int(usage["total_tokens"])
    return None


class FallbackLLM:
    """Same `.invoke(prompt).content` interface as a chat model, with rate-limit-aware fallback."""

    def __init__(self, chain, manager):
        self.chain = list(chain)
        self.manager = manager
        self.last_model = None
        self._used: list[str] = []

    def reset(self) -> None:
        self._used = []

    def used_models(self) -> list[str]:
        return list(self._used)

    def invoke(self, prompt):
        estimate = count_tokens(prompt) + OUTPUT_RESERVE
        candidates = [(m, llm) for m, llm in self.chain if self.manager.can_use(m, estimate)]
        if not candidates:
            models = [m for m, _ in self.chain]
            if not any(self.manager.fits_ever(m, estimate) for m in models):
                raise PromptTooLarge("The prompt exceeds every model's per-request token limit.")
            if all(self.manager.daily_blocked(m, estimate) or not self.manager.fits_ever(m, estimate)
                   for m in models):
                raise RateLimitExhausted("Every model in the fallback chain has hit its daily token limit.",
                                         daily=True)
            raise RateLimitExhausted("Every model in the fallback chain is at its rate limit.")
        last_exc = None
        for model, llm in candidates:
            try:
                response = llm.invoke(prompt)
            except Exception as exc:
                if is_rate_limit(exc):
                    last_exc = exc
                    continue
                raise
            self.manager.record(model, _reported_tokens(response) or estimate)
            self.last_model = model
            if model not in self._used:
                self._used.append(model)
            return response
        raise last_exc


def build_llm(models: list[str], manager) -> FallbackLLM:
    return FallbackLLM([(m, get_llm(m)) for m in models], manager)
