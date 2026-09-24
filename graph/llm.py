import re

from langchain_groq import ChatGroq

from config import MissingKeyError, get_api_key


def get_llm(model: str, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=get_api_key(),
        timeout=60,
        max_retries=1,
    )


TOO_LARGE_MSG = "The data slice is too large for the model. Try a narrower question."


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
    if name == "RateLimitError" or status == 429:
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
