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


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    name = type(exc).__name__
    if isinstance(exc, MissingKeyError):
        return msg
    if "rate limit" in low or "429" in low or name == "RateLimitError":
        return "Groq rate limit reached. Wait a minute and try again."
    if "401" in low or "invalid api key" in low or name == "AuthenticationError":
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    if "timeout" in low or "timed out" in low or name in ("APITimeoutError", "APIConnectionError"):
        return "Could not reach Groq (timeout or network). Try again."
    return f"Something went wrong: {msg}"
