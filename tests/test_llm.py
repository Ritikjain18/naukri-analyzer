import config
from graph.llm import friendly_error, get_llm


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


def test_friendly_rate_limit():
    assert "rate limit" in friendly_error(RateLimitError("429 Too Many Requests")).lower()


def test_friendly_auth():
    assert "GROQ_API_KEY" in friendly_error(AuthenticationError("401 invalid api key"))


def test_friendly_timeout():
    assert "reach Groq" in friendly_error(TimeoutError("request timed out"))


def test_friendly_missing_key():
    msg = "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
    assert friendly_error(config.MissingKeyError(msg)) == msg


def test_friendly_generic():
    assert friendly_error(ValueError("boom")) == "Something went wrong: boom"


def test_get_llm_uses_requested_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    llm = get_llm(config.MODEL_SMART)
    assert llm.model_name == config.MODEL_SMART
