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


class PayloadTooLarge(Exception):
    pass


class StatusError(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


def test_friendly_413_by_type_name_and_status():
    assert "too large" in friendly_error(PayloadTooLarge("boom")).lower()
    assert "narrower question" in friendly_error(StatusError("nope", 413))
    assert "too large" in friendly_error(ValueError("Please reduce your message size")).lower()
    assert "too large" in friendly_error(ValueError("Request too large for model")).lower()


def test_friendly_status_code_beats_substring():
    assert "rate limit" in friendly_error(StatusError("mentions 401 in body", 429)).lower()


def test_generic_error_redacts_api_key():
    out = friendly_error(ValueError("bad header gsk_abc123-XYZ_9 rejected"))
    assert "gsk_abc123" not in out and "[redacted]" in out


def test_get_llm_passes_reasoning_effort_without_network(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    llm = get_llm(config.MODEL_FAST)
    assert llm.model_name == config.MODEL_FAST
    assert config.REASONING_EFFORT == "low"
    assert llm.reasoning_effort == config.REASONING_EFFORT


MODEL_MSG = "The configured Groq model isn't available for this account. Check MODEL_SMART / MODEL_FAST in config.py."


class NotFoundError(Exception):
    pass


def test_friendly_model_not_found_by_type_name():
    assert friendly_error(NotFoundError("Error code: 404")) == MODEL_MSG


def test_friendly_model_not_found_by_status_code():
    assert friendly_error(StatusError("whatever", 404)) == MODEL_MSG


def test_friendly_model_not_found_by_message():
    assert friendly_error(ValueError("model_not_found: nope")) == MODEL_MSG
    assert friendly_error(ValueError("The model `x` does not exist or you do not have access to it.")) == MODEL_MSG


def test_other_errors_containing_keys_still_redacted():
    out = friendly_error(ValueError("failed with gsk_abc123 in header"))
    assert "gsk_abc123" not in out and "[redacted]" in out
