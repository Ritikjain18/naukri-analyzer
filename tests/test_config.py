import pytest

import config


def test_get_api_key_missing_raises(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(config.MissingKeyError) as exc:
        config.get_api_key()
    assert ".env" in str(exc.value)


def test_get_api_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", " gsk_test ")
    assert config.get_api_key() == "gsk_test"


def test_constants():
    assert config.MODEL_FAST == "llama-3.1-8b-instant"
    assert config.MODEL_SMART == "llama-3.3-70b-versatile"
    assert config.ROW_CAP == 200
    assert config.HISTORY_TURNS == 6
