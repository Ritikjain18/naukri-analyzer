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
    assert config.MODEL_FAST == "openai/gpt-oss-20b"
    assert config.MODEL_SMART == "openai/gpt-oss-120b"
    assert config.ROW_CAP == 200
    assert config.HISTORY_TURNS == 6
