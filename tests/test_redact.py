import pytest

from accounts.redact import REDACTED, redact_text, scrub_value


def test_redact_text_gsk_key():
    text = "use gsk_abcdef123456 and gsk_xyz-_AB9"
    result = redact_text(text)
    assert "gsk_" not in result and REDACTED in result
    assert result.count(REDACTED) == 2


def test_redact_text_password_patterns():
    text = "password=hunter2 and passwd:secret123 and pwd = pass123"
    result = redact_text(text)
    assert "hunter2" not in result and "secret123" not in result and "pass123" not in result
    assert "password=" in result and "passwd:" in result and "pwd =" in result
    assert result.count(REDACTED) == 3


def test_redact_text_bearer_token():
    text = "Bearer ABC123DEF456GHI789JKL012 is the token"
    result = redact_text(text)
    assert "ABC123DEF456GHI789JKL012" not in result
    assert REDACTED in result


def test_redact_text_api_key_formats():
    text = "api_key=sk_xyz123 and api-key: gsk_abc or apikey=test"
    result = redact_text(text)
    assert "sk_xyz123" not in result and "gsk_abc" not in result
    assert "api_key=" in result and "api-key:" in result
    # apikey (without underscore/dash) is not matched, only api_key/api-key
    assert "apikey=" in result


def test_redact_text_idempotent():
    text = "use gsk_abcdef123456 please"
    result1 = redact_text(text)
    result2 = redact_text(result1)
    assert result1 == result2


def test_redact_text_case_insensitive():
    text = "PASSWORD=secret AND Secret=value Token=abc"
    result = redact_text(text)
    assert "secret" not in result.lower() or REDACTED in result
    assert result.count(REDACTED) >= 1


def test_scrub_value_string_redacted():
    value = "use gsk_abcdef123456 and password=hunter2"
    result = scrub_value(value)
    assert isinstance(result, str)
    assert "gsk_" not in result and "hunter2" not in result
    assert REDACTED in result


def test_scrub_value_string_truncated():
    value = "x" * 5000
    result = scrub_value(value)
    assert len(result) == 2000


def test_scrub_value_dict_drop_secrets_top_level():
    value = {"password": "hunter2", "ok": "yes", "api_key": "sk_xyz"}
    result = scrub_value(value)
    assert "password" not in result and "api_key" not in result
    assert result["ok"] == "yes"


def test_scrub_value_nested_dict_drop_secrets_at_depth():
    value = {"payload": {"password": "x", "ok": 1}, "lst": [{"token": "t", "n": 2}]}
    result = scrub_value(value)
    assert "password" not in result.get("payload", {})
    assert result["payload"]["ok"] == 1
    assert "token" not in result["lst"][0]
    assert result["lst"][0]["n"] == 2


def test_scrub_value_list_truncated():
    value = list(range(100000))
    result = scrub_value(value)
    assert len(result) == 50


def test_scrub_value_max_items_dict():
    value = {f"key{i}": i for i in range(100)}
    result = scrub_value(value)
    assert len(result) == 50


def test_scrub_value_max_depth():
    value = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
    result = scrub_value(value, max_depth=3)
    assert result["a"]["b"]["c"] == "[TRUNCATED]"


def test_scrub_value_object_stringified():
    obj = object()
    result = scrub_value(obj)
    assert isinstance(result, str) and "object at" in result


def test_scrub_value_numbers_and_bool_unchanged():
    assert scrub_value(42) == 42
    assert scrub_value(3.14) == 3.14
    assert scrub_value(True) is True
    assert scrub_value(None) is None


def test_scrub_value_tuple_and_set_to_list():
    result_tuple = scrub_value((1, 2, 3))
    assert result_tuple == [1, 2, 3]
    result_set = scrub_value({1, 2, 3})
    assert isinstance(result_set, list) and len(result_set) == 3
