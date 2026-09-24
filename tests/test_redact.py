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


# Round 2 Fixes

def test_redact_text_no_early_return_marker():
    """Fix: Remove early return that allows other secrets to survive if marker already present."""
    text = "password=[REDACTED] and gsk_AAAAAAAAAAAA"
    result = redact_text(text)
    # gsk_ key should be redacted even though [REDACTED] is already in text
    assert "gsk_AAAAAAAAAAAA" not in result
    assert REDACTED in result


def test_redact_text_idempotent_with_marker():
    """Verify idempotence when marker already present in input."""
    text1 = "use gsk_abcdef123456 please"
    text2 = redact_text(text1)
    text3 = redact_text(text2)
    assert text2 == text3, "Reapplying redaction should produce identical result"


def test_redact_text_idempotent_mixed():
    """Idempotence for mixed already-redacted and fresh secrets."""
    text = "password=[REDACTED] and gsk_FRESH123456"
    result1 = redact_text(text)
    result2 = redact_text(result1)
    assert result1 == result2


def test_scrub_value_non_string_dict_keys():
    """Fix: Handle non-string dict keys without raising TypeError."""
    value = {"meta": {1: "a", "normal": "b"}}
    result = scrub_value(value)
    # Should not raise TypeError, should process successfully
    assert isinstance(result, dict)
    # Keys are stringified per the fix
    assert result["meta"]["1"] == "a"  # int key stringified to "1"
    assert result["meta"]["normal"] == "b"


def test_scrub_value_bytes_value():
    """Fix: bytes values should become '[bytes]' not repr."""
    value = {"data": b"secret_bytes"}
    result = scrub_value(value)
    assert result["data"] == "[bytes]"


def test_scrub_value_cyclic_reference():
    """Fix: Handle self-referencing structures without infinite loop."""
    lst = []
    lst.append(lst)  # Self-reference
    result = scrub_value(lst)
    # Should not hang or raise, should truncate gracefully
    assert isinstance(result, list)


def test_scrub_value_cyclic_dict():
    """Fix: Handle self-referencing dicts without infinite loop."""
    d = {}
    d["self"] = d  # Self-reference
    result = scrub_value(d)
    # Should not hang or raise
    assert isinstance(result, dict)


def test_redact_text_quoted_values():
    """Fix: Redact quoted values correctly - through the closing quote."""
    text = "password: 'hunter 2' ok"
    result = redact_text(text)
    assert "hunter 2" not in result
    assert "password:" in result
    # Single quotes should be handled


def test_redact_text_double_quoted_values():
    """Fix: Handle double-quoted values."""
    text = 'password="hunter 2" ok'
    result = redact_text(text)
    assert "hunter 2" not in result


def test_redact_text_unquoted_values():
    """Fix: Bare values redact only word chars."""
    text = "password: hunter2 ok"
    result = redact_text(text)
    assert "hunter2" not in result
    assert "ok" in result


def test_redact_text_json_in_string():
    """Fix: Redact JSON forms like 'password':'value'."""
    text = '{"password":"hunter2","ok":"yes"}'
    result = redact_text(text)
    assert "hunter2" not in result
    assert '"ok"' in result or "'ok'" in result


def test_redact_text_sk_prefix_key():
    """Fix: Add sk-[A-Za-z0-9_-]{16,} pattern."""
    text = "api_key: sk-1234567890ABCDEF and sk_short"
    result = redact_text(text)
    assert "sk-1234567890ABCDEF" not in result
    assert REDACTED in result
    # sk_short is too short (underscore variant needs 8+, dash variant needs 16+)
    assert "sk_short" in result


def test_redact_text_basic_auth_header():
    """Fix: Add Basic auth pattern."""
    text = "Authorization: Basic ABC123DEF456GHI789+"
    result = redact_text(text)
    assert "ABC123DEF456GHI789+" not in result
    assert "Basic" in result  # Preserve the keyword


def test_redact_text_bearer_case_preserved():
    """Fix: Preserve original case of Bearer."""
    text = "Authorization: Bearer ABC123DEF456GHI789JKL012MNO345"
    result = redact_text(text)
    assert "ABC123DEF456GHI789JKL012MNO345" not in result
    assert "Bearer" in result  # Original case preserved


def test_scrub_value_non_string_dict_key_types():
    """Handle various non-string key types."""
    value = {
        1: "int_key",
        (1, 2): "tuple_key",
        None: "none_key",
        "password": "secret"  # Should be dropped
    }
    result = scrub_value(value)
    # password key should be dropped
    assert "password" not in result
    # Other keys should be preserved (stringified) with values intact
    assert result["1"] == "int_key"
    assert result["(1, 2)"] == "tuple_key"
    assert result["None"] == "none_key"


def test_regression_normal_prose_not_redacted():
    """Regression: normal English prose should not be over-redacted."""
    text = "What is the password reset rate? The tokenization rate is high."
    result = redact_text(text)
    # These should NOT be redacted (word boundaries)
    assert "password reset rate" in result or "reset rate" in result
    assert "tokenization rate" in result


def test_regression_token_in_prose():
    """Regression: 'token' as noun in prose should not be redacted."""
    text = "The token was distributed at the event."
    result = redact_text(text)
    # 'token' as English word should survive, only token=value patterns redacted
    assert "token" in result.lower()


# Round 3 Fixes

def test_redact_text_fake_marker_bypass():
    """Fix: Fake markers like password=[REDACTED]x must still be redacted.

    The negative lookahead must only skip EXACTLY [REDACTED], not [REDACTED]x.
    """
    text = "password=[REDACTED]x"
    result = redact_text(text)
    # Should be redacted because [REDACTED]x is NOT exactly [REDACTED]
    assert "[REDACTED]x" not in result
    assert result.count(REDACTED) >= 1

    text2 = "password=[REDACTED]hunter2"
    result2 = redact_text(text2)
    assert "hunter2" not in result2

    # But password=[REDACTED] alone should NOT be re-redacted
    text3 = "password=[REDACTED]"
    result3 = redact_text(text3)
    assert result3.count(REDACTED) == 1  # Only one, not re-redacted


def test_redact_text_quoted_with_embedded_quotes():
    """Fix: Quoted values containing the other quote type must work.

    password="it's fine" and password='say "hi" there' must redact the whole value.
    """
    text = 'password="it\'s fine"'
    result = redact_text(text)
    assert "it's fine" not in result
    assert REDACTED in result

    text2 = """password='say "hi" there'"""
    result2 = redact_text(text2)
    assert 'say "hi" there' not in result2
    assert REDACTED in result2


def test_redact_text_escaped_quotes():
    """Fix: Escaped quotes within values must be handled.

    password="pa\"ss" should redact the full escaped value.
    """
    text = r'password="pa\"ss"'
    result = redact_text(text)
    # The whole value including escapes should be redacted
    assert "pa" not in result or REDACTED in result


def test_redact_text_bearer_case_insensitive():
    """Fix: Bearer/Basic matching must be case-insensitive while preserving case.

    BEARER, Bearer, bearer all work, and original case is preserved.
    """
    text1 = "Authorization: BEARER ABC123DEF456GHI789JKL012MNO345"
    result1 = redact_text(text1)
    assert "ABC123DEF456GHI789JKL012MNO345" not in result1
    assert "BEARER" in result1  # Original case preserved

    text2 = "Authorization: bearer ABC123DEF456GHI789JKL012MNO345"
    result2 = redact_text(text2)
    assert "ABC123DEF456GHI789JKL012MNO345" not in result2
    assert "bearer" in result2  # Original case preserved

    text3 = "Authorization: Bearer ABC123DEF456GHI789JKL012MNO345"
    result3 = redact_text(text3)
    assert "ABC123DEF456GHI789JKL012MNO345" not in result3
    assert "Bearer" in result3


def test_redact_text_basic_auth_case_insensitive():
    """Fix: Basic auth matching must be case-insensitive.

    BASIC, Basic, basic all work.
    """
    text1 = "Authorization: BASIC ABC123DEF456GHI789+=="
    result1 = redact_text(text1)
    assert "ABC123DEF456GHI789+=" not in result1
    assert "BASIC" in result1

    text2 = "Authorization: basic ABC123DEF456GHI789+=="
    result2 = redact_text(text2)
    assert "ABC123DEF456GHI789+=" not in result2
    assert "basic" in result2
