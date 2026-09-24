import json
import re

REDACTED = "[REDACTED]"

# Patterns for various secret formats
GSK_PATTERN = re.compile(r'gsk_[A-Za-z0-9_-]{8,}')
SK_PATTERN = re.compile(r'sk-[A-Za-z0-9_-]{16,}')

# Credential patterns for quoted and unquoted values
# Designed to be idempotent: already-redacted values won't be matched again
QUOTED_CREDENTIAL_PATTERN = re.compile(
    r'(?i)(password|passwd|pwd|token|secret|api[_-]?key)(\s*)([=:])(\s*)'
    r'([\"\'])([^\'\"]*?)\5'  # Quoted values: 'value' or "value"
)
# Unquoted: NOT [REDACTED] exactly
# Skip matching only if value is EXACTLY [REDACTED] followed by delimiter/space/end
UNQUOTED_CREDENTIAL_PATTERN = re.compile(
    r'(?i)(password|passwd|pwd|token|secret|api[_-]?key)(\s*)([=:])(\s*)'
    r'(?!\[REDACTED\](?=[\s,;"\'\}\]$]|$))(\S+)'
)
# JSON-style credentials: "password":"value" or "password": 'value'
JSON_CREDENTIAL_PATTERN = re.compile(
    r'([\"\'])(password|passwd|pwd|token|secret|api[_-]?key)\1\s*:\s*'
    r'([\"\'])([^\'\"]*?)\3'
)

# Bearer/Basic auth patterns - case-insensitive but preserve original case
BEARER_PATTERN = re.compile(r'(Bearer|bearer|BEARER)\s+[A-Za-z0-9._-]{16,}')
BASIC_PATTERN = re.compile(r'(Basic|basic|BASIC)\s+[A-Za-z0-9+/=]{12,}')

# Secret key pattern for dicts
SECRET_KEY = re.compile(r"pass|secret|token|api_?key|hash", re.I)


def redact_text(text: str) -> str:
    """Redact secrets from text: API keys, credentials, bearer tokens. Idempotent.

    Patterns are designed to be idempotent: reapplying redaction produces the same result.
    """
    if not isinstance(text, str):
        return text

    original = text

    # Redact gsk_* keys (case matters for this pattern)
    text = GSK_PATTERN.sub(REDACTED, text)

    # Redact sk-* keys
    text = SK_PATTERN.sub(REDACTED, text)

    # Redact quoted credentials: password: 'value', password="value"
    def redact_quoted_credential(match):
        key = match.group(1)
        space_before = match.group(2)
        delimiter = match.group(3)
        space_after = match.group(4)
        quote = match.group(5)
        return f"{key}{space_before}{delimiter}{space_after}{quote}{REDACTED}{quote}"

    text = QUOTED_CREDENTIAL_PATTERN.sub(redact_quoted_credential, text)

    # Redact unquoted credentials: password=value, pwd: test
    def redact_unquoted_credential(match):
        key = match.group(1)
        space_before = match.group(2)
        delimiter = match.group(3)
        space_after = match.group(4)
        return f"{key}{space_before}{delimiter}{space_after}{REDACTED}"

    text = UNQUOTED_CREDENTIAL_PATTERN.sub(redact_unquoted_credential, text)

    # Redact JSON-style credentials: "password":"value"
    def redact_json_credential(match):
        key_quote = match.group(1)
        key = match.group(2)
        value_quote = match.group(3)
        return f"{key_quote}{key}{key_quote}: {value_quote}{REDACTED}{value_quote}"

    text = JSON_CREDENTIAL_PATTERN.sub(redact_json_credential, text)

    # Redact Bearer tokens (preserve case)
    text = BEARER_PATTERN.sub(r'\1 ' + REDACTED, text)

    # Redact Basic auth
    text = BASIC_PATTERN.sub(r'\1 ' + REDACTED, text)

    return text


def scrub_value(value, max_str=2000, max_items=50, max_depth=4, _depth=0, _visited=None):
    """Recursively scrub values: redact text, drop secret keys from dicts, truncate.

    Beyond max_depth, return "[TRUNCATED]".
    Handles cyclic references gracefully.
    """
    if _visited is None:
        _visited = set()

    if _depth >= max_depth:
        return "[TRUNCATED]"

    # Handle None
    if value is None:
        return None

    # Handle booleans
    if isinstance(value, bool):
        return value

    # Handle numbers
    if isinstance(value, (int, float)):
        return value

    # Handle bytes: return "[bytes]" placeholder
    if isinstance(value, bytes):
        return "[bytes]"

    # Handle strings: redact and truncate
    if isinstance(value, str):
        redacted = redact_text(value)
        return redacted[:max_str]

    # Handle dicts: drop secret keys at this level, recursively process values
    if isinstance(value, dict):
        # Check for cycles
        obj_id = id(value)
        if obj_id in _visited:
            return "[TRUNCATED]"
        _visited.add(obj_id)

        result = {}
        for key, val in value.items():
            # Stringify key for checking and output
            str_key = str(key)
            # Drop secret keys at every level
            if SECRET_KEY.search(str_key):
                continue
            result[str_key] = scrub_value(val, max_str, max_items, max_depth, _depth + 1, _visited)
            if len(result) >= max_items:
                break

        _visited.discard(obj_id)
        return result

    # Handle lists, tuples, sets
    if isinstance(value, (list, tuple, set)):
        # Check for cycles
        obj_id = id(value)
        if obj_id in _visited:
            return "[TRUNCATED]"
        _visited.add(obj_id)

        result = []
        for item in value:
            result.append(scrub_value(item, max_str, max_items, max_depth, _depth + 1, _visited))
            if len(result) >= max_items:
                break

        _visited.discard(obj_id)
        return result

    # Handle other objects: stringify then process as string
    # For bytes-like or unusual objects, just return placeholder or stringified
    stringified = str(value)[:max_str]
    return redact_text(stringified)
