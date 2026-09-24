import json
import re

REDACTED = "[REDACTED]"

# Patterns for various secret formats
GSK_PATTERN = re.compile(r'gsk_[A-Za-z0-9_-]{8,}')
# Captures: group 1 = key name, group 2 = spaces before delimiter, group 3 = delimiter (= or :), group 4 = spaces after delimiter
CREDENTIAL_PATTERN = re.compile(r'(?i)(password|passwd|pwd|token|secret|api[_-]?key)(\s*)([=:])(\s*)(\S+)')
BEARER_PATTERN = re.compile(r'(?i)bearer\s+[A-Za-z0-9._-]{16,}')

# Secret key pattern for dicts
SECRET_KEY = re.compile(r"pass|secret|token|api_?key|hash", re.I)


def redact_text(text: str) -> str:
    """Redact secrets from text: API keys, credentials, bearer tokens. Idempotent."""
    if not isinstance(text, str):
        return text

    # Already redacted, return as-is
    if REDACTED in text:
        return text

    # Redact gsk_* keys
    text = GSK_PATTERN.sub(REDACTED, text)

    # Redact credentials: password=xxx, token:yyy, etc.
    # Keep the key name and spacing/delimiter, replace the value
    def redact_credential(match):
        key = match.group(1)
        space_before = match.group(2)
        delimiter = match.group(3)
        space_after = match.group(4)
        return f"{key}{space_before}{delimiter}{space_after}{REDACTED}"

    text = CREDENTIAL_PATTERN.sub(redact_credential, text)

    # Redact bearer tokens
    text = BEARER_PATTERN.sub(f"bearer {REDACTED}", text)

    return text


def scrub_value(value, max_str=2000, max_items=50, max_depth=4, _depth=0):
    """Recursively scrub values: redact text, drop secret keys from dicts, truncate.

    Beyond max_depth, return "[TRUNCATED]".
    """
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

    # Handle strings: redact and truncate
    if isinstance(value, str):
        redacted = redact_text(value)
        return redacted[:max_str]

    # Handle dicts: drop secret keys at this level, recursively process values
    if isinstance(value, dict):
        result = {}
        for key, val in value.items():
            # Drop secret keys at every level
            if SECRET_KEY.search(key):
                continue
            result[key] = scrub_value(val, max_str, max_items, max_depth, _depth + 1)
            if len(result) >= max_items:
                break
        return result

    # Handle lists, tuples, sets
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            result.append(scrub_value(item, max_str, max_items, max_depth, _depth + 1))
            if len(result) >= max_items:
                break
        return result

    # Handle other objects: stringify then process as string
    stringified = str(value)[:max_str]
    return redact_text(stringified)
