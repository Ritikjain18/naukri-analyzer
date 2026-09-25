import json
import re

REDACTED = "[REDACTED]"

# Patterns for various secret formats
GSK_PATTERN = re.compile(r'gsk_[A-Za-z0-9_-]{8,}')
SK_PATTERN = re.compile(r'sk-[A-Za-z0-9_-]{16,}')

# Credential patterns.
#
# One pass handles `key=value`, `key: value` and JSON-style `"key": value`. The value is:
#   * double-quoted:  "(?:\\[^\n] | [^"\\\n] | "(?!DELIM))* \\? "?
#   * single-quoted:  the same with '
#   * unquoted:       \S+ (first whitespace-delimited word only, by design)
#
# Quoting rule: backslash escapes stay inside the value; a matching quote ends the value
# only when followed by whitespace, one of , ; } ) ] or end of text, otherwise it is part of
# the value; an unterminated value runs to the end of the line. The whole value is replaced
# by the marker wrapped in the original opening quote: password="..." -> password="[REDACTED]".
# Every loop alternative starts with a distinct character class (backslash / quote / other),
# so matching is linear-time with no catastrophic backtracking.
_CRED_KEY = r'(?:password|passwd|pwd|token|secret|api[_-]?key)'
_QUOTE_END = r'(?=[\s,;})\]]|$)'


def _quoted(q: str) -> str:
    return rf'{q}(?:\\[^\n]|[^{q}\\\n]|{q}(?!{_QUOTE_END}))*\\?{q}?'


CREDENTIAL_PATTERN = re.compile(
    rf'(?i)(?P<prefix>{_CRED_KEY}["\']?\s*[=:]\s*)'
    rf'(?:(?P<quoted>{_quoted(chr(34))}|{_quoted(chr(39))})'
    # Unquoted: skip a value that is EXACTLY the marker (idempotence), but not [REDACTED]x.
    rf'|(?!\[REDACTED\](?=[\s,;"\'}}\]]|$))\S+)'
)

# Bearer/Basic auth: scheme matched case-insensitively, original spelling kept in the output.
BEARER_PATTERN = re.compile(r'(?i)(bearer)\s+[A-Za-z0-9._-]{16,}')
# Basic: redacted after an `authorization:` prefix, or when the token has base64 shape (>= 16 chars with a
# digit, + / or =); plain HR prose such as "basic qualifications" is left alone.
BASIC_AUTH_PATTERN = re.compile(r'(?i)(\bauthorization\s*[:=]\s*basic)\s+[A-Za-z0-9+/=]{12,}')
BASIC_PATTERN = re.compile(r'(?i)(\bbasic)\s+(?=[A-Za-z0-9+/=]*[0-9+/=])[A-Za-z0-9+/=]{16,}')

# Secret key pattern for dicts
SECRET_KEY = re.compile(r"pass|secret|token|api_?key|hash", re.I)


def redact_text(text: str) -> str:
    """Redact secrets from text: API keys, credentials, bearer tokens. Idempotent.

    Patterns are designed to be idempotent: reapplying redaction produces the same result.
    """
    if not isinstance(text, str):
        return text

    # Redact gsk_* keys (case matters for this pattern)
    text = GSK_PATTERN.sub(REDACTED, text)

    # Redact sk-* keys
    text = SK_PATTERN.sub(REDACTED, text)

    # Redact credentials (quoted, JSON-style and unquoted) in a single pass
    def redact_credential(match):
        quoted = match.group("quoted")
        if quoted:
            return f"{match.group('prefix')}{quoted[0]}{REDACTED}{quoted[0]}"
        return f"{match.group('prefix')}{REDACTED}"

    text = CREDENTIAL_PATTERN.sub(redact_credential, text)

    # Redact Bearer tokens (preserve case)
    text = BEARER_PATTERN.sub(r'\1 ' + REDACTED, text)

    # Redact Basic auth
    text = BASIC_AUTH_PATTERN.sub(r'\1 ' + REDACTED, text)
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
