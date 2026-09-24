import json
import re


_LANGS = r"(?:sql|sqlite|python|py|pandas)"


def _strip_fence(text: str) -> str:
    m = re.search(r"```\w*[ \t]*\n(.*?)(?:```|\Z)", text, re.S)  # multi-line, closed or unclosed
    if m:
        return m.group(1)
    m = re.search(r"```(.*?)```", text, re.S)  # single-line
    if m:
        return re.sub(rf"^\s*{_LANGS}\s+", "", m.group(1), flags=re.I)
    return text.replace("```", "")


def extract_code(text: str) -> str:
    return _strip_fence(text).strip().rstrip(";").strip()


def extract_sql(text: str) -> str:
    body = _strip_fence(text)
    m = re.search(r"\b(with|select)\b", body, re.I)
    if m:
        body = body[m.start():]
    return body.strip().rstrip(";").strip()


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model output: {exc}") from exc
