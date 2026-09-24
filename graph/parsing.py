import json
import re


def extract_code(text: str) -> str:
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    return body.strip().rstrip(";").strip()


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model output: {exc}") from exc
