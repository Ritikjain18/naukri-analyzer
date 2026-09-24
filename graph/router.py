import re


def _column_matches(column: str, question: str) -> bool:
    if len(column) < 3:
        return False
    pattern = r"\b" + r"[ _]".join(re.escape(part) for part in column.lower().split("_")) + r"\b"
    return re.search(pattern, question) is not None


def choose_tool(question: str, columns: list[str] | None) -> str:
    if not columns:
        return "sql"
    q = question.lower()
    return "pandas" if any(_column_matches(c, q) for c in columns) else "sql"
