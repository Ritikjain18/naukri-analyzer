def choose_tool(question: str, columns: list[str] | None) -> str:
    if not columns:
        return "sql"
    q = question.lower()
    for col in columns:
        c = col.lower()
        if c in q or c.replace("_", " ") in q:
            return "pandas"
    return "sql"
