MARKER = "--- model output ---\n"


def generated_sql(prompts) -> str:
    """SQL text the model produced in the last retrieve-sql prompt entry, or '' when unavailable."""
    for entry in reversed(prompts or []):
        if entry.get("node") == "retrieve-sql" and MARKER in entry.get("prompt", ""):
            return entry["prompt"].split(MARKER, 1)[1]
    return ""
