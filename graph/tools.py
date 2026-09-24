import ast
import re

import pandas as pd

from config import ROW_CAP
from graph.parsing import extract_code, extract_sql
from graph.prompts import render


class RetrievalError(RuntimeError):
    pass


def _record(trace: list | None, node: str, prompt: str, output: str) -> None:
    if trace is not None:
        trace.append({"node": node, "prompt": f"{prompt}\n\n--- model output ---\n{output}"})


def make_sql_tool(store, llm, row_cap: int = ROW_CAP):
    def run(question: str, schema: str, history: str = "(none)", trace: list | None = None,
            correction: str = "") -> pd.DataFrame:
        error_note = ""
        for _ in range(2):
            prompt = render("sql", "v2", schema=schema, limit=row_cap, history=history, question=question,
                            correction=(f"A reviewer asked for this change: {correction}" if correction else ""),
                            error_note=error_note)
            sql = extract_sql(llm.invoke(prompt).content)
            _record(trace, "retrieve-sql", prompt, sql)
            if not re.match(r"\s*(select|with)\b", sql, re.I):
                error_note = f"Your previous output was not a SELECT statement: {sql!r}. Return only one SELECT."
                continue
            try:
                return store.run_sql(sql).head(row_cap)
            except Exception as exc:
                error_note = f"Your previous SQL failed.\nSQL: {sql}\nError: {exc}\nFix it."
        raise RetrievalError(f"Could not produce a working SQL query. {error_note}")

    return run


IO_NAME = re.compile(
    r"^(read_\w*|to_(csv|pickle|parquet|excel|json|sql|hdf|feather|stata|html|xml|clipboard|markdown|latex|orc|gbq))$"
)
DENIED_NAMES = frozenset({
    "io", "eval", "query", "pipe", "apply", "applymap", "map", "exec", "compile", "open", "globals",
    "locals", "getattr", "setattr", "delattr", "vars", "type", "__import__",
})
SAFE_BUILTINS = {"len": len, "sum": sum, "min": min, "max": max, "round": round,
                 "abs": abs, "sorted": sorted, "str": str, "int": int, "float": float}


def _bound_names(tree: ast.AST) -> set[str]:
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
    return bound


def check_expression(expr: str) -> str | None:
    """Return a description of the first rejected construct, or None if the expression is allowed."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return f"syntax error ({exc.msg})"
    free = {"df", "pd"} | set(SAFE_BUILTINS) | _bound_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            ident = node.attr
        elif isinstance(node, ast.Name):
            ident = node.id
        else:
            continue
        if ident.startswith("_") or ident in DENIED_NAMES or IO_NAME.match(ident):
            return f"'{ident}'"
        if isinstance(node, ast.Name) and ident not in free:
            return f"unknown name '{ident}'"
    return None


def _to_frame(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.reset_index()
    return pd.DataFrame({"value": [result]})


# NOT wired into the app: eval of model-written code is unsafe with a deny-list filter
# (proven bypassable). Needs an AST allowlist/sandbox before it is used anywhere.
def make_pandas_tool(llm, row_cap: int = ROW_CAP):
    def run(question: str, df: pd.DataFrame, history: str = "(none)", trace: list | None = None) -> pd.DataFrame:
        error_note = ""
        for _ in range(2):
            sample = df.head(3).to_csv(index=False)
            prompt = render("pandas", columns=", ".join(f"{c} ({df[c].dtype})" for c in df.columns),
                            sample=sample, history=history, question=question, error_note=error_note)
            expr = extract_code(llm.invoke(prompt).content)
            _record(trace, "retrieve-pandas", prompt, expr)
            rejected = check_expression(expr)
            if rejected:
                error_note = (f"Your previous expression used a forbidden construct ({rejected}): {expr!r}. "
                              "Use only df and pd, no file I/O, query, apply or map.")
                continue
            try:
                result = eval(expr, {"__builtins__": SAFE_BUILTINS}, {"df": df, "pd": pd})
                return _to_frame(result).head(row_cap)
            except Exception as exc:
                error_note = f"Your previous expression failed.\nExpression: {expr}\nError: {exc}\nFix it."
        raise RetrievalError(f"Could not produce a working pandas expression. {error_note}")

    return run
