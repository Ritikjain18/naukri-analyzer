import re

import pandas as pd

from config import ROW_CAP
from graph.parsing import extract_code
from graph.prompts import render


class RetrievalError(RuntimeError):
    pass


def make_sql_tool(store, llm, row_cap: int = ROW_CAP):
    def run(question: str, schema: str) -> pd.DataFrame:
        error_note = ""
        for _ in range(2):
            prompt = render("sql", schema=schema, limit=row_cap, question=question, error_note=error_note)
            sql = extract_code(llm.invoke(prompt).content)
            if not re.match(r"\s*(select|with)\b", sql, re.I):
                error_note = f"Your previous output was not a SELECT statement: {sql!r}. Return only one SELECT."
                continue
            try:
                return store.run_sql(sql).head(row_cap)
            except Exception as exc:
                error_note = f"Your previous SQL failed.\nSQL: {sql}\nError: {exc}\nFix it."
        raise RetrievalError(f"Could not produce a working SQL query. {error_note}")

    return run


FORBIDDEN = ("__", "import", "open(", "exec(", "eval(", "compile(", "globals", "getattr", "os.", "sys.")
SAFE_BUILTINS = {"len": len, "sum": sum, "min": min, "max": max, "round": round,
                 "abs": abs, "sorted": sorted, "str": str, "int": int, "float": float}


def _to_frame(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.reset_index()
    return pd.DataFrame({"value": [result]})


def make_pandas_tool(llm, row_cap: int = ROW_CAP):
    def run(question: str, df: pd.DataFrame) -> pd.DataFrame:
        error_note = ""
        for _ in range(2):
            sample = df.head(3).to_csv(index=False)
            prompt = render("pandas", columns=", ".join(f"{c} ({df[c].dtype})" for c in df.columns),
                            sample=sample, question=question, error_note=error_note)
            expr = extract_code(llm.invoke(prompt).content)
            if any(tok in expr for tok in FORBIDDEN):
                error_note = f"Your previous expression used a forbidden construct: {expr!r}. Use only df and pd."
                continue
            try:
                result = eval(expr, {"__builtins__": SAFE_BUILTINS}, {"df": df, "pd": pd})
                return _to_frame(result).head(row_cap)
            except Exception as exc:
                error_note = f"Your previous expression failed.\nExpression: {expr}\nError: {exc}\nFix it."
        raise RetrievalError(f"Could not produce a working pandas expression. {error_note}")

    return run
