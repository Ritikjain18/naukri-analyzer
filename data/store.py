import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect


def sanitize_name(name: str, prefix: str = "t_") -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    if not s:
        return "unnamed"
    if s[0].isdigit():
        s = prefix + s
    return s


class SchemaMismatchError(ValueError):
    pass


class SQLiteStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{self.path}"
        self._engine = create_engine(url)
        # Touch the DB file via write engine so it exists before creating read-only connection
        with self._engine.connect():
            pass
        # Use read-only URI for truly read-only access
        ro_url = f"sqlite:///file:{self.path}?mode=ro&uri=true"
        self._ro_engine = create_engine(ro_url)

    def list_tables(self) -> list[str]:
        return sorted(inspect(self._engine).get_table_names())

    def get_schema(self) -> dict[str, list[tuple[str, str]]]:
        insp = inspect(self._engine)
        return {
            t: [(c["name"], str(c["type"])) for c in insp.get_columns(t)]
            for t in sorted(insp.get_table_names())
        }

    def schema_text(self) -> str:
        return "\n".join(
            f"{t}({', '.join(f'{c} {ty}' for c, ty in cols)})"
            for t, cols in self.get_schema().items()
        )

    def replace_table(self, df: pd.DataFrame, table: str) -> None:
        table = sanitize_name(table)
        df = df.copy()
        df.columns = [sanitize_name(c) for c in df.columns]
        df.to_sql(table, self._engine, if_exists="replace", index=False)

    def append_or_create(self, df: pd.DataFrame, table: str) -> str:
        table = sanitize_name(table)
        df = df.copy()
        df.columns = [sanitize_name(c) for c in df.columns]
        if table in self.list_tables():
            existing = {c for c, _ in self.get_schema()[table]}
            if set(df.columns) != existing:
                raise SchemaMismatchError(
                    f"Table '{table}' exists with columns {sorted(existing)}, "
                    f"but the upload has {sorted(df.columns)}."
                )
            df.to_sql(table, self._engine, if_exists="append", index=False)
            return "appended"
        df.to_sql(table, self._engine, if_exists="fail", index=False)
        return "created"

    def run_sql(self, query: str) -> pd.DataFrame:
        # Validate that only SELECT/WITH queries are allowed
        stripped = query.lstrip().upper()
        if not (stripped.startswith("SELECT") or stripped.startswith("WITH")):
            raise ValueError("Only SELECT queries are allowed")
        with self._ro_engine.connect() as conn:
            return pd.read_sql_query(query, conn)
