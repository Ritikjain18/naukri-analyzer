from pathlib import Path

from config import DESCRIBE_COLUMNS, SAMPLE_CHARS
from data.parsing import parse_upload
from data.store import sanitize_name
from graph.prompts import render
from graph.skills import load_domain_skills


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _clip(text: str, limit: int = SAMPLE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _sample_for_df(df) -> str:
    described = df.iloc[:, :DESCRIBE_COLUMNS].describe(include="all").round(2).to_csv()
    return _clip(df.head(5).to_csv(index=False) + "\n" + described)


def _sample_for_db(store) -> str:
    parts = []
    for table in store.list_tables():
        parts.append(f"-- {table}\n" + store.run_sql(f"SELECT * FROM {quote_ident(table)} LIMIT 3").to_csv(index=False))
    return _clip("\n".join(parts)) or "(database is empty)"


def make_ingest_node(store, llm):
    def ingest(state):
        update = {}
        df = None
        upload = state.get("upload")
        if upload:
            df = parse_upload(upload["name"], upload["bytes"])
            df.columns = [sanitize_name(c) for c in df.columns]
            table = sanitize_name(Path(upload["name"]).stem)
            action = store.append_or_create(df, table)
            update["ingest_action"] = action
            update["table_name"] = table
            if action == "appended":
                df = store.run_sql(f"SELECT * FROM {quote_ident(table)}")
            update["df"] = df

        schema = store.schema_text()
        sample = _sample_for_df(df) if df is not None else _sample_for_db(store)
        prompt = render("data_understanding", skills=load_domain_skills(), schema=schema, sample=sample)
        summary = llm.invoke(prompt).content
        update["schema"] = schema
        update["data_summary"] = summary
        update["prompts"] = list(state.get("prompts", [])) + [{"node": "ingest", "prompt": prompt}]
        return update

    return ingest
