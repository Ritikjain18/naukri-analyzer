from pathlib import Path

from data.parsing import parse_upload
from data.store import sanitize_name
from graph.prompts import render
from graph.skills import load_domain_skills


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sample_for_df(df) -> str:
    return df.head(5).to_csv(index=False) + "\n" + df.describe(include="all").round(2).to_csv()


def _sample_for_db(store) -> str:
    parts = []
    for table in store.list_tables():
        parts.append(f"-- {table}\n" + store.run_sql(f"SELECT * FROM {quote_ident(table)} LIMIT 3").to_csv(index=False))
    return "\n".join(parts) or "(database is empty)"


def make_ingest_node(store, llm):
    def ingest(state):
        update = {}
        df = None
        upload = state.get("upload")
        if upload:
            df = parse_upload(upload["name"], upload["bytes"])
            df.columns = [sanitize_name(c) for c in df.columns]
            table = sanitize_name(Path(upload["name"]).stem)
            update["ingest_action"] = store.append_or_create(df, table)
            update["table_name"] = table
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
