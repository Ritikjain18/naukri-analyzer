import json

import pandas as pd
import pytest

from data.parsing import parse_upload
from data.store import SchemaMismatchError
from graph.nodes.ingest import make_ingest_node
from tests.fakes import FakeLLM

CSV = b"Job ID,Views,Applications\n1,100,10\n2,200,30\n"


def test_parse_csv_json_and_unsupported():
    assert list(parse_upload("a.csv", CSV).columns) == ["Job ID", "Views", "Applications"]
    js = json.dumps({"rows": [{"a": 1}, {"a": 2}]}).encode()
    assert len(parse_upload("a.json", js)) == 2
    js2 = json.dumps({"metrics": {"a": 1}}).encode()
    assert len(parse_upload("m.json", js2)) == 1
    with pytest.raises(ValueError):
        parse_upload("a.pdf", b"x")


def test_parse_xlsx_roundtrip(tmp_path):
    p = tmp_path / "x.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(p, index=False)
    assert len(parse_upload("x.xlsx", p.read_bytes())) == 2


def test_upload_creates_then_appends(store):
    llm = FakeLLM(["summary one", "summary two"])
    node = make_ingest_node(store, llm)
    out = node({"upload": {"name": "Jobs Q1.csv", "bytes": CSV}, "prompts": []})
    assert out["table_name"] == "jobs_q1"
    assert out["ingest_action"] == "created"
    assert list(out["df"].columns) == ["job_id", "views", "applications"]
    assert out["data_summary"] == "summary one"
    assert out["prompts"][0]["node"] == "ingest"
    assert "jobs_q1" in out["schema"]

    out2 = node({"upload": {"name": "Jobs Q1.csv", "bytes": CSV}})
    assert out2["ingest_action"] == "appended"
    assert len(store.run_sql("SELECT * FROM jobs_q1")) == 4


def test_upload_schema_mismatch_raises(store):
    node = make_ingest_node(store, FakeLLM(["s"]))
    node({"upload": {"name": "t.csv", "bytes": CSV}})
    with pytest.raises(SchemaMismatchError):
        node({"upload": {"name": "t.csv", "bytes": b"x,y\n1,2\n"}})


def test_no_upload_summarises_database(store):
    store.replace_table(pd.DataFrame({"a": [1, 2, 3]}), "widgets")
    llm = FakeLLM(["db summary"])
    out = make_ingest_node(store, llm)({})
    assert out["data_summary"] == "db summary"
    assert "df" not in out
    assert "widgets" in llm.prompts[0]


def test_reserved_word_table_names_do_not_break_ingest(store):
    store.replace_table(pd.DataFrame({"a": [1]}), "order")
    store.replace_table(pd.DataFrame({"b": [2]}), "group")
    llm = FakeLLM(["s"])
    out = make_ingest_node(store, llm)({})
    assert out["data_summary"] == "s"
    assert "-- order" in llm.prompts[0] and "-- group" in llm.prompts[0]


def test_append_returns_full_table_in_df(store):
    node = make_ingest_node(store, FakeLLM(["a", "b"]))
    first = node({"upload": {"name": "jobs.csv", "bytes": CSV}, "prompts": []})
    assert len(first["df"]) == 2
    second = node({"upload": {"name": "jobs.csv", "bytes": CSV}, "prompts": []})
    assert second["ingest_action"] == "appended"
    assert len(second["df"]) == 4
