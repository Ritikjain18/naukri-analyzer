import pandas as pd
import pytest

from data.store import SchemaMismatchError, sanitize_name


def test_sanitize_name():
    assert sanitize_name("Job Postings (Q1)") == "job_postings_q1"
    assert sanitize_name("2024 data") == "t_2024_data"
    assert sanitize_name("!!!") == "unnamed"


def test_create_then_append(store):
    df = pd.DataFrame({"Job ID": [1, 2], "Views": [10, 20]})
    assert store.append_or_create(df, "My Jobs") == "created"
    assert store.list_tables() == ["my_jobs"]
    assert store.append_or_create(df, "My Jobs") == "appended"
    assert len(store.run_sql("SELECT * FROM my_jobs")) == 4
    assert [c for c, _ in store.get_schema()["my_jobs"]] == ["job_id", "views"]


def test_schema_mismatch_raises(store):
    store.append_or_create(pd.DataFrame({"a": [1]}), "t")
    with pytest.raises(SchemaMismatchError):
        store.append_or_create(pd.DataFrame({"b": [1]}), "t")


def test_replace_table(store):
    store.replace_table(pd.DataFrame({"a": [1, 2]}), "t")
    store.replace_table(pd.DataFrame({"a": [9]}), "t")
    assert len(store.run_sql("SELECT * FROM t")) == 1


def test_run_sql_is_read_only(store):
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    with pytest.raises(Exception):
        store.run_sql("DROP TABLE t")
    assert store.list_tables() == ["t"]


def test_schema_text(store):
    store.replace_table(pd.DataFrame({"a": [1], "b": ["x"]}), "t")
    text = store.schema_text()
    assert text.startswith("t(a ") and "b " in text
