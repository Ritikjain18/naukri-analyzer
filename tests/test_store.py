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


def test_run_sql_drop_table_raises(store):
    """DROP TABLE via run_sql should raise ValueError."""
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        store.run_sql("DROP TABLE t")
    # Table should still exist
    assert store.list_tables() == ["t"]
    assert len(store.run_sql("SELECT * FROM t")) == 1


def test_run_sql_insert_raises(store):
    """INSERT via run_sql should raise ValueError."""
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        store.run_sql("INSERT INTO t VALUES (2)")
    # Row count should be unchanged
    assert len(store.run_sql("SELECT * FROM t")) == 1


def test_run_sql_delete_raises(store):
    """DELETE via run_sql should raise ValueError."""
    store.replace_table(pd.DataFrame({"a": [1, 2, 3]}), "t")
    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        store.run_sql("DELETE FROM t WHERE a = 2")
    # Row count should be unchanged
    assert len(store.run_sql("SELECT * FROM t")) == 3


def test_run_sql_pragma_raises(store):
    """PRAGMA query_only OFF via run_sql should raise ValueError."""
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    with pytest.raises(ValueError, match="Only SELECT queries are allowed"):
        store.run_sql("PRAGMA query_only = OFF")
    # Verify table is still protected and readable
    assert store.list_tables() == ["t"]
    result = store.run_sql("SELECT * FROM t")
    assert len(result) == 1


def test_run_sql_select_works(store):
    """SELECT should work even after failed attempts."""
    store.replace_table(pd.DataFrame({"a": [1, 2, 3]}), "t")
    # Attempt failed writes
    with pytest.raises(ValueError):
        store.run_sql("DELETE FROM t WHERE a = 1")
    with pytest.raises(ValueError):
        store.run_sql("INSERT INTO t VALUES (99)")
    # SELECT should still work and data unchanged
    result = store.run_sql("SELECT * FROM t")
    assert len(result) == 3
    assert list(result["a"]) == [1, 2, 3]


def test_run_sql_with_query_works(store):
    """WITH (CTE) queries should work."""
    store.replace_table(pd.DataFrame({"a": [1, 2, 3]}), "t")
    result = store.run_sql("WITH cte AS (SELECT * FROM t) SELECT * FROM cte WHERE a > 1")
    assert len(result) == 2


def test_fresh_store_with_run_sql(tmp_path):
    """Fresh SQLiteStore should allow run_sql immediately after creation."""
    from data.store import SQLiteStore
    store = SQLiteStore(tmp_path / "fresh.db")
    # Create a table and query immediately
    store.replace_table(pd.DataFrame({"x": [42]}), "test")
    result = store.run_sql("SELECT * FROM test")
    assert len(result) == 1
    assert result["x"][0] == 42


def test_replace_table_sanitizes_columns(store):
    """replace_table should sanitize column names."""
    df = pd.DataFrame({"Job ID": [1, 2], "Views": [10, 20]})
    store.replace_table(df, "My Jobs")
    schema = store.get_schema()["my_jobs"]
    column_names = [c for c, _ in schema]
    assert column_names == ["job_id", "views"]


def test_replace_then_append_same_headers(store):
    """replace_table then append_or_create with same headers should work."""
    df = pd.DataFrame({"Job ID": [1, 2], "Views": [10, 20]})
    # Replace with Job ID, Views
    store.replace_table(df, "My Jobs")
    # Append with Job ID, Views (same unsanitized names)
    result = store.append_or_create(df, "My Jobs")
    assert result == "appended"
    # Check total rows
    assert len(store.run_sql("SELECT * FROM my_jobs")) == 4
    # Check columns are consistent
    schema = store.get_schema()["my_jobs"]
    column_names = [c for c, _ in schema]
    assert column_names == ["job_id", "views"]


def test_schema_text(store):
    store.replace_table(pd.DataFrame({"a": [1], "b": ["x"]}), "t")
    text = store.schema_text()
    assert text.startswith("t(a ") and "b " in text
