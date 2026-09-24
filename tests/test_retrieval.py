import pandas as pd
import pytest

from graph.nodes.retrieve import make_retrieve_node
from graph.tools import RetrievalError, make_pandas_tool, make_sql_tool
from tests.fakes import FakeLLM


@pytest.fixture
def seeded(store):
    store.replace_table(pd.DataFrame({"category": ["a", "a", "b"], "views": [10, 20, 30]}), "jobs")
    return store


def test_sql_tool_happy_path(seeded):
    llm = FakeLLM(["```sql\nSELECT category, SUM(views) AS v FROM jobs GROUP BY category;\n```"])
    out = make_sql_tool(seeded, llm)("views by category", seeded.schema_text())
    assert dict(zip(out["category"], out["v"])) == {"a": 30, "b": 30}


def test_sql_tool_retries_with_error_note(seeded):
    llm = FakeLLM(["SELECT nope FROM jobs", "SELECT COUNT(*) AS n FROM jobs"])
    out = make_sql_tool(seeded, llm)("how many", seeded.schema_text())
    assert out["n"][0] == 3
    assert "failed" in llm.prompts[1].lower()


def test_sql_tool_rejects_non_select_then_gives_up(seeded):
    llm = FakeLLM(["DROP TABLE jobs", "DELETE FROM jobs"])
    with pytest.raises(RetrievalError):
        make_sql_tool(seeded, llm)("x", seeded.schema_text())
    assert seeded.list_tables() == ["jobs"]


def test_sql_tool_caps_rows(store):
    store.replace_table(pd.DataFrame({"a": range(500)}), "t")
    llm = FakeLLM(["SELECT a FROM t"])
    assert len(make_sql_tool(store, llm, row_cap=50)("all", store.schema_text())) == 50


def test_pandas_tool_dataframe_series_scalar():
    df = pd.DataFrame({"g": ["x", "x", "y"], "v": [1, 2, 3]})
    llm = FakeLLM(["df.groupby('g')['v'].sum()", "df['v'].sum()", "df[df['v'] > 1]"])
    tool = make_pandas_tool(llm)
    s = tool("q", df)
    assert list(s.columns) == ["g", "v"] and list(s["v"]) == [3, 3]
    assert tool("q", df)["value"][0] == 6
    assert len(tool("q", df)) == 2


def test_pandas_tool_blocks_dangerous_code():
    df = pd.DataFrame({"v": [1]})
    llm = FakeLLM(["__import__('os').system('ls')", "df.__class__"])
    with pytest.raises(RetrievalError):
        make_pandas_tool(llm)("q", df)


def test_retrieve_node_routes_and_caps():
    calls = []

    def sql_tool(q, schema):
        calls.append("sql")
        return pd.DataFrame({"a": range(300)})

    def pandas_tool(q, df):
        calls.append("pandas")
        return df

    node = make_retrieve_node(sql_tool, pandas_tool)
    out = node({"question": "total views", "schema": "s"})
    assert out["query_type"] == "sql" and len(out["data_slice"]) == 200

    df = pd.DataFrame({"salary": [1, 2]})
    out = node({"question": "avg salary", "schema": "s", "df": df})
    assert out["query_type"] == "pandas"
    assert calls == ["sql", "pandas"]
