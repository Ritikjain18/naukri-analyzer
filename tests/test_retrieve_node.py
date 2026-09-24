import pandas as pd

from graph.nodes.retrieve import make_retrieve_node


def _sql(question, schema, history="(none)", trace=None):
    return pd.DataFrame({"n": [1]})


def _boom(*a, **k):
    raise AssertionError("pandas tool must not be called")


def _state():
    return {"question": "average salary by city", "schema": "s", "df": pd.DataFrame({"salary": [1], "city": ["x"]})}


def test_no_pandas_tool_always_uses_sql():
    out = make_retrieve_node(_sql)(_state())
    assert out["query_type"] == "sql"
    assert list(out["data_slice"].columns) == ["n"]


def test_explicit_pandas_tool_keeps_routing():
    called = []

    def pd_tool(q, df, history="(none)", trace=None):
        called.append(1)
        return pd.DataFrame({"n": [2]})

    out = make_retrieve_node(_boom, pd_tool)(_state())
    assert out["query_type"] == "pandas" and called
