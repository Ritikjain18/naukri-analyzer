import json

import pandas as pd

from graph.build_graph import build_graph, route_entry
from tests.fakes import FakeLLM

INSIGHT = json.dumps({"finding": "Eng leads.", "evidence": ["20% vs 10%"], "recommendation": "Invest."})
CHART = json.dumps({"type": "bar", "x": "category", "y": "conversion", "title": "Conversion"})


def sql_tool(question, schema, history="(none)", trace=None):
    if trace is not None:
        trace.append({"node": "retrieve-sql", "prompt": "p"})
    return pd.DataFrame({"category": ["Eng", "Sales"], "conversion": [0.2, 0.1]})


def no_pandas(question, df, history="(none)", trace=None):
    raise AssertionError("pandas tool should not be used")


def make(store, fast, smart, sql=sql_tool):
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    return build_graph(store, fast, smart, sql, no_pandas)


def test_route_entry():
    assert route_entry({}) == "ingest"
    assert route_entry({"data_summary": ""}) == "ingest"
    assert route_entry({"data_summary": "x"}) == "retrieve"


def test_first_run_ingests_then_answers(store):
    fast = FakeLLM(["DB summary", CHART])
    smart = FakeLLM([INSIGHT])
    result = make(store, fast, smart).invoke({"question": "q1", "prompts": [], "errors": []})
    assert result["data_summary"] == "DB summary"
    assert result["insight"].finding == "Eng leads."
    assert result["chart_config"].type == "bar"
    assert [p["node"] for p in result["prompts"]] == ["ingest", "retrieve-sql", "analyst", "visualization"]
    assert len(result["chat_history"]) == 1


def test_follow_up_skips_ingest_and_uses_history(store):
    fast = FakeLLM(["DB summary", CHART, CHART])
    smart = FakeLLM([INSIGHT, INSIGHT])
    graph = make(store, fast, smart)
    first = graph.invoke({"question": "q1", "prompts": [], "errors": []})
    second = graph.invoke({**first, "question": "q2", "prompts": [], "errors": []})
    assert [p["node"] for p in second["prompts"]] == ["retrieve-sql", "analyst", "visualization"]
    assert len(second["chat_history"]) == 2
    assert "Q: q1" in smart.prompts[1]
    assert len(second["insight_memory"]) == 2


def test_empty_slice_skips_llm_calls(store):
    fast = FakeLLM(["DB summary"])
    smart = FakeLLM([])
    graph = make(store, fast, smart, sql=lambda q, s, history="(none)", trace=None: pd.DataFrame({"a": []}))
    result = graph.invoke({"question": "q", "prompts": [], "errors": []})
    assert "No data matched" in result["insight"].finding
    assert result["chart_config"] is None
    assert smart.prompts == []


def test_chart_failure_is_reported_not_raised(store):
    fast = FakeLLM(["DB summary", "bad", "bad"])
    result = make(store, fast, FakeLLM([INSIGHT])).invoke({"question": "q", "prompts": [], "errors": []})
    assert result["chart_config"] is None
    assert result["errors"] == ["Chart could not be generated."]
    assert result["insight"].finding == "Eng leads."


def test_build_graph_without_pandas_tool_uses_sql_for_uploaded_df(store):
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    df = pd.DataFrame({"salary": [1, 2], "city": ["x", "y"]})
    graph = build_graph(store, FakeLLM(["DB summary", CHART]), FakeLLM([INSIGHT]), sql_tool)
    result = graph.invoke({"question": "average salary by city", "df": df, "prompts": [], "errors": []})
    assert result["query_type"] == "sql"
    assert "retrieve-sql" in [p["node"] for p in result["prompts"]]
