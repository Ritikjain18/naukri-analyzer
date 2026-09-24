from graph.trace import generated_sql

M = "--- model output ---\n"


def test_returns_sql_after_marker_from_last_retrieve_entry():
    prompts = [{"node": "retrieve-sql", "prompt": f"p{M}SELECT 0"}, {"node": "analyst", "prompt": "x"},
               {"node": "retrieve-sql", "prompt": f"p2\n{M}SELECT 1"}]
    assert generated_sql(prompts) == "SELECT 1"


def test_empty_when_missing_or_no_marker():
    assert generated_sql([]) == ""
    assert generated_sql(None) == ""
    assert generated_sql([{"node": "retrieve-sql", "prompt": "no marker"}]) == ""
    assert generated_sql([{"node": "analyst", "prompt": M + "x"}]) == ""
