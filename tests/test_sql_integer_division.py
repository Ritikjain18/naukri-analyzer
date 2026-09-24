from data.store import SQLiteStore
from graph.tools import make_sql_tool
from tests.fakes import FakeLLM

BUGGY = "SELECT category, SUM(applications) / SUM(views) AS r FROM job_postings GROUP BY category"
FIXED = ("SELECT category, CAST(SUM(applications) AS REAL) / NULLIF(SUM(views), 0) AS r "
         "FROM job_postings GROUP BY category ORDER BY r DESC")


def _seed(store):
    import pandas as pd
    df = pd.DataFrame({"category": ["Sales", "Sales", "Tech", "Tech"],
                       "views": [100, 100, 50, 50], "applications": [10, 20, 40, 30]})
    store.replace_table(df, "job_postings")


def test_integer_division_bug_and_fix(tmp_path):
    store = SQLiteStore(tmp_path / "t.db")
    _seed(store)
    bug = store.run_sql(BUGGY)
    assert (bug["r"] == 0).all()  # documents the bug: 30/200 and 70/100 truncate to 0
    fixed = store.run_sql(FIXED).set_index("category")["r"]
    assert fixed["Sales"] == 0.15 and fixed["Tech"] == 0.7


def test_sql_tool_prompt_contains_real_division_guidance():
    llm = FakeLLM(["SELECT 1"])

    class S:
        def run_sql(self, q):
            import pandas as pd
            return pd.DataFrame({"a": [1]})

    make_sql_tool(S(), llm)("q", "schema")
    assert "CAST" in llm.prompts[0] and "NULLIF" in llm.prompts[0]


def test_sql_tool_with_cast_form_returns_nonzero_rates(tmp_path):
    store = SQLiteStore(tmp_path / "t.db")
    _seed(store)
    df = make_sql_tool(store, FakeLLM([FIXED]))("which category converts best?", store.schema_text())
    assert (df["r"] > 0).all() and df.iloc[0]["category"] == "Tech"
