import pytest
from pydantic import ValidationError

from graph.models import ChartConfig, Insight
from graph.parsing import extract_code, extract_json, extract_sql
from graph.router import choose_tool


def test_insight_requires_text():
    with pytest.raises(ValidationError):
        Insight(finding="", evidence=[], recommendation="x")
    assert Insight(finding="f", evidence=["e"], recommendation="r").finding == "f"


def test_chart_type_restricted():
    with pytest.raises(ValidationError):
        ChartConfig(type="donut", x="a", y="b", title="t")


def test_extract_code_strips_fence_and_semicolon():
    assert extract_code("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_code("SELECT 2;") == "SELECT 2"


def test_extract_json_finds_object_in_prose():
    assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.parametrize("q,cols,expected", [
    ("total views by category", None, "sql"),
    ("total views by category", [], "sql"),
    ("average salary by dept", ["salary", "dept"], "pandas"),
    ("show Time To Hire trend", ["time_to_hire"], "pandas"),
    ("something unrelated", ["salary"], "sql"),
])
def test_choose_tool(q, cols, expected):
    assert choose_tool(q, cols) == expected


@pytest.mark.parametrize("q,cols,expected", [
    ("How many candidates applied", ["id"], "sql"),
    ("average salary by dept", ["salary", "dept"], "pandas"),
    ("show time to hire trend", ["time_to_hire"], "pandas"),
    ("provide averages", ["age"], "sql"),
])
def test_router_whole_word_matching(q, cols, expected):
    assert choose_tool(q, cols) == expected


@pytest.mark.parametrize("text,expected", [
    ("Here is the query: SELECT a FROM t;", "SELECT a FROM t"),
    ("```sql\nSELECT a FROM t;\n```", "SELECT a FROM t"),
    ("```SELECT 1```", "SELECT 1"),
    ("```sql\nSELECT 1;", "SELECT 1"),
    ("Sure!\n```sql\nWITH x AS (SELECT 1) SELECT * FROM x;\n```\nHope it helps", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("  nothing here;  ", "nothing here"),
])
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


def test_extract_code_single_line_and_unclosed_fences():
    assert extract_code("```df['a']```") == "df['a']"
    assert extract_code("```python\ndf.head()") == "df.head()"
