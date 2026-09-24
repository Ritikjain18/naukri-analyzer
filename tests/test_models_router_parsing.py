import pytest
from pydantic import ValidationError

from graph.models import ChartConfig, Insight
from graph.parsing import extract_code, extract_json
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
