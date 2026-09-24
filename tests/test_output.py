import json

import pandas as pd
import pytest

from graph.models import ChartConfig, Insight
from graph.nodes.output import make_output_node
from graph.viz import build_figure, validate_chart
from tests.fakes import FakeLLM

SLICE = pd.DataFrame({"category": ["Eng", "Sales"], "rate": [0.2, 0.1]})
INSIGHT = Insight(finding="Eng leads.", evidence=["20% vs 10%"], recommendation="Invest.")
GOOD = json.dumps({"type": "bar", "x": "category", "y": "rate", "title": "Rate by category"})


def test_validate_chart_rules():
    assert validate_chart(json.loads(GOOD), SLICE).type == "bar"
    with pytest.raises(ValueError):
        validate_chart({"type": "bar", "x": "nope", "y": "rate", "title": "t"}, SLICE)
    with pytest.raises(ValueError):
        validate_chart({"type": "bar", "x": "rate", "y": "category", "title": "t"}, SLICE)


@pytest.mark.parametrize("kind", ["bar", "line", "scatter", "pie"])
def test_build_figure_types(kind):
    fig = build_figure(ChartConfig(type=kind, x="category", y="rate", title="T"), SLICE)
    assert len(fig.data) == 1


def state(**over):
    s = {"question": "q", "insight": INSIGHT, "data_slice": SLICE, "chat_history": [],
         "insight_memory": [], "prompts": [], "errors": []}
    s.update(over)
    return s


def test_output_happy_path():
    out = make_output_node(FakeLLM([GOOD]))(state())
    assert out["chart_config"].y == "rate"
    assert out["chat_history"] == [{"question": "q", "finding": "Eng leads."}]
    assert out["insight_memory"][0]["insight"]["finding"] == "Eng leads."
    assert out["prompts"][-1]["node"] == "visualization"
    assert out["errors"] == []


def test_output_retries_then_succeeds():
    llm = FakeLLM(["nonsense", GOOD])
    out = make_output_node(llm)(state())
    assert out["chart_config"] is not None
    assert "invalid" in llm.prompts[1].lower()


def test_output_falls_back_after_two_failures():
    out = make_output_node(FakeLLM(["bad", "bad"]))(state())
    assert out["chart_config"] is None
    assert out["errors"] == ["Chart could not be generated."]


def test_output_empty_slice_no_llm():
    llm = FakeLLM([])
    out = make_output_node(llm)(state(data_slice=SLICE.iloc[0:0]))
    assert out["chart_config"] is None and out["errors"] == [] and llm.prompts == []


def test_history_trimmed_to_six():
    hist = [{"question": f"q{i}", "finding": f"f{i}"} for i in range(6)]
    out = make_output_node(FakeLLM([GOOD]))(state(chat_history=hist))
    assert len(out["chat_history"]) == 6
    assert out["chat_history"][-1]["question"] == "q"
    assert out["chat_history"][0]["question"] == "q1"


def test_memory_entry_has_chart_slice_and_index():
    out = make_output_node(FakeLLM([GOOD]))(state(insight_memory=[{"question": "old"}]))
    entry = out["insight_memory"][-1]
    assert out["memory_index"] == 1 and entry["approved"] is False and entry["question"] == "q"
    assert entry["insight"]["finding"] == "Eng leads." and entry["chart"]["type"] == "bar"
    assert entry["slice"] == [{"category": "Eng", "rate": 0.2}, {"category": "Sales", "rate": 0.1}]


def test_memory_slice_is_capped_and_chart_may_be_none():
    import pandas as pd

    big = pd.DataFrame({"category": [f"c{i}" for i in range(80)], "rate": [0.1] * 80})
    out = make_output_node(FakeLLM(["bad", "bad"]))(state(data_slice=big))
    entry = out["insight_memory"][-1]
    assert len(entry["slice"]) == 50 and entry["chart"] is None
