import json

import pandas as pd
import pytest

import config
from graph.models import Insight
from graph.nodes.judge import make_judge_node, route_after_judge
from tests.fakes import FakeLLM

SLICE = pd.DataFrame({"category": ["Eng", "Sales"], "conversion": [0.2, 0.1]})
INSIGHT = Insight(finding="Engineering has the highest conversion rate.", evidence=["20% vs 10%"],
                  recommendation="Invest.")


def scores(r, s, a, correction=""):
    return json.dumps({"relevance": r, "specificity": s, "actionability": a, "correction": correction})


def state(**over):
    base = {"question": "Which category converts best?", "data_slice": SLICE, "insight": INSIGHT,
            "prompts": [{"node": "retrieve-sql", "prompt": "p\n\n--- model output ---\nSELECT 1"}],
            "errors": [], "judge_scores": []}
    base.update(over)
    return base


def test_accept_records_scores_and_prompt():
    llm = FakeLLM([scores(4, 3, 5)])
    out = make_judge_node(llm, "analyst")(state())
    assert out["judge_verdict"] == "accept" and route_after_judge(out) == "accept"
    assert out["judge_scores"][-1] == {"stage": "analyst", "relevance": 4, "specificity": 3,
                                       "actionability": 5, "accepted": True}
    assert out["prompts"][-1]["node"] == "judge-analyst"
    assert "Engineering has the highest" in llm.prompts[0]


def test_reject_sets_correction_and_counter():
    out = make_judge_node(FakeLLM([scores(4, 2, 4, "Cite the numbers.")]), "analyst")(state())
    assert out["judge_verdict"] == "reject" and route_after_judge(out) == "reject"
    assert out["judge_correction"] == "Cite the numbers." and out["analyst_rejections"] == 1


def test_retrieval_judge_sees_sql_and_rows():
    llm = FakeLLM([scores(2, 2, 2, "Group by category.")])
    out = make_judge_node(llm, "retrieval")(state())
    assert out["retrieval_correction"] == "Group by category." and out["retrieval_rejections"] == 1
    assert "SELECT 1" in llm.prompts[0] and "Rows returned: 2" in llm.prompts[0]


def test_default_correction_text_when_judge_gives_none():
    out = make_judge_node(FakeLLM([scores(1, 1, 1)]), "analyst")(state())
    assert "relevance" in out["judge_correction"]


def test_cap_accepts_with_low_confidence_warning():
    out = make_judge_node(FakeLLM([scores(2, 2, 2, "Still vague.")]), "analyst")(state(analyst_rejections=2))
    assert out["judge_verdict"] == "accept"
    assert out["errors"] == ["Low-confidence answer: Still vague."]
    assert out["judge_correction"] == ""


def test_unparsable_judge_output_accepts_with_warning():
    out = make_judge_node(FakeLLM(["no json"]), "analyst")(state())
    assert out["judge_verdict"] == "accept" and out["errors"] == ["Judge could not score this answer."]


def test_scores_are_clamped_and_non_numeric_is_unparsable():
    out = make_judge_node(FakeLLM([scores(9, 0, 5)]), "analyst")(state())
    assert (out["judge_scores"][-1]["relevance"], out["judge_scores"][-1]["specificity"]) == (5, 1)
    out = make_judge_node(FakeLLM(['{"relevance":"high","specificity":3,"actionability":3}']), "analyst")(state())
    assert out["errors"] == ["Judge could not score this answer."]


@pytest.mark.parametrize("stage,flag", [("analyst", "JUDGE_ENABLED"), ("retrieval", "JUDGE_RETRIEVAL")])
def test_switches_skip_the_call(monkeypatch, stage, flag):
    monkeypatch.setattr(config, flag, False)
    llm = FakeLLM([])
    assert make_judge_node(llm, stage)(state()) == {"judge_verdict": "accept"} and llm.prompts == []


def test_empty_slice_skips_the_call():
    llm = FakeLLM([])
    assert make_judge_node(llm, "analyst")(state(data_slice=SLICE.iloc[0:0])) == {"judge_verdict": "accept"}
