import json

import config
from graph.nodes.orchestrate import make_orchestrate_node
from tests.fakes import FakeLLM

STATE = {"question": "Which category converts best?", "schema": "job_postings(category TEXT)",
         "data_summary": "S", "chat_history": [], "prompts": []}


def test_orchestrate_parses_and_logs_prompt():
    llm = FakeLLM([json.dumps({"intent": "best category", "retrieval_instruction": "conversion by category"})])
    out = make_orchestrate_node(llm)(STATE)
    assert out["orchestration"] == {"intent": "best category", "retrieval_instruction": "conversion by category"}
    assert out["query_type"] == "sql" and out["prompts"][-1]["node"] == "orchestrate"
    assert "job_postings" in llm.prompts[0]


def test_orchestrate_falls_back_on_bad_json():
    out = make_orchestrate_node(FakeLLM(["nonsense"]))(STATE)
    assert out["orchestration"]["retrieval_instruction"] == STATE["question"]


def test_orchestrate_skips_llm_when_judge_disabled(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_ENABLED", False)
    llm = FakeLLM([])
    out = make_orchestrate_node(llm)(STATE)
    assert out["orchestration"]["retrieval_instruction"] == STATE["question"] and llm.prompts == []


class _Limited:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, prompt):
        raise self.exc


def test_orchestrate_rate_limit_uses_fallback_plan_and_warns():
    from graph.llm import RateLimitExhausted

    out = make_orchestrate_node(_Limited(RateLimitExhausted("busy")))({**STATE, "errors": ["earlier"]})
    assert out["orchestration"]["retrieval_instruction"] == STATE["question"]
    assert out["errors"] == ["earlier", "Orchestrator skipped: rate limit."]


def test_orchestrate_other_errors_propagate():
    import pytest

    with pytest.raises(RuntimeError):
        make_orchestrate_node(_Limited(RuntimeError("boom")))(STATE)
