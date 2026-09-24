"""Whole graph through the REAL FallbackLLM + RateLimitManager (only the providers are fakes)."""
import json

import pandas as pd
import pytest

from config import MODEL_FAST, MODEL_SMART, OUTPUT_RESERVE, RATE_HEADROOM
from graph.budget import count_tokens
from graph.build_graph import build_graph
from graph.llm import FallbackLLM, PromptTooLarge, RateLimitExhausted, friendly_error
from graph.ratelimit import RateLimitManager
from graph.state import new_turn
from tests.fakes import FakeLLM


class ReportingLLM(FakeLLM):
    """FakeLLM that reports Groq-style usage_metadata, as the real provider does."""

    def invoke(self, prompt):
        resp = super().invoke(prompt)
        resp.usage_metadata = {"total_tokens": 300}
        return resp

QUESTION = "Which job category has the best conversion rate?"
ORCH = json.dumps({"intent": "best category", "retrieval_instruction": "conversion rate by category"})
OK = json.dumps({"relevance": 4, "specificity": 4, "actionability": 4, "correction": ""})
INSIGHT = json.dumps({"finding": "Engineering has the highest conversion rate.",
                      "evidence": ["Eng 20% vs Sales 10%"], "recommendation": "Invest in Engineering."})
CHART = json.dumps({"type": "bar", "x": "category", "y": "conversion", "title": "Conversion"})
SMART_SCRIPT = [ORCH, OK, INSIGHT, OK]
SHARED = {"data_summary": "S", "schema": "job_postings(category TEXT, conversion REAL)"}


def small_frame():
    return pd.DataFrame({"category": ["Eng", "Sales"], "conversion": [0.2, 0.1]})


def wide_frame():
    extra = {f"attr{c}": [f"value{r:07d}" for r in range(200)] for c in range(18)}
    return pd.DataFrame({"category": ["Eng", "Sales"] + [f"Cat{r}" for r in range(198)],
                         "conversion": [0.2, 0.1] + [0.05] * 198, **extra})


def setup(store, tmp_path, frame=small_frame, fast_script=(CHART,), smart_script=SMART_SCRIPT):
    store.replace_table(pd.DataFrame({"a": [1]}), "job_postings")
    manager = RateLimitManager(tmp_path / "u.db")
    smart_fake, smart_backup = ReportingLLM(smart_script), ReportingLLM(smart_script)
    fast = FallbackLLM([(MODEL_FAST, ReportingLLM(list(fast_script)))], manager)
    smart = FallbackLLM([(MODEL_SMART, smart_fake), (MODEL_FAST, smart_backup)], manager)

    def sql_tool(question, schema, history="(none)", trace=None, correction=""):
        if trace is not None:
            trace.append({"node": "retrieve-sql", "prompt": "p"})
        return frame()

    return build_graph(store, fast, smart, sql_tool), fast, smart, manager, smart_fake


def run(graph):
    return graph.invoke(new_turn(SHARED, QUESTION), config={"recursion_limit": 60})


def test_happy_path_uses_smart_model_and_records_usage(store, tmp_path):
    graph, fast, smart, manager, _ = setup(store, tmp_path)
    result = run(graph)
    assert result["insight"].finding == "Engineering has the highest conversion rate."
    assert result["chart_config"].type == "bar"
    assert len(result["judge_scores"]) == 2 and all(s["accepted"] for s in result["judge_scores"])
    assert [p["node"] for p in result["prompts"]] == [
        "orchestrate", "retrieve-sql", "judge-retrieval", "analyst", "judge-analyst", "visualization"]
    assert smart.used_models() == [MODEL_SMART]
    assert manager.used_last_minute(MODEL_SMART) > 0


def test_exhausted_70b_falls_back_to_8b_for_the_same_question(store, tmp_path):
    graph, fast, smart, manager, smart_fake = setup(store, tmp_path)
    manager.record(MODEL_SMART, 11000)
    result = run(graph)
    assert result["insight"].finding == "Engineering has the highest conversion rate."
    assert smart.used_models() == [MODEL_FAST]
    assert smart_fake.prompts == []


def test_wide_slice_completes_without_rate_limit_error(store, tmp_path):
    graph, fast, smart, manager, smart_fake = setup(store, tmp_path, frame=wide_frame)
    result = run(graph)
    assert result["insight"].finding.startswith("Engineering")
    assert smart.used_models() == [MODEL_SMART]
    analyst_prompt = next(p["prompt"] for p in result["prompts"] if p["node"] == "analyst")
    assert count_tokens(analyst_prompt) + OUTPUT_RESERVE <= RATE_HEADROOM * 12000


def test_whole_chain_exhausted_raises_and_maps_to_friendly_message(store, tmp_path):
    graph, fast, smart, manager, _ = setup(store, tmp_path)
    manager.record(MODEL_SMART, 11000)
    manager.record(MODEL_FAST, 5500)
    with pytest.raises(RateLimitExhausted) as info:
        run(graph)
    assert not isinstance(info.value, PromptTooLarge)
    assert "wait a minute" in friendly_error(info.value).lower()
