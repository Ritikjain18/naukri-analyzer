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


def ReportingLLM(script):
    """FakeLLM reporting prompt-proportional usage_metadata, as the real provider does."""
    return FakeLLM(script, usage=True)


QUESTION = "Which job category has the best conversion rate?"
ORCH = json.dumps({"intent": "best category", "retrieval_instruction": "conversion rate by category"})
OK = json.dumps({"relevance": 4, "specificity": 4, "actionability": 4, "correction": ""})
INSIGHT = json.dumps({"finding": "Engineering has the highest conversion rate.",
                      "evidence": ["Eng 20% vs Sales 10%"], "recommendation": "Invest in Engineering."})
CHART = json.dumps({"type": "bar", "x": "category", "y": "conversion", "title": "Conversion"})
SMART_SCRIPT = [ORCH, OK, INSIGHT, OK]
BACKUP_SCRIPT = [OK, INSIGHT, OK, OK]  # 8B backup: judges score cleanly, analyst can answer
SHARED = {"data_summary": "S", "schema": "job_postings(category TEXT, conversion REAL)"}


def small_frame():
    return pd.DataFrame({"category": ["Eng", "Sales"], "conversion": [0.2, 0.1]})


def wide_frame():
    extra = {f"attr{c}": [f"value{r:07d}" for r in range(200)] for c in range(18)}
    return pd.DataFrame({"category": ["Eng", "Sales"] + [f"Cat{r}" for r in range(198)],
                         "conversion": [0.2, 0.1] + [0.05] * 198, **extra})


def setup(store, tmp_path, frame=small_frame, fast_script=(CHART,), smart_script=SMART_SCRIPT, backup_script=None):
    store.replace_table(pd.DataFrame({"a": [1]}), "job_postings")
    manager = RateLimitManager(tmp_path / "u.db")
    smart_fake = ReportingLLM(smart_script)
    smart_backup = ReportingLLM(BACKUP_SCRIPT if backup_script is None else backup_script)
    fast = FallbackLLM([(MODEL_FAST, ReportingLLM(list(fast_script)))], manager)
    smart = FallbackLLM([(MODEL_SMART, smart_fake), (MODEL_FAST, smart_backup)], manager)

    def sql_tool(question, schema, history="(none)", trace=None, correction=""):
        if trace is not None:
            trace.append({"node": "retrieve-sql", "prompt": "p"})
        return frame()

    return build_graph(store, fast, smart, sql_tool), fast, smart, manager, smart_fake


def run(graph, shared=SHARED, **extra):
    return graph.invoke(new_turn(shared, QUESTION, **extra), config={"recursion_limit": 60})


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
    graph, fast, smart, manager, smart_fake = setup(store, tmp_path, backup_script=SMART_SCRIPT)
    manager.record(MODEL_SMART, 11000)
    result = run(graph)
    assert result["insight"].finding == "Engineering has the highest conversion rate."
    assert smart.used_models() == [MODEL_FAST]
    assert smart_fake.prompts == []


def test_wide_slice_completes_without_rate_limit_error(store, tmp_path):
    graph, fast, smart, manager, smart_fake = setup(store, tmp_path, frame=wide_frame)
    result = run(graph)
    assert result["insight"].finding.startswith("Engineering")
    # the analyst is served by 70B; later judges may fall to 8B once its per-minute budget is spent
    assert len(smart_fake.prompts) >= 3 and "Data slice (CSV)" in smart_fake.prompts[2]
    assert "Judge could not score this answer." not in result.get("errors", [])
    analyst_prompt = next(p["prompt"] for p in result["prompts"] if p["node"] == "analyst")
    assert "# truncated" in analyst_prompt
    assert count_tokens(analyst_prompt) + OUTPUT_RESERVE <= RATE_HEADROOM * 12000


def test_whole_chain_exhausted_raises_and_maps_to_friendly_message(store, tmp_path):
    graph, fast, smart, manager, _ = setup(store, tmp_path)
    manager.record(MODEL_SMART, 11000)
    manager.record(MODEL_FAST, 5500)
    with pytest.raises(RateLimitExhausted) as info:
        run(graph)
    assert not isinstance(info.value, PromptTooLarge)
    assert "wait a minute" in friendly_error(info.value).lower()


HISTORY = [{"question": f"Earlier HR question {i}?",
            "finding": f"Earlier finding {i}: Engineering had a conversion rate of 20% against Sales at 10%, "
                       "driven by recruiter response times and job description quality across regions."}
           for i in range(6)]
SUMMARY_LONG = "job_postings summary: " + "category and conversion rate columns cover postings per job family. " * 40
FABRICATED = json.dumps({"finding": "Engineering converts at 87.31% overall.", "evidence": ["Eng 87.31%"],
                         "recommendation": "Invest in Engineering."})


def description_frame():
    desc = "Senior role covering hiring funnel ownership, recruiter coordination and stakeholder reporting. " * 4
    return pd.DataFrame({"category": ["Eng", "Sales"] + [f"Cat{r}" for r in range(198)],
                         "conversion": [0.2, 0.1] + [0.05] * 198,
                         "description": [f"{desc[:340]}{r:04d}" for r in range(200)]})


def test_realistic_inputs_wide_slice_with_summary_and_history_complete(store, tmp_path):
    graph, fast, smart, manager, _ = setup(store, tmp_path, frame=wide_frame)
    result = run(graph, {**SHARED, "data_summary": SUMMARY_LONG}, chat_history=HISTORY)
    assert result["insight"].finding.startswith("Engineering")
    assert not result.get("degraded")
    assert "Retry skipped" not in " ".join(result.get("errors", []))


def test_long_description_column_without_history_completes(store, tmp_path):
    graph, fast, smart, manager, _ = setup(store, tmp_path, frame=description_frame)
    result = run(graph)
    assert result["insight"].finding.startswith("Engineering")
    assert not result.get("degraded")


def test_guard_retry_after_full_size_analyst_call_ends_with_corrected_answer(store, tmp_path):
    graph, fast, smart, manager, smart_fake = setup(
        store, tmp_path, frame=wide_frame, smart_script=[ORCH, OK, FABRICATED, INSIGHT, OK],
        backup_script=[INSIGHT, OK])
    result = run(graph, {**SHARED, "data_summary": SUMMARY_LONG}, chat_history=HISTORY)
    assert result["insight"].finding == "Engineering has the highest conversion rate."
    assert not result.get("degraded")
    assert not any("Retry skipped" in e for e in result.get("errors", []))
    assert "Judge could not score this answer." not in result.get("errors", [])
    analyst_prompts = [p["prompt"] for p in result["prompts"] if p["node"] == "analyst"]
    assert len(analyst_prompts) == 2 and count_tokens(analyst_prompts[1]) < count_tokens(analyst_prompts[0])


def test_guard_retry_with_8b_exhausted_degrades_without_raising(store, tmp_path):
    graph, fast, smart, manager, smart_fake = setup(
        store, tmp_path, frame=wide_frame, smart_script=[ORCH, OK, FABRICATED, INSIGHT, OK])
    manager.record(MODEL_FAST, 5300)
    result = run(graph, {**SHARED, "data_summary": SUMMARY_LONG}, chat_history=HISTORY)
    assert result["degraded"] is True
    assert any("Retry skipped" in e for e in result["errors"])


def test_malformed_first_attempt_is_retried_with_a_smaller_slice(store, tmp_path):
    graph, fast, smart, manager, smart_fake = setup(
        store, tmp_path, frame=wide_frame, smart_script=[ORCH, OK, "not json", INSIGHT, OK],
        backup_script=[INSIGHT, OK])
    result = run(graph, {**SHARED, "data_summary": SUMMARY_LONG})
    assert result["insight"].finding == "Engineering has the highest conversion rate."
    analyst_prompts = [p["prompt"] for p in result["prompts"] if p["node"] == "analyst"]
    assert len(analyst_prompts) == 2 and count_tokens(analyst_prompts[1]) < count_tokens(analyst_prompts[0])


def test_first_attempt_rate_limit_at_the_analyst_raises_and_maps_to_friendly_message(store, tmp_path):
    graph, fast, smart, manager, smart_fake = setup(store, tmp_path, frame=wide_frame)
    backup_fake = smart.chain[1][1]
    manager.record(MODEL_SMART, 9000)
    manager.record(MODEL_FAST, 3000)
    with pytest.raises(RateLimitExhausted) as info:
        run(graph)
    assert not isinstance(info.value, PromptTooLarge)
    sent = smart_fake.prompts + backup_fake.prompts
    assert sent, "orchestrator/judge calls must have been served before the analyst"
    assert not any("Data slice (CSV)" in p for p in sent)   # the analyst prompt was never sent
    assert "wait a minute" in friendly_error(info.value).lower()


def test_prompt_that_can_never_fit_raises_prompt_too_large(store, tmp_path):
    graph, *_ = setup(store, tmp_path)
    huge = {**SHARED, "data_summary": "columns and figures " * 3000}
    with pytest.raises(PromptTooLarge) as info:
        run(graph, huge)
    assert "narrower" in friendly_error(info.value).lower()
