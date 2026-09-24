import json

import pandas as pd
import pytest

from graph.budget import count_tokens, effective_limit
from graph.nodes.analyst import AnalysisError, format_history, make_analyst_node
from tests.fakes import FakeLLM

GOOD = json.dumps({"finding": "Engineering converts best.", "evidence": ["Eng 20% vs Sales 10%"],
                   "recommendation": "Shift budget to Engineering."})
SLICE = pd.DataFrame({"category": ["Eng", "Sales"], "views": [100, 100], "applications": [20, 10]})


def base_state(**over):
    state = {"question": "which converts best?", "data_summary": "SUMMARY", "data_slice": SLICE,
             "chat_history": [], "prompts": []}
    state.update(over)
    return state


def test_format_history():
    assert format_history([]) == "(none)"
    text = format_history([{"question": "q1", "finding": "f1"}])
    assert "Q: q1" in text and "A: f1" in text


def test_analyst_happy_path_injects_context():
    llm = FakeLLM([GOOD])
    out = make_analyst_node(llm)(base_state(chat_history=[{"question": "old", "finding": "older"}]))
    assert out["insight"].finding == "Engineering converts best."
    prompt = llm.prompts[0]
    assert "SUMMARY" in prompt and "category,views,applications" in prompt
    assert "Job Posting Analytics" in prompt          # domain skill injected
    assert "Q: old" in prompt
    assert out["prompts"][-1]["node"] == "analyst"


def test_analyst_retries_once_on_bad_json():
    llm = FakeLLM(["not json", GOOD])
    out = make_analyst_node(llm)(base_state())
    assert out["insight"].recommendation.startswith("Shift")
    assert "invalid" in llm.prompts[1].lower()


def test_analyst_gives_up_after_second_failure():
    with pytest.raises(AnalysisError):
        make_analyst_node(FakeLLM(["bad", "worse"]))(base_state())


def test_analyst_empty_slice_skips_llm():
    llm = FakeLLM([])
    out = make_analyst_node(llm)(base_state(data_slice=SLICE.iloc[0:0]))
    assert "No data matched" in out["insight"].finding
    assert llm.prompts == []


def test_analyst_truncates_large_slice_to_fit_token_budget():
    import config
    big = pd.DataFrame({f"col_{i}": [f"value_{r}_{i}" for r in range(200)] for i in range(40)})
    llm = FakeLLM([GOOD])
    make_analyst_node(llm)(base_state(data_slice=big))
    prompt = llm.prompts[0]
    assert count_tokens(prompt) < effective_limit(config.MODEL_SMART) - config.OUTPUT_RESERVE
    assert "# truncated: showing " in prompt and " of 200 rows" in prompt


def test_analyst_small_slice_not_truncated():
    llm = FakeLLM([GOOD])
    make_analyst_node(llm)(base_state())
    assert "truncated" not in llm.prompts[0]


def _long_error_llm():
    # extract_json raises ValueError embedding the (unbounded) offending text
    return FakeLLM(["{" + "z" * 5000, GOOD])


def test_analyst_retry_prompt_stays_within_budget_with_long_error():
    import config
    big = pd.DataFrame({f"col_{i}": [f"value_{r}_{i}" for r in range(200)] for i in range(30)})
    llm = _long_error_llm()
    make_analyst_node(llm)(base_state(data_slice=big))
    assert len(llm.prompts) == 2
    assert count_tokens(llm.prompts[1]) <= effective_limit(config.MODEL_SMART) - config.OUTPUT_RESERVE


def test_analyst_retry_error_note_is_clipped():
    from graph.nodes.analyst import ERROR_NOTE
    llm = _long_error_llm()
    make_analyst_node(llm)(base_state())
    extra = len(llm.prompts[1]) - len(llm.prompts[0])
    assert 0 < extra <= len(ERROR_NOTE) + 400
    assert "z" * 400 not in llm.prompts[1]
