import math

import pandas as pd
import pytest

import config
from graph.budget import available_tokens, clip_to_tokens, count_tokens, effective_limit, fit_rows


def test_count_tokens_applies_margin():
    assert count_tokens("a" * 400) == math.ceil(100 * 1.10)


def test_effective_limit_uses_tpm_or_context_window():
    assert effective_limit(config.MODEL_SMART) == config.MODEL_LIMITS[config.MODEL_SMART]["tpm"] == 8000
    assert effective_limit(config.MODEL_FAST) == config.MODEL_LIMITS[config.MODEL_FAST]["tpm"] == 8000
    assert effective_limit("unknown-model") == config.CONTEXT_WINDOW


def test_available_tokens_math():
    system, history, question = "a" * 400, "b" * 40, "c" * 40
    expected = config.MODEL_LIMITS[config.MODEL_SMART]["tpm"] - config.OUTPUT_RESERVE - count_tokens(system) - count_tokens(history) - count_tokens(question)
    assert available_tokens(config.MODEL_SMART, system, history, question) == expected


def test_available_tokens_never_negative():
    assert available_tokens(config.MODEL_FAST, "x" * 100000, "", "") == 0


def test_clip_to_tokens():
    assert clip_to_tokens("short", 100) == "short"
    clipped = clip_to_tokens("x" * 4000, 100)
    assert clipped.endswith("...") and count_tokens(clipped) <= 100


def test_fit_rows_small_frame_untouched():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert fit_rows(df, 1000) == df.to_csv(index=False)


def test_fit_rows_truncates_with_note():
    df = pd.DataFrame({f"c{i}": range(200) for i in range(5)})
    text = fit_rows(df, 300)
    assert "# truncated: showing" in text and "of 200 rows" in text
    shown = int(text.split("showing ")[1].split(" of")[0])
    assert 5 <= shown < 200
    assert count_tokens(text) <= 300


@pytest.mark.live
def test_real_tiktoken_counts_tokens():
    from graph import budget

    budget.use_estimator(None)  # network on first use: the BPE file is downloaded
    assert budget.count_tokens("How many job applications did each category get?") > 5


def test_slice_budget_fresh_minute():
    from config import MODEL_SMART, OUTPUT_RESERVE
    from graph.budget import request_budget, slice_budget
    assert slice_budget(MODEL_SMART, 1000, 0) == request_budget(MODEL_SMART) - OUTPUT_RESERVE - 1000


def test_slice_budget_partially_used_minute():
    from config import MODEL_SMART, OUTPUT_RESERVE
    from graph.budget import request_budget, slice_budget
    assert slice_budget(MODEL_SMART, 1000, 3000) == request_budget(MODEL_SMART) - 3000 - OUTPUT_RESERVE - 1000


def test_slice_budget_floor_when_used_beyond_budget():
    from config import MODEL_SMART
    from graph.budget import request_budget, slice_budget
    assert slice_budget(MODEL_SMART, 1000, request_budget(MODEL_SMART) + 5000) == 200


def test_slice_budget_floor_when_overhead_exceeds_remaining():
    from config import MODEL_SMART
    from graph.budget import request_budget, slice_budget
    assert slice_budget(MODEL_SMART, request_budget(MODEL_SMART), 0) == 200
