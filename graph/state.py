import copy
from typing import Optional, TypedDict

import pandas as pd

from graph.models import ChartConfig, Insight


class AnalyzerState(TypedDict, total=False):
    upload: dict
    df: pd.DataFrame
    table_name: str
    ingest_action: str
    schema: str
    data_summary: str
    question: str
    query_type: str
    chat_history: list
    data_slice: pd.DataFrame
    insight: Insight
    chart_config: Optional[ChartConfig]
    insight_memory: list
    prompts: list
    errors: list
    guard_rejected: bool
    guard_error: str
    guard_failures: int
    revision_note: str
    judge_correction: str
    degraded: bool


TURN_FIELDS = {
    "prompts": [], "errors": [], "guard_rejected": False, "guard_error": "", "guard_failures": 0,
    "revision_note": "", "judge_correction": "", "degraded": False,
}


def new_turn(shared: dict, question: str, **extra) -> dict:
    return {**shared, **copy.deepcopy(TURN_FIELDS), "question": question, **extra}
