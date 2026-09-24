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
