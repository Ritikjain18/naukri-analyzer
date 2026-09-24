import json
import os
import time
from pathlib import Path

import pytest

import config
from data.seed import seed_database
from graph.build_graph import build_graph
from graph.llm import RateLimitExhausted, build_llm, friendly_error, is_rate_limit
from graph.ratelimit import RateLimitManager
from graph.state import new_turn
from graph.tools import make_sql_tool

QUESTIONS = json.loads((Path(__file__).parent / "eval_questions.json").read_text())

RATE_LIMIT_BACKOFF_SECONDS = 65
PACE_SECONDS = 10
_SQL_MARK = "--- model output ---\n"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    from data.store import SQLiteStore

    base = tmp_path_factory.mktemp("live")
    store = SQLiteStore(base / "live.db")
    seed_database(store, scale=0.2)
    manager = RateLimitManager(base / "usage.db")  # fresh log, never data/usage.db
    fast = build_llm([config.MODEL_FAST], manager)
    smart = build_llm([config.MODEL_SMART, config.MODEL_FAST], manager)
    graph = build_graph(store, fast, smart, make_sql_tool(store, fast))
    return graph, (fast, smart)


def _invoke(graph, llms, question):
    for llm in llms:
        llm.reset()
    return graph.invoke(new_turn({}, question), config={"recursion_limit": 60})


def _generated_sql(prompts):
    for entry in reversed(prompts):
        if entry.get("node") == "retrieve-sql" and _SQL_MARK in entry.get("prompt", ""):
            return entry["prompt"].split(_SQL_MARK, 1)[1]
    return None


def _write_report(question, result, llms):
    path = os.environ.get("LIVE_REPORT_PATH")
    if not path:
        return
    insight = result["insight"]
    chart = result.get("chart_config")
    row = {
        "question": question,
        "finding": insight.finding,
        "evidence": list(insight.evidence),
        "recommendation": insight.recommendation,
        "chart": chart.model_dump() if chart is not None else None,
        "errors": result.get("errors", []),
        "judge_scores": result.get("judge_scores", []),
        "models": sorted({m for llm in llms for m in llm.used_models()}),
        "degraded": bool(result.get("degraded")),
        "num_prompts": len(result.get("prompts", [])),
        "sql": _generated_sql(result.get("prompts", [])),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


@pytest.mark.parametrize("item", QUESTIONS, ids=[q["question"][:40] for q in QUESTIONS])
def test_live_question(runtime, item):
    graph, llms = runtime
    question = item["question"]
    time.sleep(PACE_SECONDS)
    try:
        try:
            result = _invoke(graph, llms, question)
        except Exception as exc:
            if not (isinstance(exc, RateLimitExhausted) or is_rate_limit(exc)):
                raise
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
            result = _invoke(graph, llms, question)
    except Exception as exc:
        pytest.fail(friendly_error(exc))
    _write_report(question, result, llms)
    assert not result.get("degraded"), f"degraded answer; guard error: {result.get('guard_error')}"
    assert not result.get("guard_rejected"), f"guard rejected: {result.get('guard_error')}"
    assert result["query_type"] == item["expect_tool"]
    assert len(result["data_slice"]) > 0
    assert result["insight"].finding.strip()
    assert result["insight"].recommendation.strip()
