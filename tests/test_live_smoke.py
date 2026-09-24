import json
import os
from pathlib import Path

import pytest

import config
from data.seed import seed_database
from graph.build_graph import build_graph
from graph.llm import get_llm
from graph.tools import make_sql_tool

QUESTIONS = json.loads((Path(__file__).parent / "eval_questions.json").read_text())

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    from data.store import SQLiteStore

    store = SQLiteStore(tmp_path_factory.mktemp("live") / "live.db")
    seed_database(store, scale=0.2)
    fast, smart = get_llm(config.MODEL_FAST), get_llm(config.MODEL_SMART)
    return build_graph(store, fast, smart, make_sql_tool(store, fast))


@pytest.mark.parametrize("item", QUESTIONS, ids=[q["question"][:40] for q in QUESTIONS])
def test_live_question(graph, item):
    result = graph.invoke({"question": item["question"], "prompts": [], "errors": []})
    assert result["query_type"] == item["expect_tool"]
    assert len(result["data_slice"]) > 0
    assert result["insight"].finding.strip()
    assert result["insight"].recommendation.strip()
