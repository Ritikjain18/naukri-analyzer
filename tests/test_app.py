import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import graph.build_graph as bg
from graph.models import Insight


def test_app_shows_setup_error_without_key(monkeypatch, tmp_path):
    import config

    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)


def test_graph_error_is_persisted_across_reruns(monkeypatch, tmp_path):
    import streamlit as st

    import config
    import graph.build_graph as bg

    class BoomGraph:
        def invoke(self, state, config=None):
            raise RuntimeError("429 rate limit")

    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(bg, "build_graph", lambda *a, **k: BoomGraph())
    st.cache_resource.clear()
    try:
        at = AppTest.from_file("../app.py").run(timeout=60)
        assert not at.exception
        at.chat_input[0].set_value("q").run(timeout=60)
        assert not at.exception
        expected = "Groq rate limit reached"
        assert sum(expected in e.value for e in at.error) == 1
        at.run(timeout=60)
        assert not at.exception
        assert sum(expected in e.value for e in at.error) == 1
        assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant"]
    finally:
        st.cache_resource.clear()


def test_app_does_not_construct_pandas_tool():
    from pathlib import Path

    assert "make_pandas_tool" not in (Path(__file__).parent.parent / "app.py").read_text()


class StubGraph:
    def __init__(self):
        self.states = []
        self.overrides = {}

    def invoke(self, state, config=None):
        self.states.append(state)
        insight = Insight(finding="Engineering has the highest conversion rate.", evidence=["20% vs 10%"],
                          recommendation="Invest.")
        memory = list(state.get("insight_memory", [])) + [
            {"question": state["question"], "insight": insight.model_dump(), "chart": None,
             "slice": [{"category": "Eng", "rate": 0.2}], "approved": False}]
        return {**state, "insight": insight, "data_slice": pd.DataFrame({"category": ["Eng"], "rate": [0.2]}),
                "chart_config": None, "prompts": [{"node": "analyst", "prompt": "p"}], "errors": [],
                "insight_memory": memory, "memory_index": len(memory) - 1, "judge_scores": [], "degraded": False,
                **self.overrides}


@pytest.fixture
def app_with_stub(monkeypatch, tmp_path):
    stub = StubGraph()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(bg, "build_graph", lambda *a, **k: stub)
    st.cache_resource.clear()
    try:
        at = AppTest.from_file("../app.py").run(timeout=60)
        at.chat_input[0].set_value("Which job category has the best conversion rate?").run(timeout=60)
        yield at, stub
    finally:
        st.cache_resource.clear()


def test_answer_carries_memory_index_and_can_be_approved(app_with_stub):
    at, _ = app_with_stub
    assert not at.exception
    assistant = [m for m in at.session_state["messages"] if m["role"] == "assistant"]
    assert len(assistant) == 1 and assistant[0]["memory_index"] == 0
    at.button(key="approve_1").click().run(timeout=60)
    assert not at.exception
    assert at.session_state["shared"]["insight_memory"][0]["approved"] is True
    assert any("Approved" in c.value for c in at.caption)


def test_revise_sends_note_and_previous_finding(app_with_stub):
    at, stub = app_with_stub
    at.text_input(key="note_1").set_value("focus on Mumbai").run(timeout=60)
    at.button(key="revise_1").click().run(timeout=60)
    assert not at.exception
    revised = stub.states[-1]
    assert "focus on Mumbai" in revised["revision_note"]
    assert "Engineering has the highest conversion rate." in revised["revision_note"]
    assert revised["question"] == "Which job category has the best conversion rate?"
    assert isinstance(revised["data_slice"], pd.DataFrame)
    msgs = at.session_state["messages"]
    assert [m["role"] for m in msgs][-2:] == ["user", "assistant"]
    assert msgs[-2]["content"] == "Revise: focus on Mumbai"
