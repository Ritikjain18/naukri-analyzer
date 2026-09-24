from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import graph.build_graph as bg
from accounts.auth import AuthService
from accounts.db import AppDB
from graph.models import Insight

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


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


def make_user(tmp_path, role="analyst", username=None, password="correct horse") -> dict:
    username = username or f"{role}_user"
    uid = AuthService(AppDB(tmp_path / "app.db")).create_user(username, password, role)
    return {"user_id": uid, "username": username, "role": role}


def start_app(monkeypatch, tmp_path, role="analyst", graph=None, question=None, key="gsk_fake") -> AppTest:
    """Run app.py fully against tmp paths as a logged-in user. Callers must clear st.cache_resource in a finally."""
    monkeypatch.setenv("GROQ_API_KEY", key)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.db")
    graph = graph if graph is not None else StubGraph()
    monkeypatch.setattr(bg, "build_graph", lambda *a, **k: graph)
    st.cache_resource.clear()
    auth = make_user(tmp_path, role)
    at = AppTest.from_file(APP_FILE)
    at.stub = graph
    at.session_state["auth"] = auth
    at.run(timeout=60)
    if question is not None:
        at.chat_input[0].set_value(question).run(timeout=60)
    return at
