import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import graph.build_graph as bg
from accounts.db import AppDB
from accounts.sessions import SessionTracker
from tests.helpers import StubGraph, make_user, start_app


@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    st.cache_resource.clear()


def rows(tmp_path, sql="SELECT * FROM audit_log ORDER BY id"):
    return AppDB(tmp_path / "app.db").query(sql)


def test_question_writes_audit_history_and_session(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    assert not at.exception
    q = [r for r in rows(tmp_path) if r["action"] == "question"]
    assert len(q) == 1 and '"question": "Which category has the best conversion rate?"' in q[0]["detail_json"]
    assert q[0]["username"] and q[0]["session_id"]
    h = rows(tmp_path, "SELECT * FROM insight_history")
    assert len(h) == 1 and h[0]["question_norm"] == "which category has the best conversion rate"
    assert rows(tmp_path, "SELECT question_count FROM sessions")[0]["question_count"] == 1
    msg = [m for m in at.session_state["messages"] if m["role"] == "assistant"][0]
    assert msg["history_id"] == h[0]["id"]


def test_approve_marks_history_and_audits(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    at.button(key="approve_1").click().run(timeout=60)
    assert rows(tmp_path, "SELECT approved FROM insight_history")[0]["approved"] == 1
    assert "approve" in [r["action"] for r in rows(tmp_path)]


def test_revision_is_audited_and_not_counted_as_a_new_question(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    at.text_input(key="note_1").set_value("focus on Mumbai").run(timeout=60)
    at.button(key="revise_1").click().run(timeout=60)
    actions = [r["action"] for r in rows(tmp_path)]
    assert "revise" in actions
    assert rows(tmp_path, "SELECT question_count FROM sessions")[0]["question_count"] == 1


def test_guard_rejected_answers_are_not_stored_in_history(monkeypatch, tmp_path):
    stub = StubGraph()
    stub.overrides = {"guard_rejected": True}
    start_app(monkeypatch, tmp_path, role="analyst", graph=stub, question="What is the weather in Paris today?")
    assert rows(tmp_path, "SELECT * FROM insight_history") == []
    q = [r for r in rows(tmp_path) if r["action"] == "question"]
    assert '"guard_rejected": true' in q[0]["detail_json"]


def _ctx_script():
    import streamlit as st
    from accounts.services import build_services
    from tests.fakes import FakeLLM
    from ui import analyze
    from ui.context import Ctx

    services = build_services(st.session_state["_db"], FakeLLM([]))
    ctx = Ctx(store=None, ingest_node=None, graph=None, llms=(), services=services,
              user={"id": 1, "username": "eve", "role": "nobody"}, session_id="s1", shared={}, messages=[])
    st.session_state["ok"] = analyze.ask(ctx, "Which category has the best conversion rate?")
    st.session_state["msgs"] = ctx.messages


def test_unauthorised_role_is_refused_audited_and_not_shown(tmp_path):
    at = AppTest.from_function(_ctx_script)
    at.session_state["_db"] = str(tmp_path / "app.db")
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["ok"] is False and at.session_state["msgs"] == []
    assert any("does not allow" in e.value for e in at.error)
    assert "denied:ask" in [r["action"] for r in rows(tmp_path)]
    assert rows(tmp_path, "SELECT * FROM insight_history") == []


def test_failing_history_and_audit_writes_never_lose_the_answer(monkeypatch, tmp_path):
    from accounts.audit import AuditLog
    from accounts.history import InsightHistory

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(InsightHistory, "add", boom)
    monkeypatch.setattr(AuditLog, "record", boom)
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    assert not at.exception
    assert [m for m in at.session_state["messages"] if m["role"] == "assistant" and "insight" in m]
    assert any("Could not record" in w.value for w in at.warning)


def test_upload_is_audited_without_contents(monkeypatch, tmp_path):
    from ui.analyze import audit_upload
    from accounts.services import build_services
    from tests.fakes import FakeLLM
    from ui.context import Ctx

    services = build_services(tmp_path / "app.db", FakeLLM([]))
    ctx = Ctx(store=None, ingest_node=None, graph=None, llms=(), services=services,
              user={"id": 1, "username": "a", "role": "analyst"}, session_id="s1", shared={}, messages=[])
    audit_upload(ctx, "cands.csv", "candidates", "created", 5)
    d = services.audit.query(action="upload")[0]["detail"]
    assert d == {"file": "cands.csv", "table": "candidates", "action": "created", "rows": 5}


def test_prior_context_reaches_the_graph_state(monkeypatch, tmp_path):
    stub = StubGraph()
    at = start_app(monkeypatch, tmp_path, role="analyst", graph=stub)
    at.session_state["shared"]["prior_context"] = "Session 2026-09-24: topics: conversion."
    at.chat_input[0].set_value("Which category has the best conversion rate?").run(timeout=60)
    assert stub.states[-1]["prior_context"].startswith("Session 2026-09-24")
    assert at.session_state["shared"]["prior_context"].startswith("Session 2026-09-24")


def test_login_loads_prior_context_from_stored_summaries(monkeypatch, tmp_path):
    import json

    user = make_user(tmp_path, "analyst", username="alice", password="correct horse")
    db = AppDB(tmp_path / "app.db")
    sid = SessionTracker(db).start(user["user_id"])
    db.insert("INSERT INTO session_summaries (user_id, session_id, ts_utc, summary_json) VALUES (?,?,?,?)",
              (user["user_id"], sid, "2026-09-24T10:00:00.000000+00:00",
               json.dumps({"topics": ["conversion by category"], "key_findings": ["Eng leads"],
                           "open_questions": [], "data_loaded": []})))
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(bg, "build_graph", lambda *a, **k: StubGraph())
    st.cache_resource.clear()
    try:
        at = AppTest.from_file("../app.py").run(timeout=60)
        at.text_input(key="login_username").set_value("alice")
        at.text_input(key="login_password").set_value("correct horse")
        at.button(key="login_submit").click().run(timeout=60)
        assert not at.exception
        ctx = at.session_state["shared"]["prior_context"]
        assert "conversion by category" in ctx and "Eng leads" in ctx and ctx.startswith("Session 2026-09-24")
    finally:
        st.cache_resource.clear()


def test_export_audit_callback_records_count(tmp_path):
    from accounts.services import build_services
    from tests.fakes import FakeLLM
    from ui.analyze import audit_export
    from ui.context import Ctx

    services = build_services(tmp_path / "app.db", FakeLLM([]))
    uid = services.auth.create_user("alice", "correct horse", "analyst")
    ctx = Ctx(store=None, ingest_node=None, graph=None, llms=(), services=services,
              user={"id": uid, "username": "alice", "role": "analyst"}, session_id="s1", shared={}, messages=[])
    audit_export(ctx, 2)
    row = services.audit.query(action="export")[0]
    assert row["username"] == "alice" and row["detail"]["insights"] == 2 and row["session_id"] == "s1"
    assert row["detail"]["source"] == "analyze"
