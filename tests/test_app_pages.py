import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import graph.build_graph as bg
from accounts.db import AppDB
from accounts.history import InsightHistory
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


def seed_history(tmp_path, user_ids):
    db = AppDB(tmp_path / "app.db")
    hist, sess = InsightHistory(db), SessionTracker(db)
    ins = lambda f: {"finding": f, "evidence": ["e1"], "recommendation": "r1"}
    s1, s2 = sess.start(user_ids[0]), sess.start(user_ids[1])
    hist.add(user_ids[0], s1, "Which category converts best?", ins("Eng leads in June."), None, [{"a": 1}])
    hist.add(user_ids[1], s2, "which category converts BEST", ins("Sales leads in July."), None, [{"a": 2}])
    hist.add(user_ids[1], s2, "Bounce by source?", ins("Organic bounces most."), None, [])
    return s1, s2


def test_analyst_has_only_the_analyze_page(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst")
    assert at.sidebar.radio(key="page").options == ["Analyze"]


def test_manager_sees_history_but_not_admin(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="manager")
    assert at.sidebar.radio(key="page").options == ["Analyze", "History"]


def test_history_lists_all_users_and_filters(monkeypatch, tmp_path):
    a = make_user(tmp_path, "analyst", username="alice")
    b = make_user(tmp_path, "analyst", username="bob")
    seed_history(tmp_path, [a["user_id"], b["user_id"]])
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    assert not at.exception
    df = at.dataframe[0].value
    assert len(df) == 3 and set(df["user"]) == {"alice", "bob"}
    at.text_input(key="hist_text").set_value("bounce").run(timeout=60)
    assert len(at.dataframe[0].value) == 1
    at.text_input(key="hist_text").set_value("")
    at.selectbox(key="hist_user").set_value("alice").run(timeout=60)
    assert set(at.dataframe[0].value["user"]) == {"alice"}
    assert "history_view" in [r["action"] for r in rows(tmp_path)]


def test_history_view_is_audited_once_per_session(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    at.text_input(key="hist_text").set_value("x").run(timeout=60)
    assert [r["action"] for r in rows(tmp_path)].count("history_view") == 1


def test_comparative_report_shows_same_question_across_sessions(monkeypatch, tmp_path):
    a = make_user(tmp_path, "analyst", username="alice")
    b = make_user(tmp_path, "analyst", username="bob")
    seed_history(tmp_path, [a["user_id"], b["user_id"]])
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    assert not at.exception
    options = at.selectbox(key="cmp_question").options
    assert len(options) == 1 and "2 sessions" in options[0] and "Bounce" not in options[0]
    page_text = " ".join(m.value for m in at.markdown)
    assert "Eng leads in June." in page_text and "Sales leads in July." in page_text


def test_history_is_refused_even_if_the_router_offered_it(monkeypatch, tmp_path):
    import ui.router as router

    monkeypatch.setattr(router, "allowed_pages", lambda role: ["Analyze", "History", "Admin"])
    at = start_app(monkeypatch, tmp_path, role="analyst")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    assert not at.exception
    assert any("does not allow" in e.value for e in at.error)
    assert len(at.dataframe) == 0                                  # nothing sensitive rendered
    assert "denied:view_history" in [r["action"] for r in rows(tmp_path)]


def test_history_rows_feed_the_deck_builder(tmp_path):
    from io import BytesIO

    from pptx import Presentation

    from export.deck import build_deck

    user = make_user(tmp_path, "analyst", username="alice")
    db = AppDB(tmp_path / "app.db")
    InsightHistory(db).add(user["user_id"], "s", "q one two",
                           {"finding": "Eng leads conversion.", "evidence": ["e"], "recommendation": "r"},
                           None, [{"a": 1}])
    entries = InsightHistory(db).list()
    assert len(Presentation(BytesIO(build_deck(entries))).slides) == 1


def test_history_deck_export_builds_and_audits(monkeypatch, tmp_path):
    a = make_user(tmp_path, "analyst", username="alice")
    b = make_user(tmp_path, "analyst", username="bob")
    seed_history(tmp_path, [a["user_id"], b["user_id"]])
    import export.deck as deck

    seen = []
    monkeypatch.setattr(deck, "build_deck", lambda entries: seen.append(len(entries)) or b"PK")
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    pick = at.multiselect(key="hist_deck_pick")
    pick.set_value(pick.options[:2]).run(timeout=60)
    assert not at.exception and seen and seen[-1] == 2


def test_history_deck_failure_shows_friendly_warning(monkeypatch, tmp_path):
    a = make_user(tmp_path, "analyst", username="alice")
    seed_history(tmp_path, [a["user_id"], a["user_id"]])
    import export.deck as deck

    def boom(entries):
        raise RuntimeError("SECRET-INTERNAL request timed out")

    monkeypatch.setattr(deck, "build_deck", boom)
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    pick = at.multiselect(key="hist_deck_pick")
    pick.set_value(pick.options[:1]).run(timeout=60)
    assert not at.exception
    assert any("Could not build the slide deck" in w.value and "SECRET-INTERNAL" not in w.value for w in at.warning)


def test_admin_creates_a_user_who_can_log_in(monkeypatch, tmp_path):
    from accounts.auth import AuthService

    at = start_app(monkeypatch, tmp_path, role="admin")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    assert not at.exception
    at.text_input(key="new_username").set_value("carol")
    at.text_input(key="new_password").set_value("carol password 1")
    at.selectbox(key="new_role").set_value("manager")
    at.button(key="create_user").click().run(timeout=60)
    assert not at.exception
    db = AppDB(tmp_path / "app.db")
    assert AuthService(db).authenticate("carol", "carol password 1").ok
    created = db.query("SELECT detail_json FROM audit_log WHERE action = 'user_created'")[-1]["detail_json"]
    assert "carol password 1" not in created and '"role": "manager"' in created
    assert at.text_input(key="new_password").value == ""


def test_manager_is_refused_on_the_admin_page_even_if_offered(monkeypatch, tmp_path):
    import ui.router as router

    monkeypatch.setattr(router, "allowed_pages", lambda role: ["Analyze", "History", "Admin"])
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    assert not at.exception
    assert any("does not allow" in e.value for e in at.error)
    assert not [t for t in at.text_input if t.key == "new_username"]
    assert [r["action"] for r in rows(tmp_path)].count("denied:manage_users") == 1


def test_admin_prompt_and_config_changes_apply_and_are_audited(monkeypatch, tmp_path):
    from graph import prompts

    monkeypatch.setattr(config, "JUDGE_MIN_SCORE", config.JUDGE_MIN_SCORE)      # restored at teardown
    try:
        at = start_app(monkeypatch, tmp_path, role="admin")
        at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
        assert not at.exception
        assert set(at.selectbox(key="prompt_sql").options) == {"default", "v2", "v3"}   # v1 lacks required variables
        at.selectbox(key="prompt_sql").set_value("v2")
        at.button(key="prompt_apply_sql").click().run(timeout=60)
        assert prompts.get_overrides()["sql"] == "v2"
        at.number_input(key="cfg_JUDGE_MIN_SCORE").set_value(4)
        at.button(key="cfg_save").click().run(timeout=60)
        assert not at.exception and config.JUDGE_MIN_SCORE == 4
        actions = [r["action"] for r in rows(tmp_path)]
        assert "prompt_version_changed" in actions and "config_changed" in actions
    finally:
        prompts.set_overrides({})


def test_audit_tab_lists_actions_and_filters(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="admin", question="Which category has the best conversion rate?")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    assert not at.exception

    def audit_frame():
        return [d.value for d in at.dataframe if "action" in d.value.columns][0]

    assert {"question"} <= set(audit_frame()["action"])
    at.selectbox(key="audit_action").set_value("question").run(timeout=60)
    assert set(audit_frame()["action"]) == {"question"}


def test_admin_memory_tab_writes_file_and_shows_it(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "mem")
    at = start_app(monkeypatch, tmp_path, role="admin")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    at.button(key="write_memory").click().run(timeout=60)
    assert not at.exception
    assert (tmp_path / "mem" / "PROJECT_MEMORY.md").is_file() and not (tmp_path / "CLAUDE.md").exists()
    assert any("# Project memory" in c.value for c in at.code)
    assert "config_changed" in [r["action"] for r in rows(tmp_path)]


def test_admin_cannot_disable_own_account_in_ui(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="admin")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    uid = at.session_state["auth"]["user_id"]
    at.button(key=f"toggle_active_{uid}").click().run(timeout=60)
    assert not at.exception
    assert any("cannot disable your own account" in e.value for e in at.error)
