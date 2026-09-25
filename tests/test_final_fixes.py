"""Regression tests for the phase 3a final-review fix wave."""
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
import streamlit as st

import config
from accounts.audit import AuditLog
from accounts.auth import AccountError, AuthService
from accounts.db import AppDB
from accounts.history import InsightHistory
from accounts.memory import MemoryService
from accounts.permissions import PermissionDenied
from accounts.redact import redact_text
from accounts.sessions import SessionTracker
from accounts.settings import PromptSettings
from tests.fakes import FakeLLM
from tests.helpers import make_user, start_app

INS = {"finding": "Eng leads.", "evidence": ["20%"], "recommendation": "Invest."}
GOOD = json.dumps({"topics": ["t"], "key_findings": ["f"], "open_questions": [], "data_loaded": []})
HR_PHRASES = ["Which candidates lack basic qualifications?", "What is the basic compensation by city?",
              "basic infrastructure requirements", "basic understanding of hiring"]


@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    st.cache_resource.clear()


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    db = AppDB(tmp_path / "a.db", clock=clock)
    uid = AuthService(db).create_user("alice", "correct horse", "analyst")
    return {"clock": clock, "db": db, "uid": uid, "hist": InsightHistory(db), "sess": SessionTracker(db)}


def add_session(env, q="Which category converts best?"):
    sid = env["sess"].start(env["uid"])
    env["hist"].add(env["uid"], sid, q, INS, None, [])
    env["sess"].touch(sid, questions=1)
    return sid


# ---- I1
@pytest.mark.parametrize("phrase", HR_PHRASES)
def test_basic_hr_prose_is_not_redacted(phrase, env):
    assert redact_text(phrase) == phrase
    AuditLog(env["db"]).record(None, "question", question=phrase)
    assert json.loads(env["db"].query("SELECT detail_json FROM audit_log")[0]["detail_json"])["question"] == phrase
    hid = env["hist"].add(env["uid"], "s", phrase, INS, None, [])
    assert env["hist"].get(hid)["question"] == phrase


@pytest.mark.parametrize("text", ["Authorization: Basic dXNlcjpwYXNzd29yZA==", "basic dXNlcjpwYXNzd29yZDEyMw==",
                                  "BASIC dXNlcjpwYXNzd29yZDEyMw=="])
def test_basic_credentials_are_redacted_and_idempotent(text):
    out = redact_text(text)
    assert "dXNlcj" not in out and "[REDACTED]" in out
    assert out.split()[0 if not text.startswith("Auth") else 1].lower() in ("basic", "authorization:")
    assert redact_text(out) == out


def test_authorization_prefix_kept():
    assert redact_text("Authorization: Basic dXNlcjpwYXNz") == "Authorization: Basic [REDACTED]"


# ---- I2
def test_prior_context_is_required_and_rollback_is_refused(tmp_path):
    ps = PromptSettings(AppDB(tmp_path / "a.db"))
    assert ps.compatible_versions("orchestrator") == ["v2"]
    assert ps.compatible_versions("query") == ["v3"]
    with pytest.raises(AccountError, match="prior_context"):
        ps.set_active("query", "v2", 1)
    with pytest.raises(AccountError, match="prior_context"):
        ps.set_active("orchestrator", "v1", 1)


def test_stored_incompatible_override_is_skipped_by_apply(tmp_path):
    from graph import prompts
    db = AppDB(tmp_path / "a.db")
    db.execute("INSERT INTO prompt_settings (name, active_version, updated_by, ts_utc) VALUES ('query','v2',1,'x')")
    try:
        PromptSettings(db).apply()
        assert "query" not in prompts.get_overrides()
    finally:
        prompts.set_overrides({})


# ---- M4 / M5
def test_summarise_pending_caps_per_login_oldest_first(env):
    sids = []
    for i in range(5):
        sids.append(add_session(env, f"Question number {i} please"))
        env["clock"].now += timedelta(minutes=1)
    env["clock"].now += timedelta(minutes=40)
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([GOOD] * 5))
    assert mem.summarise_pending(env["uid"]) == config.MAX_SUMMARIES_PER_LOGIN == 2
    assert [env["sess"].get(s)["summarised"] for s in sids] == [1, 1, 0, 0, 0]
    assert len(env["sess"].pending_for_user(env["uid"], 30)) == 3


def test_summarise_pending_stops_after_first_failure(env):
    for i in range(3):
        add_session(env, f"Question number {i} please")
    env["clock"].now += timedelta(minutes=40)

    class Boom:
        calls = 0

        def invoke(self, prompt):
            Boom.calls += 1
            raise RuntimeError("x")

    mem = MemoryService(env["db"], env["hist"], env["sess"], Boom())
    assert mem.summarise_pending(env["uid"]) == 0
    assert Boom.calls == 1


def test_summarise_pending_stops_after_deferral(env):
    from graph.llm import RateLimitExhausted
    for i in range(3):
        add_session(env, f"Question number {i} please")
    env["clock"].now += timedelta(minutes=40)

    class Limited:
        calls = 0

        def invoke(self, prompt):
            Limited.calls += 1
            raise RateLimitExhausted("x")

    mem = MemoryService(env["db"], env["hist"], env["sess"], Limited())
    assert mem.summarise_pending(env["uid"]) == 0
    assert Limited.calls == 1


def test_session_with_count_but_no_entries_is_not_pending_again(env):
    sid = env["sess"].start(env["uid"])
    env["sess"].touch(sid, questions=1)                     # e.g. only guard-rejected questions
    env["clock"].now += timedelta(minutes=40)
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([]))
    assert mem.summarise_pending(env["uid"]) == 0
    assert env["sess"].pending_for_user(env["uid"], 30) == []
    assert env["sess"].get(sid)["summarised"] == 1


# ---- bidi / zero-width
def test_bidi_and_zero_width_are_stripped_from_prior_context_and_summaries(env):
    from accounts.memory import _clean
    bad = "a​b‏c‪d‮e⁦f⁩g﻿h"
    assert _clean(bad) == "abcdefgh"
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([]))
    sid = add_session(env)
    env["db"].insert("INSERT INTO session_summaries (user_id, session_id, ts_utc, summary_json) VALUES (?,?,?,?)",
                     (env["uid"], sid, "2026-09-20T00:00:00.000000+00:00",
                      json.dumps({"topics": [bad], "key_findings": [], "open_questions": [], "data_loaded": []})))
    ctx = mem.prior_context(env["uid"])
    assert "abcdefgh" in ctx
    assert not any(c in ctx for c in "​‏‪‮⁦⁩﻿")


# ---- M10
def test_permission_denied_is_not_an_oserror():
    assert not issubclass(PermissionDenied, OSError) and issubclass(PermissionDenied, Exception)


# ---- M8
def test_query_v3_puts_prior_context_before_the_data_slice():
    text = (config.ROOT / "prompts" / "query.v3.txt").read_text()
    assert text.index("$prior_context") < text.index("$slice")


# ---- M1
def test_approve_mark_and_audit_fail_independently(monkeypatch, tmp_path):
    q = "Which category has the best conversion rate?"
    for target in ("mark", "audit"):
        st.cache_resource.clear()
        sub = tmp_path / target
        sub.mkdir()
        at = start_app(monkeypatch, sub, role="analyst", question=q)
        if target == "mark":
            monkeypatch.setattr(InsightHistory, "mark_approved", lambda self, hid: (_ for _ in ()).throw(RuntimeError("x")))
        else:
            orig = AuditLog.record

            def rec(self, user, action, *a, **k):
                if action == "approve":
                    raise RuntimeError("x")
                return orig(self, user, action, *a, **k)
            monkeypatch.setattr(AuditLog, "record", rec)
        at.button(key="approve_1").click().run(timeout=60)
        db = AppDB(sub / "app.db")
        actions = [r["action"] for r in db.query("SELECT action FROM audit_log")]
        approved = db.query("SELECT approved FROM insight_history")[0]["approved"]
        if target == "mark":
            assert "approve" in actions and approved == 0
        else:
            assert "approve" not in actions and approved == 1
        monkeypatch.undo()


# ---- M2
def _login_attempt(at, name):
    at.text_input(key="login_username").set_value(name)
    at.text_input(key="login_password").set_value("x")
    at.button(key="login_submit").click().run(timeout=60)


def test_username_tried_only_stores_existing_names(monkeypatch, tmp_path):
    from tests.test_app_auth import fresh_app
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)
    _login_attempt(at, "Hunter2Password")
    _login_attempt(at, "alice")
    rows = AppDB(tmp_path / "app.db").query("SELECT detail_json FROM audit_log ORDER BY id")
    assert "Hunter2Password" not in "".join(r["detail_json"] for r in rows)
    assert '"username_tried": "[unknown]"' in rows[0]["detail_json"]
    assert '"username_tried": "alice"' in rows[1]["detail_json"]


# ---- M6
def test_logout_still_resets_when_audit_fails(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst")

    def boom(self, *a, **k):
        raise RuntimeError("audit down")
    monkeypatch.setattr(AuditLog, "record", boom)
    at.button(key="logout").click().run(timeout=60)
    assert not at.exception and "auth" not in at.session_state and not at.chat_input


def test_login_succeeds_when_audit_fails(monkeypatch, tmp_path):
    from tests.test_app_auth import fresh_app
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)

    def boom(self, user, action, *a, **k):
        if action == "login":
            raise RuntimeError("audit down")
        return 0
    monkeypatch.setattr(AuditLog, "record", boom)
    at.text_input(key="login_username").set_value("alice")
    at.text_input(key="login_password").set_value("correct horse")
    at.button(key="login_submit").click().run(timeout=60)
    assert not at.exception and at.chat_input and at.session_state["auth"]["username"] == "alice"


# ---- M11 / filter collision / M12
def _seed(tmp_path, uid, session_id="abcdefgh12"):
    db = AppDB(tmp_path / "app.db")
    h = InsightHistory(db)
    for sid in (session_id, "zzzzzzzz99", "nullnull00"):
        h.add(uid, sid, "Which category converts best?", INS, None, [])
    return db


def test_comparative_tab_tolerates_null_session_id(monkeypatch, tmp_path):
    a = make_user(tmp_path, "analyst", username="alice")
    db = _seed(tmp_path, a["user_id"])
    db.execute("UPDATE insight_history SET session_id = NULL WHERE id = 3")
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    assert not at.exception


def test_history_user_named_all_is_distinct_from_all_users(monkeypatch, tmp_path):
    a = make_user(tmp_path, "analyst", username="All")
    b = make_user(tmp_path, "analyst", username="bob")
    h = InsightHistory(AppDB(tmp_path / "app.db"))
    h.add(a["user_id"], "s1", "Question from All user", INS, None, [])
    h.add(b["user_id"], "s2", "Question from bob user", INS, None, [])
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    assert len(at.dataframe[0].value) == 2
    at.selectbox(key="hist_user").set_value("All").run(timeout=60)
    assert set(at.dataframe[0].value["user"]) == {"All"}


def test_history_deck_failure_uses_fixed_message(monkeypatch, tmp_path, caplog):
    import export.deck as deck
    a = make_user(tmp_path, "analyst", username="alice")
    _seed(tmp_path, a["user_id"])
    monkeypatch.setattr(deck, "build_deck", lambda e: (_ for _ in ()).throw(RuntimeError("SECRET-INTERNAL **x**")))
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    pick = at.multiselect(key="hist_deck_pick")
    with caplog.at_level(logging.WARNING):
        pick.set_value(pick.options[:1]).run(timeout=60)
    assert [w.value for w in at.warning] == ["Could not build the slide deck."]
    assert "RuntimeError" in caplog.text and "SECRET-INTERNAL" not in caplog.text


def test_deck_failure_message_helper():
    from ui.analyze import deck_failure_message
    assert deck_failure_message(RuntimeError("boom **x**")) == "Could not build the slide deck."
    from graph.llm import RateLimitExhausted
    assert deck_failure_message(RateLimitExhausted("x")).startswith("Could not build the slide deck: Groq rate limit")


def test_admin_prompt_options_exclude_versions_without_prior_context(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="admin")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    assert not at.exception
    assert set(at.selectbox(key="prompt_query").options) == {"default", "v3"}
    assert set(at.selectbox(key="prompt_orchestrator").options) == {"default", "v2"}
