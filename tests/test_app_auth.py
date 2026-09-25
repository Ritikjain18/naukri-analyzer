from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import graph.build_graph as bg
from accounts.auth import AuthService
from accounts.db import AppDB
from tests.helpers import APP_FILE, StubGraph, make_user, start_app


@pytest.fixture(autouse=True)
def _clear_cache():
    yield
    st.cache_resource.clear()


def fresh_app(monkeypatch, tmp_path, graph=None):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(bg, "build_graph", lambda *a, **k: graph or StubGraph())
    st.cache_resource.clear()
    return AppTest.from_file(APP_FILE).run(timeout=60)


def actions(tmp_path):
    return [r["action"] for r in AppDB(tmp_path / "app.db").query("SELECT action FROM audit_log ORDER BY id")]


def test_first_launch_shows_only_admin_creation(monkeypatch, tmp_path):
    at = fresh_app(monkeypatch, tmp_path)
    assert not at.exception
    assert at.text_input(key="bootstrap_username") and not at.chat_input
    assert not at.get("text_input") or all(t.key.startswith("bootstrap_") for t in at.text_input)
    at.text_input(key="bootstrap_username").set_value("root")
    at.text_input(key="bootstrap_password").set_value("correct horse")
    at.text_input(key="bootstrap_confirm").set_value("different pw!")
    at.button(key="bootstrap_create").click().run(timeout=60)
    assert any("Passwords do not match." == e.value for e in at.error)
    at.text_input(key="bootstrap_confirm").set_value("correct horse")
    at.button(key="bootstrap_create").click().run(timeout=60)
    assert not at.exception and at.chat_input                      # created and logged in
    db = AppDB(tmp_path / "app.db")
    assert AuthService(db).list_users()[0].role == "admin"
    rows = db.query("SELECT action, detail_json FROM audit_log")
    created = [r for r in rows if r["action"] == "user_created"]
    assert created and '"bootstrap": true' in created[0]["detail_json"] and "correct horse" not in created[0]["detail_json"]
    assert "login" in [r["action"] for r in rows]


def test_bootstrap_shows_account_errors(monkeypatch, tmp_path):
    at = fresh_app(monkeypatch, tmp_path)
    at.text_input(key="bootstrap_username").set_value("root")
    at.text_input(key="bootstrap_password").set_value("short")
    at.text_input(key="bootstrap_confirm").set_value("short")
    at.button(key="bootstrap_create").click().run(timeout=60)
    assert at.error and not at.chat_input and not at.exception


def test_login_flow_and_failures_are_audited(monkeypatch, tmp_path):
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)
    assert at.text_input(key="login_username") and not at.chat_input
    at.text_input(key="login_username").set_value("alice")
    at.text_input(key="login_password").set_value("wrong password")
    at.button(key="login_submit").click().run(timeout=60)
    assert any("Invalid username or password" in e.value for e in at.error) and not at.chat_input
    at.text_input(key="login_password").set_value("correct horse")
    at.button(key="login_submit").click().run(timeout=60)
    assert at.chat_input and not at.exception
    assert actions(tmp_path) == ["login_failed", "login"]


def test_clear_passwords_removes_every_password_key(monkeypatch):
    from ui import auth_ui

    state = {k: "secret pw" for k in auth_ui.PASSWORD_KEYS}
    state["login_username"] = "alice"
    monkeypatch.setattr(st, "session_state", state)
    auth_ui._clear_passwords()
    assert state == {"login_username": "alice"}


def test_login_calls_clear_passwords_and_keeps_no_secret(monkeypatch, tmp_path):
    from ui import auth_ui

    calls = []
    real = auth_ui._clear_passwords
    monkeypatch.setattr(auth_ui, "_clear_passwords", lambda: (calls.append(1), real()))
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)
    at.text_input(key="login_username").set_value("alice")
    at.text_input(key="login_password").set_value("correct horse")
    at.button(key="login_submit").click().run(timeout=60)
    assert at.chat_input and calls
    assert "correct horse" not in repr(at.session_state)


def test_username_tried_never_stores_a_typed_password(monkeypatch, tmp_path):
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)
    for typed in ("my secret password!", "ghost"):
        at.text_input(key="login_username").set_value(typed)
        at.text_input(key="login_password").set_value("x")
        at.button(key="login_submit").click().run(timeout=60)
    rows = AppDB(tmp_path / "app.db").query("SELECT detail_json FROM audit_log ORDER BY id")
    assert "secret" not in "".join(r["detail_json"] for r in rows)
    assert '"username_tried": "[invalid]"' in rows[0]["detail_json"]
    assert '"username_tried": "ghost"' in rows[1]["detail_json"]


def test_unknown_and_disabled_users_get_the_generic_message(monkeypatch, tmp_path):
    user = make_user(tmp_path, "analyst", username="dora", password="correct horse")
    make_user(tmp_path, "admin", username="boss", password="correct horse")
    AuthService(AppDB(tmp_path / "app.db")).set_active(user["user_id"], False)
    at = fresh_app(monkeypatch, tmp_path)
    for name in ("ghost", "dora"):
        at.text_input(key="login_username").set_value(name)
        at.text_input(key="login_password").set_value("correct horse")
        at.button(key="login_submit").click().run(timeout=60)
        assert [e.value for e in at.error] == ["Invalid username or password."]
    assert actions(tmp_path) == ["login_failed", "login_failed"]


def test_lockout_message_and_audit(monkeypatch, tmp_path):
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)
    for _ in range(7):
        at.text_input(key="login_username").set_value("alice")
        at.text_input(key="login_password").set_value("wrong password")
        at.button(key="login_submit").click().run(timeout=60)
    assert any("Too many failed attempts" in e.value for e in at.error)
    rows = AppDB(tmp_path / "app.db").query("SELECT action, detail_json FROM audit_log ORDER BY id")
    assert [r["action"] for r in rows].count("account_locked") == 1
    assert [r["action"] for r in rows] == ["login_failed"] * 4 + ["account_locked"] + ["login_failed"] * 2
    assert '"reason": "locked"' in rows[-1]["detail_json"]
    assert not any("wrong password" in r["detail_json"] for r in rows)


def test_logout_clears_session_and_summarises(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    assert at.session_state["auth"]["role"] == "analyst"
    at.button(key="logout").click().run(timeout=60)
    assert not at.exception and "auth" not in at.session_state and not at.chat_input
    for key in ("session_id", "shared", "messages"):
        assert key not in at.session_state or not at.session_state[key]
    db = AppDB(tmp_path / "app.db")
    assert actions(tmp_path)[-2:] == ["session_end", "logout"]
    assert db.one("SELECT ended_at FROM sessions")["ended_at"]


def test_end_session_starts_a_fresh_session_and_stays_logged_in(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    first = at.session_state["session_id"]
    at.button(key="end_session").click().run(timeout=60)
    assert not at.exception and at.chat_input and at.session_state["auth"]["role"] == "analyst"
    assert at.session_state["session_id"] != first
    assert not at.session_state["messages"]
    assert actions(tmp_path)[-1] == "session_end"
    assert AppDB(tmp_path / "app.db").one("SELECT ended_at FROM sessions WHERE id = ?", (first,))["ended_at"]


def test_summarise_failure_never_blocks_logout(monkeypatch, tmp_path):
    from accounts.memory import MemoryService

    def boom(self, *a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(MemoryService, "end_session", boom)
    at = start_app(monkeypatch, tmp_path, role="analyst")
    at.button(key="logout").click().run(timeout=60)
    assert not at.exception and "auth" not in at.session_state and not at.chat_input
    assert actions(tmp_path)[-2:] == ["session_end", "logout"]


def test_change_password(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst")
    at.text_input(key="pw_old").set_value("correct horse")
    at.text_input(key="pw_new").set_value("brand new password")
    at.button(key="pw_save").click().run(timeout=60)
    assert not at.exception
    svc = AuthService(AppDB(tmp_path / "app.db"))
    name = at.session_state["auth"]["username"]
    assert svc.authenticate(name, "brand new password").ok
    assert "password_changed" in actions(tmp_path)
    assert "brand new password" not in repr(at.session_state)


def test_pages_follow_role(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst")
    assert at.sidebar.radio(key="page").options == ["Analyze"]
    st.cache_resource.clear()
    (tmp_path / "x").mkdir()
    at = start_app(monkeypatch, tmp_path / "x", role="admin")
    assert at.sidebar.radio(key="page").options == ["Analyze", "History", "Admin"]
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    assert any("Coming soon" in i.value for i in at.info)


def test_key_check_still_precedes_login(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.db")
    at = AppTest.from_file(APP_FILE).run(timeout=30)
    assert any("GROQ_API_KEY" in e.value for e in at.error)
    assert not (tmp_path / "app.db").exists()


def test_no_pandas_tool_anywhere_in_app_code():
    root = Path(__file__).parent.parent
    for f in [root / "app.py", *(root / "ui").glob("*.py")]:
        assert "make_pandas_tool" not in f.read_text()


def test_manager_sees_analyze_and_history_only(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="manager")
    assert at.sidebar.radio(key="page").options == ["Analyze", "History"]
    at.sidebar.radio(key="page").set_value("History").run(timeout=60)
    assert not at.exception and len(at.tabs) == 2   # the real History page, comparative tab included


def test_bootstrap_unreachable_once_users_exist(monkeypatch, tmp_path):
    make_user(tmp_path, "analyst", username="alice")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.db")
    monkeypatch.setattr(bg, "build_graph", lambda *a, **k: StubGraph())
    st.cache_resource.clear()
    at = AppTest.from_file(APP_FILE)
    at.session_state["bootstrap_username"] = "evil"
    at.session_state["bootstrap_password"] = "correct horse"
    at.session_state["bootstrap_confirm"] = "correct horse"
    at.run(timeout=60)
    assert at.text_input(key="login_username") and not at.chat_input
    assert len(AuthService(AppDB(tmp_path / "app.db")).list_users()) == 1


def test_bootstrap_race_shows_error_and_creates_nothing(monkeypatch, tmp_path):
    from accounts.auth import AuthService as AS

    at = fresh_app(monkeypatch, tmp_path)
    make_user(tmp_path, "admin", username="first", password="correct horse")   # someone else won the race
    monkeypatch.setattr(AS, "needs_bootstrap", lambda self: True)              # this session's stale view
    at.text_input(key="bootstrap_username").set_value("second")
    at.text_input(key="bootstrap_password").set_value("correct horse")
    at.text_input(key="bootstrap_confirm").set_value("correct horse")
    at.button(key="bootstrap_create").click().run(timeout=60)
    assert any(e.value == "Setup is already complete." for e in at.error)
    assert [u.username for u in AuthService(AppDB(tmp_path / "app.db")).list_users()] == ["first"]


def mutate(tmp_path):
    return AuthService(AppDB(tmp_path / "app.db"))


def test_demotion_takes_effect_on_next_run(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="admin")
    assert at.sidebar.radio(key="page").options == ["Analyze", "History", "Admin"]
    svc = mutate(tmp_path)
    make_user(tmp_path, "admin", username="second_admin")                       # keep an active admin
    svc.set_role(at.session_state["auth"]["user_id"], "analyst")
    at.run(timeout=60)
    assert not at.exception
    assert at.sidebar.radio(key="page").options == ["Analyze"]
    assert at.session_state["auth"]["role"] == "analyst"


@pytest.mark.parametrize("how,reason", [("disable", "disabled"), ("delete", "deleted")])
def test_revoked_user_is_logged_out_with_message_and_audit(monkeypatch, tmp_path, how, reason):
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    uid, sid = at.session_state["auth"]["user_id"], at.session_state["session_id"]
    db = AppDB(tmp_path / "app.db")
    make_user(tmp_path, "analyst", username="someone_else")      # keep the login form (not bootstrap) reachable
    if how == "disable":
        db.execute("UPDATE users SET active = 0 WHERE id = ?", (uid,))
    else:
        db.execute("DELETE FROM users WHERE id = ?", (uid,))
    at.run(timeout=60)
    assert not at.exception and not at.chat_input and at.text_input(key="login_username")
    assert [w.value for w in at.warning] == ["Your session has ended. Please log in again."]
    assert "auth" not in at.session_state and "session_id" not in at.session_state
    row = [r for r in db.query("SELECT * FROM audit_log") if r["action"] == "session_revoked"]
    assert len(row) == 1 and row[0]["session_id"] == sid and f'"reason": "{reason}"' in row[0]["detail_json"]
    at.run(timeout=60)
    assert not at.warning                                                       # flash shown once


def test_unchanged_user_unaffected_and_one_lookup_per_run(monkeypatch, tmp_path):
    calls = []
    real = AuthService.get_user
    monkeypatch.setattr(AuthService, "get_user", lambda self, uid: (calls.append(uid), real(self, uid))[1])
    at = start_app(monkeypatch, tmp_path, role="manager")
    calls.clear()
    at.run(timeout=60)
    assert not at.exception and at.chat_input and calls == [at.session_state["auth"]["user_id"]]
    assert at.sidebar.radio(key="page").options == ["Analyze", "History"]


def test_failed_end_session_warning_is_flashed_once(monkeypatch, tmp_path):
    from accounts.memory import MemoryService

    def boom(self, *a, **k):
        raise RuntimeError("secret llm detail")

    monkeypatch.setattr(MemoryService, "end_session", boom)
    at = start_app(monkeypatch, tmp_path, role="analyst")
    at.button(key="logout").click().run(timeout=60)
    msg = "Could not save the session summary (it will be retried later)."
    assert [w.value for w in at.warning] == [msg] and "secret" not in msg
    at.run(timeout=60)
    assert not at.warning


def test_failed_summarise_pending_at_login_is_flashed_once(monkeypatch, tmp_path):
    from accounts.memory import MemoryService

    def boom(self, *a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(MemoryService, "summarise_pending", boom)
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)
    at.text_input(key="login_username").set_value("alice")
    at.text_input(key="login_password").set_value("correct horse")
    at.button(key="login_submit").click().run(timeout=60)
    assert at.chat_input
    assert [w.value for w in at.warning] == ["Could not save the session summary (it will be retried later)."]
    at.run(timeout=60)
    assert not at.warning


def test_router_dispatches_through_the_module(monkeypatch, tmp_path):
    import ui.analyze as analyze

    monkeypatch.setattr(analyze, "render_analyze", lambda ctx: st.write("PATCHED PAGE"))
    at = start_app(monkeypatch, tmp_path, role="analyst")
    assert any("PATCHED PAGE" in m.value for m in at.markdown)
