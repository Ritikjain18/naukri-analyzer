from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from accounts.audit import AuditLog
from accounts.db import AppDB


def run_guard(tmp_path, role, permission):
    db = AppDB(tmp_path / "app.db")
    ctx = SimpleNamespace(user={"id": 1, "username": "u", "role": role}, session_id="s1",
                          services=SimpleNamespace(audit=AuditLog(db)))

    def script():
        from ui.permissions_ui import guard
        import streamlit as st
        st.session_state["allowed"] = guard(st.session_state["_ctx"], st.session_state["_perm"])

    at = AppTest.from_function(script)
    at.session_state["_ctx"] = ctx
    at.session_state["_perm"] = permission
    at.run(timeout=30)
    return at, db


def test_guard_allows_and_is_silent(tmp_path):
    at, db = run_guard(tmp_path, "analyst", "ask")
    assert at.session_state["allowed"] is True and not at.error
    assert db.query("SELECT * FROM audit_log") == []


def test_guard_denies_shows_error_and_audits(tmp_path):
    at, db = run_guard(tmp_path, "analyst", "view_audit")
    assert at.session_state["allowed"] is False
    assert any("view_audit" in e.value for e in at.error)
    assert [r["action"] for r in db.query("SELECT action FROM audit_log")] == ["denied:view_audit"]
