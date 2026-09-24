import csv
import io
from datetime import datetime, timedelta, timezone

import pytest

from accounts.audit import AuditLog
from accounts.auth import User
from accounts.db import AppDB
from accounts.permissions import PERMISSIONS, PermissionDenied, can, require

ALL = {"upload", "ask", "export", "view_history", "view_comparative",
       "manage_users", "manage_prompts", "manage_config", "view_audit"}


def test_permission_matrix_exact():
    assert PERMISSIONS["analyst"] == {"upload", "ask", "export"}
    assert PERMISSIONS["manager"] == {"upload", "ask", "export", "view_history", "view_comparative"}
    assert PERMISSIONS["admin"] == ALL


@pytest.mark.parametrize("role", ["analyst", "manager", "admin"])
@pytest.mark.parametrize("perm", sorted(ALL))
def test_can_and_require_agree(role, perm):
    allowed = perm in PERMISSIONS[role]
    assert can(role, perm) is allowed
    if allowed:
        require(role, perm)
    else:
        with pytest.raises(PermissionDenied):
            require(role, perm)


def test_unknown_role_or_permission_is_denied():
    assert not can("root", "ask") and not can("admin", "launch_rockets")
    with pytest.raises(PermissionDenied):
        require(None, "ask")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


@pytest.fixture
def audit(tmp_path):
    clock = Clock()
    log = AuditLog(AppDB(tmp_path / "a.db", clock=clock))
    log.clock = clock
    return log


def test_record_and_query_newest_first(audit):
    alice = User(1, "alice", "analyst", True)
    audit.record(alice, "login", session_id="s1")
    audit.clock.now += timedelta(minutes=1)
    audit.record(alice, "question", session_id="s1", question="How many jobs?", models=["m"])
    rows = audit.query()
    assert [r["action"] for r in rows] == ["question", "login"]
    assert rows[0]["username"] == "alice" and rows[0]["session_id"] == "s1"
    assert rows[0]["detail"] == {"question": "How many jobs?", "models": ["m"]}


def test_failed_login_has_no_user_id(audit):
    audit.record(None, "login_failed", username_tried="ghost")
    row = audit.query()[0]
    assert row["user_id"] is None and row["detail"]["username_tried"] == "ghost"


def test_secrets_are_scrubbed_and_long_values_cut(audit):
    audit.record({"id": 2, "username": "bob"}, "user_created", password="hunter2", API_KEY="gsk_x",
                 password_hash="h", token="t", note="x" * 5000, keep="ok", obj=object())
    detail = audit.query()[0]["detail"]
    assert set(detail) == {"note", "keep", "obj"} and len(detail["note"]) == 2000 and isinstance(detail["obj"], str)
    raw = audit.db.one("SELECT detail_json FROM audit_log")["detail_json"]
    assert "hunter2" not in raw and "gsk_x" not in raw


def test_query_filters(audit):
    alice, bob = User(1, "Alice", "analyst", True), User(2, "bob", "admin", True)
    audit.record(alice, "login")
    audit.clock.now += timedelta(days=2)
    audit.record(bob, "config_changed")
    audit.record(bob, "login")
    assert {r["username"] for r in audit.query(user="ali")} == {"Alice"}
    assert {r["action"] for r in audit.query(action="login")} == {"login"} and len(audit.query(action="login")) == 2
    cutoff = audit.clock.now - timedelta(days=1)
    assert len(audit.query(since=cutoff)) == 2 and len(audit.query(until=cutoff)) == 1
    assert len(audit.query(limit=1)) == 1


def test_to_csv_round_trip(audit):
    audit.record(User(1, "alice", "analyst", True), "upload", file="a.csv", rows=3)
    rows = list(csv.DictReader(io.StringIO(audit.to_csv(audit.query()))))
    assert rows[0]["username"] == "alice" and rows[0]["action"] == "upload" and '"rows": 3' in rows[0]["detail"]
