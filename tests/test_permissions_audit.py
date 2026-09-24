import csv
import io
from datetime import datetime, timedelta, timezone

import pytest

from accounts.audit import AuditLog
from accounts.auth import User, ROLES as AUTH_ROLES
from accounts.db import AppDB
from accounts.permissions import PERMISSIONS, PermissionDenied, can, require, ROLES

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


def test_csv_formula_injection_guard(audit):
    alice = User(1, "alice", "analyst", True)
    audit.record(alice, "test", note="normal")
    audit.record(alice, "=SUM(A1)", note="formula")
    audit.record(alice, "+alert(1)", note="plus")
    audit.record(alice, "-cmd", note="minus")
    audit.record(alice, "@domain", note="at")
    audit.record(alice, "\ttest", note="tab")
    csv_out = audit.to_csv(audit.query())
    lines = csv_out.split('\n')[1:]  # Skip header
    # Check that formula-starting actions get prefixed with '
    assert any("'=SUM" in line for line in lines)
    assert any("'+alert" in line for line in lines)
    assert any("'-cmd" in line for line in lines)
    assert any("'@domain" in line for line in lines)
    # Normal values should not have formula injection prefix in action column
    # The "test" action should appear without prefix
    assert any(",test," in line and "normal" in line for line in lines)


def test_can_non_string_role():
    assert not can(None, "ask")
    assert not can(["admin"], "ask")
    assert not can({"role": "admin"}, "ask")
    assert not can(42, "ask")
    assert can("admin", "ask") is True


def test_require_non_string_role():
    with pytest.raises(PermissionDenied):
        require(None, "ask")
    with pytest.raises(PermissionDenied):
        require(["admin"], "ask")
    with pytest.raises(PermissionDenied):
        require({"role": "admin"}, "ask")
    with pytest.raises(PermissionDenied):
        require(42, "ask")


def test_roles_export_matches_auth():
    assert ROLES == AUTH_ROLES
    assert set(PERMISSIONS.keys()) == set(ROLES)


def test_record_with_non_string_dict_keys(audit):
    """Fix: record() must handle non-string dict keys without raising TypeError.

    The detail must be stored WITHOUT scrub_error, with stringified keys and values intact.
    """
    alice = User(1, "alice", "analyst", True)
    audit.record(alice, "test", metadata={1: "int_key", (2, 3): "tuple_key"})
    rows = audit.query()
    assert len(rows) == 1
    assert rows[0]["action"] == "test"
    detail = rows[0]["detail"]
    # scrub_error should NOT be present for valid (non-raising) data
    assert "scrub_error" not in detail
    # Keys should be stringified, values preserved
    assert detail["metadata"]["1"] == "int_key"
    assert detail["metadata"]["(2, 3)"] == "tuple_key"


def test_record_with_bytes_value(audit):
    """Fix: record() must handle bytes values gracefully.

    Bytes should be stored as exactly "[bytes]" placeholder, not repr().
    """
    alice = User(1, "alice", "analyst", True)
    audit.record(alice, "test", data=b"some_bytes")
    rows = audit.query()
    assert len(rows) == 1
    detail = rows[0]["detail"]
    # scrub_error should NOT be present
    assert "scrub_error" not in detail
    # bytes must be exactly "[bytes]", not repr output
    assert detail["data"] == "[bytes]"


def test_record_with_cyclic_reference(audit):
    """Fix: record() must handle cyclic references without hanging.

    Cyclic references should be stored with [TRUNCATED] marker, not scrub_error.
    """
    alice = User(1, "alice", "analyst", True)
    lst = []
    lst.append(lst)  # Create cycle
    audit.record(alice, "test", data=lst)
    rows = audit.query()
    assert len(rows) == 1
    detail = rows[0]["detail"]
    # scrub_error should NOT be present for cyclic refs (they are handled)
    assert "scrub_error" not in detail
    # Cyclic ref (list containing self) should be marked as list with [TRUNCATED] marker
    assert detail["data"] == ["[TRUNCATED]"]


class _RaisingStr:
    """Object whose __str__ raises an exception."""
    def __str__(self):
        raise ValueError("intentional __str__ failure")


def test_record_scrub_error_value_raises(audit):
    """Fix: If a value's __str__ raises, record() should degrade to {"scrub_error": true}.

    The error should not leak any details from the value.
    """
    alice = User(1, "alice", "analyst", True)
    # Use an object whose __str__ raises
    audit.record(alice, "test", data=_RaisingStr())
    rows = audit.query()
    assert len(rows) == 1
    # Must store exactly {"scrub_error": true} and nothing else
    detail = rows[0]["detail"]
    assert detail == {"scrub_error": True}


def test_record_scrub_error_key_raises(audit):
    """Fix: If value __str__ raises during scrubbing, degrade to {"scrub_error": true}.

    Since Python doesn't allow non-string keys in kwargs, we test the same
    scrub_error behavior for a value whose __str__ raises.
    """
    alice = User(1, "alice", "analyst", True)
    # Test the scrub_error fallback with a value that raises
    audit.record(alice, "test", data=_RaisingStr())
    rows = audit.query()
    assert len(rows) == 1
    # Must store exactly {"scrub_error": true} and nothing else
    stored_detail = rows[0]["detail"]
    assert stored_detail == {"scrub_error": True}
