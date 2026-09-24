from datetime import datetime, timedelta, timezone

import pytest

from accounts import auth
from accounts.auth import AccountError, AuthService
from accounts.db import AppDB

PW = "correct horse"


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def svc(tmp_path, clock):
    return AuthService(AppDB(tmp_path / "a.db", clock=clock))


def test_hash_verify_and_salt_uniqueness():
    h1, h2 = auth.hash_password(PW), auth.hash_password(PW)
    assert h1 != h2 and h1.startswith("scrypt$")
    assert auth.verify_password(PW, h1) and not auth.verify_password("wrong password", h1)
    assert not auth.verify_password(PW, "garbage") and not auth.verify_password(PW, h1[:-4] + "AAAA")
    assert PW not in h1


def test_validation_messages():
    with pytest.raises(AccountError, match="3-32 characters"):
        auth.validate_username("a b")
    with pytest.raises(AccountError, match="at least 8"):
        auth.validate_password("short")
    auth.validate_username("Ritik.Jain-1")


def test_bootstrap_and_create_user(svc):
    assert svc.needs_bootstrap()
    uid = svc.create_user("Alice", PW, "admin")
    assert not svc.needs_bootstrap() and svc.get_user(uid).role == "admin"
    with pytest.raises(AccountError, match="already taken"):
        svc.create_user("alice", PW, "analyst")
    with pytest.raises(AccountError, match="Unknown role"):
        svc.create_user("bob", PW, "root")


def test_authenticate_success_updates_last_login(svc):
    svc.create_user("alice", PW, "manager")
    res = svc.authenticate("ALICE", PW)
    assert res.ok and res.user.username == "alice" and res.reason == "ok"
    assert svc.db.one("SELECT last_login FROM users")["last_login"]


def test_wrong_password_unknown_user_look_the_same(svc):
    svc.create_user("alice", PW, "analyst")
    a, b = svc.authenticate("alice", "nope nope nope"), svc.authenticate("ghost", "nope nope nope")
    assert (a.ok, a.reason, a.user) == (b.ok, b.reason, b.user) == (False, "invalid", None)


def test_lockout_after_five_failures_and_expiry(svc, clock):
    svc.create_user("alice", PW, "analyst")
    reasons = [svc.authenticate("alice", "bad password!").reason for _ in range(5)]
    assert reasons == ["invalid"] * 4 + ["locked"]
    assert svc.authenticate("alice", PW).reason == "locked"            # right password still refused
    clock.now += timedelta(minutes=14, seconds=59)
    assert svc.authenticate("alice", PW).reason == "locked"
    clock.now += timedelta(seconds=2)
    assert svc.authenticate("alice", PW).ok                            # lock expired, counter reset
    assert svc.authenticate("alice", "bad password!").reason == "invalid"


def test_success_resets_failed_counter(svc):
    svc.create_user("alice", PW, "analyst")
    for _ in range(4):
        svc.authenticate("alice", "bad password!")
    assert svc.authenticate("alice", PW).ok
    for _ in range(4):
        assert svc.authenticate("alice", "bad password!").reason == "invalid"


def test_disabled_account_cannot_login(svc):
    svc.create_user("root", PW, "admin")
    uid = svc.create_user("alice", PW, "analyst")
    svc.set_active(uid, False)
    assert svc.authenticate("alice", PW).reason == "disabled"


def test_last_admin_protection(svc):
    a = svc.create_user("root", PW, "admin")
    with pytest.raises(AccountError, match="active admin"):
        svc.set_active(a, False)
    with pytest.raises(AccountError, match="active admin"):
        svc.set_role(a, "manager")
    b = svc.create_user("root2", PW, "admin")
    svc.set_role(a, "manager")
    with pytest.raises(AccountError):
        svc.set_active(b, False)


def test_change_and_reset_password(svc):
    svc.create_user("root", PW, "admin")
    uid = svc.create_user("alice", PW, "analyst")
    with pytest.raises(AccountError, match="incorrect"):
        svc.change_password(uid, "wrong password", "new password 1")
    svc.change_password(uid, PW, "new password 1")
    assert svc.authenticate("alice", "new password 1").ok and not svc.authenticate("alice", PW).ok
    for _ in range(5):
        svc.authenticate("alice", "bad password!")
    svc.reset_password(uid, "reset password 2")
    assert svc.authenticate("alice", "reset password 2").ok       # reset clears the lock


def test_list_users_has_no_hashes(svc):
    svc.create_user("alice", PW, "analyst")
    assert [(u.username, u.role, u.active) for u in svc.list_users()] == [("alice", "analyst", True)]
