import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import config
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
    with pytest.raises(AccountError, match=r"^Username must be 3-32 characters: letters, digits, '_', '\.', '-'\.$"):
        auth.validate_username("a b")
    with pytest.raises(AccountError, match=r"^Password must be at least 8 characters\.$"):
        auth.validate_password("short")
    auth.validate_username("Ritik.Jain-1")


def test_bootstrap_and_create_user(svc):
    assert svc.needs_bootstrap()
    uid = svc.create_user("Alice", PW, "admin")
    assert not svc.needs_bootstrap() and svc.get_user(uid).role == "admin"
    with pytest.raises(AccountError, match=r"^That username is already taken\.$"):
        svc.create_user("alice", PW, "analyst")
    with pytest.raises(AccountError, match=r"^Unknown role\.$"):
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
    with pytest.raises(AccountError, match=r"^At least one active admin is required\.$"):
        svc.set_active(a, False)
    with pytest.raises(AccountError, match=r"^At least one active admin is required\.$"):
        svc.set_role(a, "manager")
    b = svc.create_user("root2", PW, "admin")
    svc.set_role(a, "manager")
    with pytest.raises(AccountError):
        svc.set_active(b, False)


def test_change_and_reset_password(svc):
    svc.create_user("root", PW, "admin")
    uid = svc.create_user("alice", PW, "analyst")
    with pytest.raises(AccountError, match=r"^Current password is incorrect\.$"):
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


def _hammer(fn, n):
    barrier, out = threading.Barrier(n), []

    def run(i):
        barrier.wait()
        try:
            out.append(fn(i))
        except Exception as exc:
            out.append(exc)

    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def test_concurrent_wrong_guesses_are_all_counted(svc, monkeypatch):
    svc.create_user("alice", PW, "analyst")

    def slow_false(password, stored):
        time.sleep(0.02)
        return False

    monkeypatch.setattr(auth, "verify_password", slow_false)
    res = _hammer(lambda i: svc.authenticate("alice", "bad password!"), 20)
    reasons = [r.reason for r in res]
    assert reasons.count("invalid") == config.LOCKOUT_ATTEMPTS - 1
    assert reasons.count("locked") == 20 - (config.LOCKOUT_ATTEMPTS - 1)
    row = svc.db.one("SELECT failed_attempts, locked_until FROM users")
    assert row["failed_attempts"] == config.LOCKOUT_ATTEMPTS and row["locked_until"]
    monkeypatch.undo()
    assert svc.authenticate("alice", PW).reason == "locked"


def test_slow_verification_cannot_erase_a_lock(svc, monkeypatch):
    svc.create_user("alice", PW, "analyst")
    started = threading.Event()

    def verify(password, stored):
        if password == "slow one!!!":
            started.set()
            time.sleep(0.3)
        return False

    monkeypatch.setattr(auth, "verify_password", verify)
    slow = threading.Thread(target=lambda: svc.authenticate("alice", "slow one!!!"))
    slow.start()
    started.wait(2)
    for _ in range(6):
        svc.authenticate("alice", "fast one!!!")
    slow.join()
    monkeypatch.undo()
    assert svc.db.one("SELECT locked_until FROM users")["locked_until"]
    assert svc.authenticate("alice", PW).reason == "locked"


def test_concurrent_admin_disable_keeps_one_admin(svc, monkeypatch):
    a = svc.create_user("root", PW, "admin")
    b = svc.create_user("root2", PW, "admin")
    real = svc._active_admins

    def slow_count():
        n = real()
        time.sleep(0.05)
        return n

    monkeypatch.setattr(svc, "_active_admins", slow_count)
    res = _hammer(lambda i: svc.set_active(b if i == 0 else a, False), 2)
    assert sum(isinstance(r, AccountError) for r in res) == 1
    assert real() == 1


def test_unknown_user_id_raises(svc):
    svc.create_user("root", PW, "admin")
    for call in (
        lambda: svc.set_role(999, "analyst"),
        lambda: svc.set_active(999, False),
        lambda: svc.reset_password(999, "reset password 2"),
        lambda: svc.change_password(999, PW, "new password 1"),
    ):
        with pytest.raises(AccountError, match=r"^Unknown user\.$"):
            call()


def _parts(h):
    return h.split("$")


def test_verify_rejects_tampered_parameters():
    h = auth.hash_password(PW)
    scheme, n, r, p, salt, dig = _parts(h)
    t0 = time.perf_counter()
    assert not auth.verify_password(PW, "$".join([scheme, n, r, "100", salt, dig]))
    assert time.perf_counter() - t0 < 1
    assert not auth.verify_password(PW, "$".join([scheme, str(2**30), r, p, salt, dig]))
    assert not auth.verify_password(PW, "$".join([scheme, "17", r, p, salt, dig]))
    assert not auth.verify_password(PW, "$".join([scheme, n, "64", p, salt, dig]))
    short = auth._b64(b"x" * 16)
    assert not auth.verify_password(PW, "$".join([scheme, n, r, p, salt, short]))
    flipped = ("B" if dig[0] != "B" else "C") + dig[1:]
    assert len(flipped) == len(dig)
    assert not auth.verify_password(PW, "$".join([scheme, n, r, p, salt, flipped]))
    assert auth.verify_password(PW, h)


def test_lone_surrogate_password_is_rejected_cleanly(svc):
    bad = "abc\ud800defgh"
    msg = r"^Password contains unsupported characters\.$"
    with pytest.raises(AccountError, match=msg):
        auth.validate_password(bad)
    with pytest.raises(AccountError, match=msg):
        svc.create_user("alice", bad, "analyst")
    uid = svc.create_user("alice", PW, "analyst")
    with pytest.raises(AccountError, match=msg):
        svc.change_password(uid, PW, bad)
    with pytest.raises(AccountError, match=msg):
        svc.reset_password(uid, bad)
    assert not auth.verify_password(bad, auth.hash_password(PW))
    assert svc.authenticate("alice", bad).reason == "invalid"


def test_newly_locked_only_on_the_triggering_attempt(svc):
    svc.create_user("alice", PW, "analyst")
    results = [svc.authenticate("alice", "bad password!") for _ in range(7)]
    assert [r.newly_locked for r in results] == [False] * 4 + [True] + [False] * 2
    assert [r.reason for r in results] == ["invalid"] * 4 + ["locked"] * 3
    assert svc.authenticate("alice", PW).newly_locked is False


def test_create_first_admin_only_when_empty(svc):
    uid = svc.create_first_admin("root", PW)
    assert svc.get_user(uid).role == "admin"
    with pytest.raises(AccountError) as exc:
        svc.create_first_admin("other", PW)
    assert str(exc.value) == "Setup is already complete."
    assert [u.username for u in svc.list_users()] == ["root"]


def test_create_first_admin_race_creates_exactly_one(svc):
    barrier, outcomes = threading.Barrier(2), []
    assert svc.needs_bootstrap()

    def go(name):
        barrier.wait()
        try:
            svc.create_first_admin(name, PW)
            outcomes.append("ok")
        except AccountError as exc:
            outcomes.append(str(exc))

    threads = [threading.Thread(target=go, args=(n,)) for n in ("root1", "root2")]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(outcomes) == ["Setup is already complete.", "ok"]
    assert len(svc.list_users()) == 1
