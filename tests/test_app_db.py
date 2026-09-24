import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

import config
from accounts.db import AppDB, iso, parse_iso, utcnow

TABLES = {"users", "audit_log", "sessions", "insight_history", "session_summaries", "prompt_settings", "app_settings"}


def test_config_constants():
    assert config.APP_DB_PATH.name == "app.db" and config.MEMORY_DIR.name == "memory"
    assert (config.LOCKOUT_ATTEMPTS, config.LOCKOUT_MINUTES, config.MIN_PASSWORD_LENGTH) == (5, 15, 8)
    assert (config.PRIOR_SESSIONS, config.PRIOR_CONTEXT_TOKENS, config.ABANDONED_SESSION_MINUTES) == (3, 600, 30)


def test_schema_created_and_idempotent(tmp_path):
    path = tmp_path / "app.db"
    db = AppDB(path)
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert TABLES <= names and db.version == 1
    AppDB(path)  # opening again must not fail or duplicate


def test_username_unique_case_insensitive_and_role_checked(tmp_path):
    db = AppDB(tmp_path / "a.db")
    ts = iso(utcnow())
    db.insert("INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)", ("Alice", "h", "admin", ts))
    with pytest.raises(sqlite3.IntegrityError):
        db.insert("INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)", ("alice", "h", "analyst", ts))
    with pytest.raises(sqlite3.IntegrityError):
        db.insert("INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)", ("bob", "h", "root", ts))


def test_iso_roundtrip_is_utc_and_orderable():
    a = datetime(2026, 9, 25, 10, 0, 0, tzinfo=timezone.utc)
    ist = timezone(timedelta(hours=5, minutes=30))
    assert iso(a.astimezone(ist)) == iso(a) and parse_iso(iso(a)) == a
    assert iso(a) < iso(a + timedelta(microseconds=1))


def test_injected_clock_and_helpers(tmp_path):
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db = AppDB(tmp_path / "a.db", clock=lambda: fixed)
    assert db.now() == fixed
    assert db.one("SELECT 1 AS x") == {"x": 1} and db.one("SELECT * FROM users") is None


def test_threads_can_share_the_connection(tmp_path):
    db = AppDB(tmp_path / "a.db")
    errors = []

    def work(i):
        try:
            for j in range(20):
                db.insert("INSERT INTO app_settings (key, value_json, ts_utc) VALUES (?,?,?)", (f"k{i}_{j}", "1", "t"))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and db.one("SELECT COUNT(*) AS n FROM app_settings")["n"] == 80
