import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('analyst','manager','admin')),
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_login TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    user_id INTEGER,
    username TEXT,
    action TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    session_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log (ts_utc);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log (action);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    ended_at TEXT,
    question_count INTEGER NOT NULL DEFAULT 0,
    summarised INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS insight_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    session_id TEXT,
    question TEXT NOT NULL,
    question_norm TEXT NOT NULL,
    insight_json TEXT NOT NULL,
    chart_json TEXT,
    slice_json TEXT NOT NULL DEFAULT '[]',
    approved INTEGER NOT NULL DEFAULT 0,
    degraded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_history_norm ON insight_history (question_norm);
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    ts_utc TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prompt_settings (
    name TEXT PRIMARY KEY,
    active_version TEXT NOT NULL,
    updated_by INTEGER,
    ts_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_by INTEGER,
    ts_utc TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text)


class AppDB:
    def __init__(self, path, clock=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._clock = clock or utcnow
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def now(self) -> datetime:
        return self._clock()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
            except Exception:
                self._conn.rollback()
                raise

    def insert(self, sql: str, params=()) -> int:
        return self.execute(sql, params).lastrowid

    def query(self, sql: str, params=()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    @property
    def version(self) -> int:
        return self.one("PRAGMA user_version")["user_version"]
