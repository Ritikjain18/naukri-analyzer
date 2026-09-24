# Naukri Personal Data Analyzer — Phase 3a Implementation Plan (accounts, audit, memory, admin)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add login with three roles, a permission matrix enforced in code, an audit trail, persistent insight history, per-user session summaries injected as prior context, and Manager/Admin pages, on top of the phase 1+2 app.

**Architecture:** A new `accounts/` package (stdlib `sqlite3` + `hashlib.scrypt`, own file `data/app.db`) provides services (auth, permissions, audit, history, sessions, memory, prompt/app settings). The Streamlit shell (`app.py`) becomes a gate + sidebar router; the existing chat moves into `ui/analyze.py`; new `ui/history_page.py` and `ui/admin_page.py` render Manager/Admin pages. Prior context reaches the graph through a new persistent state key and two new prompt versions.

**Tech Stack:** Python 3.13 (`.venv`), Streamlit, LangGraph, sqlite3, hashlib, pytest, existing deck builder.

**Spec:** `docs/superpowers/specs/2026-09-25-phase3a-design.md` (read it first). Baseline: 312 offline tests pass, 0 warnings; 7 `live` tests deselected.

## Global Constraints
- Models: fast `openai/gpt-oss-20b`, smart `openai/gpt-oss-120b` (see `config.py`). Tests never call Groq; live tests stay opt-in and are not run without the user's OK.
- **No new third-party dependencies.** Passwords: `hashlib.scrypt`; storage: `sqlite3` with one `RLock`, `check_same_thread=False`.
- Accounts data lives only in `data/app.db` (gitignored via `data/*.db`); `memory/` is gitignored. Nothing ever writes a repo-root `CLAUDE.md`. Tests use tmp paths only (never the real `data/*.db`, `memory/`, `.env`).
- Never read, print or create `.env`. Audit/log/history/memory content must never contain passwords, hashes, API keys or uploaded-file contents.
- The pandas route stays disabled. SQL stays read-only. Model-derived or user-derived text rendered with `st.markdown` goes through `graph.textsafe.safe_text`.
- Permissions: `upload, ask, export` (Analyst); + `view_history, view_comparative` (Manager); + `manage_users, manage_prompts, manage_config, view_audit` (Admin). Code checks `require(...)`; hiding widgets alone is never the barrier. Denials are audited as `denied:<permission>`.
- Passwords >= 8 chars; usernames 3-32 chars `[A-Za-z0-9_.-]` (case-insensitive unique); 5 consecutive failures lock the account for 15 minutes; the last active admin cannot be disabled or demoted.
- Prior context: last 3 session summaries, clipped to 600 tokens, whitespace-collapsed, wrapped in a labelled block; empty when there are none.
- 0 warnings in the offline test run. Run commands from `/Users/ritikjain/Downloads/python_scripts/naukri_analyzer` with `.venv/bin/pytest`. Commit after each task; never commit `.superpowers/`, `data/*.db`, `memory/`.

## Deviations from the spec (decided during planning)
1. Login failure messages: unknown user and wrong password both show "Invalid username or password."; a locked account shows "Too many failed attempts. Try again later." (this reveals existence only to someone who already caused five failures).
2. A new prompt `session_summary.v1.txt` (variable `$record`) drives the summariser; `REQUIRED_VARS` covers every prompt the nodes use.
3. Prompt override compatibility is checked with `REQUIRED_VARS` (variables the node depends on must appear as `$var` in the chosen version file).
4. Export is audited from `st.download_button(on_click=...)`; AppTest cannot click it, so the callback function is unit-tested directly.
5. `graph/trace.py::generated_sql(prompts)` is extracted from the judge module so the audit hook and judge share it.

## File Structure
```
config.py                       # + APP_DB_PATH, MEMORY_DIR, lockout/prior-context constants (T1)
.gitignore                      # + memory/ (T1)
accounts/__init__.py
accounts/db.py                  # AppDB, schema, iso/parse_iso/utcnow (T1)
accounts/auth.py                # hashing, AuthService, User, AccountError (T2)
accounts/permissions.py         # ROLES, PERMISSIONS, can, require, PermissionDenied (T3)
accounts/audit.py               # AuditLog (T3)
accounts/history.py             # InsightHistory, normalise_question (T4)
accounts/sessions.py            # SessionTracker (T4)
accounts/settings.py            # REQUIRED_VARS, PromptSettings, AppSettings (T5)
accounts/memory.py              # summarise_session, MemoryService, write_project_memory (T7)
accounts/services.py            # Services container + build_services (T8)
graph/prompts.py                # + override registry (T5)
graph/trace.py                  # generated_sql (T9)
prompts/                        # + orchestrator.v2, query.v3, session_summary.v1 (T6, T7)
ui/__init__.py  ui/context.py  ui/auth_ui.py  ui/router.py  ui/analyze.py  ui/history_page.py  ui/admin_page.py
app.py                          # gate + router (T8)
tests/                          # test_app_db, test_auth, test_permissions_audit, test_history_sessions,
                                # test_settings, test_prior_context, test_memory, test_app_auth, test_app_pages, helpers.py
```

---

### Task 1: Config constants and AppDB

**Files:**
- Modify: `config.py`, `.gitignore`
- Create: `accounts/__init__.py`, `accounts/db.py`
- Test: `tests/test_app_db.py`

**Interfaces:**
- Produces: `config.APP_DB_PATH = ROOT/"data"/"app.db"`, `config.MEMORY_DIR = ROOT/"memory"`, `LOCKOUT_ATTEMPTS = 5`, `LOCKOUT_MINUTES = 15`, `MIN_PASSWORD_LENGTH = 8`, `PRIOR_SESSIONS = 3`, `PRIOR_CONTEXT_TOKENS = 600`, `ABANDONED_SESSION_MINUTES = 30`.
- `accounts.db`: `utcnow() -> datetime` (aware UTC), `iso(dt) -> str` (UTC, `timespec="microseconds"`), `parse_iso(s) -> datetime`; `class AppDB(path, clock=None)` with `now()`, `execute(sql, params=()) -> Cursor` (locked, commits), `insert(sql, params=()) -> int`, `query(sql, params=()) -> list[dict]`, `one(sql, params=()) -> dict | None`, `version -> int` (`PRAGMA user_version`, 1). Tables: `users, audit_log, sessions, insight_history, session_summaries, prompt_settings, app_settings` with the columns in the spec (`users.username` is `UNIQUE COLLATE NOCASE`, `users.role` has a CHECK on the three roles). Creating the schema is idempotent.

- [ ] **Step 1: Write the failing tests**

`tests/test_app_db.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_app_db.py -v` — Expected: FAIL (`ModuleNotFoundError: accounts`).

- [ ] **Step 3: Implement**

`config.py` (append after `MEMORY_SLICE_ROWS`): 
```python
APP_DB_PATH = ROOT / "data" / "app.db"
MEMORY_DIR = ROOT / "memory"
LOCKOUT_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
MIN_PASSWORD_LENGTH = 8
PRIOR_SESSIONS = 3
PRIOR_CONTEXT_TOKENS = 600
ABANDONED_SESSION_MINUTES = 30
```
`.gitignore`: add `memory/`. `accounts/__init__.py` empty. `accounts/db.py`:
```python
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

    def now(self) -> datetime:
        return self._clock()

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest -q` — Expected: all pass (312 + new), 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add config.py .gitignore accounts tests/test_app_db.py
git commit -m "feat: add app database and phase 3a config constants"
```

---

### Task 2: Authentication service

**Files:**
- Create: `accounts/auth.py`
- Modify: `tests/conftest.py` (autouse fast-scrypt fixture)
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `AppDB`, `iso`, `parse_iso`, `config.LOCKOUT_*`, `config.MIN_PASSWORD_LENGTH`.
- Produces (`accounts/auth.py`): `ROLES = ("analyst", "manager", "admin")`; `class AccountError(ValueError)`; `@dataclass(frozen=True) User(id, username, role, active)`; `@dataclass(frozen=True) AuthResult(ok: bool, user: User | None, reason: str)` with `reason` in `ok|invalid|locked|disabled`; module constants `SCRYPT_N=2**14, SCRYPT_R=8, SCRYPT_P=1`; `hash_password(password) -> str` (`scrypt$n$r$p$salt_b64$hash_b64`, random 16-byte salt); `verify_password(password, stored) -> bool` (constant-time; malformed stored value -> False); `validate_username`, `validate_password` (raise `AccountError`); `class AuthService(db)`: `needs_bootstrap()`, `create_user(username, password, role) -> int`, `get_user(user_id)`, `get_by_username(username)`, `list_users()`, `authenticate(username, password) -> AuthResult`, `change_password(user_id, old, new)`, `reset_password(user_id, new)`, `set_role(user_id, role)`, `set_active(user_id, active)`.
- Error text (exact): `"Username must be 3-32 characters: letters, digits, '_', '.', '-'."`, `f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters."`, `"That username is already taken."`, `"Unknown role."`, `"At least one active admin is required."`, `"Current password is incorrect."`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/conftest.py`:
```python
@pytest.fixture(autouse=True)
def _fast_scrypt(monkeypatch):
    from accounts import auth

    monkeypatch.setattr(auth, "SCRYPT_N", 16)
```
`tests/test_auth.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_auth.py -v` — Expected: FAIL (`ModuleNotFoundError: accounts.auth`).

- [ ] **Step 3: Implement `accounts/auth.py`**

```python
import base64
import hashlib
import hmac
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import timedelta

import config
from accounts.db import AppDB, iso, parse_iso

ROLES = ("analyst", "manager", "admin")
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_dummy_hash: str | None = None


class AccountError(ValueError):
    pass


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: str
    active: bool


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    user: User | None
    reason: str


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = base64.urlsafe_b64decode(salt_b64), base64.urlsafe_b64decode(digest_b64)
        actual = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def validate_username(username: str) -> None:
    if not USERNAME_RE.match(username):
        raise AccountError("Username must be 3-32 characters: letters, digits, '_', '.', '-'.")


def validate_password(password: str) -> None:
    if len(password) < config.MIN_PASSWORD_LENGTH:
        raise AccountError(f"Password must be at least {config.MIN_PASSWORD_LENGTH} characters.")


def _dummy() -> str:
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password("dummy-password-for-timing")
    return _dummy_hash


def _user(row: dict) -> User:
    return User(row["id"], row["username"], row["role"], bool(row["active"]))


class AuthService:
    def __init__(self, db: AppDB):
        self.db = db

    def needs_bootstrap(self) -> bool:
        return self.db.one("SELECT COUNT(*) AS n FROM users")["n"] == 0

    def create_user(self, username: str, password: str, role: str) -> int:
        username = username.strip()
        validate_username(username)
        validate_password(password)
        if role not in ROLES:
            raise AccountError("Unknown role.")
        try:
            return self.db.insert(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?,?,?,?)",
                (username, hash_password(password), role, iso(self.db.now())),
            )
        except sqlite3.IntegrityError:
            raise AccountError("That username is already taken.") from None

    def get_user(self, user_id: int) -> User | None:
        row = self.db.one("SELECT * FROM users WHERE id = ?", (user_id,))
        return _user(row) if row else None

    def get_by_username(self, username: str) -> User | None:
        row = self.db.one("SELECT * FROM users WHERE username = ?", (username.strip(),))
        return _user(row) if row else None

    def list_users(self) -> list[User]:
        return [_user(r) for r in self.db.query("SELECT * FROM users ORDER BY username")]

    def authenticate(self, username: str, password: str) -> AuthResult:
        now = self.db.now()
        row = self.db.one("SELECT * FROM users WHERE username = ?", (username.strip(),))
        if row is None:
            verify_password(password, _dummy())
            return AuthResult(False, None, "invalid")
        if row["locked_until"]:
            if now < parse_iso(row["locked_until"]):
                return AuthResult(False, None, "locked")
            self.db.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", (row["id"],))
            row = {**row, "failed_attempts": 0, "locked_until": None}
        if not row["active"]:
            verify_password(password, _dummy())
            return AuthResult(False, None, "disabled")
        if not verify_password(password, row["password_hash"]):
            attempts = row["failed_attempts"] + 1
            locked = attempts >= config.LOCKOUT_ATTEMPTS
            until = iso(now + timedelta(minutes=config.LOCKOUT_MINUTES)) if locked else None
            self.db.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                            (attempts, until, row["id"]))
            return AuthResult(False, None, "locked" if locked else "invalid")
        self.db.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?",
                        (iso(now), row["id"]))
        return AuthResult(True, _user(row), "ok")

    def change_password(self, user_id: int, old: str, new: str) -> None:
        row = self.db.one("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        if row is None or not verify_password(old, row["password_hash"]):
            raise AccountError("Current password is incorrect.")
        validate_password(new)
        self.db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new), user_id))

    def reset_password(self, user_id: int, new: str) -> None:
        validate_password(new)
        self.db.execute(
            "UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = NULL WHERE id = ?",
            (hash_password(new), user_id),
        )

    def _active_admins(self) -> int:
        return self.db.one("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1")["n"]

    def set_role(self, user_id: int, role: str) -> None:
        if role not in ROLES:
            raise AccountError("Unknown role.")
        target = self.get_user(user_id)
        if target and target.role == "admin" and role != "admin" and target.active and self._active_admins() <= 1:
            raise AccountError("At least one active admin is required.")
        self.db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))

    def set_active(self, user_id: int, active: bool) -> None:
        target = self.get_user(user_id)
        if target and not active and target.role == "admin" and target.active and self._active_admins() <= 1:
            raise AccountError("At least one active admin is required.")
        self.db.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add accounts/auth.py tests/conftest.py tests/test_auth.py
git commit -m "feat: add scrypt authentication with lockout and last-admin protection"
```

---

### Task 3: Permissions and audit log

**Files:**
- Create: `accounts/permissions.py`, `accounts/audit.py`
- Test: `tests/test_permissions_audit.py`

**Interfaces:**
- Consumes: `AppDB`, `iso`, `User`.
- Produces: `PERMISSIONS: dict[str, frozenset[str]]` (role -> permissions per the Global Constraints), `class PermissionDenied(PermissionError)`, `can(role, permission) -> bool` (unknown role/permission -> False), `require(role, permission) -> None`. `class AuditLog(db)`: `record(user, action, session_id=None, **detail) -> int` (`user` is a `User`, a dict with `id`/`username`, or `None`; detail is scrubbed: keys matching `pass|secret|token|api_?key|hash` (case-insensitive) are dropped, string values are cut to 2000 characters, non-JSON values are stringified); `query(user=None, action=None, since=None, until=None, limit=500) -> list[dict]` (newest first; each row has `detail` as a dict; `user` filters by username substring case-insensitively; `since`/`until` are datetimes); `to_csv(rows) -> str` (columns `ts_utc,username,action,session_id,detail`).

- [ ] **Step 1: Write the failing tests**

`tests/test_permissions_audit.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_permissions_audit.py -v` — Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`accounts/permissions.py`:
```python
ANALYST = frozenset({"upload", "ask", "export"})
MANAGER = ANALYST | {"view_history", "view_comparative"}
ADMIN = MANAGER | {"manage_users", "manage_prompts", "manage_config", "view_audit"}
PERMISSIONS = {"analyst": ANALYST, "manager": MANAGER, "admin": ADMIN}


class PermissionDenied(PermissionError):
    pass


def can(role, permission: str) -> bool:
    return permission in PERMISSIONS.get(role, frozenset())


def require(role, permission: str) -> None:
    if not can(role, permission):
        raise PermissionDenied(f"Your role does not allow: {permission}.")
```
`accounts/audit.py`:
```python
import csv
import io
import json
import re

from accounts.db import AppDB, iso

SECRET_KEY = re.compile(r"pass|secret|token|api_?key|hash", re.I)
MAX_VALUE = 2000


def _scrub(detail: dict) -> dict:
    clean = {}
    for key, value in detail.items():
        if SECRET_KEY.search(key):
            continue
        if isinstance(value, str):
            value = value[:MAX_VALUE]
        elif not isinstance(value, (int, float, bool, list, dict, type(None))):
            value = str(value)[:MAX_VALUE]
        clean[key] = value
    return clean


def _identity(user):
    if user is None:
        return None, None
    if isinstance(user, dict):
        return user.get("id"), user.get("username")
    return user.id, user.username


class AuditLog:
    def __init__(self, db: AppDB):
        self.db = db

    def record(self, user, action: str, session_id: str | None = None, **detail) -> int:
        user_id, username = _identity(user)
        return self.db.insert(
            "INSERT INTO audit_log (ts_utc, user_id, username, action, detail_json, session_id) VALUES (?,?,?,?,?,?)",
            (iso(self.db.now()), user_id, username, action,
             json.dumps(_scrub(detail), default=str), session_id),
        )

    def query(self, user=None, action=None, since=None, until=None, limit: int = 500) -> list[dict]:
        sql, params = "SELECT * FROM audit_log WHERE 1=1", []
        if user:
            sql += " AND username LIKE ? COLLATE NOCASE"
            params.append(f"%{user}%")
        if action:
            sql += " AND action = ?"
            params.append(action)
        if since:
            sql += " AND ts_utc >= ?"
            params.append(iso(since))
        if until:
            sql += " AND ts_utc <= ?"
            params.append(iso(until))
        sql += " ORDER BY ts_utc DESC, id DESC LIMIT ?"
        params.append(limit)
        return [{**r, "detail": json.loads(r["detail_json"])} for r in self.db.query(sql, tuple(params))]

    @staticmethod
    def to_csv(rows: list[dict]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ts_utc", "username", "action", "session_id", "detail"])
        for r in rows:
            writer.writerow([r["ts_utc"], r["username"] or "", r["action"], r["session_id"] or "",
                             json.dumps(r["detail"], ensure_ascii=False)])
        return buf.getvalue()
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add accounts/permissions.py accounts/audit.py tests/test_permissions_audit.py
git commit -m "feat: add permission matrix and audit log"
```

---

### Task 4: Insight history and session tracking

**Files:**
- Create: `accounts/history.py`, `accounts/sessions.py`
- Test: `tests/test_history_sessions.py`

**Interfaces:**
- Consumes: `AppDB`, `iso`, `parse_iso`, `config.MEMORY_SLICE_ROWS`.
- Produces: `normalise_question(text) -> str` (lower-case, non-alphanumerics -> spaces, collapsed whitespace, stripped). `class InsightHistory(db)`: `add(user_id, session_id, question, insight: dict, chart: dict | None, data_slice: list[dict], approved=False, degraded=False) -> int` (slice capped at `MEMORY_SLICE_ROWS`, JSON with `default=str`); `get(history_id) -> dict | None`; `list(user_id=None, text=None, since=None, until=None, limit=200) -> list[dict]` (newest first; rows include parsed `insight`, `chart`, `slice`, `approved`/`degraded` as bools, plus `username` via join with users, `"?"` if missing); `mark_approved(history_id) -> None`; `by_question(question_norm, limit=20) -> list[dict]` (newest first); `repeated_questions(min_sessions=2) -> list[dict]` (`{"question_norm", "sample", "sessions", "count"}` for questions asked in at least `min_sessions` distinct sessions, most recent first; `sample` = latest original text). `class SessionTracker(db)`: `start(user_id) -> str` (uuid4 hex), `touch(session_id, questions=0)` (updates `last_activity_at`, adds to `question_count`), `end(session_id)` (sets `ended_at` once), `get(session_id) -> dict | None`, `pending_for_user(user_id, older_than_minutes) -> list[str]` (sessions with `summarised=0`, `question_count>0`, and either ended or inactive for longer than the threshold), `mark_summarised(session_id)`, `entries_for(session_id) -> list[dict]` (history rows for that session, oldest first).

- [ ] **Step 1: Write the failing tests**

`tests/test_history_sessions.py`:
```python
from datetime import datetime, timedelta, timezone

import pytest

from accounts.auth import AuthService
from accounts.db import AppDB
from accounts.history import InsightHistory, normalise_question
from accounts.sessions import SessionTracker

INSIGHT = {"finding": "Eng leads.", "evidence": ["20% vs 10%"], "recommendation": "Invest."}


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    db = AppDB(tmp_path / "a.db", clock=clock)
    auth = AuthService(db)
    return {"clock": clock, "db": db, "hist": InsightHistory(db), "sess": SessionTracker(db),
            "alice": auth.create_user("alice", "correct horse", "analyst"),
            "bob": auth.create_user("bob", "correct horse", "manager")}


def test_normalise_question():
    assert normalise_question("  Which CATEGORY, has the best rate?? ") == "which category has the best rate"


def test_add_get_list_with_username_and_cap(env):
    rows = [{"a": i} for i in range(80)]
    hid = env["hist"].add(env["alice"], "s1", "Which category?", INSIGHT, {"type": "bar"}, rows)
    got = env["hist"].get(hid)
    assert got["insight"] == INSIGHT and got["chart"] == {"type": "bar"} and len(got["slice"]) == 50
    assert got["approved"] is False and got["degraded"] is False
    assert env["hist"].list()[0]["username"] == "alice"


def test_list_filters_and_order(env):
    h = env["hist"]
    h.add(env["alice"], "s1", "Bounce by source?", INSIGHT, None, [])
    env["clock"].now += timedelta(days=2)
    h.add(env["bob"], "s2", "Views by category?", INSIGHT, None, [], degraded=True)
    assert [r["question"] for r in h.list()] == ["Views by category?", "Bounce by source?"]
    assert [r["username"] for r in h.list(user_id=env["alice"])] == ["alice"]
    assert [r["question"] for r in h.list(text="BOUNCE")] == ["Bounce by source?"]
    cutoff = env["clock"].now - timedelta(days=1)
    assert len(h.list(since=cutoff)) == 1 and len(h.list(until=cutoff)) == 1
    assert h.list()[0]["degraded"] is True


def test_mark_approved(env):
    hid = env["hist"].add(env["alice"], "s1", "q one two", INSIGHT, None, [])
    env["hist"].mark_approved(hid)
    assert env["hist"].get(hid)["approved"] is True


def test_by_question_and_repeated_questions(env):
    h = env["hist"]
    h.add(env["alice"], "s1", "Which category is best?", INSIGHT, None, [])
    env["clock"].now += timedelta(hours=1)
    h.add(env["bob"], "s2", "which category is BEST", INSIGHT, None, [])
    h.add(env["alice"], "s2", "Only once here", INSIGHT, None, [])
    norm = normalise_question("Which category is best")
    assert len(h.by_question(norm)) == 2 and h.by_question(norm)[0]["username"] == "bob"
    rep = h.repeated_questions()
    assert [(r["question_norm"], r["sessions"], r["count"]) for r in rep] == [(norm, 2, 2)]
    assert rep[0]["sample"] == "which category is BEST"


def test_session_lifecycle_and_pending(env):
    s, hist = env["sess"], env["hist"]
    sid = s.start(env["alice"])
    assert len(sid) == 32 and s.get(sid)["question_count"] == 0
    hist.add(env["alice"], sid, "q one two", INSIGHT, None, [])
    s.touch(sid, questions=1)
    assert s.get(sid)["question_count"] == 1
    assert s.pending_for_user(env["alice"], 30) == []                 # active recently
    env["clock"].now += timedelta(minutes=31)
    assert s.pending_for_user(env["alice"], 30) == [sid]              # abandoned
    s.mark_summarised(sid)
    assert s.pending_for_user(env["alice"], 30) == []
    assert [e["question"] for e in s.entries_for(sid)] == ["q one two"]


def test_ended_session_is_pending_immediately_and_empty_ones_never(env):
    s = env["sess"]
    a, empty = s.start(env["alice"]), s.start(env["alice"])
    s.touch(a, questions=2)
    s.end(a)
    s.end(empty)
    assert s.pending_for_user(env["alice"], 30) == [a]
    first_end = s.get(a)["ended_at"]
    env["clock"].now += timedelta(minutes=5)
    s.end(a)
    assert s.get(a)["ended_at"] == first_end                          # ended_at is set once
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_history_sessions.py -v` — Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`accounts/history.py`:
```python
import json
import re

import config
from accounts.db import AppDB, iso


def normalise_question(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _row(r: dict) -> dict:
    return {
        "id": r["id"], "ts_utc": r["ts_utc"], "user_id": r["user_id"], "username": r.get("username") or "?",
        "session_id": r["session_id"], "question": r["question"], "question_norm": r["question_norm"],
        "insight": json.loads(r["insight_json"]), "chart": json.loads(r["chart_json"]) if r["chart_json"] else None,
        "slice": json.loads(r["slice_json"]), "approved": bool(r["approved"]), "degraded": bool(r["degraded"]),
    }


BASE = ("SELECT h.*, u.username AS username FROM insight_history h "
        "LEFT JOIN users u ON u.id = h.user_id WHERE 1=1")


class InsightHistory:
    def __init__(self, db: AppDB):
        self.db = db

    def add(self, user_id, session_id, question, insight, chart, data_slice, approved=False, degraded=False) -> int:
        return self.db.insert(
            "INSERT INTO insight_history (ts_utc, user_id, session_id, question, question_norm, insight_json,"
            " chart_json, slice_json, approved, degraded) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (iso(self.db.now()), user_id, session_id, question, normalise_question(question),
             json.dumps(insight, default=str), json.dumps(chart, default=str) if chart else None,
             json.dumps(list(data_slice)[: config.MEMORY_SLICE_ROWS], default=str),
             1 if approved else 0, 1 if degraded else 0),
        )

    def get(self, history_id: int):
        row = self.db.one(BASE + " AND h.id = ?", (history_id,))
        return _row(row) if row else None

    def list(self, user_id=None, text=None, since=None, until=None, limit: int = 200) -> list[dict]:
        sql, params = BASE, []
        if user_id is not None:
            sql += " AND h.user_id = ?"
            params.append(user_id)
        if text:
            sql += " AND h.question LIKE ? COLLATE NOCASE"
            params.append(f"%{text}%")
        if since:
            sql += " AND h.ts_utc >= ?"
            params.append(iso(since))
        if until:
            sql += " AND h.ts_utc <= ?"
            params.append(iso(until))
        sql += " ORDER BY h.ts_utc DESC, h.id DESC LIMIT ?"
        params.append(limit)
        return [_row(r) for r in self.db.query(sql, tuple(params))]

    def mark_approved(self, history_id: int) -> None:
        self.db.execute("UPDATE insight_history SET approved = 1 WHERE id = ?", (history_id,))

    def by_question(self, question_norm: str, limit: int = 20) -> list[dict]:
        rows = self.db.query(BASE + " AND h.question_norm = ? ORDER BY h.ts_utc DESC, h.id DESC LIMIT ?",
                             (question_norm, limit))
        return [_row(r) for r in rows]

    def repeated_questions(self, min_sessions: int = 2) -> list[dict]:
        groups = self.db.query(
            "SELECT question_norm, COUNT(DISTINCT session_id) AS sessions, COUNT(*) AS count, MAX(ts_utc) AS last_ts"
            " FROM insight_history GROUP BY question_norm HAVING COUNT(DISTINCT session_id) >= ?"
            " ORDER BY last_ts DESC", (min_sessions,))
        out = []
        for g in groups:
            latest = self.db.one("SELECT question FROM insight_history WHERE question_norm = ?"
                                 " ORDER BY ts_utc DESC, id DESC LIMIT 1", (g["question_norm"],))
            out.append({"question_norm": g["question_norm"], "sample": latest["question"],
                        "sessions": g["sessions"], "count": g["count"]})
        return out
```
`accounts/sessions.py`:
```python
import uuid
from datetime import timedelta

from accounts.db import AppDB, iso
from accounts.history import _row


class SessionTracker:
    def __init__(self, db: AppDB):
        self.db = db

    def start(self, user_id: int) -> str:
        sid, now = uuid.uuid4().hex, iso(self.db.now())
        self.db.execute("INSERT INTO sessions (id, user_id, started_at, last_activity_at) VALUES (?,?,?,?)",
                        (sid, user_id, now, now))
        return sid

    def get(self, session_id: str):
        return self.db.one("SELECT * FROM sessions WHERE id = ?", (session_id,))

    def touch(self, session_id: str, questions: int = 0) -> None:
        self.db.execute("UPDATE sessions SET last_activity_at = ?, question_count = question_count + ? WHERE id = ?",
                        (iso(self.db.now()), questions, session_id))

    def end(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                        (iso(self.db.now()), session_id))

    def mark_summarised(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET summarised = 1 WHERE id = ?", (session_id,))

    def pending_for_user(self, user_id: int, older_than_minutes: int) -> list[str]:
        cutoff = iso(self.db.now() - timedelta(minutes=older_than_minutes))
        rows = self.db.query(
            "SELECT id FROM sessions WHERE user_id = ? AND summarised = 0 AND question_count > 0"
            " AND (ended_at IS NOT NULL OR last_activity_at <= ?) ORDER BY started_at", (user_id, cutoff))
        return [r["id"] for r in rows]

    def entries_for(self, session_id: str) -> list[dict]:
        rows = self.db.query(
            "SELECT h.*, u.username AS username FROM insight_history h LEFT JOIN users u ON u.id = h.user_id"
            " WHERE h.session_id = ? ORDER BY h.ts_utc, h.id", (session_id,))
        return [_row(r) for r in rows]
```

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add accounts/history.py accounts/sessions.py tests/test_history_sessions.py
git commit -m "feat: add insight history and session tracking"
```

---

### Task 5: Prompt overrides and app settings

**Files:**
- Create: `accounts/settings.py`
- Modify: `graph/prompts.py`, `tests/conftest.py` (autouse override reset)
- Test: `tests/test_settings.py`

**Interfaces:**
- Consumes: `AppDB`, `iso`, `config.PROMPTS_DIR` (defined in `graph/prompts.py`), `AccountError`.
- Produces:
  - `graph/prompts.py`: `set_overrides(overrides: dict[str, str]) -> None`, `get_overrides() -> dict`, `effective_version(name, requested) -> str` (returns the override only if `prompts/<name>.<override>.txt` exists, else `requested`); `render` applies `effective_version` before loading. `load_prompt` is unchanged (explicit version).
  - `accounts/settings.py`: `REQUIRED_VARS` (dict below), `placeholders(text) -> set[str]` (`\$(\w+)`), `class PromptSettings(db, prompts_dir=PROMPTS_DIR)`: `versions(name) -> list[str]` (sorted numerically, e.g. `["v1","v2","v3"]`), `compatible_versions(name)`, `active(name) -> str | None`, `set_active(name, version, user_id)` (raises `AccountError` if the name is unknown, the version file is missing, or it lacks any `REQUIRED_VARS[name]` placeholder), `clear(name)`, `overrides() -> dict`, `apply()` (calls `set_overrides(self.overrides())`), `names() -> list[str]`. `class AppSettings(db)`: `ALLOWED = {"JUDGE_ENABLED": bool, "JUDGE_RETRIEVAL": bool, "JUDGE_MIN_SCORE": int}`, `get(key)` (stored value else the `config` default), `set(key, value, user_id)` (validates type, `JUDGE_MIN_SCORE` in 1..5, unknown key -> `AccountError`), `all() -> dict`, `apply(config_module=config)` (sets attributes from stored values).
  - `REQUIRED_VARS`: `data_understanding: {skills, schema, sample}`, `orchestrator: {schema, summary, history, question}`, `sql: {schema, limit, history, question, correction, error_note}`, `query: {summary, skill, history, slice, question, error_note, guard_error, judge_correction, revision_note}`, `judge: {stage, question, output}`, `visualization: {insight, columns, error_note}`, `session_summary: {record}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/conftest.py`:
```python
@pytest.fixture(autouse=True)
def _reset_prompt_overrides():
    from graph import prompts

    prompts.set_overrides({})
    yield
    prompts.set_overrides({})
```
`tests/test_settings.py`:
```python
import pytest

import config
from accounts.auth import AccountError
from accounts.db import AppDB
from accounts.settings import REQUIRED_VARS, AppSettings, PromptSettings, placeholders
from graph import prompts


def write(dirpath, name, version, text):
    (dirpath / f"{name}.{version}.txt").write_text(text)


@pytest.fixture
def pdir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    write(d, "sql", "v1", "old $schema $question")
    write(d, "sql", "v2", "$schema $limit $history $question $correction $error_note")
    write(d, "sql", "v10", "$schema $limit $history $question $correction $error_note more")
    write(d, "sql", "v3", "$schema $limit $history $question $correction $error_note")
    return d


@pytest.fixture
def ps(tmp_path, pdir):
    return PromptSettings(AppDB(tmp_path / "a.db"), prompts_dir=pdir)


def test_placeholders():
    assert placeholders("a $x and ${y} and $z_1 $$") >= {"x", "z_1"}


def test_versions_sorted_numerically_and_compatibility(ps):
    assert ps.versions("sql") == ["v1", "v2", "v3", "v10"]
    assert ps.compatible_versions("sql") == ["v2", "v3", "v10"]
    assert ps.versions("nothing") == []


def test_set_active_validates(ps):
    ps.set_active("sql", "v2", user_id=1)
    assert ps.active("sql") == "v2" and ps.overrides() == {"sql": "v2"}
    with pytest.raises(AccountError, match="missing"):
        ps.set_active("sql", "v1", user_id=1)          # lacks $limit etc
    with pytest.raises(AccountError, match="not found"):
        ps.set_active("sql", "v99", user_id=1)
    with pytest.raises(AccountError, match="Unknown prompt"):
        ps.set_active("bogus", "v1", user_id=1)
    assert ps.active("sql") == "v2"


def test_clear_and_apply_sets_render_override(ps, pdir, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    ps.set_active("sql", "v3", user_id=1)
    ps.apply()
    assert prompts.get_overrides() == {"sql": "v3"}
    assert prompts.effective_version("sql", "v2") == "v3"
    out = prompts.render("sql", "v2", schema="S", limit=5, history="h", question="Q", correction="", error_note="")
    assert out.startswith("S 5")
    ps.clear("sql")
    ps.apply()
    assert prompts.effective_version("sql", "v2") == "v2"


def test_effective_version_ignores_override_without_file(monkeypatch, pdir):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    prompts.set_overrides({"sql": "v77"})
    assert prompts.effective_version("sql", "v2") == "v2"


def test_required_vars_cover_real_prompt_files():
    real = PromptSettings(AppDB(":memory:"))
    for name, needed in REQUIRED_VARS.items():
        assert real.compatible_versions(name), f"no compatible version for {name}"
        assert needed


def test_app_settings_validate_store_and_apply(tmp_path, monkeypatch):
    s = AppSettings(AppDB(tmp_path / "a.db"))
    assert s.get("JUDGE_ENABLED") is config.JUDGE_ENABLED
    s.set("JUDGE_ENABLED", False, user_id=1)
    s.set("JUDGE_MIN_SCORE", 4, user_id=1)
    with pytest.raises(AccountError):
        s.set("JUDGE_MIN_SCORE", 9, user_id=1)
    with pytest.raises(AccountError):
        s.set("JUDGE_ENABLED", "yes", user_id=1)
    with pytest.raises(AccountError, match="Unknown setting"):
        s.set("MODEL_SMART", "x", user_id=1)
    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    monkeypatch.setattr(config, "JUDGE_MIN_SCORE", 3)
    s.apply(config)
    assert config.JUDGE_ENABLED is False and config.JUDGE_MIN_SCORE == 4
    assert s.all() == {"JUDGE_ENABLED": False, "JUDGE_RETRIEVAL": config.JUDGE_RETRIEVAL, "JUDGE_MIN_SCORE": 4}
```
(`test_required_vars_cover_real_prompt_files` needs `session_summary.v1.txt`, added in Task 7; until then mark that name's assertion with `pytest.skip` inside the loop **only** for `session_summary`, and remove the skip in Task 7. `AppDB(":memory:")` works for `PromptSettings` because it only touches the file system for prompt files; use a tmp path instead if `:memory:` proves awkward.)

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_settings.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`graph/prompts.py` (replace `render`, keep `load_prompt`, add registry):
```python
from string import Template

from config import ROOT

PROMPTS_DIR = ROOT / "prompts"
_OVERRIDES: dict[str, str] = {}


def set_overrides(overrides: dict[str, str]) -> None:
    global _OVERRIDES
    _OVERRIDES = dict(overrides)


def get_overrides() -> dict[str, str]:
    return dict(_OVERRIDES)


def effective_version(name: str, requested: str) -> str:
    override = _OVERRIDES.get(name)
    if override and (PROMPTS_DIR / f"{name}.{override}.txt").exists():
        return override
    return requested


def load_prompt(name: str, version: str = "v1") -> str:
    return (PROMPTS_DIR / f"{name}.{version}.txt").read_text()


def render(name: str, version: str = "v1", **values) -> str:
    return Template(load_prompt(name, effective_version(name, version))).safe_substitute(
        {k: str(v) for k, v in values.items()}
    )
```
`accounts/settings.py`:
```python
import json
import re

import config
from accounts.auth import AccountError
from accounts.db import AppDB, iso
from graph import prompts as prompt_registry
from graph.prompts import PROMPTS_DIR

REQUIRED_VARS = {
    "data_understanding": {"skills", "schema", "sample"},
    "orchestrator": {"schema", "summary", "history", "question"},
    "sql": {"schema", "limit", "history", "question", "correction", "error_note"},
    "query": {"summary", "skill", "history", "slice", "question", "error_note",
              "guard_error", "judge_correction", "revision_note"},
    "judge": {"stage", "question", "output"},
    "visualization": {"insight", "columns", "error_note"},
    "session_summary": {"record"},
}
_PLACEHOLDER = re.compile(r"\$(\w+)|\$\{(\w+)\}")


def placeholders(text: str) -> set[str]:
    return {a or b for a, b in _PLACEHOLDER.findall(text)}


def _version_key(version: str) -> int:
    return int(version[1:])


class PromptSettings:
    def __init__(self, db: AppDB, prompts_dir=None):
        self.db = db
        self.dir = prompts_dir if prompts_dir is not None else PROMPTS_DIR

    def names(self) -> list[str]:
        return sorted(REQUIRED_VARS)

    def versions(self, name: str) -> list[str]:
        found = [p.name[len(name) + 1:-4] for p in self.dir.glob(f"{name}.v*.txt")]
        return sorted((v for v in found if re.fullmatch(r"v\d+", v)), key=_version_key)

    def _missing(self, name: str, version: str) -> set[str]:
        text = (self.dir / f"{name}.{version}.txt").read_text()
        return REQUIRED_VARS[name] - placeholders(text)

    def compatible_versions(self, name: str) -> list[str]:
        return [v for v in self.versions(name) if not self._missing(name, v)]

    def active(self, name: str):
        row = self.db.one("SELECT active_version FROM prompt_settings WHERE name = ?", (name,))
        return row["active_version"] if row else None

    def overrides(self) -> dict:
        return {r["name"]: r["active_version"] for r in self.db.query("SELECT * FROM prompt_settings")}

    def set_active(self, name: str, version: str, user_id) -> None:
        if name not in REQUIRED_VARS:
            raise AccountError("Unknown prompt.")
        if version not in self.versions(name):
            raise AccountError(f"Prompt version not found: {name}.{version}")
        missing = self._missing(name, version)
        if missing:
            raise AccountError(f"{name}.{version} is missing variables: {', '.join(sorted(missing))}")
        self.db.execute(
            "INSERT INTO prompt_settings (name, active_version, updated_by, ts_utc) VALUES (?,?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET active_version=excluded.active_version,"
            " updated_by=excluded.updated_by, ts_utc=excluded.ts_utc",
            (name, version, user_id, iso(self.db.now())))

    def clear(self, name: str) -> None:
        self.db.execute("DELETE FROM prompt_settings WHERE name = ?", (name,))

    def apply(self) -> None:
        prompt_registry.set_overrides(self.overrides())


class AppSettings:
    ALLOWED = {"JUDGE_ENABLED": bool, "JUDGE_RETRIEVAL": bool, "JUDGE_MIN_SCORE": int}

    def __init__(self, db: AppDB):
        self.db = db

    def get(self, key: str):
        row = self.db.one("SELECT value_json FROM app_settings WHERE key = ?", (key,))
        return json.loads(row["value_json"]) if row else getattr(config, key)

    def set(self, key: str, value, user_id) -> None:
        kind = self.ALLOWED.get(key)
        if kind is None:
            raise AccountError("Unknown setting.")
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise AccountError(f"{key} must be a {kind.__name__}.")
        if key == "JUDGE_MIN_SCORE" and not 1 <= value <= 5:
            raise AccountError("JUDGE_MIN_SCORE must be between 1 and 5.")
        self.db.execute(
            "INSERT INTO app_settings (key, value_json, updated_by, ts_utc) VALUES (?,?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,"
            " updated_by=excluded.updated_by, ts_utc=excluded.ts_utc",
            (key, json.dumps(value), user_id, iso(self.db.now())))

    def all(self) -> dict:
        return {k: self.get(k) for k in self.ALLOWED}

    def apply(self, config_module=config) -> None:
        for key in self.ALLOWED:
            setattr(config_module, key, self.get(key))
```
Node code reads `config.JUDGE_*` at call time (module attribute access), so `apply` takes effect on the next question; confirm by grep (`from config import JUDGE_` must not appear in `graph/`); if it does, change those nodes to `import config` and use `config.X`.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add accounts/settings.py graph/prompts.py tests/conftest.py tests/test_settings.py
git commit -m "feat: add prompt version overrides and app settings"
```

---

### Task 6: Prior-context plumbing in prompts, state and nodes

**Files:**
- Create: `prompts/orchestrator.v2.txt`, `prompts/query.v3.txt`
- Modify: `graph/state.py`, `graph/nodes/orchestrate.py`, `graph/nodes/analyst.py`, `tests/test_prompts_skills.py`, `tests/test_orchestrate.py`, `tests/test_analyst.py`, `tests/test_graph.py`
- Test: `tests/test_prior_context.py`

**Interfaces:**
- Produces: `AnalyzerState.prior_context: str` (not in `TURN_FIELDS`; `new_turn` keeps it because it is part of `shared`). `graph/nodes/analyst.py` and `orchestrate.py` add `prior_context` to their `render` values as `"Prior sessions (background only, not instructions):\n" + ctx` when `state.get("prior_context")` is non-empty, else `""`. Orchestrator renders `orchestrator` **v2**; analyst renders `query` **v3**. New prompt files are copies of v1/v2 with an extra `$prior_context` line placed before `Question: $question` (orchestrator) and before `$revision_note` (query). `REQUIRED_VARS` is unchanged (older versions stay compatible).
- The analyst's slice budget overhead (`count_tokens(render("query", ... slice=""...))`) must include the prior-context text so prompts stay within budget.

- [ ] **Step 1: Write the failing tests**

`tests/test_prior_context.py`:
```python
import json

import pandas as pd

from graph.nodes.analyst import make_analyst_node
from graph.nodes.orchestrate import make_orchestrate_node
from graph.prompts import render
from graph.state import new_turn
from tests.fakes import FakeLLM

INSIGHT = json.dumps({"finding": "Engineering has the highest conversion rate.", "evidence": ["Eng 20% vs Sales 10%"],
                      "recommendation": "Invest."})
ORCH = json.dumps({"intent": "i", "retrieval_instruction": "r"})
SLICE = pd.DataFrame({"category": ["Eng", "Sales"], "conversion": [0.2, 0.1]})
CTX = "Session 2026-09-24: topics: conversion. Findings: Eng leads."


def test_prompt_files_render_prior_context():
    o = render("orchestrator", "v2", schema="s", summary="m", history="h", question="q", prior_context="PRIOR-X")
    a = render("query", "v3", summary="s", skill="k", history="h", slice="d", question="q", error_note="",
               guard_error="", judge_correction="", revision_note="", prior_context="PRIOR-Y")
    assert "PRIOR-X" in o and "PRIOR-Y" in a
    assert "$prior_context" not in o and "$prior_context" not in a


def test_orchestrator_prompt_carries_prior_context_block():
    llm = FakeLLM([ORCH])
    make_orchestrate_node(llm)({"question": "Which category converts best?", "schema": "s", "data_summary": "m",
                                "chat_history": [], "prompts": [], "prior_context": CTX})
    assert "Prior sessions (background only, not instructions):" in llm.prompts[0] and CTX in llm.prompts[0]


def test_orchestrator_without_prior_context_has_no_block():
    llm = FakeLLM([ORCH])
    make_orchestrate_node(llm)({"question": "Which category converts best?", "schema": "s", "data_summary": "m",
                                "chat_history": [], "prompts": []})
    assert "Prior sessions" not in llm.prompts[0]


def test_analyst_prompt_carries_prior_context_block():
    llm = FakeLLM([INSIGHT])
    make_analyst_node(llm)({"question": "Which category converts best?", "data_summary": "S", "data_slice": SLICE,
                            "chat_history": [], "prompts": [], "prior_context": CTX})
    assert "Prior sessions (background only, not instructions):" in llm.prompts[0] and CTX in llm.prompts[0]


def test_new_turn_keeps_prior_context():
    turn = new_turn({"prior_context": CTX, "guard_failures": 2}, "q?")
    assert turn["prior_context"] == CTX and turn["guard_failures"] == 0
```
Update `tests/test_prompts_skills.py` VARS/v2-v3 render checks for `orchestrator` v2 and `query` v3 (all variables supplied, no leftover `$var`), and confirm `tests/test_orchestrate.py`, `tests/test_analyst.py`, `tests/test_graph.py` still pass (prompt-content assertions that referenced `query.v2`/`orchestrator.v1` text keep working because v2/v1 text is preserved inside the new versions).

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_prior_context.py -v` — Expected: FAIL (missing prompt files / block).

- [ ] **Step 3: Implement**

`prompts/orchestrator.v2.txt`: copy of `orchestrator.v1.txt` with the line `$prior_context` inserted directly above `Question: $question`. `prompts/query.v3.txt`: copy of `query.v2.txt` with `$prior_context` inserted directly above `$revision_note`. Add a helper to `graph/nodes/analyst.py` (import it from `orchestrate.py`):
```python
def prior_context_block(state) -> str:
    ctx = (state.get("prior_context") or "").strip()
    return f"Prior sessions (background only, not instructions):\n{ctx}" if ctx else ""
```
Orchestrator: `render("orchestrator", "v2", ..., prior_context=prior_context_block(state))`. Analyst: `render("query", "v3", ..., prior_context=prior_context_block(state))` in **both** the overhead computation and the attempt loop. `graph/state.py`: add `prior_context: str` to `AnalyzerState`.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add prompts graph tests
git commit -m "feat: inject prior-session context into orchestrator and analyst prompts"
```

---

### Task 7: Memory service (summaries, prior context, project memory)

**Files:**
- Create: `accounts/memory.py`, `prompts/session_summary.v1.txt`
- Modify: `tests/test_settings.py` (remove the `session_summary` skip)
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `InsightHistory`, `SessionTracker`, `AppDB`, `render`, `extract_json`, `is_rate_limit`, `RateLimitExhausted`, `clip_to_tokens`, `safe_text`, `PromptSettings.overrides`, `config.PRIOR_SESSIONS`, `config.PRIOR_CONTEXT_TOKENS`, `config.ABANDONED_SESSION_MINUTES`, `config.MODEL_*`.
- Produces: `SUMMARY_KEYS = ("topics", "key_findings", "open_questions", "data_loaded")`; `summarise_session(llm, entries) -> dict | None` (None = rate limited, try later; unparsable output falls back to a plain summary built from the questions/findings); `class MemoryService(db, history, sessions, llm)`: `end_session(session_id, user_id) -> str` (returns `"summarised"`, `"skipped"` (no questions), or `"deferred"` (rate limited)), `summarise_pending(user_id) -> int` (count summarised; failures other than rate limits are re-raised only after all pending were attempted, or captured: choose to skip and continue), `prior_context(user_id) -> str`, `summaries_for(user_id, limit) -> list[dict]`; `write_project_memory(path, store, prompt_settings, history, models) -> Path` (creates parent dir; content sections `# Project memory`, `## Data sources`, `## Active prompt versions`, `## Recent approved insights`, `## Models`, `## Evaluation` ("n/a — LangSmith evaluations arrive in phase 3b")). All user/model-derived strings are passed through `safe_text` and whitespace-collapsed.
- Prompt: `prompts/session_summary.v1.txt` (variable `$record`).

- [ ] **Step 1: Write the failing tests**

`tests/test_memory.py`:
```python
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import config
from accounts.auth import AuthService
from accounts.db import AppDB
from accounts.history import InsightHistory
from accounts.memory import MemoryService, summarise_session, write_project_memory
from accounts.sessions import SessionTracker
from accounts.settings import PromptSettings
from graph.llm import RateLimitExhausted
from tests.fakes import FakeLLM

INS = {"finding": "Eng leads conversion.", "evidence": ["20%"], "recommendation": "Invest."}
GOOD = json.dumps({"topics": ["conversion"], "key_findings": ["Eng leads"], "open_questions": ["why?"],
                   "data_loaded": ["job_postings"]})


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


class RaisingLLM:
    def __init__(self, exc):
        self.exc = exc

    def invoke(self, prompt):
        raise self.exc


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    db = AppDB(tmp_path / "a.db", clock=clock)
    uid = AuthService(db).create_user("alice", "correct horse", "analyst")
    return {"clock": clock, "db": db, "uid": uid, "hist": InsightHistory(db), "sess": SessionTracker(db)}


def add_session(env, questions=("Which category converts best?",)):
    sid = env["sess"].start(env["uid"])
    for q in questions:
        env["hist"].add(env["uid"], sid, q, INS, None, [])
        env["sess"].touch(sid, questions=1)
    return sid


def test_summarise_session_good_json_is_clamped():
    entries = [{"question": "q", "insight": INS}]
    big = json.dumps({"topics": ["t"] * 9, "key_findings": ["x" * 500], "open_questions": [], "data_loaded": []})
    out = summarise_session(FakeLLM([big]), entries)
    assert len(out["topics"]) == 5 and len(out["key_findings"][0]) == 200 and set(out) == set(("topics", "key_findings", "open_questions", "data_loaded"))


def test_summarise_session_bad_json_falls_back_and_rate_limit_returns_none():
    entries = [{"question": "Which category converts best?", "insight": INS}]
    fb = summarise_session(FakeLLM(["not json"]), entries)
    assert fb["topics"] == ["Which category converts best?"] and fb["key_findings"] == ["Eng leads conversion."]
    assert summarise_session(RaisingLLM(RateLimitExhausted("x")), entries) is None
    with pytest.raises(RuntimeError):
        summarise_session(RaisingLLM(RuntimeError("boom")), entries)


def test_end_session_summarises_and_stores(env):
    sid = add_session(env)
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([GOOD]))
    assert mem.end_session(sid, env["uid"]) == "summarised"
    assert env["sess"].get(sid)["summarised"] == 1 and env["sess"].get(sid)["ended_at"]
    assert mem.summaries_for(env["uid"], 5)[0]["key_findings"] == ["Eng leads"]


def test_end_session_without_questions_and_when_rate_limited(env):
    empty = env["sess"].start(env["uid"])
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([]))
    assert mem.end_session(empty, env["uid"]) == "skipped"
    sid = add_session(env)
    limited = MemoryService(env["db"], env["hist"], env["sess"], RaisingLLM(RateLimitExhausted("x")))
    assert limited.end_session(sid, env["uid"]) == "deferred"
    assert env["sess"].get(sid)["summarised"] == 0 and env["sess"].get(sid)["ended_at"]
    later = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([GOOD]))
    assert later.summarise_pending(env["uid"]) == 1                     # retried at next login


def test_summarise_pending_only_abandoned_sessions(env):
    fresh, old = add_session(env), add_session(env, ("Show job views by month please",))
    env["clock"].now += timedelta(minutes=31)
    env["sess"].touch(fresh)                                            # active again just now
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([GOOD]))
    assert mem.summarise_pending(env["uid"]) == 1
    assert env["sess"].get(old)["summarised"] == 1 and env["sess"].get(fresh)["summarised"] == 0


def test_summarise_pending_continues_after_a_failure(env):
    add_session(env)
    add_session(env, ("Show job views by month please",))
    env["clock"].now += timedelta(minutes=31)
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM(["not json", GOOD]))
    assert mem.summarise_pending(env["uid"]) == 2


def test_prior_context_last_three_clipped_and_collapsed(env):
    mem = MemoryService(env["db"], env["hist"], env["sess"], FakeLLM([]))
    assert mem.prior_context(env["uid"]) == ""
    for i in range(5):
        sid = add_session(env)
        env["sess"].end(sid)
        env["db"].insert("INSERT INTO session_summaries (user_id, session_id, ts_utc, summary_json) VALUES (?,?,?,?)",
                         (env["uid"], sid, f"2026-09-2{i}T00:00:00.000000+00:00",
                          json.dumps({"topics": [f"topic{i}\nwith newline"], "key_findings": ["f" * 150] * 5,
                                      "open_questions": [], "data_loaded": []})))
    ctx = mem.prior_context(env["uid"])
    assert ctx.count("Session ") == 3 and "topic4" in ctx and "topic1" not in ctx and "\n\n" not in ctx
    from graph.budget import count_tokens
    assert count_tokens(ctx) <= config.PRIOR_CONTEXT_TOKENS


def test_write_project_memory(tmp_path, env):
    from data.store import SQLiteStore

    store = SQLiteStore(tmp_path / "an.db")
    store.replace_table(pd.DataFrame({"a": [1]}), "job_postings")
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "sql.v2.txt").write_text("$schema $limit $history $question $correction $error_note")
    ps = PromptSettings(env["db"], prompts_dir=tmp_path / "p")
    ps.set_active("sql", "v2", user_id=1)
    hid = env["hist"].add(env["uid"], "s", "Which # category?", {"finding": "**Bold** finding", "evidence": [],
                                                                   "recommendation": "r"}, None, [])
    env["hist"].mark_approved(hid)
    path = write_project_memory(tmp_path / "memory" / "PROJECT_MEMORY.md", store, ps, env["hist"],
                                {"smart": "m1", "fast": "m2"})
    text = path.read_text()
    for section in ("# Project memory", "## Data sources", "## Active prompt versions",
                    "## Recent approved insights", "## Models", "## Evaluation"):
        assert section in text
    assert "job_postings" in text and "sql: v2" in text and "m1" in text and "phase 3b" in text
    assert "**Bold**" not in text and "\n# category" not in text          # markdown from data is escaped
    assert path.name != "CLAUDE.md"
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_memory.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`prompts/session_summary.v1.txt`:
```
You summarise an analytics session for a talent team so the next session can continue where this one left off.

Session record (question and finding pairs):
$record

Return ONLY a JSON object: {"topics": ["..."], "key_findings": ["..."], "open_questions": ["..."], "data_loaded": ["table names or files mentioned"]}
Each list has at most 5 short strings. Use only facts present in the record.
```
`accounts/memory.py`:
```python
import json
from pathlib import Path

import config
from accounts.db import AppDB, iso
from accounts.history import InsightHistory
from accounts.sessions import SessionTracker
from graph.budget import clip_to_tokens
from graph.llm import RateLimitExhausted, is_rate_limit
from graph.parsing import extract_json
from graph.prompts import render
from graph.textsafe import safe_text

SUMMARY_KEYS = ("topics", "key_findings", "open_questions", "data_loaded")
ITEM_CHARS, MAX_ITEMS = 200, 5


def _flat(text) -> str:
    return " ".join(str(text).split())


def _fallback(entries: list[dict]) -> dict:
    return {
        "topics": [_flat(e["question"])[:ITEM_CHARS] for e in entries[:MAX_ITEMS]],
        "key_findings": [_flat(e["insight"]["finding"])[:ITEM_CHARS] for e in entries[:MAX_ITEMS]],
        "open_questions": [], "data_loaded": [],
    }


def summarise_session(llm, entries: list[dict]):
    record = "\n".join(f"Q: {_flat(e['question'])}\nA: {_flat(e['insight']['finding'])}" for e in entries[:20])
    prompt = render("session_summary", "v1", record=record)
    try:
        raw = llm.invoke(prompt).content
    except Exception as exc:
        if isinstance(exc, RateLimitExhausted) or is_rate_limit(exc):
            return None
        raise
    try:
        data = extract_json(raw)
        return {k: [_flat(x)[:ITEM_CHARS] for x in list(data.get(k, []))[:MAX_ITEMS]] for k in SUMMARY_KEYS}
    except (ValueError, TypeError, AttributeError):
        return _fallback(entries)


class MemoryService:
    def __init__(self, db: AppDB, history: InsightHistory, sessions: SessionTracker, llm):
        self.db, self.history, self.sessions, self.llm = db, history, sessions, llm

    def _summarise(self, session_id: str, user_id: int) -> str:
        entries = self.sessions.entries_for(session_id)
        if not entries:
            return "skipped"
        summary = summarise_session(self.llm, entries)
        if summary is None:
            return "deferred"
        self.db.insert("INSERT INTO session_summaries (user_id, session_id, ts_utc, summary_json) VALUES (?,?,?,?)",
                       (user_id, session_id, iso(self.db.now()), json.dumps(summary)))
        self.sessions.mark_summarised(session_id)
        return "summarised"

    def end_session(self, session_id: str, user_id: int) -> str:
        self.sessions.end(session_id)
        return self._summarise(session_id, user_id)

    def summarise_pending(self, user_id: int) -> int:
        done = 0
        for sid in self.sessions.pending_for_user(user_id, config.ABANDONED_SESSION_MINUTES):
            try:
                if self._summarise(sid, user_id) == "summarised":
                    done += 1
            except Exception:  # one bad session must not block the others
                continue
        return done

    def summaries_for(self, user_id: int, limit: int) -> list[dict]:
        rows = self.db.query("SELECT ts_utc, summary_json FROM session_summaries WHERE user_id = ?"
                             " ORDER BY ts_utc DESC, id DESC LIMIT ?", (user_id, limit))
        return [{**json.loads(r["summary_json"]), "ts_utc": r["ts_utc"]} for r in rows]

    def prior_context(self, user_id: int) -> str:
        lines = []
        for s in self.summaries_for(user_id, config.PRIOR_SESSIONS):
            parts = [f"Session {s['ts_utc'][:10]}:"]
            for label, key in (("topics", "topics"), ("findings", "key_findings"), ("open questions", "open_questions")):
                if s.get(key):
                    parts.append(f"{label}: " + "; ".join(_flat(x) for x in s[key]) + ".")
            lines.append(" ".join(parts))
        return clip_to_tokens("\n".join(lines), config.PRIOR_CONTEXT_TOKENS) if lines else ""


def write_project_memory(path, store, prompt_settings, history: InsightHistory, models: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overrides = prompt_settings.overrides()
    approved = [r for r in history.list(limit=200) if r["approved"]][:10]
    lines = ["# Project memory", "", f"_Generated {iso(prompt_settings.db.now())}_", "", "## Data sources"]
    lines += [f"- {safe_text(t)}" for t in store.list_tables()] or ["- (none)"]
    lines += ["", "## Active prompt versions"]
    lines += [f"- {safe_text(n)}: {safe_text(overrides.get(n, 'default'))}" for n in prompt_settings.names()]
    lines += ["", "## Recent approved insights"]
    lines += [f"- {r['ts_utc'][:10]} — {safe_text(_flat(r['question']))} → {safe_text(_flat(r['insight']['finding']))}"
              for r in approved] or ["- (none yet)"]
    lines += ["", "## Models"] + [f"- {safe_text(k)}: {safe_text(v)}" for k, v in models.items()]
    lines += ["", "## Evaluation", "- n/a — LangSmith evaluations arrive in phase 3b"]
    path.write_text("\n".join(lines) + "\n")
    return path
```
Remove the temporary `session_summary` skip from `tests/test_settings.py::test_required_vars_cover_real_prompt_files`.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add accounts/memory.py prompts/session_summary.v1.txt tests
git commit -m "feat: add session summaries, prior context and project memory file"
```

---

### Task 8: App shell — services, login gate, session lifecycle, Analyze module

**Files:**
- Create: `accounts/services.py`, `ui/__init__.py`, `ui/context.py`, `ui/auth_ui.py`, `ui/router.py`, `ui/analyze.py`, `tests/helpers.py`
- Modify: `app.py`, `tests/test_app.py` (all runtime-starting tests), `tests/test_app_*.py` if present
- Test: `tests/test_app_auth.py`

**Interfaces:**
- `accounts/services.py`: `@dataclass Services(db, auth, audit, history, sessions, memory, prompts, settings)` and `build_services(path, llm, clock=None) -> Services` (creates `AppDB(path)`, wires all services, calls `settings.apply(config)` and `prompts.apply()` so persisted overrides are live).
- `ui/router.py`: `allowed_pages(role) -> list[str]` (`Analyze` always; `History` if `can(role, "view_history")`; `Admin` if the role has any of `manage_users/manage_prompts/manage_config/view_audit`) and `render_page(ctx, page)` dispatching to `render_analyze`, `render_history`, `render_admin` (the History/Admin renderers arrive in Tasks 10-11 and show `st.info("Coming soon")` until then). `app.py` must call `router.allowed_pages(...)` and `router.render_page(...)` through the module (`from ui import router`) so tests can monkeypatch `ui.router.allowed_pages`.
- `ui/context.py`: `@dataclass Ctx(store, ingest_node, graph, llms, services, user: dict, session_id: str, shared: dict, messages: list)` where `user = {"id", "username", "role"}`.
- `app.py` flow: (1) page config/title and API-key check as today; (2) `store, ingest_node, graph, llms, services = get_runtime()` (cached; `get_runtime` builds `AppDB` at `config.APP_DB_PATH` and `build_services(..., fast)`); (3) if `services.auth.needs_bootstrap()` → `render_bootstrap(services)` and stop; (4) if `st.session_state.get("auth")` is missing → `render_login(services)` and stop; (5) else ensure a session (`st.session_state["session_id"]` via `services.sessions.start` when missing, at first render after login, including for pre-seeded test logins), build `Ctx`, render the sidebar account box + page radio (`key="page"`, options from `router.allowed_pages(role)`), and call `router.render_page(ctx, page)`.
- `ui/auth_ui.py`: `render_bootstrap(services)` (keys `bootstrap_username`, `bootstrap_password`, `bootstrap_confirm`, button `bootstrap_create`; creates the first admin, audits `user_created` with `role`, then logs them in); `render_login(services)` (keys `login_username`, `login_password`, button `login_submit`; on success sets `st.session_state["auth"] = {"user_id", "username", "role"}`, starts a session, runs `memory.summarise_pending(user_id)` best-effort and stores `shared["prior_context"] = memory.prior_context(user_id)`, audits `login`; on failure audits `login_failed` (or `account_locked` when the result reason is `locked`) and shows "Invalid username or password." / "Too many failed attempts. Try again later."); `render_account_box(ctx)` (sidebar: `Signed in as <username> (<role>)`, buttons `end_session` and `logout`, expander `Change password` with keys `pw_old`, `pw_new`, `pw_save`). `logout`/`end_session` call `memory.end_session(...)` (warning on failure, never blocking), audit `session_end` / `logout`, then clear `auth`, `session_id`, `shared`, `messages`, `pending_revision`, `ingest_prompts` from `st.session_state` and `st.rerun()` (End session then starts a fresh session for the same login).
- `ui/analyze.py`: the current chat/upload/deck code from `app.py` moved into `render_analyze(ctx)` and its helper functions, behaviour unchanged in this task (permission checks, audit and history hooks come in Task 9). It uses `ctx.shared`/`ctx.messages` instead of module-level globals.
- `tests/helpers.py`: `make_user(tmp_path, role="analyst", username=None, password="correct horse") -> dict` (creates the user in `tmp_path/"app.db"` with a separate `AppDB`, returns `{"user_id", "username", "role"}`); `start_app(monkeypatch, tmp_path, role="analyst", graph=None, question=None, key="gsk_fake") -> AppTest` (sets `GROQ_API_KEY`, patches `config.DB_PATH`, `USAGE_DB_PATH`, `APP_DB_PATH`, optionally patches `graph.build_graph.build_graph` to return `graph`, clears `st.cache_resource`, creates the user, pre-seeds `at.session_state["auth"]`, runs the app, and if `question` is given submits it). Tests must clear `st.cache_resource` in a `finally`. All existing app tests are migrated to `start_app` (analyst role unless a test needs more), keeping every assertion.

- [ ] **Step 1: Write the failing tests**

`tests/test_app_auth.py` (uses `helpers.start_app` and a `StubGraph` moved to `tests/helpers.py` from `tests/test_app.py`):
```python
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import config
import graph.build_graph as bg
from accounts.db import AppDB
from accounts.auth import AuthService
from tests.helpers import StubGraph, make_user, start_app


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
    return AppTest.from_file("../app.py").run(timeout=60)


def test_first_launch_shows_only_admin_creation(monkeypatch, tmp_path):
    at = fresh_app(monkeypatch, tmp_path)
    assert not at.exception
    assert at.text_input(key="bootstrap_username") and not at.chat_input
    at.text_input(key="bootstrap_username").set_value("root")
    at.text_input(key="bootstrap_password").set_value("correct horse")
    at.text_input(key="bootstrap_confirm").set_value("different pw!")
    at.button(key="bootstrap_create").click().run(timeout=60)
    assert any("match" in e.value.lower() for e in at.error)
    at.text_input(key="bootstrap_confirm").set_value("correct horse")
    at.button(key="bootstrap_create").click().run(timeout=60)
    assert not at.exception and at.chat_input                      # created and logged in
    db = AppDB(tmp_path / "app.db")
    assert AuthService(db).list_users()[0].role == "admin"
    assert any(r["action"] == "user_created" for r in db.query("SELECT action FROM audit_log"))


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
    actions = [r["action"] for r in AppDB(tmp_path / "app.db").query("SELECT action FROM audit_log ORDER BY id")]
    assert actions == ["login_failed", "login"]


def test_lockout_message_and_audit(monkeypatch, tmp_path):
    make_user(tmp_path, "analyst", username="alice", password="correct horse")
    at = fresh_app(monkeypatch, tmp_path)
    for _ in range(5):
        at.text_input(key="login_username").set_value("alice")
        at.text_input(key="login_password").set_value("wrong password")
        at.button(key="login_submit").click().run(timeout=60)
    assert any("Too many failed attempts" in e.value for e in at.error)
    actions = [r["action"] for r in AppDB(tmp_path / "app.db").query("SELECT action FROM audit_log")]
    assert "account_locked" in actions


def test_logout_clears_session_and_summarises(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst", question="Which category has the best conversion rate?")
    assert at.session_state["auth"]["role"] == "analyst"
    at.button(key="logout").click().run(timeout=60)
    assert not at.exception and "auth" not in at.session_state and not at.chat_input
    db = AppDB(tmp_path / "app.db")
    actions = [r["action"] for r in db.query("SELECT action FROM audit_log ORDER BY id")]
    assert actions[-2:] == ["session_end", "logout"]
    assert db.one("SELECT ended_at FROM sessions")["ended_at"]


def test_change_password(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="analyst")
    at.text_input(key="pw_old").set_value("correct horse")
    at.text_input(key="pw_new").set_value("brand new password")
    at.button(key="pw_save").click().run(timeout=60)
    assert not at.exception
    svc = AuthService(AppDB(tmp_path / "app.db"))
    name = at.session_state["auth"]["username"]
    assert svc.authenticate(name, "brand new password").ok
    assert "password_changed" in [r["action"] for r in AppDB(tmp_path / "app.db").query("SELECT action FROM audit_log")]


def test_key_check_still_precedes_login(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(config, "USAGE_DB_PATH", tmp_path / "u.db")
    monkeypatch.setattr(config, "APP_DB_PATH", tmp_path / "app.db")
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert any("GROQ_API_KEY" in e.value for e in at.error)


def test_no_pandas_tool_anywhere_in_app_code():
    from pathlib import Path

    root = Path(__file__).parent.parent
    for f in [root / "app.py", *(root / "ui").glob("*.py")]:
        assert "make_pandas_tool" not in f.read_text()
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_app_auth.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement**

`accounts/services.py`:
```python
from dataclasses import dataclass

import config
from accounts.audit import AuditLog
from accounts.auth import AuthService
from accounts.db import AppDB
from accounts.history import InsightHistory
from accounts.memory import MemoryService
from accounts.sessions import SessionTracker
from accounts.settings import AppSettings, PromptSettings


@dataclass
class Services:
    db: AppDB
    auth: AuthService
    audit: AuditLog
    history: InsightHistory
    sessions: SessionTracker
    memory: MemoryService
    prompts: PromptSettings
    settings: AppSettings


def build_services(path, llm, clock=None) -> Services:
    db = AppDB(path, clock=clock)
    history, sessions = InsightHistory(db), SessionTracker(db)
    services = Services(db, AuthService(db), AuditLog(db), history, sessions,
                        MemoryService(db, history, sessions, llm), PromptSettings(db), AppSettings(db))
    services.settings.apply(config)
    services.prompts.apply()
    return services
```
`ui/context.py`: the `Ctx` dataclass (fields as listed). `ui/auth_ui.py`, `ui/analyze.py` and `app.py` per the interface description; `app.py` keeps `PERSISTED` and `run_turn` etc. inside `ui/analyze.py`. Sidebar page radio:
```python
pages = ["Analyze"] + (["History"] if can(role, "view_history") else []) + \
        (["Admin"] if any(can(role, p) for p in ("manage_users", "manage_prompts", "manage_config", "view_audit")) else [])
page = st.sidebar.radio("Page", pages, key="page")
```
`tests/helpers.py`: move `StubGraph` and `run_app_with_stub` logic here (parametrised by role via `start_app`). Migrate every test in `tests/test_app.py` (setup error, error persistence, approve/revise, judge scores, degraded, deck failure, markdown safety) to `start_app`/`make_user`; keep their assertions. `test_app_does_not_construct_pandas_tool` is superseded by the new `test_no_pandas_tool_anywhere_in_app_code` (delete the old one). If `AppTest` needs the auth dict shape `{"user_id", "username", "role"}` under `at.session_state["auth"]`, the helper sets exactly that.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings, and `ls data/` shows no new `app.db`/`usage.db`/`memory/` created by tests (all paths are tmp).

- [ ] **Step 5: Commit**

```bash
git add accounts/services.py ui app.py tests
git commit -m "feat: add login gate, session lifecycle and modular Analyze page"
```

---

### Task 9: Analyze hooks — permissions, audit, history, sessions, prior context

**Files:**
- Create: `graph/trace.py`
- Modify: `ui/analyze.py`, `graph/nodes/judge.py` (import `generated_sql`), `tests/test_app_pages.py` (new)
- Test: `tests/test_app_pages.py` (part 1), `tests/test_trace.py`

**Interfaces:**
- `graph/trace.py`: `generated_sql(prompts) -> str` (the text after `"--- model output ---\n"` in the last `retrieve-sql` prompt entry, else `""`); `graph/nodes/judge.py::_generated_sql(state)` becomes `generated_sql(state.get("prompts", [])) or "(unknown)"`.
- `ui/analyze.py` behaviour:
  - `guard(ctx, permission) -> bool`: `require(ctx.user["role"], permission)`; on `PermissionDenied` show `st.error(str(exc))`, audit `denied:<permission>`, return False.
  - Upload: `guard(ctx, "upload")` before ingesting; audit `upload` with `file`, `table`, `action` (created/appended) and `rows` (never contents).
  - Question/revision: `guard(ctx, "ask")` first. After a successful turn: audit `question` (or `revise` for revisions) with `question`, `query_type`, `sql=generated_sql(result["prompts"])`, `models`, `guard_rejected`, `degraded`, `judge=[s["accepted"] for s in judge_scores]`; when the answer is not guard-rejected, `history.add(...)` (with `degraded` flag) and store the returned id as `history_id` in the message dict; `sessions.touch(session_id, questions=1)` for every asked question (not revisions or rejected ones).
  - Approve: sets the in-session entry approved (as today), `history.mark_approved(history_id)` when present, audits `approve`.
  - Export: `guard(ctx, "export")` before showing the deck section; the download button uses `on_click=` a function `audit_export(ctx, count)` that audits `export` with `insights=count`.
  - Sidebar `Answered by`, judge scores etc. unchanged.

- [ ] **Step 1: Write the failing tests**

`tests/test_trace.py`: `generated_sql([...])` returns the SQL after the marker from the last `retrieve-sql` entry, `""` when none/marker missing; the judge still reports `SQL:\nSELECT 1` (existing judge test keeps passing).
`tests/test_app_pages.py` (part 1; uses `start_app`, `StubGraph`, `AppDB`):
```python
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
    at = start_app(monkeypatch, tmp_path, role="analyst", graph=stub, question="What is the weather in Paris today?")
    assert rows(tmp_path, "SELECT * FROM insight_history") == []
    q = [r for r in rows(tmp_path) if r["action"] == "question"]
    assert '"guard_rejected": true' in q[0]["detail_json"]


def test_permission_denied_is_audited(monkeypatch, tmp_path):
    from ui import analyze
    from accounts.permissions import PermissionDenied
    # an unauthorised role is refused by the guard helper even if a widget were somehow reachable
    at = start_app(monkeypatch, tmp_path, role="analyst")
    at.session_state["auth"] = {**at.session_state["auth"], "role": "nobody"}
    at.chat_input[0].set_value("Which category has the best conversion rate?").run(timeout=60)
    assert any("does not allow" in e.value for e in at.error)
    assert "denied:ask" in [r["action"] for r in rows(tmp_path)]
    assert rows(tmp_path, "SELECT * FROM insight_history") == []


def test_prior_context_reaches_the_graph_state(monkeypatch, tmp_path):
    stub = StubGraph()
    at = start_app(monkeypatch, tmp_path, role="analyst", graph=stub)
    at.session_state["shared"]["prior_context"] = "Session 2026-09-24: topics: conversion."
    at.chat_input[0].set_value("Which category has the best conversion rate?").run(timeout=60)
    assert stub.states[-1]["prior_context"].startswith("Session 2026-09-24")


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
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_trace.py tests/test_app_pages.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement** the hooks as specified; add `generated_sql` and use it from `judge.py`. Keep `PERSISTED` unchanged (do not add `prior_context`: it lives in `shared` set at login and is not overwritten by graph results because `PERSISTED` excludes it).

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add graph ui tests
git commit -m "feat: enforce permissions and record audit, history and sessions in the Analyze page"
```

---

### Task 10: History page and comparative report

**Files:**
- Create: `ui/history_page.py`
- Modify: `app.py` (History branch), `tests/test_app_pages.py`
- Test: additions to `tests/test_app_pages.py`

**Interfaces:**
- `render_history(ctx)`: begins with `guard(ctx, "view_history")` (return on denial); audits `history_view` once per page render session (`st.session_state["history_viewed"]` flag reset on logout); two tabs `History` and `Comparative report` (the latter requires `can(role, "view_comparative")`, else the tab is omitted).
  - **History tab:** filters (`hist_user` selectbox of `All` + usernames, `hist_from`/`hist_to` `st.date_input` (optional, default empty via checkboxes or `None`), `hist_text` text input), a dataframe of rows (`id, when (ts_utc[:16]), user, question, finding, approved, degraded`), an expander per row (limit 25) showing evidence/recommendation via `safe_text`, the chart (`build_figure`) when present, and the data slice; a multiselect `hist_deck_pick` of row labels (`"<id>. <first 60 chars of finding>"`) and a `st.download_button("Export selected as slide deck")` built with `build_deck` from `{"question","insight","chart","slice","approved"}` entries (history rows already have these keys), guarded with try/except like the Analyze deck; export audited via `on_click` as `export` with `source: "history"`.
  - **Comparative report tab:** `repeated_questions()` in a selectbox `cmp_question` (label = sample question + `(N sessions)`); shows one column per version (newest first, up to 4 via `st.columns`), each with date, user, session id (first 8 chars), finding, evidence bullets and recommendation (all through `safe_text`), plus a compact table of the same rows. When there are no repeated questions show `st.info("No question has been asked in more than one session yet.")`.
- `app.py`: `History` → `render_history(ctx)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_app_pages.py`; seed data through `InsightHistory`/`SessionTracker` on a separate `AppDB` at `tmp_path/"app.db"` **before** the app starts, using `make_user` for users)

```python
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
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_app_pages.py -v` — Expected: new tests FAIL.

- [ ] **Step 3: Implement** `ui/history_page.py` as described and wire `app.py`. For date filters use `st.date_input` with `value=None` (Streamlit >= 1.36 supports an empty default); convert to UTC datetime bounds (`since = datetime.combine(d, time.min, tzinfo=UTC)`, `until = ... time.max`).

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add ui/history_page.py app.py tests
git commit -m "feat: add manager history page and comparative report"
```

---

### Task 11: Admin pages (users, prompts, config, audit, memory)

**Files:**
- Create: `ui/admin_page.py`
- Modify: `app.py` (Admin branch), `tests/test_app_pages.py`
- Test: additions to `tests/test_app_pages.py`, plus `tests/test_admin_actions.py` for the pure action helpers

**Interfaces:**
- `ui/admin_page.py`: `render_admin(ctx)` with tabs shown only when the role has the permission: **Users** (`manage_users`), **Prompts** (`manage_prompts`), **Config** (`manage_config`), **Audit** (`view_audit`), **Memory** (`manage_config`). Each tab starts with `guard(ctx, permission)`.
- Pure helpers (unit-tested, used by the tabs so behaviour is testable without clicking): `admin_create_user(ctx, username, password, role) -> str` (returns a status message; raises nothing: `AccountError` -> message; audits `user_created` with `role` and target username), `admin_set_role(ctx, user_id, role)`, `admin_set_active(ctx, user_id, active)`, `admin_reset_password(ctx, user_id, new)` (audit `user_updated`/`password_changed` with target ids, never the password), `admin_set_prompt(ctx, name, version)` (audit `prompt_version_changed` with `name`, `version`, `previous`; applies overrides immediately via `services.prompts.apply()`), `admin_clear_prompt(ctx, name)`, `admin_set_config(ctx, key, value)` (audit `config_changed` with `key`, `old`, `new`; applies immediately), `admin_write_memory(ctx) -> Path` (writes `config.MEMORY_DIR/PROJECT_MEMORY.md`, audit `config_changed` with `key: "project_memory"`). Every helper calls `require(role, ...)` first and raises `PermissionDenied` (the tab catches it, shows the error and audits `denied:<perm>`).
- Users tab widgets: table of users; create form keys `new_username`, `new_password`, `new_role`, button `create_user`; per-user role `selectbox` (key `role_<id>`) + button `apply_role_<id>`, `toggle_active_<id>` button (label Disable/Enable), and password reset (`reset_pw_<id>` text input + `reset_pw_apply_<id>` button).
- Prompts tab: for each prompt name a selectbox `prompt_<name>` of compatible versions with the active one (or "default") preselected, an "Apply" button `prompt_apply_<name>`, a "Use default" button `prompt_clear_<name>`, and a `st.code` preview of the selected version file.
- Config tab: checkboxes `cfg_JUDGE_ENABLED`, `cfg_JUDGE_RETRIEVAL`, number input `cfg_JUDGE_MIN_SCORE` (1-5) and a Save button `cfg_save`.
- Audit tab: filters `audit_user`, `audit_action` (selectbox of known actions + All), date inputs, `st.dataframe` of rows and `st.download_button("Download CSV")` with `AuditLog.to_csv` (audit `export` with `source: "audit"` via `on_click`).
- Memory tab: button `write_memory` and the current file contents in an expander.

- [ ] **Step 1: Write the failing tests**

`tests/test_admin_actions.py` (helpers on a real `Services` over a tmp DB; `Ctx` built with `Services`, a stub store exposing `list_tables()`, `user={"id","username","role"}`):
- create-user helper creates + audits without the password, shows validation errors as messages, refuses non-admins (`PermissionDenied`), duplicate username message;
- role change / disable respects last-admin protection (message, no audit of success);
- prompt change: incompatible version refused with message; compatible sets the override, `graph.prompts.get_overrides()` reflects it, audit has `previous` and new version; clear removes it;
- config change validates, applies to `config`, audits old/new;
- `admin_write_memory` writes `PROJECT_MEMORY.md` under a tmp `config.MEMORY_DIR` (monkeypatched), never a `CLAUDE.md`, and refuses non-admins;
- audit rows never contain the new user's password (grep the JSON).
AppTest flows in `tests/test_app_pages.py`:
```python
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


def test_manager_is_refused_on_the_admin_page_even_if_offered(monkeypatch, tmp_path):
    import ui.router as router

    monkeypatch.setattr(router, "allowed_pages", lambda role: ["Analyze", "History", "Admin"])
    at = start_app(monkeypatch, tmp_path, role="manager")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    assert not at.exception
    assert any("does not allow" in e.value for e in at.error)
    assert not [t for t in at.text_input if t.key == "new_username"]
    assert any(r["action"].startswith("denied:") for r in rows(tmp_path))


def test_admin_prompt_and_config_changes_apply_and_are_audited(monkeypatch, tmp_path):
    from graph import prompts

    monkeypatch.setattr(config, "JUDGE_MIN_SCORE", config.JUDGE_MIN_SCORE)      # restored at teardown
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


def test_audit_tab_lists_actions_and_filters(monkeypatch, tmp_path):
    at = start_app(monkeypatch, tmp_path, role="admin", question="Which category has the best conversion rate?")
    at.sidebar.radio(key="page").set_value("Admin").run(timeout=60)
    assert not at.exception

    def audit_frame():
        return [d.value for d in at.dataframe if "action" in d.value.columns][0]

    assert {"question"} <= set(audit_frame()["action"])
    at.selectbox(key="audit_action").set_value("question").run(timeout=60)
    assert set(audit_frame()["action"]) == {"question"}
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_admin_actions.py tests/test_app_pages.py -v` — Expected: FAIL.

- [ ] **Step 3: Implement** `ui/admin_page.py` (helpers first, then tabs) and wire `app.py`. Every helper takes `ctx` and uses `ctx.services`. Messages are returned to the tab and shown with `st.success`/`st.error`. Never render passwords back.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest -q` — Expected: all pass, 0 warnings; `git status` shows no `memory/` or `data/*.db` changes.

- [ ] **Step 5: Commit**

```bash
git add ui/admin_page.py app.py tests
git commit -m "feat: add admin pages for users, prompts, config, audit and project memory"
```

---

### Task 12: Docs and final verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_docs_and_layout.py`

**Interfaces:**
- README gains a "Phase 3a: accounts, audit, memory" section: first launch creates the admin; roles and the permission table; lockout rules; the audit trail (what is and is not logged); history/comparative report; session summaries and prior context (600-token cap, when summaries are written); prompt-version and config admin pages; `memory/PROJECT_MEMORY.md` (why not `CLAUDE.md`); limits (no HTTPS, login lost on refresh, uploaded tables shared, history slices are visible to Managers/Admins); how to reset a forgotten admin password (stop the app and use a one-line `python -c` snippet that calls `AuthService(AppDB(config.APP_DB_PATH)).reset_password(...)`; give the exact snippet, no password inside).

- [ ] **Step 1: Write the failing tests** (`tests/test_docs_and_layout.py`)
```python
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_documents_phase_3a():
    text = (ROOT / "README.md").read_text()
    for needle in ("Phase 3a", "PROJECT_MEMORY.md", "lock", "audit", "Manager", "Admin", "reset_password"):
        assert needle in text, needle
    assert "gsk_" not in text


def test_gitignore_covers_runtime_state():
    ignore = (ROOT / ".gitignore").read_text()
    assert "memory/" in ignore and "data/*.db" in ignore and ".env" in ignore


def test_nothing_writes_a_repo_root_claude_md():
    for f in list((ROOT / "accounts").glob("*.py")) + list((ROOT / "ui").glob("*.py")) + [ROOT / "app.py"]:
        assert "CLAUDE.md" not in f.read_text().replace("`CLAUDE.md`", ""), f
    assert not (ROOT / "CLAUDE.md").exists()


def test_accounts_package_uses_only_the_standard_library_for_crypto_and_storage():
    for f in (ROOT / "accounts").glob("*.py"):
        src = f.read_text()
        for banned in ("bcrypt", "passlib", "argon2", "sqlalchemy", "streamlit_authenticator"):
            assert banned not in src, (f, banned)
```
(Adjust the `CLAUDE.md` check if a docstring must mention the name: it should only appear in README/spec explanations, not in code paths that write files.)

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest tests/test_docs_and_layout.py -v` — Expected: FAIL (README lacks the section).

- [ ] **Step 3: Implement** the README section per the interface list.

- [ ] **Step 4: Full verification** — Run `.venv/bin/pytest -q` (0 warnings), `.venv/bin/pip check`, `ls data/ memory/ 2>&1` (tests must not have created `app.db`/`usage.db`/`memory/` in the repo; if they did, fix the leaking test), and an offline `AppTest` smoke (scratch script, not committed): bootstrap an admin in a temp dir, create an analyst, log in as each role, and confirm the page radio options are `["Analyze"]`, `["Analyze","History"]`, `["Analyze","History","Admin"]`.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_docs_and_layout.py
git commit -m "docs: document phase 3a accounts, audit and memory"
```

---

## Self-Review (spec coverage)

| Spec requirement | Task |
|---|---|
| `data/app.db` with the seven tables; stdlib only | 1 |
| Scrypt hashing, validation, lockout, disabled accounts, last-admin protection, bootstrap | 2 |
| Permission matrix enforced in code; denials audited | 3, 9, 10, 11 |
| Audit trail incl. secret scrubbing and CSV | 3, 9, 11 |
| Insight history, comparative queries, session tracking | 4 |
| Prompt version control (compatibility-checked) and judge/config settings applied live | 5, 11 |
| Prior-context prompts (`orchestrator.v2`, `query.v3`), state key, 600-token cap | 6, 7 |
| Session summaries at End session/logout/next login; rate-limit deferral | 7, 8 |
| `memory/PROJECT_MEMORY.md`, never `CLAUDE.md` | 7, 11, 12 |
| Login gate, bootstrap form, sidebar router by role, change password | 8 |
| Audit/history/session hooks in the Analyze page; approve/revise/export audited | 9 |
| Manager history page, deck from history, comparative report | 10 |
| Admin users/prompts/config/audit/memory pages | 11 |
| README limits and reset-admin instructions | 12 |
| Success criteria 1-9 | 8 (1, 3), 8-11 (2), 9-11 (4), 10 (5), 7-9 (6), 11 (7), 7/11/12 (8), all (9) |

Type/name consistency checked: `AppDB`, `AuthService`, `User`, `AuthResult`, `AccountError`, `PermissionDenied`, `can`/`require`, `AuditLog.record/query/to_csv`, `InsightHistory`, `SessionTracker`, `PromptSettings`, `AppSettings`, `REQUIRED_VARS`, `MemoryService`, `Services`, `Ctx`, `new_turn`/`prior_context`, widget keys (`login_*`, `bootstrap_*`, `logout`, `end_session`, `pw_*`, `page`, `hist_*`, `cmp_question`, `new_*`, `create_user`, `prompt_*`, `cfg_*`) and audit action names are used identically wherever they appear.
