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
    """reason is ok|invalid|locked|disabled. The UI must show the generic
    "Invalid username or password." for "disabled" too; the distinct reason
    is kept for auditing only."""

    ok: bool
    user: User | None
    reason: str
    newly_locked: bool = False


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
        n, r, p = int(n), int(r), int(p)
        if n < 2 or n > 2**20 or n & (n - 1) or not 1 <= r <= 32 or not 1 <= p <= 16:
            return False
        salt, expected = base64.urlsafe_b64decode(salt_b64), base64.urlsafe_b64decode(digest_b64)
        if len(expected) != 32:
            return False
        actual = hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def validate_username(username: str) -> None:
    if not USERNAME_RE.match(username):
        raise AccountError("Username must be 3-32 characters: letters, digits, '_', '.', '-'.")


def validate_password(password: str) -> None:
    try:
        password.encode("utf-8")
    except UnicodeEncodeError:
        raise AccountError("Password contains unsupported characters.") from None
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
    # Mutating operations hold db.lock across their read -> check -> write
    # sequence. This serialises within one process (one AppDB instance);
    # cross-process / second-instance races are out of scope for this local app.
    def __init__(self, db: AppDB):
        self.db = db

    def needs_bootstrap(self) -> bool:
        return self.db.one("SELECT COUNT(*) AS n FROM users")["n"] == 0

    def create_user(self, username: str, password: str, role: str) -> int:
        with self.db.lock:
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

    def create_first_admin(self, username: str, password: str) -> int:
        with self.db.lock:
            if self.db.one("SELECT COUNT(*) AS n FROM users")["n"] != 0:
                raise AccountError("Setup is already complete.")
            return self.create_user(username, password, "admin")

    def get_user(self, user_id: int) -> User | None:
        row = self.db.one("SELECT * FROM users WHERE id = ?", (user_id,))
        return _user(row) if row else None

    def get_by_username(self, username: str) -> User | None:
        row = self.db.one("SELECT * FROM users WHERE username = ?", (username.strip(),))
        return _user(row) if row else None

    def list_users(self) -> list[User]:
        return [_user(r) for r in self.db.query("SELECT * FROM users ORDER BY username")]

    def authenticate(self, username: str, password: str) -> AuthResult:
        with self.db.lock:
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
                return AuthResult(False, None, "locked" if locked else "invalid", newly_locked=locked)
            self.db.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?",
                            (iso(now), row["id"]))
            return AuthResult(True, _user(row), "ok")

    def change_password(self, user_id: int, old: str, new: str) -> None:
        with self.db.lock:
            row = self.db.one("SELECT password_hash FROM users WHERE id = ?", (user_id,))
            if row is None:
                raise AccountError("Unknown user.")
            if not verify_password(old, row["password_hash"]):
                raise AccountError("Current password is incorrect.")
            validate_password(new)
            self.db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new), user_id))

    def reset_password(self, user_id: int, new: str) -> None:
        with self.db.lock:
            validate_password(new)
            if self.get_user(user_id) is None:
                raise AccountError("Unknown user.")
            self.db.execute(
                "UPDATE users SET password_hash = ?, failed_attempts = 0, locked_until = NULL WHERE id = ?",
                (hash_password(new), user_id),
            )

    def _active_admins(self) -> int:
        return self.db.one("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1")["n"]

    def set_role(self, user_id: int, role: str) -> None:
        with self.db.lock:
            if role not in ROLES:
                raise AccountError("Unknown role.")
            target = self.get_user(user_id)
            if target is None:
                raise AccountError("Unknown user.")
            if target.role == "admin" and role != "admin" and target.active and self._active_admins() <= 1:
                raise AccountError("At least one active admin is required.")
            self.db.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))

    def set_active(self, user_id: int, active: bool) -> None:
        with self.db.lock:
            target = self.get_user(user_id)
            if target is None:
                raise AccountError("Unknown user.")
            if not active and target.role == "admin" and target.active and self._active_admins() <= 1:
                raise AccountError("At least one active admin is required.")
            self.db.execute("UPDATE users SET active = ? WHERE id = ?", (1 if active else 0, user_id))

