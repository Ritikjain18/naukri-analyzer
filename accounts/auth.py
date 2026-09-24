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
