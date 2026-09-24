import csv
import io
import json
import logging
import re

from accounts.db import AppDB, iso
from accounts.redact import scrub_value

logger = logging.getLogger(__name__)

SECRET_KEY = re.compile(r"pass|secret|token|api_?key|hash", re.I)
MAX_VALUE = 2000
MAX_DETAIL_JSON = 20000


def _scrub(detail: dict) -> dict:
    """Scrub detail dict by dropping secret-key entries and redacting/truncating values."""
    clean = {}
    for key, value in detail.items():
        if SECRET_KEY.search(key):
            continue
        # Use scrub_value for deep redaction and truncation
        clean[key] = scrub_value(value, max_str=MAX_VALUE)
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

        # Wrap scrubbing and serialization in try-catch to ensure record always succeeds
        # Keep DB insert outside the try to not mask database errors
        try:
            scrubbed = _scrub(detail)
            detail_json = json.dumps(scrubbed, default=str)

            # If detail_json exceeds max size, store truncated marker instead
            if len(detail_json) > MAX_DETAIL_JSON:
                detail_json = json.dumps({"truncated": True, "size": len(detail_json)})
        except Exception:
            # If scrubbing or JSON encoding fails, degrade gracefully and log
            logger.debug("Audit detail scrubbing failed", exc_info=True)
            detail_json = json.dumps({"scrub_error": True})

        return self.db.insert(
            "INSERT INTO audit_log (ts_utc, user_id, username, action, detail_json, session_id) VALUES (?,?,?,?,?,?)",
            (iso(self.db.now()), user_id, username, action, detail_json, session_id),
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
    def _escape_formula_injection(value: str) -> str:
        """Prefix cells that start with =, +, -, @, tab or CR with single quote."""
        if not value:
            return value
        if value[0] in ('=', '+', '-', '@', '\t', '\r'):
            return "'" + value
        return value

    @staticmethod
    def to_csv(rows: list[dict]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ts_utc", "username", "action", "session_id", "detail"])
        for r in rows:
            ts = AuditLog._escape_formula_injection(r["ts_utc"])
            username = AuditLog._escape_formula_injection(r["username"] or "")
            action = AuditLog._escape_formula_injection(r["action"])
            session_id = AuditLog._escape_formula_injection(r["session_id"] or "")
            detail = json.dumps(r["detail"], ensure_ascii=False)
            writer.writerow([ts, username, action, session_id, detail])
        return buf.getvalue()
