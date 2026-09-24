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
