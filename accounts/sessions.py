import uuid
from datetime import timedelta

from accounts.auth import AccountError
from accounts.db import AppDB, iso
from accounts.history import _row


class SessionTracker:
    def __init__(self, db: AppDB):
        self.db = db

    def _require(self, session_id: str) -> None:
        if self.get(session_id) is None:
            raise AccountError("Unknown session.")

    def start(self, user_id: int) -> str:
        sid, now = uuid.uuid4().hex, iso(self.db.now())
        self.db.execute("INSERT INTO sessions (id, user_id, started_at, last_activity_at) VALUES (?,?,?,?)",
                        (sid, user_id, now, now))
        return sid

    def get(self, session_id: str):
        return self.db.one("SELECT * FROM sessions WHERE id = ?", (session_id,))

    def touch(self, session_id: str, questions: int = 0) -> None:
        with self.db.lock:
            self._require(session_id)
            self.db.execute("UPDATE sessions SET last_activity_at = ?, question_count = question_count + ? WHERE id = ?",
                            (iso(self.db.now()), questions, session_id))

    def end(self, session_id: str) -> None:
        with self.db.lock:
            self._require(session_id)
            self.db.execute("UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                            (iso(self.db.now()), session_id))

    def mark_summarised(self, session_id: str) -> None:
        with self.db.lock:
            self._require(session_id)
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
