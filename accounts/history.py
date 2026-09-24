from __future__ import annotations

import json
import re

import config
from accounts import redact
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
        question = redact.redact_text(question)
        insight = redact.scrub_value(insight, max_str=2000)
        chart = redact.scrub_value(chart, max_str=2000) if chart else None
        rows = [{k: redact.scrub_value(v, max_str=2000) for k, v in dict(r).items()}
                for r in list(data_slice)[: config.MEMORY_SLICE_ROWS]]
        return self.db.insert(
            "INSERT INTO insight_history (ts_utc, user_id, session_id, question, question_norm, insight_json,"
            " chart_json, slice_json, approved, degraded) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (iso(self.db.now()), user_id, session_id, question, normalise_question(question),
             json.dumps(insight, default=str), json.dumps(chart, default=str) if chart else None,
             json.dumps(rows, default=str), 1 if approved else 0, 1 if degraded else 0),
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
            escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += " AND h.question LIKE ? ESCAPE '\\' COLLATE NOCASE"
            params.append(f"%{escaped}%")
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
        with self.db.lock:
            return self._repeated(min_sessions)

    def _repeated(self, min_sessions: int) -> list[dict]:
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
