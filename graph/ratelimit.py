import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import MODEL_LIMITS

HEADROOM = 0.9


class RateLimitManager:
    def __init__(self, path, limits=None, clock=None):
        self.limits = limits if limits is not None else MODEL_LIMITS
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.Lock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_usage (ts_utc TEXT NOT NULL, model TEXT NOT NULL, tokens INTEGER NOT NULL)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_model_ts ON llm_usage (model, ts_utc)")
        self._conn.commit()

    @staticmethod
    def _iso(moment: datetime) -> str:
        return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")

    def record(self, model: str, tokens: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO llm_usage (ts_utc, model, tokens) VALUES (?, ?, ?)",
                (self._iso(self._clock()), model, int(tokens)),
            )
            self._conn.commit()

    def _used_since(self, model: str, since: datetime) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM llm_usage WHERE model = ? AND ts_utc >= ?",
                (model, self._iso(since)),
            ).fetchone()
        return int(row[0])

    def used_last_minute(self, model: str) -> int:
        return self._used_since(model, self._clock() - timedelta(seconds=60))

    def used_today(self, model: str) -> int:
        now = self._clock().astimezone(timezone.utc)
        return self._used_since(model, now.replace(hour=0, minute=0, second=0, microsecond=0))

    def can_use(self, model: str, est_tokens: int) -> bool:
        limit = self.limits.get(model)
        if not limit:
            return True
        if est_tokens > limit["tpm"]:
            return False
        if self.used_last_minute(model) + est_tokens > HEADROOM * limit["tpm"]:
            return False
        return self.used_today(model) + est_tokens <= HEADROOM * limit["tpd"]
