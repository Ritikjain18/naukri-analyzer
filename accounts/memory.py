import json
import logging
import os
import re
from datetime import timedelta
import tempfile
from pathlib import Path

import config
from accounts.db import AppDB, iso
from accounts.history import InsightHistory
from accounts.redact import redact_text
from accounts.sessions import SessionTracker
from graph.budget import clip_to_tokens
from graph.llm import RateLimitExhausted, is_rate_limit
from graph.parsing import extract_json
from graph.prompts import render
from graph.textsafe import safe_text

log = logging.getLogger(__name__)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

SUMMARY_KEYS = ("topics", "key_findings", "open_questions", "data_loaded")
ITEM_CHARS, MAX_ITEMS = 200, 5
# sessions.summarised: 0 = pending, 1 = done, 2 = claimed by a running summarisation
PENDING, DONE, IN_PROGRESS = 0, 1, 2


def _flat(text) -> str:
    return " ".join(str(text).split())


def _clean(text) -> str:
    """Whitespace-collapse, redact secrets, then clamp (redact first so a cut never leaves a partial secret)."""
    return redact_text(_flat(text))[:ITEM_CHARS]


def _items(value) -> list[str]:
    """Model output for a summary list. A bare string is one item; only str and int items are kept."""
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list):
        return []
    return [_clean(x) for x in value if isinstance(x, (str, int)) and not isinstance(x, bool)][:MAX_ITEMS]


def _neutralise(text) -> str:
    """Light prompt-control neutralisation for text injected into prompts: strip control characters, drop
    backticks, turn angle brackets into spaces (no tag-like text), collapse whitespace, then redact.
    This plus the labelled block from Task 6 is how the spec's "escaped of prompt-control text" is met."""
    text = _CONTROL.sub("", str(text)).replace("`", "")
    return _flat(text.replace("<", " ").replace(">", " "))


def _fallback(entries: list[dict]) -> dict:
    return {
        "topics": [_clean(e["question"]) for e in entries[:MAX_ITEMS]],
        "key_findings": [_clean(e["insight"]["finding"]) for e in entries[:MAX_ITEMS]],
        "open_questions": [], "data_loaded": [],
    }


def summarise_session(llm, entries: list[dict]):
    record = "\n".join(f"Q: {_flat(e['question'])}\nA: {_flat(e['insight']['finding'])}" for e in entries[:20])
    prompt = render("session_summary", "v1", record=redact_text(record))
    try:
        raw = llm.invoke(prompt).content
    except Exception as exc:
        if isinstance(exc, RateLimitExhausted) or is_rate_limit(exc):
            return None
        raise
    try:
        data = extract_json(raw)
        return {k: _items(data.get(k, [])) for k in SUMMARY_KEYS}
    except (ValueError, TypeError, AttributeError):
        return _fallback(entries)


class MemoryService:
    def __init__(self, db: AppDB, history: InsightHistory, sessions: SessionTracker, llm):
        self.db, self.history, self.sessions, self.llm = db, history, sessions, llm

    def _claim(self, session_id: str) -> bool:
        """Atomically claim a pending session. A claim implies the session is over, so ended_at is stamped."""
        cur = self.db.execute(
            "UPDATE sessions SET summarised = ?, ended_at = COALESCE(ended_at, ?) WHERE id = ? AND summarised = ?",
            (IN_PROGRESS, iso(self.db.now()), session_id, PENDING))
        return cur.rowcount == 1

    def _release(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET summarised = ? WHERE id = ? AND summarised = ?",
                        (PENDING, session_id, IN_PROGRESS))

    def _reset_stale_claims(self, user_id: int) -> None:
        """A summary call takes seconds, so a claim older than ABANDONED_SESSION_MINUTES belongs to a killed
        process and is certainly stale; release it so the session is summarised again."""
        cutoff = iso(self.db.now() - timedelta(minutes=config.ABANDONED_SESSION_MINUTES))
        with self.db.lock:
            self.db.execute("UPDATE sessions SET summarised = ? WHERE user_id = ? AND summarised = ?"
                            " AND ended_at <= ?", (PENDING, user_id, IN_PROGRESS, cutoff))

    def _summarise(self, session_id: str, user_id: int) -> str:
        """Returns "summarised", "skipped" (no questions), "deferred" (rate limited) or "in_progress"
        (another caller holds the claim; an already-summarised session reports "summarised")."""
        if not self._claim(session_id):
            row = self.sessions.get(session_id)
            return "summarised" if row and row["summarised"] == DONE else "in_progress"
        try:
            entries = self.sessions.entries_for(session_id)
            if not entries:
                self._release(session_id)
                return "skipped"
            summary = summarise_session(self.llm, entries)
            if summary is None:
                self._release(session_id)
                return "deferred"
            with self.db.lock:                                 # insert + mark as one unit
                row_id = self.db.insert("INSERT INTO session_summaries (user_id, session_id, ts_utc, summary_json)"
                                        " VALUES (?,?,?,?)",
                                        (user_id, session_id, iso(self.db.now()), json.dumps(summary)))
                try:
                    self.sessions.mark_summarised(session_id)
                except BaseException:
                    self.db.execute("DELETE FROM session_summaries WHERE id = ?", (row_id,))
                    raise
            return "summarised"
        except BaseException:
            self._release(session_id)
            raise

    def end_session(self, session_id: str, user_id: int) -> str:
        self.sessions.end(session_id)                          # raises AccountError for an unknown session
        return self._summarise(session_id, user_id)

    def summarise_pending(self, user_id: int) -> int:
        done = 0
        self._reset_stale_claims(user_id)
        for sid in self.sessions.pending_for_user(user_id, config.ABANDONED_SESSION_MINUTES):
            try:
                if self._summarise(sid, user_id) == "summarised":
                    done += 1
            except Exception as exc:  # one bad session must not block the others
                log.warning("summary failed for session %s: %s", sid, type(exc).__name__)   # no message: may hold secrets
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
                    parts.append(f"{label}: " + "; ".join(_neutralise(x) for x in s[key]) + ".")
            if len(parts) > 1:
                lines.append(" ".join(parts))
        # Redact again on read so rows stored before redaction existed cannot leak into prompts.
        return clip_to_tokens(redact_text("\n".join(lines)), config.PRIOR_CONTEXT_TOKENS) if lines else ""


def _md(text) -> str:
    return safe_text(redact_text(_flat(text)))


def write_project_memory(path, store, prompt_settings, history: InsightHistory, models: dict) -> Path:
    path = Path(path)
    if path.name.lower() == "claude.md":
        raise ValueError("Refusing to write project memory to CLAUDE.md.")
    if path.is_symlink():
        raise ValueError("Refusing to write project memory through a symlink.")
    path.parent.mkdir(parents=True, exist_ok=True)
    overrides = prompt_settings.overrides()
    approved = [r for r in history.list(limit=200) if r["approved"]][:10]
    lines = ["# Project memory", "", f"_Generated {iso(prompt_settings.db.now())}_", "", "## Data sources"]
    lines += [f"- `{_flat(redact_text(str(t))).replace(chr(96), chr(39))}`" for t in store.list_tables()] or ["- (none)"]
    lines += ["", "## Active prompt versions"]
    lines += [f"- {_md(n)}: {_md(overrides.get(n, 'default'))}" for n in prompt_settings.names()]
    lines += ["", "## Recent approved insights"]
    lines += [f"- {r['ts_utc'][:10]} — {_md(r['question'])} → {_md(r['insight']['finding'])}"
              for r in approved] or ["- (none yet)"]
    lines += ["", "## Models"] + [f"- {_md(k)}: {_md(v)}" for k, v in models.items()]
    lines += ["", "## Evaluation", "- n/a — LangSmith evaluations arrive in phase 3b"]
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pm-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path
