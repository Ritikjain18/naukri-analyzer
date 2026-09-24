import json
import os
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

SUMMARY_KEYS = ("topics", "key_findings", "open_questions", "data_loaded")
ITEM_CHARS, MAX_ITEMS = 200, 5
# sessions.summarised: 0 = pending, 1 = done, 2 = claimed by a running summarisation
PENDING, DONE, IN_PROGRESS = 0, 1, 2


def _flat(text) -> str:
    return " ".join(str(text).split())


def _clean(text) -> str:
    """Whitespace-collapse, redact secrets, then clamp (redact first so a cut never leaves a partial secret)."""
    return redact_text(_flat(text))[:ITEM_CHARS]


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
        return {k: [_clean(x) for x in list(data.get(k, []))[:MAX_ITEMS]] for k in SUMMARY_KEYS}
    except (ValueError, TypeError, AttributeError):
        return _fallback(entries)


class MemoryService:
    def __init__(self, db: AppDB, history: InsightHistory, sessions: SessionTracker, llm):
        self.db, self.history, self.sessions, self.llm = db, history, sessions, llm

    def _claim(self, session_id: str) -> bool:
        cur = self.db.execute("UPDATE sessions SET summarised = ? WHERE id = ? AND summarised = ?",
                              (IN_PROGRESS, session_id, PENDING))
        return cur.rowcount == 1

    def _release(self, session_id: str) -> None:
        self.db.execute("UPDATE sessions SET summarised = ? WHERE id = ? AND summarised = ?",
                        (PENDING, session_id, IN_PROGRESS))

    def _summarise(self, session_id: str, user_id: int) -> str:
        if not self._claim(session_id):
            return "skipped"                                   # done already, or another caller is on it
        try:
            entries = self.sessions.entries_for(session_id)
            summary = summarise_session(self.llm, entries) if entries else None
            if not entries:
                self._release(session_id)
                return "skipped"
            if summary is None:
                self._release(session_id)
                return "deferred"
            with self.db.lock:                                 # store + mark atomically w.r.t. other threads
                self.db.insert("INSERT INTO session_summaries (user_id, session_id, ts_utc, summary_json)"
                               " VALUES (?,?,?,?)", (user_id, session_id, iso(self.db.now()), json.dumps(summary)))
                self.sessions.mark_summarised(session_id)
            return "summarised"
        except BaseException:
            self._release(session_id)
            raise

    def end_session(self, session_id: str, user_id: int) -> str:
        self.sessions.end(session_id)                          # raises AccountError for an unknown session
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
