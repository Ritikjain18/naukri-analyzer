import json
import logging
import re

import config
from accounts.auth import AccountError
from accounts.db import AppDB, iso
from graph import prompts as prompt_registry
from graph.prompts import placeholders  # noqa: F401  (re-exported)

log = logging.getLogger(__name__)

REQUIRED_VARS = {
    "data_understanding": {"skills", "schema", "sample"},
    "orchestrator": {"schema", "summary", "history", "question", "prior_context"},
    "sql": {"schema", "limit", "history", "question", "correction", "error_note"},
    "query": {"summary", "skill", "history", "slice", "question", "error_note",
              "guard_error", "judge_correction", "revision_note", "prior_context"},
    "judge": {"stage", "question", "output"},
    "visualization": {"insight", "columns", "error_note"},
    "session_summary": {"record"},
}
_VERSION = re.compile(r"v[0-9]+")


def _version_key(version: str) -> int:
    return int(version[1:])


class PromptSettings:
    def __init__(self, db: AppDB, prompts_dir=None):
        self.db = db
        self._dir = prompts_dir

    @property
    def dir(self):
        # Looked up at call time so tests can monkeypatch graph.prompts.PROMPTS_DIR.
        return self._dir if self._dir is not None else prompt_registry.PROMPTS_DIR

    def names(self) -> list[str]:
        return sorted(REQUIRED_VARS)

    def versions(self, name: str) -> list[str]:
        if name not in REQUIRED_VARS:
            return []
        found = [p.name[len(name) + 1:-4] for p in self.dir.glob(f"{name}.v*.txt")]
        return sorted((v for v in found if _VERSION.fullmatch(v)), key=_version_key)

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
        # Validate name/version shape before touching the filesystem (no path traversal).
        if name not in REQUIRED_VARS:
            raise AccountError("Unknown prompt.")
        if not isinstance(version, str) or not _VERSION.fullmatch(version):
            raise AccountError(f"Prompt version not found: {name}.{version}")
        with self.db.lock:
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
        good = {}
        for name, version in self.overrides().items():
            try:
                ok = name in REQUIRED_VARS and version in self.versions(name) and not self._missing(name, version)
            except OSError:
                ok = False
            if ok:
                good[name] = version
            else:
                log.warning("Skipping unusable prompt override %s.%s", name, version)
        prompt_registry.set_overrides(good, REQUIRED_VARS)


class AppSettings:
    ALLOWED = {"JUDGE_ENABLED": bool, "JUDGE_RETRIEVAL": bool, "JUDGE_MIN_SCORE": int}

    def __init__(self, db: AppDB):
        self.db = db

    @staticmethod
    def _valid(key: str, value) -> bool:
        kind = AppSettings.ALLOWED[key]
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            return False
        return not (key == "JUDGE_MIN_SCORE" and not 1 <= value <= 5)

    def get(self, key: str):
        row = self.db.one("SELECT value_json FROM app_settings WHERE key = ?", (key,))
        if row:
            try:
                value = json.loads(row["value_json"])
            except (ValueError, TypeError):
                value = None
            if key in self.ALLOWED and self._valid(key, value):
                return value
            log.warning("Ignoring invalid stored setting %s", key)
        return getattr(config, key)

    def set(self, key: str, value, user_id) -> None:
        kind = self.ALLOWED.get(key)
        if kind is None:
            raise AccountError("Unknown setting.")
        if not self._valid(key, value):
            if key == "JUDGE_MIN_SCORE" and isinstance(value, int) and not isinstance(value, bool):
                raise AccountError("JUDGE_MIN_SCORE must be between 1 and 5.")
            raise AccountError(f"{key} must be a {kind.__name__}.")
        with self.db.lock:
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
