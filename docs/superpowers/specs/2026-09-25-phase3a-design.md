# Naukri Personal Data Analyzer — Phase 3a Design (accounts, audit, memory, admin)

Builds on the phase 1 and 2 specs (implemented; 312 offline tests). Source of intent: `Naukri_Architecture_Document.docx` sections 9, 12, 13. Models are now `openai/gpt-oss-20b` (fast) and `openai/gpt-oss-120b` (smart); see `config.py`.

## Goals
- Login with three roles (Analyst, Manager, Admin) and a permission matrix enforced in code, not only by hiding UI.
- An audit trail of who did what, with timestamps.
- Persistent insight history, per-user session summaries, and their injection as prior context in the next session.
- Admin pages: users, prompt versions, system config, audit view. Manager pages: history and comparative report.
- Zero new accounts or infrastructure: everything in SQLite and the standard library.

## Non-goals (deferred)
- Phase 3b: Tavily web search, LangSmith tracing/evals, MLflow. Phase 3c: PostgreSQL, MongoDB, pgvector, embeddings, FlashRank.
- SSO/OAuth, HTTPS, "remember me" cookies, password recovery by email, per-user data isolation of uploaded tables (uploads still write to the shared analytics DB), row-level security.
- Prompt A/B evaluation (needs LangSmith).

## Design decisions
1. **Own database file:** `data/app.db` (gitignored) holds accounts, audit, history, sessions, summaries and settings. It is separate from the analytics DB (`data/naukri.db`) so the model never sees these tables in the schema, and separate from `data/usage.db`.
2. **Standard library only** for hashing (`hashlib.scrypt`) and storage (`sqlite3` with a lock, `check_same_thread=False`, same pattern as `RateLimitManager`).
3. **Project memory file** is `memory/PROJECT_MEMORY.md` (gitignored), not `CLAUDE.md`: a repo-root `CLAUDE.md` is read by Claude Code as instructions, so writing application state there would mix the two and open an instruction-injection path.
4. **Navigation** is a sidebar radio (Analyze / History / Admin) rather than Streamlit multipage, so it stays testable with `AppTest` and keeps `app.py` the single entry point.
5. **Comparative report** = the same (normalised) question asked in different sessions, shown side by side with their findings, evidence and dates.

## Data model (`data/app.db`)
| Table | Columns |
|---|---|
| `users` | `id`, `username` (unique, case-insensitive), `password_hash`, `role` (`analyst`/`manager`/`admin`), `active`, `created_at`, `last_login`, `failed_attempts`, `locked_until` |
| `audit_log` | `id`, `ts_utc`, `user_id` (nullable for failed logins), `username`, `action`, `detail_json`, `session_id` |
| `sessions` | `id` (uuid), `user_id`, `started_at`, `last_activity_at`, `ended_at`, `question_count`, `summarised` |
| `insight_history` | `id`, `ts_utc`, `user_id`, `session_id`, `question`, `question_norm`, `insight_json`, `chart_json`, `slice_json` (<= 50 rows), `approved`, `degraded` |
| `session_summaries` | `id`, `user_id`, `session_id`, `ts_utc`, `summary_json` |
| `prompt_settings` | `name`, `active_version`, `updated_by`, `ts_utc` |
| `app_settings` | `key`, `value_json`, `updated_by`, `ts_utc` |

## Components (new package `accounts/`)
- `db.py` — `AppDB(path)`: creates the schema, exposes a locked connection helper and small typed query helpers. Bumps a `schema_version` pragma.
- `auth.py` — `hash_password`/`verify_password` (scrypt, per-user random salt, constant-time compare, parameters stored in the hash string), `create_user`, `authenticate(username, password, now) -> AuthResult` (five consecutive failures lock the account for 15 minutes; a locked account fails even with the right password until the lock expires; disabled accounts cannot log in; successful login resets the counter), `change_password`, `set_active`, `set_role`, `needs_bootstrap()` (true when no user exists). Passwords: minimum 8 characters. Usernames: 3-32 chars, `[A-Za-z0-9_.-]`. Error messages never reveal whether the username exists.
- `permissions.py` — `ROLES`, `PERMISSIONS` matrix and `can(role, permission) -> bool`, plus `require(role, permission)` that raises `PermissionDenied`. Permissions: `upload`, `ask`, `export`, `view_history`, `view_comparative`, `manage_users`, `manage_prompts`, `manage_config`, `view_audit`. Analyst: upload, ask, export. Manager: Analyst plus view_history, view_comparative. Admin: everything. The graph entry in the app calls `require(...)` before uploads/questions/exports/history/admin actions, so hiding a widget is never the only barrier.
- `audit.py` — `AuditLog.record(user, action, session_id, **detail)` and `query(filters, limit)`. Actions: `login`, `login_failed`, `logout`, `account_locked`, `user_created`, `user_updated`, `password_changed`, `upload`, `question`, `approve`, `revise`, `export`, `session_end`, `prompt_version_changed`, `config_changed`, `history_view`. Details never contain passwords, hashes, API keys or file contents (uploads log only file name, table, action and row count; questions log the question text, query type, generated SQL when available, models used, guard rejection/degraded flags and judge outcome).
- `history.py` — `InsightHistory.add(...)`, `list(filters)` (date range, user, text search), `by_question(question_norm)` for the comparative report, `mark_approved(id)`. `normalise_question(text)` lower-cases, strips punctuation and collapses whitespace.
- `memory.py` — `SessionTracker` (start/touch/end sessions), `summarise_session(llm, insights) -> dict` (one fast-model call; JSON keys `topics`, `key_findings`, `open_questions`, `data_loaded`; unparsable output falls back to a plain-text summary built from the questions; rate-limit errors skip the summary and leave the session unsummarised for a later retry), `prior_context(user_id, max_tokens=600) -> str` (last three summaries, clipped with `clip_to_tokens`, escaped of prompt-control text), `summarise_pending(user_id, llm)` (called at login for sessions with at least one question that were never ended and are older than 30 minutes), and `write_project_memory(path, ...)` which regenerates `memory/PROJECT_MEMORY.md` (tables loaded, active prompt versions, last 10 approved insights, the models in use). The file is written by an Admin button and after each session end.
- `settings.py` — `PromptSettings`: `list_versions(name)` (from `prompts/<name>.vN.txt`), `active_version(name, default)`, `set_active(name, version, user)` (refuses versions missing any variable listed in `REQUIRED_VARS[name]`, so an admin cannot select an incompatible prompt); `AppSettings`: `get`/`set` for `JUDGE_ENABLED`, `JUDGE_RETRIEVAL`, `JUDGE_MIN_SCORE` (1-5), applied to `config` at app start and immediately on save.
- `graph/prompts.py` — `render(name, version, ...)` applies an active-version override registry (`set_overrides({name: version})`) when the override file exists and is compatible, else uses the version the node asked for. Nodes keep passing their default version.

## Prompts
New versions (older kept): `orchestrator.v2.txt` (adds `$prior_context`) and `query.v3.txt` (adds `$prior_context`), used by default by the orchestrator and analyst nodes. When `prior_context` is empty the block renders as an empty line. New state key `prior_context` (persistent across turns; reset only at login/new session). `new_turn` does not reset it.

## UI
- `app.py`: gate → bootstrap form (no users) or login form; after login a sidebar shows username, role, an "End session" button, "Change password", and a radio of pages the role may see.
- **Analyze:** the existing chat, upload, deck export, approve/revise. Uploads, questions and exports call `require(...)` and write audit records; every answer is stored in `insight_history`; the session's `question_count` and `last_activity_at` are updated.
- **History (Manager, Admin):** filterable list, expandable details, "add to deck" selection, and the **Comparative report** tab (pick a question that was asked more than once; show the versions side by side).
- **Admin (Admin):** tabs Users (create, change role, disable/enable, reset password), Prompts (version picker per prompt with a preview of the selected file), Config (judge toggles), Audit (filters by user, action, date; CSV download), Memory (button to regenerate `PROJECT_MEMORY.md`).
- Logout and "End session" trigger `summarise_session` (errors are shown as a warning, never block logout).

## Errors and security
- Unauthorized attempts raise `PermissionDenied`, show a clear message and are audited as `denied:<permission>`.
- Rate-limit/model errors during summarisation are non-fatal.
- Admin cannot disable or demote the last active admin (enforced in `auth`).
- Audit and history are append-only from the application's point of view (no UI to edit or delete).
- History slices contain data from uploads; only Managers and Admins can read them. The README states this.
- Known limits: no HTTPS, login lost on browser refresh, single-machine SQLite, uploaded tables are shared by all users.

## Testing
- Unit: scrypt hash/verify (wrong password, tampered hash, salt uniqueness), username/password validation, lockout timing with an injected clock, disabled accounts, last-admin protection, permission matrix (every role x permission), audit records (fields present, no secrets), history filters and `normalise_question`, summariser with `FakeLLM` (good JSON, bad JSON fallback, rate-limit skip), `prior_context` clipping and escaping, prompt override compatibility (rejects an incompatible version), settings applied to `config`.
- Graph/app: `new_turn` keeps `prior_context`; orchestrator/analyst prompts contain the prior-context block; AppTest flows: bootstrap admin, login/logout, wrong password and lockout message, analyst cannot see History/Admin, manager can see History but not Admin, admin creates a user and that user logs in, question flow writes audit + history, approve/revise still work, unauthorized call path is refused and audited, comparative report shows two versions.
- No network, no real `data/*.db`: all tests use tmp paths.

## Success criteria
1. First launch with no users shows only the admin-creation form; afterwards the app requires login.
2. Analysts cannot reach History or Admin (UI and `require`), Managers can reach History but not Admin, Admins reach everything; refusals are audited.
3. Five wrong passwords lock the account for 15 minutes with a clear message; audit shows the failures.
4. Every upload, question, approve, revise, export and admin change appears in the audit page with user and time; no secrets in any record.
5. Answers persist in insight history; a Manager can filter it and build a deck from selected rows; the comparative report shows the same question across sessions.
6. Ending a session (or the next login after an abandoned one) stores a summary; the next session's prompts contain the prior-context block within the 600-token cap.
7. An Admin can switch a prompt to a compatible version and toggle judge settings; the change takes effect on the next question and is audited; incompatible versions are refused.
8. `memory/PROJECT_MEMORY.md` is generated and gitignored, and nothing writes to a repo-root `CLAUDE.md`.
9. All offline tests pass with 0 warnings; live tests stay opt-in.

## Risks
- Streamlit reruns and threads: `AppDB` uses one locked connection shared via `st.cache_resource`.
- `AppTest` cannot click `download_button`; CSV/deck downloads are verified through their builder functions.
- Prior-context text is model-generated from user data: it is escaped/clipped and placed in a clearly labelled block, but it remains a prompt-injection surface (mitigated by the existing input guard on questions, not by this block).
- Summaries cost one fast-model call each; at 8,000 tokens per minute they can be skipped under load and retried later.
