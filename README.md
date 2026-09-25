# Naukri Personal Data Analyzer

Multi-agent HR analytics on Groq + LangGraph + Streamlit. Phases 1, 2 and 3a run on SQLite with no external services (only the Groq API).

> **Note:** Groq retired the Llama 3.1 8B / 3.3 70B models for this account; the models are now openai/gpt-oss-20b (fast) and openai/gpt-oss-120b (smart) — see config.py.

## Setup
    python3.13 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then put your Groq key in .env
    python -m data.seed         # optional; the app seeds an empty DB itself
    streamlit run app.py

The app is a local, single-machine app with user accounts (see Phase 3a below): all users share one local SQLite database, so uploaded tables are visible to everyone who can log in.

**Security:** Questions about uploaded files run as read-only SQL; the pandas expression tool is disabled because it evaluates model-written code.

## Phase 2
- **Guardrails:** questions are checked before running (relevance, prompt injection, known columns, coherence) and answers after (non-empty, no placeholders, numbers traceable to the data). A failed answer is self-corrected; after 3 failures the app shows a degraded answer (a warning plus the raw data). Guardrails AI is used when installed, otherwise a pure-Python fallback.
- **Judge and revise/approve:** an LLM judge scores retrieval and analyst output (relevance, specificity, actionability) and can reject and re-dispatch (max 2 times). Each answer has Approve and Revise buttons; Revise re-enters at the analyst step, reusing the previous data slice.
- **Fallback chain and token limits:** the smart model (`openai/gpt-oss-120b`) falls back to the fast model (`openai/gpt-oss-20b`) on rate limits or when its budget is spent. Per-model limits live in `config.py` (`MODEL_LIMITS`): measured from Groq response headers on this account, both models have 8,000 tokens/minute and 1,000 requests/day (the two may share one rate-limit bucket, which is not modelled); the daily token limit (200,000 each) is an assumption, so verify it in the Groq console. Both models are reasoning models: `REASONING_EFFORT` (`low`) in `config.py` keeps hidden reasoning from burning the small per-minute budget, and `OUTPUT_RESERVE` (1,500 tokens) accounts for reasoning tokens as output. If a configured model is retired the app says: "The configured Groq model isn't available for this account. Check MODEL_SMART / MODEL_FAST in config.py." Usage is tracked in a local SQLite file.
- **Switches:** `JUDGE_ENABLED` and `JUDGE_RETRIEVAL` in `config.py`. Free-tier cost note: a question makes about 7 model calls on the happy path (orchestrate, 2 judges, analyst, visualization, plus ingest on first load) and about 27 in the worst retry case; when the smart model's budget runs low the manager falls back to the fast model. The analyst slice adapts to the smart model's remaining per-minute capacity (headroom minus tokens already used this minute), and a rate-limited retry keeps the previous answer instead of failing.
- **Slide-deck export:** approved insights (or all, if none approved) are listed in the sidebar under "Slide deck"; pick some and click "Export slide deck" to download a .pptx with one insight per slide and native charts. Answers also show which models produced them and the judge scores.
- Together AI is not configured.
- The live tests (`pytest -m live`) now make about 7 Groq calls per question (orchestrate, judge x2, analyst, visualization, plus retries). They build the app's fallback chain, pace 10 s between questions, wait 65 s and retry a question once on a rate limit, and set `LIVE_REPORT_PATH` to append a JSONL report (one line per question).

**tiktoken note:** token counting uses tiktoken, which downloads its ~1.7 MB `cl100k_base` data file on first use (internet needed once; macOS may clear the cache). Offline, it falls back to a character-based estimate.

## Phase 3a: accounts, audit, memory

**First launch.** When no users exist, the app shows a bootstrap form instead of the login page; the account you create there is the first Admin. The form never appears again once any user exists. Admins create further accounts on the Admin page.

**Login.** Usernames are 3-32 characters (letters, digits, `_`, `.`, `-`); passwords must be at least 8 characters and are stored only as scrypt hashes. A wrong username, a wrong password and a disabled account all show the same generic "Invalid username or password." message. After 5 failed attempts an account is locked for 15 minutes (`LOCKOUT_ATTEMPTS`, `LOCKOUT_MINUTES` in `config.py`). Users can change their own password from the sidebar.

**Roles.**

| Role | Can do |
|---|---|
| Analyst | upload files, ask questions (and revise/approve), export slide decks |
| Manager | everything an Analyst can, plus the History page and the comparative report |
| Admin | everything a Manager can, plus manage users, prompt versions, judge/config settings, the audit log and project memory |

Every action is checked in code, not only hidden in the UI; a denied attempt is written to the audit trail as `denied:<permission>`. Roles are re-read from the database on every run, so a disabled or demoted user loses access on their next interaction. The last active Admin cannot be demoted or disabled.

**Audit trail.** Recorded actions: `login`, `login_failed`, `account_locked`, `logout`, `session_end`, `session_revoked`, `user_created`, `user_updated`, `password_changed`, `upload`, `question`, `revise`, `approve`, `export`, `prompt_version_changed`, `config_changed`, `history_view` and `denied:*`. Not logged: passwords, password hashes, API keys and uploaded file contents (an upload records the file name, table and row count only); secret-looking strings pasted into free text are redacted before storage. The Admin audit page filters by user, action and date and offers a CSV download that is safe to open in a spreadsheet (cells that could be read as formulas are escaped).

**History and comparative report (Manager/Admin).** Every answer is stored as an insight with its question, finding, approval and degraded flags. The History page lists and filters them, builds a slide deck from selected history entries, and has a comparative report tab: for a question asked in more than one session, it shows the answers side by side (date, user, session, finding, recommendation) so you can see how findings changed. Opening the page is audited as `history_view`.

**Session summaries and prior context.** A short summary of each session is written when the user clicks End session or logs out, or, for sessions that were abandoned (browser closed), at that user's next login. Summaries are stored redacted. The most recent 3 summaries (`PRIOR_SESSIONS`) are injected into the orchestrator and analyst prompts as "Prior sessions" background context, capped at about 600 tokens (`PRIOR_CONTEXT_TOKENS`). If the model is rate limited the summary is deferred rather than lost.

**Admin pages.** *Prompt versions:* pick the active version of each prompt; only versions compatible with the placeholders the code supplies can be selected, and "Use default" reverts. *Config:* the judge switches (`JUDGE_ENABLED`, `JUDGE_RETRIEVAL`) and minimum judge score (`JUDGE_MIN_SCORE`). Changes apply process-wide immediately (to all users, in the running app) and are audited as `prompt_version_changed` / `config_changed`.

**Project memory.** An Admin button writes `memory/PROJECT_MEMORY.md` (data sources, active prompt versions, up to 10 recent approved insights, models). The `memory/` directory is gitignored. It is deliberately not `CLAUDE.md`: that file is read by coding assistants as instructions, and this file contains generated, data-derived text; the app refuses to write to any file with that name.

**Storage.** Accounts, audit, history and summaries live in `data/app.db` (SQLite, created on first run); model usage stays in `data/usage.db`. Both are gitignored (`data/*.db`).

**Limits.**
- There is no HTTPS: run it on localhost or a trusted network and do not expose it to the internet.
- Login is held in the browser session and is lost on refresh; log in again.
- Uploaded tables are shared by all users.
- History slices hold data taken from uploads; only Managers and Admins can read them.
- Concurrency safety is per process; run one instance against a given `data/app.db`.

**Forgotten admin password.** Stop the app, then run this from the project root (the new password is read from the environment, so it is not stored in shell history as an argument or in this file; replace `CHOOSE_A_NEW_PASSWORD` and `admin_username`):

    NEW_PW='CHOOSE_A_NEW_PASSWORD' .venv/bin/python -c "import os, config; from accounts.db import AppDB; from accounts.auth import AuthService; a = AuthService(AppDB(config.APP_DB_PATH)); u = a.get_by_username('admin_username'); a.reset_password(u.id, os.environ['NEW_PW'])"

This also clears any lockout. Start the app and log in with the new password (then change it from the sidebar).

## Tests
    pytest                      # always offline (live tests are deselected); uses temp paths only, never data/app.db or memory/
    pytest -m live tests/test_live_smoke.py   # live eval, needs GROQ_API_KEY, uses Groq tokens

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.

The SQL prompt (v3) forces real division for rates, ratios and percentages (`CAST(... AS REAL)` with `NULLIF`), avoiding SQLite's integer truncation.
