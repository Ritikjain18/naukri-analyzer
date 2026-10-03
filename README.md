# Naukri Personal Data Analyzer

Multi-agent HR analytics on Groq + LangGraph + Streamlit. Phases 1, 2 and 3a run on SQLite with no external services (only the Groq API).

> **Note:** Groq retired the Llama 3.1 8B / 3.3 70B models for this account; the models are now openai/gpt-oss-20b (fast) and openai/gpt-oss-120b (smart) — see config.py.

## Setup
    python3.13 -m venv .venv && source .venv/bin/activate
    pip install -r requirements-dev.txt   # requirements.txt is the slim runtime set used for deployment
    cp .env.example .env        # then put your Groq key in .env
    python -m data.seed         # optional; the app seeds an empty DB itself
    streamlit run app.py --server.address 127.0.0.1   # http://localhost:8501, this machine only

Always start it with `--server.address 127.0.0.1` locally. Without it Streamlit listens on every network interface, and because there is no HTTPS and the first run offers a "create admin" form, anyone who could reach the port before the first admin exists could take over the app. To allow LAN access deliberately, run `streamlit run app.py --server.address 0.0.0.0`, only on a trusted network or behind a reverse proxy with HTTPS, and only after the first admin exists.

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

Every action is checked in code, not only hidden in the UI; a denied attempt is written to the audit trail as `denied:<permission>`. Roles are re-read from the database on every run, so a disabled or demoted user loses access on their next interaction. The last active Admin cannot be demoted or disabled. Disable an account to revoke its sessions; resetting a password does not end existing sessions.

**Audit trail.** Recorded actions: `login`, `login_failed`, `account_locked`, `logout`, `session_end`, `session_revoked`, `user_created`, `user_updated`, `password_changed`, `upload`, `question`, `revise`, `approve`, `export`, `prompt_version_changed`, `config_changed`, `history_view` and `denied:*`. Writing project memory is audited as `config_changed` with `key=project_memory`; every export (a deck from Analyze, a deck from History, the audit CSV) is recorded as `export` with a `source`. Not logged: passwords, password hashes, API keys and uploaded file contents (an upload records the file name, table and row count only); secret-looking strings pasted into free text are redacted before storage. The Admin audit page filters by user, action and date and offers a CSV download that is safe to open in a spreadsheet (cells that could be read as formulas are escaped).

**History and comparative report (Manager/Admin).** Every answer is stored as an insight with its question, finding, approval and degraded flags. The History page lists and filters them, builds a slide deck from selected history entries, and has a comparative report tab: for a question asked in more than one session, it shows the answers side by side (date, user, session, finding, recommendation) so you can see how findings changed. Opening the page is audited as `history_view`.

**Session summaries and prior context.** A short summary of each session is written when the user clicks End session or logs out, or, for sessions that were abandoned (browser closed), at that user's next login. Summaries are stored redacted. The most recent 3 summaries (`PRIOR_SESSIONS`) are injected into the orchestrator and analyst prompts as "Prior sessions" background context, capped at about 600 tokens (`PRIOR_CONTEXT_TOKENS`). If the model is rate limited the summary is deferred rather than lost.

**Admin pages.** *Prompt versions:* pick the active version of each prompt; only versions compatible with the placeholders the code supplies can be selected, and "Use default" reverts. *Config:* the judge switches (`JUDGE_ENABLED`, `JUDGE_RETRIEVAL`) and minimum judge score (`JUDGE_MIN_SCORE`). Changes apply process-wide immediately (to all users, in the running app) and are audited as `prompt_version_changed` / `config_changed`.

**Project memory.** It is regenerated by an Admin on demand (the Admin page button), not automatically after each session, so it can be stale between regenerations. The button writes `memory/PROJECT_MEMORY.md` (data sources, active prompt versions, up to 10 recent approved insights, models). The `memory/` directory is gitignored. It is deliberately not `CLAUDE.md`: that file is read by coding assistants as instructions, and this file contains generated, data-derived text; the app refuses to write to any file with that name.

**Storage.** Accounts, audit, history and summaries live in `data/app.db` (SQLite, created on first run); model usage stays in `data/usage.db`. Both are gitignored (`data/*.db`).

**Limits.**
- Network exposure: run locally with `--server.address 127.0.0.1`; passing `--server.address 0.0.0.0` exposes an app with no HTTPS to the network.
- There is no HTTPS: run it on localhost or a trusted network and do not expose it to the internet.
- Login is held in the browser session and is lost on refresh; log in again.
- Uploaded tables are shared by all users.
- History slices hold data taken from uploads; only Managers and Admins can read them.
- Concurrency safety is per process; run one instance against a given `data/app.db`.

**Forgotten admin password.** Stop the app, then run this from the project root. The new password is typed at a hidden prompt (`read -rs`), so it never appears on the command line, in shell history or in this file (the `export` only puts it in that shell's environment; close the terminal afterwards). Replace `admin_username`:

    read -rs NEW_PW && export NEW_PW
    .venv/bin/python - <<'PY'
    import os, sys, config
    from accounts.db import AppDB
    from accounts.auth import AuthService
    a = AuthService(AppDB(config.APP_DB_PATH))
    u = a.get_by_username("admin_username")
    if u is None:
        print("No such user"); sys.exit(1)
    a.reset_password(u.id, os.environ["NEW_PW"])
    PY
    unset NEW_PW

This calls `AuthService.reset_password`, which also clears any lockout. It does not re-enable a disabled account, and it is not recorded in the audit log. Start the app and log in with the new password (then change it from the sidebar). An Admin who can still log in can reset any user's password from the Admin page instead.

## Tests
    pytest                      # always offline (live tests are deselected); uses temp paths only, never data/app.db or memory/
    pytest -m live tests/test_live_smoke.py   # live eval, needs GROQ_API_KEY, uses Groq tokens

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.

The SQL prompt (v3) forces real division for rates, ratios and percentages (`CAST(... AS REAL)` with `NULLIF`), avoiding SQLite's integer truncation.

## Deploying on Streamlit Community Cloud (free)

1. Push this folder to a GitHub repo (private is fine; `.env` and `data/*.db` are gitignored).
2. On share.streamlit.io choose "Create app", pick the repo, branch `main`, main file `app.py`, and Python 3.13 under advanced settings.
3. Under Advanced settings > Secrets, add:

       GROQ_API_KEY = "gsk_..."
       BOOTSTRAP_CODE = "a long random string"

4. Deploy, open the link, and create the admin immediately using the setup code. `BOOTSTRAP_CODE` stops a stranger who reaches a freshly restarted app from claiming the admin account.
5. Set the app to private (Share) and invite the viewers by email, or keep it public if you accept that anyone can reach the login page.
6. Updates: `git push` to `main` redeploys automatically.

Caveats: Community Cloud's disk is ephemeral. Accounts, history, summaries, uploads and the audit log are wiped whenever the app restarts or redeploys, and you will recreate the admin and users. Everyone shares your Groq key and its 8,000 tokens-per-minute limit. Guardrails AI is optional and not installed there (the pure checks run instead).
