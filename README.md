# Naukri Personal Data Analyzer

Multi-agent HR analytics on Groq + LangGraph + Streamlit. Phases 1 + 2 run on SQLite with no external services (only the Groq API).

> **Note:** Groq retired the Llama 3.1 8B / 3.3 70B models for this account; the models are now openai/gpt-oss-20b (fast) and openai/gpt-oss-120b (smart) — see config.py.

## Setup
    python3.13 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then put your Groq key in .env
    python -m data.seed         # optional; the app seeds an empty DB itself
    streamlit run app.py

The app assumes a single local user: uploads write to the shared local SQLite database.

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

## Tests
    pytest                      # always offline (live tests are deselected)
    pytest -m live tests/test_live_smoke.py   # live eval, needs GROQ_API_KEY, uses Groq tokens

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.

The SQL prompt (v3) forces real division for rates, ratios and percentages (`CAST(... AS REAL)` with `NULLIF`), avoiding SQLite's integer truncation.
