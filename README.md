# Naukri Personal Data Analyzer

Multi-agent HR analytics on Groq + LangGraph + Streamlit. Phase 1 runs on SQLite with no external services.

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
- **Fallback chain and token limits:** the smart model (70B) falls back to the 8B model on rate limits or when its budget is spent. Per-model limits live in `config.py` (`MODEL_LIMITS`); verify them against your account in the Groq console. Usage is tracked in a local SQLite file.
- **Switches:** `JUDGE_ENABLED` and `JUDGE_RETRIEVAL` in `config.py`. Free-tier cost note: a question can use up to 4 smart-model calls; when the budget runs low the manager falls back to the 8B model.
- **Slide-deck export:** approved insights (or all, if none approved) are listed in the sidebar under "Slide deck"; pick some and click "Export slide deck" to download a .pptx with one insight per slide and native charts. Answers also show which models produced them and the judge scores.
- Together AI is not configured.
- The live tests (`pytest -m live`) now make about 7 Groq calls per question (orchestrate, judge x2, analyst, visualization, plus retries).

## Tests
    pytest                      # always offline (live tests are deselected)
    pytest -m live tests/test_live_smoke.py   # live eval, needs GROQ_API_KEY, uses Groq tokens

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.
