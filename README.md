# Naukri Personal Data Analyzer (Phase 1)

Multi-agent HR analytics on Groq + LangGraph + Streamlit. Phase 1 runs on SQLite with no external services.

## Setup
    python3.13 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then put your Groq key in .env
    python -m data.seed         # optional; the app seeds an empty DB itself
    streamlit run app.py

The app assumes a single local user: uploads write to the shared local SQLite database.

**Security:** Questions about uploaded files run as read-only SQL; the pandas expression tool is disabled because it evaluates model-written code.

## Tests
    pytest                      # always offline (live tests are deselected)
    pytest -m live tests/test_live_smoke.py   # live eval, needs GROQ_API_KEY, uses Groq tokens

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.
