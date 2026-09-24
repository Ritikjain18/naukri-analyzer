# Naukri Personal Data Analyzer (Phase 1)

Multi-agent HR analytics on Groq + LangGraph + Streamlit. Phase 1 runs on SQLite with no external services.

## Setup
    python3.13 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env        # then put your Groq key in .env
    python -m data.seed         # optional; the app seeds an empty DB itself
    streamlit run app.py

## Tests
    pytest                      # offline unit + graph tests
    GROQ_API_KEY=... pytest tests/test_live_smoke.py   # live eval, uses Groq tokens

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.
