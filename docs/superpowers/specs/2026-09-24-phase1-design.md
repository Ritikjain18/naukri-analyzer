# Naukri Personal Data Analyzer — Phase 1 Design

Source: `Naukri_Architecture_Document.docx`. Phase 1 is a thin vertical slice: upload HR data, ask a question, get an insight and an inline chart.

## Goals
- Prove the core loop end to end: upload → retrieve → analyse → chart.
- Follow the doc's architecture (shared-state LangGraph, spoke agents, prompt-first) so later phases add nodes rather than rewrite.
- Run with zero infrastructure: no PostgreSQL, MongoDB, Docker or GPU. Only a Groq API key.

## Non-goals (deferred)
- **Phase 2:** Judge Agent, Guardrails AI (input/output, self-correction loop), .pptx export, Tiktoken budgeting, Groq→Together fallback chain.
- **Phase 3:** Tavily web RAG, embeddings + pgvector + FlashRank, real PostgreSQL and MongoDB, MLflow, LangSmith, auth/roles/audit trail, cross-session memory (`CLAUDE.md` read/write).

## Storage substitutions
| Doc | Phase 1 | Later |
|---|---|---|
| PostgreSQL + SQLAlchemy | SQLite via SQLAlchemy | Change connection URL / add `PostgresStore` |
| MongoDB (`traffic_logs`, `growth_metrics`) | Flattened SQLite tables | PyMongo-backed store behind same interface |
| pgvector | Not used | Added with embeddings in phase 3 |

## Layout
```
naukri_analyzer/
  app.py                 # Streamlit UI
  graph/                 # state.py, nodes/, build_graph.py, router.py
  data/                  # store.py (DataStore, SQLiteStore), seed.py
  prompts/               # data_understanding.v1.txt, query.v1.txt, visualization.v1.txt
  skills/                # job-posting/, hiring-funnel/, traffic/, slide-gen/ (SKILL.md each)
  tests/
  .env.example           # GROQ_API_KEY=   (real key lives in .env, gitignored)
  requirements.txt
```
Use a Python 3.13 virtualenv (system Python is 3.14; ML/LangChain wheels may lag).

## Shared state
`TypedDict` fields: `df`, `schema`, `data_summary`, `query_type` (`sql`|`pandas`), `question`, `chat_history` (last 6 turns), `data_slice`, `insight`, `chart_config`, `errors`, `new_upload` (bool). Nodes return partial updates and never call each other.

## Graph
Nodes: `ingest`, `retrieve`, `analyst`, `output`. Conditional entry: `ingest` runs when the session has no `data_summary` yet (first question), otherwise straight to `retrieve`. File uploads call the ingest node directly from the UI, outside the graph. Then `retrieve → analyst → output → END`. Follow-ups re-enter at `retrieve`. Phase 2 inserts Judge nodes around these edges.

## Agents
| Agent | Model | Job |
|---|---|---|
| Ingestion | Llama 3.1 8B (`llama-3.1-8b-instant`) | Read SKILL.md files; pandas/openpyxl parse; append to matching table or create one (types inferred); Data Understanding prompt → `data_summary`, schema to state. |
| Retrieval | Llama 3.1 8B | Router function picks tool. SQL tool: LLM writes one SELECT, run on a read-only connection, one retry on error (custom chain so a DataFrame is returned). Pandas tool: LLM writes one pandas expression, evaluated in a restricted namespace. Slice capped at 200 rows (query aggregates/sorts). |
| Analyst | Llama 3.3 70B (`llama-3.3-70b-versatile`) | Query prompt with data summary, slice, history and matching SKILL.md → structured insight: finding, evidence, recommendation. |
| Output | Llama 3.1 8B | Visualization prompt → chart JSON (`type`, `x`, `y`, `title`) validated with pydantic; one retry, then table fallback. Renders via Plotly. Pins insight to `session_state`. |

Router (plain function): Pandas tool if a file was uploaded this session and the question references its columns; otherwise SQL tool. The Judge replaces it in phase 2.

## Data layer
`DataStore` interface: `list_tables()`, `get_schema()`, `append_or_create(df, table)`, `run_sql(query)`. `SQLiteStore` implements it via SQLAlchemy. `seed.py` uses seeded Faker to fill `job_postings`, `hiring_funnel`, `traffic_metrics`, `recruiter_activity`, `candidate_profiles` (columns per the doc) with roughly 500–5,000 rows each.

Excel flow: pandas/openpyxl → DataFrame → table exists? append : create with inferred types → DataFrame into `session_state` → summary generated.

## Prompts and skills
Prompts are versioned files loaded by name + version so LangSmith hub can replace them later. The Streamlit prompt expander shows the exact final prompt sent to Groq. Four SKILL.md files hold metrics, columns, formulas and slide rules per the doc's section 12 (slide-gen is written now, used in phase 2).

## UI
Streamlit: upload panel (Excel, CSV, JSON), chat panel with history, inline `st.plotly_chart`, prompt expander. No .pptx button yet.

## Errors
Groq failures (rate limit, timeout, auth) show a clear user message; no fallback chain. Invalid SQL is retried once by the SQL agent. Missing `GROQ_API_KEY` shows a setup message on startup.

## Security
- API key only in `.env` (gitignored), loaded at runtime; never logged or shown in the prompt expander.
- SQL agent uses a read-only connection; uploaded-file table names are sanitised.

## Testing
- Unit: `DataStore`, router, chart-config validation, prompt loading.
- Graph tests with a stubbed LLM.
- One live Groq smoke test, skipped if no key.
- 5–8 eval questions with expected answer shape (grows to the doc's 15 later).

## Success criteria
1. `python seed.py` then `streamlit run app.py` starts with no external services.
2. Seeded-data questions (e.g. "conversion by category") return an insight plus a chart.
3. Uploading an Excel file appends to or creates a table, and questions about it work.
4. Follow-up questions use chat history without re-ingesting.
5. Tests pass; swapping the store is confined to `data/`.
