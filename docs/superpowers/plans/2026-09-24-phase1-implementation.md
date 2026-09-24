# Naukri Personal Data Analyzer — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit app where a user uploads HR data (or uses seeded data), asks a question, and gets a structured insight plus an inline Plotly chart, driven by a 4-node LangGraph pipeline on Groq.

**Architecture:** Shared-state LangGraph `ingest → retrieve → analyst → output`. Storage is SQLite behind a `DataStore`-style class so Postgres/Mongo can replace it later. LLM calls go through injected objects with `.invoke(prompt).content`, so every node is testable with a scripted fake.

**Tech Stack:** Python 3.13, Streamlit, LangGraph, langchain-groq, pandas, openpyxl, SQLAlchemy (SQLite), Faker, Plotly, pydantic v2, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-phase1-design.md` (read it first).

## Global Constraints
- Models: fast = `llama-3.1-8b-instant`, smart = `llama-3.3-70b-versatile`. No fallback chain in phase 1.
- Use a **Python 3.13** virtualenv (system Python is 3.14).
- No PostgreSQL, MongoDB, Docker, pgvector, embeddings, Judge, Guardrails AI, .pptx, MLflow, LangSmith or auth in phase 1.
- `GROQ_API_KEY` only in `.env` (gitignored). Never log it, never render it in the prompt expander, never commit it.
- SQL execution uses a read-only connection. Retrieval slices are capped at 200 rows. Chat history keeps the last 6 turns.
- All work happens inside `/Users/ritikjain/Downloads/python_scripts/naukri_analyzer/`. Run every command from that directory with the venv active.

## Deviations from the spec (decided during planning)
1. **Retrieval agents are LangChain-LLM-driven chains, not `create_sql_agent` / `create_pandas_dataframe_agent`.** Those return text; the pipeline needs a DataFrame slice. The SQL tool prompts the fast LLM for one SELECT, runs it read-only, and retries once on error. The Pandas tool prompts for one pandas expression and evaluates it in a restricted namespace.
2. **Ingest also runs on the first question of a session** (when `data_summary` is empty) to summarise the seeded DB. File uploads call the ingest node directly from the UI, not through the graph.
3. Two extra prompt files, `sql.v1.txt` and `pandas.v1.txt`, keep all prompts in files.

## File Structure
```
naukri_analyzer/
  app.py                     # Streamlit UI (Task 12)
  config.py                  # paths, model names, limits, get_api_key (Task 1)
  pytest.ini  requirements.txt  .gitignore  .env.example  README.md
  data/
    __init__.py
    store.py                 # SQLiteStore, sanitize_name, SchemaMismatchError (Task 2)
    seed.py                  # Faker seeding + CLI (Task 3)
    parsing.py               # parse_upload (Task 7)
  graph/
    __init__.py
    prompts.py               # load_prompt, render (Task 4)
    skills.py                # load_skill, detect_domain, load_domain_skills (Task 4)
    llm.py                   # get_llm, friendly_error (Task 5)
    models.py                # Insight, ChartConfig (Task 6)
    state.py                 # AnalyzerState (Task 6)
    parsing.py               # extract_code, extract_json (Task 6)
    router.py                # choose_tool (Task 6)
    tools.py                 # make_sql_tool, make_pandas_tool (Task 8)
    viz.py                   # validate_chart, build_figure (Task 10)
    build_graph.py           # build_graph, route_entry (Task 11)
    nodes/
      __init__.py
      ingest.py  retrieve.py  analyst.py  output.py
  prompts/                   # data_understanding.v1.txt query.v1.txt visualization.v1.txt sql.v1.txt pandas.v1.txt
  skills/                    # job-posting/ hiring-funnel/ traffic/ slide-gen/  (SKILL.md each)
  tests/
    __init__.py  conftest.py  fakes.py  eval_questions.json  test_*.py
```

---

### Task 1: Project scaffold, config, git

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `.gitignore`, `.env.example`, `config.py`, `data/__init__.py`, `graph/__init__.py`, `graph/nodes/__init__.py`, `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.ROOT: Path`, `config.DB_PATH: Path`, `config.MODEL_FAST: str`, `config.MODEL_SMART: str`, `config.ROW_CAP: int = 200`, `config.HISTORY_TURNS: int = 6`, `config.MissingKeyError`, `config.get_api_key() -> str`.

- [ ] **Step 1: Create the folder, git repo and venv**

```bash
cd /Users/ritikjain/Downloads/python_scripts/naukri_analyzer
git init
python3.13 --version || brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
```
Expected: `Python 3.13.x`. The spec and plan files already in `docs/` are untracked until the first commit.

- [ ] **Step 2: Write scaffold files**

`requirements.txt`:
```
streamlit>=1.40
pandas>=2.2
openpyxl>=3.1
sqlalchemy>=2.0
faker>=30
plotly>=5.24
langgraph>=0.2
langchain-core>=0.3
langchain-groq>=0.2
pydantic>=2.7
python-dotenv>=1.0
pytest>=8
```
`pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
```
`.gitignore`:
```
.env
.venv/
__pycache__/
.pytest_cache/
data/*.db
.DS_Store
```
`.env.example`:
```
GROQ_API_KEY=
```
Create empty files `data/__init__.py`, `graph/__init__.py`, `graph/nodes/__init__.py`, `tests/__init__.py`.

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: installs cleanly. If a package has no wheel for your Python, stop and report which one.

- [ ] **Step 4: Write the failing test**

`tests/test_config.py`:
```python
import pytest

import config


def test_get_api_key_missing_raises(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(config.MissingKeyError) as exc:
        config.get_api_key()
    assert ".env" in str(exc.value)


def test_get_api_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", " gsk_test ")
    assert config.get_api_key() == "gsk_test"


def test_constants():
    assert config.MODEL_FAST == "llama-3.1-8b-instant"
    assert config.MODEL_SMART == "llama-3.3-70b-versatile"
    assert config.ROW_CAP == 200
    assert config.HISTORY_TURNS == 6
```

- [ ] **Step 5: Run to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'config'`).

- [ ] **Step 6: Implement `config.py`**

```python
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DB_PATH = ROOT / "data" / "naukri.db"
MODEL_FAST = "llama-3.1-8b-instant"
MODEL_SMART = "llama-3.3-70b-versatile"
ROW_CAP = 200
HISTORY_TURNS = 6


class MissingKeyError(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise MissingKeyError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
```

- [ ] **Step 7: Run to verify pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add .gitignore .env.example requirements.txt pytest.ini config.py data graph tests docs
git commit -m "chore: scaffold project, config and docs"
```

---

### Task 2: SQLiteStore

**Files:**
- Create: `data/store.py`, `tests/conftest.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `sanitize_name(name: str, prefix: str = "t_") -> str`
  - `class SchemaMismatchError(ValueError)`
  - `class SQLiteStore(path)` with: `list_tables() -> list[str]`, `get_schema() -> dict[str, list[tuple[str, str]]]`, `schema_text() -> str` (one line per table: `table(col TYPE, col TYPE)`), `replace_table(df, table) -> None`, `append_or_create(df, table) -> str` (returns `"appended"` or `"created"`; sanitises table and column names; raises `SchemaMismatchError` if the table exists and column sets differ), `run_sql(query: str) -> pd.DataFrame` (read-only).
  - pytest fixture `store` (tmp-path `SQLiteStore`).

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:
```python
import pytest

from data.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "test.db")
```
`tests/test_store.py`:
```python
import pandas as pd
import pytest

from data.store import SchemaMismatchError, sanitize_name


def test_sanitize_name():
    assert sanitize_name("Job Postings (Q1)") == "job_postings_q1"
    assert sanitize_name("2024 data") == "t_2024_data"
    assert sanitize_name("!!!") == "unnamed"


def test_create_then_append(store):
    df = pd.DataFrame({"Job ID": [1, 2], "Views": [10, 20]})
    assert store.append_or_create(df, "My Jobs") == "created"
    assert store.list_tables() == ["my_jobs"]
    assert store.append_or_create(df, "My Jobs") == "appended"
    assert len(store.run_sql("SELECT * FROM my_jobs")) == 4
    assert [c for c, _ in store.get_schema()["my_jobs"]] == ["job_id", "views"]


def test_schema_mismatch_raises(store):
    store.append_or_create(pd.DataFrame({"a": [1]}), "t")
    with pytest.raises(SchemaMismatchError):
        store.append_or_create(pd.DataFrame({"b": [1]}), "t")


def test_replace_table(store):
    store.replace_table(pd.DataFrame({"a": [1, 2]}), "t")
    store.replace_table(pd.DataFrame({"a": [9]}), "t")
    assert len(store.run_sql("SELECT * FROM t")) == 1


def test_run_sql_is_read_only(store):
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    with pytest.raises(Exception):
        store.run_sql("DROP TABLE t")
    assert store.list_tables() == ["t"]


def test_schema_text(store):
    store.replace_table(pd.DataFrame({"a": [1], "b": ["x"]}), "t")
    text = store.schema_text()
    assert text.startswith("t(a ") and "b " in text
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_store.py -v`
Expected: FAIL (`ModuleNotFoundError: data.store`).

- [ ] **Step 3: Implement `data/store.py`**

```python
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, event, inspect


def sanitize_name(name: str, prefix: str = "t_") -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    if not s:
        return "unnamed"
    if s[0].isdigit():
        s = prefix + s
    return s


class SchemaMismatchError(ValueError):
    pass


class SQLiteStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{self.path}"
        self._engine = create_engine(url)
        self._ro_engine = create_engine(url)

        @event.listens_for(self._ro_engine, "connect")
        def _set_read_only(dbapi_conn, _record):
            dbapi_conn.execute("PRAGMA query_only = ON")

    def list_tables(self) -> list[str]:
        return sorted(inspect(self._engine).get_table_names())

    def get_schema(self) -> dict[str, list[tuple[str, str]]]:
        insp = inspect(self._engine)
        return {
            t: [(c["name"], str(c["type"])) for c in insp.get_columns(t)]
            for t in sorted(insp.get_table_names())
        }

    def schema_text(self) -> str:
        return "\n".join(
            f"{t}({', '.join(f'{c} {ty}' for c, ty in cols)})"
            for t, cols in self.get_schema().items()
        )

    def replace_table(self, df: pd.DataFrame, table: str) -> None:
        df.to_sql(sanitize_name(table), self._engine, if_exists="replace", index=False)

    def append_or_create(self, df: pd.DataFrame, table: str) -> str:
        table = sanitize_name(table)
        df = df.copy()
        df.columns = [sanitize_name(c) for c in df.columns]
        if table in self.list_tables():
            existing = {c for c, _ in self.get_schema()[table]}
            if set(df.columns) != existing:
                raise SchemaMismatchError(
                    f"Table '{table}' exists with columns {sorted(existing)}, "
                    f"but the upload has {sorted(df.columns)}."
                )
            df.to_sql(table, self._engine, if_exists="append", index=False)
            return "appended"
        df.to_sql(table, self._engine, if_exists="fail", index=False)
        return "created"

    def run_sql(self, query: str) -> pd.DataFrame:
        with self._ro_engine.connect() as conn:
            return pd.read_sql_query(query, conn)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_store.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add data/store.py tests/conftest.py tests/test_store.py
git commit -m "feat: add SQLiteStore with read-only SQL and append-or-create"
```

---

### Task 3: Faker seed data

**Files:**
- Create: `data/seed.py`
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: `SQLiteStore.replace_table`, `config.DB_PATH`.
- Produces: `seed_database(store, seed: int = 42, scale: float = 1.0) -> dict[str, int]` (table → row count), and a CLI: `python -m data.seed`. Constants `STAGES = ["applied","screened","interviewed","offered","hired"]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_seed.py`:
```python
from data.seed import STAGES, seed_database

DOC_COLUMNS = {
    "job_postings": {"job_id", "title", "category", "location", "date_posted", "views", "applications", "status"},
    "hiring_funnel": {"candidate_id", "job_id", "stage", "date", "drop_off_flag"},
    "traffic_metrics": {"date", "source", "visits", "session_duration", "bounce_rate"},
    "recruiter_activity": {"recruiter_id", "job_id", "actions", "response_rate"},
    "candidate_profiles": {"candidate_id", "skills", "experience_years", "education"},
}


def test_seed_creates_doc_tables(store):
    counts = seed_database(store, scale=0.05)
    assert set(counts) == set(DOC_COLUMNS)
    for table, cols in DOC_COLUMNS.items():
        assert {c for c, _ in store.get_schema()[table]} == cols
        assert counts[table] > 0


def test_seed_is_reproducible(store, tmp_path):
    from data.store import SQLiteStore

    seed_database(store, seed=7, scale=0.05)
    other = SQLiteStore(tmp_path / "other.db")
    seed_database(other, seed=7, scale=0.05)
    q = "SELECT * FROM job_postings ORDER BY job_id"
    assert store.run_sql(q).equals(other.run_sql(q))


def test_funnel_stages_are_ordered_prefix(store):
    seed_database(store, scale=0.05)
    df = store.run_sql("SELECT * FROM hiring_funnel ORDER BY candidate_id, date")
    for _, group in df.groupby("candidate_id"):
        stages = list(group["stage"])
        assert stages == STAGES[: len(stages)]
        assert list(group["drop_off_flag"])[:-1] == [0] * (len(stages) - 1)


def test_applications_never_exceed_views(store):
    seed_database(store, scale=0.05)
    bad = store.run_sql("SELECT COUNT(*) AS n FROM job_postings WHERE applications > views")
    assert bad["n"][0] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_seed.py -v`
Expected: FAIL (`ModuleNotFoundError: data.seed`).

- [ ] **Step 3: Implement `data/seed.py`**

```python
import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker

import config
from data.store import SQLiteStore

CATEGORIES = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "Design", "HR", "Customer Support"]
LOCATIONS = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Remote"]
SOURCES = ["organic_search", "direct", "referral", "social", "email", "paid_ads"]
STAGES = ["applied", "screened", "interviewed", "offered", "hired"]
DROP_PROB = [0.35, 0.40, 0.50, 0.20]  # drop chance after each non-final stage
SKILLS = ["Python", "SQL", "Excel", "Java", "Sales", "Communication", "Figma", "Accounting", "React", "Recruiting"]
EDUCATION = ["B.Tech", "B.Com", "MBA", "B.A.", "M.Tech", "BBA"]
END = date(2026, 9, 1)  # fixed so seeded data is reproducible


def seed_database(store: SQLiteStore, seed: int = 42, scale: float = 1.0) -> dict[str, int]:
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    def n(base: int) -> int:
        return max(10, int(base * scale))

    n_jobs, n_cand, n_rec = n(500), n(2000), n(100)

    jobs = []
    for job_id in range(1, n_jobs + 1):
        views = rng.randint(50, 5000)
        jobs.append({
            "job_id": job_id,
            "title": fake.job(),
            "category": rng.choice(CATEGORIES),
            "location": rng.choice(LOCATIONS),
            "date_posted": (END - timedelta(days=rng.randint(0, 364))).isoformat(),
            "views": views,
            "applications": int(views * rng.uniform(0.02, 0.25)),
            "status": rng.choice(["open", "closed", "paused"]),
        })

    funnel = []
    for cid in range(1, n_cand + 1):
        job_id = rng.randint(1, n_jobs)
        day = END - timedelta(days=rng.randint(30, 364))
        for i, stage in enumerate(STAGES):
            row = {"candidate_id": cid, "job_id": job_id, "stage": stage,
                   "date": day.isoformat(), "drop_off_flag": 0}
            funnel.append(row)
            if stage == STAGES[-1]:
                break
            if rng.random() < DROP_PROB[i]:
                row["drop_off_flag"] = 1
                break
            day += timedelta(days=rng.randint(1, 10))

    traffic = []
    for offset in range(365):
        day = END - timedelta(days=offset)
        for source in SOURCES:
            traffic.append({
                "date": day.isoformat(),
                "source": source,
                "visits": rng.randint(200, 6000),
                "session_duration": round(rng.uniform(30, 420), 1),
                "bounce_rate": round(rng.uniform(0.25, 0.85), 3),
            })

    recruiters = [{
        "recruiter_id": rng.randint(1, n_rec),
        "job_id": rng.randint(1, n_jobs),
        "actions": rng.randint(1, 200),
        "response_rate": round(rng.uniform(0.05, 0.95), 3),
    } for _ in range(n(800))]

    candidates = [{
        "candidate_id": cid,
        "skills": ", ".join(rng.sample(SKILLS, 3)),
        "experience_years": rng.randint(0, 20),
        "education": rng.choice(EDUCATION),
    } for cid in range(1, n_cand + 1)]

    tables = {
        "job_postings": jobs,
        "hiring_funnel": funnel,
        "traffic_metrics": traffic,
        "recruiter_activity": recruiters,
        "candidate_profiles": candidates,
    }
    for name, rows in tables.items():
        store.replace_table(pd.DataFrame(rows), name)
    return {name: len(rows) for name, rows in tables.items()}


if __name__ == "__main__":
    print(seed_database(SQLiteStore(config.DB_PATH)))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_seed.py -v`
Expected: 4 passed.

- [ ] **Step 5: Seed the real DB and commit**

Run: `python -m data.seed`
Expected: prints a dict with 5 table counts; `data/naukri.db` exists (gitignored).

```bash
git add data/seed.py tests/test_seed.py
git commit -m "feat: add Faker seed data for the five HR tables"
```

---

### Task 4: Prompt files, skills, loaders

**Files:**
- Create: `graph/prompts.py`, `graph/skills.py`, `prompts/{data_understanding,query,visualization,sql,pandas}.v1.txt`, `skills/{job-posting,hiring-funnel,traffic,slide-gen}/SKILL.md`
- Test: `tests/test_prompts_skills.py`

**Interfaces:**
- Produces:
  - `load_prompt(name: str, version: str = "v1") -> str`
  - `render(name: str, version: str = "v1", **values) -> str` (`string.Template.safe_substitute`, `$var` placeholders, values coerced with `str`)
  - `detect_domain(columns: list[str]) -> str | None` (returns `"job-posting"`, `"hiring-funnel"`, `"traffic"` or `None`)
  - `load_skill(domain: str) -> str`, `load_domain_skills() -> str` (the three data-domain skills concatenated with headers)
  - Prompt variables: `data_understanding`: skills, schema, sample; `query`: summary, skill, history, slice, question, error_note; `visualization`: insight, columns, error_note; `sql`: schema, limit, question, error_note; `pandas`: columns, sample, question, error_note.

- [ ] **Step 1: Write the failing tests**

`tests/test_prompts_skills.py`:
```python
import re

import pytest

from graph.prompts import load_prompt, render
from graph.skills import detect_domain, load_domain_skills, load_skill

VARS = {
    "data_understanding": dict(skills="s", schema="sc", sample="sm"),
    "query": dict(summary="a", skill="b", history="c", slice="d", question="e", error_note=""),
    "visualization": dict(insight="i", columns="c", error_note=""),
    "sql": dict(schema="s", limit=200, question="q", error_note=""),
    "pandas": dict(columns="c", sample="s", question="q", error_note=""),
}


@pytest.mark.parametrize("name", VARS)
def test_every_prompt_renders_fully(name):
    out = render(name, **VARS[name])
    assert not re.search(r"\$[A-Za-z_]+", out)


def test_render_keeps_json_braces():
    out = render("visualization", **VARS["visualization"])
    assert '{"type"' in out


def test_load_prompt_missing_version_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("query", "v99")


@pytest.mark.parametrize("cols,expected", [
    (["job_id", "title", "category", "location", "views", "applications"], "job-posting"),
    (["candidate_id", "job_id", "stage", "date", "drop_off_flag"], "hiring-funnel"),
    (["date", "source", "visits", "session_duration", "bounce_rate"], "traffic"),
    (["foo", "bar"], None),
])
def test_detect_domain(cols, expected):
    assert detect_domain(cols) == expected


def test_skills_load():
    for d in ["job-posting", "hiring-funnel", "traffic", "slide-gen"]:
        assert len(load_skill(d)) > 100
    combined = load_domain_skills()
    assert "job-posting" in combined and "traffic" in combined
    assert "slide-gen" not in combined
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_prompts_skills.py -v`
Expected: FAIL (`ModuleNotFoundError: graph.prompts`).

- [ ] **Step 3: Create the prompt files**

`prompts/data_understanding.v1.txt`:
```
You are a talent analytics specialist at Naukri (Info Edge India). Study the data schema and sample below and identify: key metrics, time dimensions, categorical columns, and anomalies (nulls, outliers, odd values).

Domain guidance:
$skills

Schema:
$schema

Sample and statistics:
$sample

Respond in plain text under exactly four headings: Key metrics, Time dimensions, Categorical columns, Anomalies. Be concise (max 200 words).
```
`prompts/query.v1.txt`:
```
You are a talent analytics specialist for Naukri (Info Edge India). Answer the question using ONLY the data slice below. Never invent numbers.

Data context:
$summary

Domain guidance:
$skill

Recent conversation:
$history

Data slice (CSV):
$slice

Question: $question

Return ONLY a JSON object with these keys:
"finding": one-sentence key finding,
"evidence": array of 1-4 short strings, each citing specific numbers from the slice,
"recommendation": one specific, actionable recommendation.
$error_note
```
`prompts/visualization.v1.txt`:
```
Given the insight and the columns available in the data slice, choose ONE chart that best supports the finding.

Insight: $insight

Available columns: $columns

Return ONLY a JSON object: {"type": "bar|line|scatter|pie", "x": "<column>", "y": "<column>", "title": "<short title>"}
x and y MUST be exact names from the available columns. y must be a numeric column.
$error_note
```
`prompts/sql.v1.txt`:
```
You write SQLite SELECT queries for an HR analytics database.

Schema:
$schema

Rules: exactly one SELECT statement; use only the tables and columns listed; aggregate (GROUP BY) when the question asks for rates, comparisons or rankings; include LIMIT $limit at most. Return ONLY the SQL.

Question: $question
$error_note
```
`prompts/pandas.v1.txt`:
```
You write a single pandas expression over a DataFrame named df.

Columns: $columns

Sample:
$sample

Return ONLY one expression (no assignment, no imports, no print) that evaluates to a DataFrame. For summaries use groupby/agg followed by reset_index().

Question: $question
$error_note
```

- [ ] **Step 4: Create the skill files**

`skills/job-posting/SKILL.md`:
```markdown
# Job Posting Analytics

**Key metrics:** views, applications, conversion rate = applications / views.
**Relevant columns:** job_id, title, category, location, date_posted, views, applications, status.

**What good insights look like:**
- Compare conversion by category and by location, not only raw counts.
- Flag postings with high views but low conversion (weak title or description) and low views but high conversion (under-promoted).
- Always state the sample size (number of postings) behind a percentage.
- Recommend a concrete action: re-promote, rewrite the title, pause, or close.
```
`skills/hiring-funnel/SKILL.md`:
```markdown
# Hiring Funnel Analytics

**Stages, in order:** applied → screened → interviewed → offered → hired.
**Formulas:** stage conversion = candidates reaching stage N+1 / candidates reaching stage N. Drop-off rate = 1 − stage conversion. Time-to-hire = date at hired − date at applied.

**Significance:** treat a drop-off above 40% at any single stage, or a 10-point change versus another period or segment, as noteworthy. Ignore segments with fewer than 30 candidates.

**What good insights look like:** name the worst stage, quantify it, compare against another segment (job, category or period), and recommend one intervention at that stage.
```
`skills/traffic/SKILL.md`:
```markdown
# Traffic Analytics

**Key metrics:** visits, session_duration (seconds), bounce_rate (0–1).
**Source attribution:** rank sources by visits and by quality; a high-visit source with a high bounce rate is low quality.
**Session quality:** low bounce rate plus long session duration is good; report both together.
**Period comparison:** compare like periods (week vs previous week, month vs previous month) and state the percentage change and the absolute values.

**What good insights look like:** separate volume from quality, call out the biggest mover, and recommend shifting effort between sources.
```
`skills/slide-gen/SKILL.md`:
```markdown
# Slide Generation (used from phase 2)

**Structure:** one insight per slide. Title = the finding as a sentence (max 12 words). Body = up to 3 evidence bullets and one recommendation line.
**Chart selection:** bar for category comparisons, line for time series, scatter for two numeric variables, pie only for shares of a whole with at most 6 slices.
**Style:** no more than 40 words of body text per slide; numbers always carry units.
```

- [ ] **Step 5: Implement `graph/prompts.py` and `graph/skills.py`**

`graph/prompts.py`:
```python
from string import Template

from config import ROOT

PROMPTS_DIR = ROOT / "prompts"


def load_prompt(name: str, version: str = "v1") -> str:
    return (PROMPTS_DIR / f"{name}.{version}.txt").read_text()


def render(name: str, version: str = "v1", **values) -> str:
    return Template(load_prompt(name, version)).safe_substitute(
        {k: str(v) for k, v in values.items()}
    )
```
`graph/skills.py`:
```python
from config import ROOT

SKILLS_DIR = ROOT / "skills"
DOMAINS = ["job-posting", "hiring-funnel", "traffic"]
DOMAIN_KEYWORDS = {
    "job-posting": ["job_id", "views", "applications", "category", "location", "date_posted", "title"],
    "hiring-funnel": ["stage", "drop_off", "candidate_id", "time_to_hire"],
    "traffic": ["visits", "bounce", "session", "source"],
}


def load_skill(domain: str) -> str:
    return (SKILLS_DIR / domain / "SKILL.md").read_text()


def load_domain_skills() -> str:
    return "\n\n".join(f"## {d}\n{load_skill(d)}" for d in DOMAINS)


def detect_domain(columns: list[str]) -> str | None:
    cols = [c.lower() for c in columns]
    scores = {
        d: sum(any(k in c for c in cols) for k in kws)
        for d, kws in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None
```

- [ ] **Step 6: Run to verify pass**

Run: `pytest tests/test_prompts_skills.py -v`
Expected: all pass (5 render params + 4 domain params + 3 others).

- [ ] **Step 7: Commit**

```bash
git add prompts skills graph/prompts.py graph/skills.py tests/test_prompts_skills.py
git commit -m "feat: add versioned prompt files, skills and loaders"
```

---

### Task 5: Groq LLM factory and friendly errors

**Files:**
- Create: `graph/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `config.get_api_key`, `config.MODEL_FAST`, `config.MODEL_SMART`, `config.MissingKeyError`.
- Produces: `get_llm(model: str, temperature: float = 0.0) -> ChatGroq` (object with `.invoke(prompt_str).content`), `friendly_error(exc: Exception) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py`:
```python
import config
from graph.llm import friendly_error, get_llm


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


def test_friendly_rate_limit():
    assert "rate limit" in friendly_error(RateLimitError("429 Too Many Requests")).lower()


def test_friendly_auth():
    assert "GROQ_API_KEY" in friendly_error(AuthenticationError("401 invalid api key"))


def test_friendly_timeout():
    assert "reach Groq" in friendly_error(TimeoutError("request timed out"))


def test_friendly_missing_key():
    msg = "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
    assert friendly_error(config.MissingKeyError(msg)) == msg


def test_friendly_generic():
    assert friendly_error(ValueError("boom")) == "Something went wrong: boom"


def test_get_llm_uses_requested_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    llm = get_llm(config.MODEL_SMART)
    assert llm.model_name == config.MODEL_SMART
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL (`ModuleNotFoundError: graph.llm`).

- [ ] **Step 3: Implement `graph/llm.py`**

```python
from langchain_groq import ChatGroq

from config import MissingKeyError, get_api_key


def get_llm(model: str, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=get_api_key(),
        timeout=60,
        max_retries=1,
    )


def friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    name = type(exc).__name__
    if isinstance(exc, MissingKeyError):
        return msg
    if "rate limit" in low or "429" in low or name == "RateLimitError":
        return "Groq rate limit reached. Wait a minute and try again."
    if "401" in low or "invalid api key" in low or name == "AuthenticationError":
        return "Groq rejected the API key. Check GROQ_API_KEY in .env."
    if "timeout" in low or "timed out" in low or name in ("APITimeoutError", "APIConnectionError"):
        return "Could not reach Groq (timeout or network). Try again."
    return f"Something went wrong: {msg}"
```
If `ChatGroq(api_key=...)` raises a validation error on your installed `langchain-groq`, switch the kwarg to `groq_api_key=` (older versions) and note it in the commit message.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_llm.py -v`
Expected: 6 passed (no network calls).

- [ ] **Step 5: Commit**

```bash
git add graph/llm.py tests/test_llm.py
git commit -m "feat: add Groq LLM factory and friendly error mapping"
```

---

### Task 6: State, models, router, parsing helpers

**Files:**
- Create: `graph/models.py`, `graph/state.py`, `graph/parsing.py`, `graph/router.py`
- Test: `tests/test_models_router_parsing.py`

**Interfaces:**
- Produces:
  - `Insight(finding: str, evidence: list[str], recommendation: str)` — pydantic; `finding` and `recommendation` non-empty (`min_length=1`).
  - `ChartConfig(type: Literal["bar","line","scatter","pie"], x: str, y: str, title: str)` — pydantic.
  - `AnalyzerState` — `TypedDict(total=False)` with keys: `upload: dict` (`{"name": str, "bytes": bytes}`), `df: pd.DataFrame`, `table_name: str`, `ingest_action: str`, `schema: str`, `data_summary: str`, `question: str`, `query_type: str`, `chat_history: list[dict]` (`{"question","finding"}`), `data_slice: pd.DataFrame`, `insight: Insight`, `chart_config: ChartConfig | None`, `insight_memory: list[dict]`, `prompts: list[dict]` (`{"node","prompt"}`), `errors: list[str]`.
  - `extract_code(text: str) -> str` (strips a Markdown code fence and trailing `;`), `extract_json(text: str) -> dict` (parses the first `{`…last `}`; raises `ValueError` on failure).
  - `choose_tool(question: str, columns: list[str] | None) -> str` (`"pandas"` if any column name appears in the question, else `"sql"`).

- [ ] **Step 1: Write the failing tests**

`tests/test_models_router_parsing.py`:
```python
import pytest
from pydantic import ValidationError

from graph.models import ChartConfig, Insight
from graph.parsing import extract_code, extract_json
from graph.router import choose_tool


def test_insight_requires_text():
    with pytest.raises(ValidationError):
        Insight(finding="", evidence=[], recommendation="x")
    assert Insight(finding="f", evidence=["e"], recommendation="r").finding == "f"


def test_chart_type_restricted():
    with pytest.raises(ValidationError):
        ChartConfig(type="donut", x="a", y="b", title="t")


def test_extract_code_strips_fence_and_semicolon():
    assert extract_code("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_code("SELECT 2;") == "SELECT 2"


def test_extract_json_finds_object_in_prose():
    assert extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


@pytest.mark.parametrize("q,cols,expected", [
    ("total views by category", None, "sql"),
    ("total views by category", [], "sql"),
    ("average salary by dept", ["salary", "dept"], "pandas"),
    ("show Time To Hire trend", ["time_to_hire"], "pandas"),
    ("something unrelated", ["salary"], "sql"),
])
def test_choose_tool(q, cols, expected):
    assert choose_tool(q, cols) == expected
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_models_router_parsing.py -v`
Expected: FAIL (`ModuleNotFoundError: graph.models`).

- [ ] **Step 3: Implement the four modules**

`graph/models.py`:
```python
from typing import Literal

from pydantic import BaseModel, Field


class Insight(BaseModel):
    finding: str = Field(min_length=1)
    evidence: list[str]
    recommendation: str = Field(min_length=1)


class ChartConfig(BaseModel):
    type: Literal["bar", "line", "scatter", "pie"]
    x: str
    y: str
    title: str
```
`graph/state.py`:
```python
from typing import Optional, TypedDict

import pandas as pd

from graph.models import ChartConfig, Insight


class AnalyzerState(TypedDict, total=False):
    upload: dict
    df: pd.DataFrame
    table_name: str
    ingest_action: str
    schema: str
    data_summary: str
    question: str
    query_type: str
    chat_history: list
    data_slice: pd.DataFrame
    insight: Insight
    chart_config: Optional[ChartConfig]
    insight_memory: list
    prompts: list
    errors: list
```
`graph/parsing.py`:
```python
import json
import re


def extract_code(text: str) -> str:
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    return body.strip().rstrip(";").strip()


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in model output: {exc}") from exc
```
`graph/router.py`:
```python
def choose_tool(question: str, columns: list[str] | None) -> str:
    if not columns:
        return "sql"
    q = question.lower()
    for col in columns:
        c = col.lower()
        if c in q or c.replace("_", " ") in q:
            return "pandas"
    return "sql"
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_models_router_parsing.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add graph/models.py graph/state.py graph/parsing.py graph/router.py tests/test_models_router_parsing.py
git commit -m "feat: add state, models, JSON/code extraction and tool router"
```

---

### Task 7: Ingestion node

**Files:**
- Create: `data/parsing.py`, `graph/nodes/ingest.py`, `tests/fakes.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `SQLiteStore` (Task 2), `render`, `load_domain_skills` (Task 4), `sanitize_name`.
- Produces:
  - `parse_upload(name: str, content: bytes) -> pd.DataFrame` — supports `.xlsx`, `.xlsm`, `.csv`, `.json`; raises `ValueError("Unsupported file type: ...")` otherwise.
  - `make_ingest_node(store, llm) -> Callable[[AnalyzerState], dict]`. With `state["upload"]` present: parses, sanitises column names, `append_or_create`s into a table named after the filename stem, returns `df`, `table_name`, `ingest_action`, `schema`, `data_summary`, `prompts`. Without an upload: summarises the whole DB (returns `schema`, `data_summary`, `prompts` only — no `df`). Prompts are appended as `{"node": "ingest", "prompt": ...}` to `state.get("prompts", [])`.
  - `tests/fakes.py::FakeLLM(responses: list[str])` — `.invoke(prompt) -> SimpleNamespace(content=...)`, records `.prompts`, raises `AssertionError` when out of responses.

- [ ] **Step 1: Write the failing tests**

`tests/fakes.py`:
```python
from types import SimpleNamespace


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return SimpleNamespace(content=self.responses.pop(0))
```
`tests/test_ingest.py`:
```python
import json

import pandas as pd
import pytest

from data.parsing import parse_upload
from data.store import SchemaMismatchError
from graph.nodes.ingest import make_ingest_node
from tests.fakes import FakeLLM

CSV = b"Job ID,Views,Applications\n1,100,10\n2,200,30\n"


def test_parse_csv_json_and_unsupported():
    assert list(parse_upload("a.csv", CSV).columns) == ["Job ID", "Views", "Applications"]
    js = json.dumps({"rows": [{"a": 1}, {"a": 2}]}).encode()
    assert len(parse_upload("a.json", js)) == 2
    js2 = json.dumps({"metrics": {"a": 1}}).encode()
    assert len(parse_upload("m.json", js2)) == 1
    with pytest.raises(ValueError):
        parse_upload("a.pdf", b"x")


def test_parse_xlsx_roundtrip(tmp_path):
    p = tmp_path / "x.xlsx"
    pd.DataFrame({"a": [1, 2]}).to_excel(p, index=False)
    assert len(parse_upload("x.xlsx", p.read_bytes())) == 2


def test_upload_creates_then_appends(store):
    llm = FakeLLM(["summary one", "summary two"])
    node = make_ingest_node(store, llm)
    out = node({"upload": {"name": "Jobs Q1.csv", "bytes": CSV}, "prompts": []})
    assert out["table_name"] == "jobs_q1"
    assert out["ingest_action"] == "created"
    assert list(out["df"].columns) == ["job_id", "views", "applications"]
    assert out["data_summary"] == "summary one"
    assert out["prompts"][0]["node"] == "ingest"
    assert "jobs_q1" in out["schema"]

    out2 = node({"upload": {"name": "Jobs Q1.csv", "bytes": CSV}})
    assert out2["ingest_action"] == "appended"
    assert len(store.run_sql("SELECT * FROM jobs_q1")) == 4


def test_upload_schema_mismatch_raises(store):
    node = make_ingest_node(store, FakeLLM(["s"]))
    node({"upload": {"name": "t.csv", "bytes": CSV}})
    with pytest.raises(SchemaMismatchError):
        node({"upload": {"name": "t.csv", "bytes": b"x,y\n1,2\n"}})


def test_no_upload_summarises_database(store):
    store.replace_table(pd.DataFrame({"a": [1, 2, 3]}), "widgets")
    llm = FakeLLM(["db summary"])
    out = make_ingest_node(store, llm)({})
    assert out["data_summary"] == "db summary"
    assert "df" not in out
    assert "widgets" in llm.prompts[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL (`ModuleNotFoundError: data.parsing`).

- [ ] **Step 3: Implement `data/parsing.py`**

```python
import io
import json
from pathlib import Path

import pandas as pd


def parse_upload(name: str, content: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(io.BytesIO(content))
    if suffix == ".json":
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, dict):
            lists = [v for v in data.values() if isinstance(v, list)]
            data = lists[0] if len(lists) == 1 else [data]
        return pd.json_normalize(data)
    raise ValueError(f"Unsupported file type: {suffix or name}")
```

- [ ] **Step 4: Implement `graph/nodes/ingest.py`**

```python
from pathlib import Path

from data.parsing import parse_upload
from data.store import sanitize_name
from graph.prompts import render
from graph.skills import load_domain_skills


def _sample_for_df(df) -> str:
    return df.head(5).to_csv(index=False) + "\n" + df.describe(include="all").round(2).to_csv()


def _sample_for_db(store) -> str:
    parts = []
    for table in store.list_tables():
        parts.append(f"-- {table}\n" + store.run_sql(f"SELECT * FROM {table} LIMIT 3").to_csv(index=False))
    return "\n".join(parts) or "(database is empty)"


def make_ingest_node(store, llm):
    def ingest(state):
        update = {}
        df = None
        upload = state.get("upload")
        if upload:
            df = parse_upload(upload["name"], upload["bytes"])
            df.columns = [sanitize_name(c) for c in df.columns]
            table = sanitize_name(Path(upload["name"]).stem)
            update["ingest_action"] = store.append_or_create(df, table)
            update["table_name"] = table
            update["df"] = df

        schema = store.schema_text()
        sample = _sample_for_df(df) if df is not None else _sample_for_db(store)
        prompt = render("data_understanding", skills=load_domain_skills(), schema=schema, sample=sample)
        summary = llm.invoke(prompt).content
        update["schema"] = schema
        update["data_summary"] = summary
        update["prompts"] = list(state.get("prompts", [])) + [{"node": "ingest", "prompt": prompt}]
        return update

    return ingest
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_ingest.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add data/parsing.py graph/nodes/ingest.py tests/fakes.py tests/test_ingest.py
git commit -m "feat: add file parsing and ingestion node"
```

---

### Task 8: Retrieval tools and node

**Files:**
- Create: `graph/tools.py`, `graph/nodes/retrieve.py`
- Test: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `SQLiteStore.run_sql`, `render`, `extract_code`, `choose_tool`, `config.ROW_CAP`.
- Produces:
  - `class RetrievalError(RuntimeError)`
  - `make_sql_tool(store, llm, row_cap=ROW_CAP) -> Callable[[str, str], pd.DataFrame]` — `(question, schema_text) -> slice`. Two attempts; second prompt carries the error in `error_note`. Non-SELECT output (must start with `select` or `with`) counts as a failed attempt.
  - `make_pandas_tool(llm, row_cap=ROW_CAP) -> Callable[[str, pd.DataFrame], pd.DataFrame]` — evaluates the LLM's single expression with `eval` and no builtins except `len, sum, min, max, round, abs, sorted, str, int, float`, namespace `{"df", "pd"}`; rejects expressions containing `__`, `import`, `open(`, `exec(`, `eval(`, `compile(`, `globals`, `getattr`, `os.`, `sys.`. A Series becomes a DataFrame via `reset_index()`; a scalar becomes `DataFrame({"value": [x]})`. Two attempts.
  - `make_retrieve_node(sql_tool, pandas_tool) -> Callable[[AnalyzerState], dict]` returning `{"query_type", "data_slice"}` with the slice capped at `ROW_CAP` rows.

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieval.py`:
```python
import pandas as pd
import pytest

from graph.nodes.retrieve import make_retrieve_node
from graph.tools import RetrievalError, make_pandas_tool, make_sql_tool
from tests.fakes import FakeLLM


@pytest.fixture
def seeded(store):
    store.replace_table(pd.DataFrame({"category": ["a", "a", "b"], "views": [10, 20, 30]}), "jobs")
    return store


def test_sql_tool_happy_path(seeded):
    llm = FakeLLM(["```sql\nSELECT category, SUM(views) AS v FROM jobs GROUP BY category;\n```"])
    out = make_sql_tool(seeded, llm)("views by category", seeded.schema_text())
    assert dict(zip(out["category"], out["v"])) == {"a": 30, "b": 30}


def test_sql_tool_retries_with_error_note(seeded):
    llm = FakeLLM(["SELECT nope FROM jobs", "SELECT COUNT(*) AS n FROM jobs"])
    out = make_sql_tool(seeded, llm)("how many", seeded.schema_text())
    assert out["n"][0] == 3
    assert "failed" in llm.prompts[1].lower()


def test_sql_tool_rejects_non_select_then_gives_up(seeded):
    llm = FakeLLM(["DROP TABLE jobs", "DELETE FROM jobs"])
    with pytest.raises(RetrievalError):
        make_sql_tool(seeded, llm)("x", seeded.schema_text())
    assert seeded.list_tables() == ["jobs"]


def test_sql_tool_caps_rows(store):
    store.replace_table(pd.DataFrame({"a": range(500)}), "t")
    llm = FakeLLM(["SELECT a FROM t"])
    assert len(make_sql_tool(store, llm, row_cap=50)("all", store.schema_text())) == 50


def test_pandas_tool_dataframe_series_scalar():
    df = pd.DataFrame({"g": ["x", "x", "y"], "v": [1, 2, 3]})
    llm = FakeLLM(["df.groupby('g')['v'].sum()", "df['v'].sum()", "df[df['v'] > 1]"])
    tool = make_pandas_tool(llm)
    s = tool("q", df)
    assert list(s.columns) == ["g", "v"] and list(s["v"]) == [3, 3]
    assert tool("q", df)["value"][0] == 6
    assert len(tool("q", df)) == 2


def test_pandas_tool_blocks_dangerous_code():
    df = pd.DataFrame({"v": [1]})
    llm = FakeLLM(["__import__('os').system('ls')", "df.__class__"])
    with pytest.raises(RetrievalError):
        make_pandas_tool(llm)("q", df)


def test_retrieve_node_routes_and_caps():
    calls = []

    def sql_tool(q, schema):
        calls.append("sql")
        return pd.DataFrame({"a": range(300)})

    def pandas_tool(q, df):
        calls.append("pandas")
        return df

    node = make_retrieve_node(sql_tool, pandas_tool)
    out = node({"question": "total views", "schema": "s"})
    assert out["query_type"] == "sql" and len(out["data_slice"]) == 200

    df = pd.DataFrame({"salary": [1, 2]})
    out = node({"question": "avg salary", "schema": "s", "df": df})
    assert out["query_type"] == "pandas"
    assert calls == ["sql", "pandas"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL (`ModuleNotFoundError: graph.tools`).

- [ ] **Step 3: Implement `graph/tools.py`**

```python
import re

import pandas as pd

from config import ROW_CAP
from graph.parsing import extract_code
from graph.prompts import render


class RetrievalError(RuntimeError):
    pass


def make_sql_tool(store, llm, row_cap: int = ROW_CAP):
    def run(question: str, schema: str) -> pd.DataFrame:
        error_note = ""
        for _ in range(2):
            prompt = render("sql", schema=schema, limit=row_cap, question=question, error_note=error_note)
            sql = extract_code(llm.invoke(prompt).content)
            if not re.match(r"\s*(select|with)\b", sql, re.I):
                error_note = f"Your previous output was not a SELECT statement: {sql!r}. Return only one SELECT."
                continue
            try:
                return store.run_sql(sql).head(row_cap)
            except Exception as exc:
                error_note = f"Your previous SQL failed.\nSQL: {sql}\nError: {exc}\nFix it."
        raise RetrievalError(f"Could not produce a working SQL query. {error_note}")

    return run


FORBIDDEN = ("__", "import", "open(", "exec(", "eval(", "compile(", "globals", "getattr", "os.", "sys.")
SAFE_BUILTINS = {"len": len, "sum": sum, "min": min, "max": max, "round": round,
                 "abs": abs, "sorted": sorted, "str": str, "int": int, "float": float}


def _to_frame(result) -> pd.DataFrame:
    if isinstance(result, pd.DataFrame):
        return result
    if isinstance(result, pd.Series):
        return result.reset_index()
    return pd.DataFrame({"value": [result]})


def make_pandas_tool(llm, row_cap: int = ROW_CAP):
    def run(question: str, df: pd.DataFrame) -> pd.DataFrame:
        error_note = ""
        for _ in range(2):
            sample = df.head(3).to_csv(index=False)
            prompt = render("pandas", columns=", ".join(f"{c} ({df[c].dtype})" for c in df.columns),
                            sample=sample, question=question, error_note=error_note)
            expr = extract_code(llm.invoke(prompt).content)
            if any(tok in expr for tok in FORBIDDEN):
                error_note = f"Your previous expression used a forbidden construct: {expr!r}. Use only df and pd."
                continue
            try:
                result = eval(expr, {"__builtins__": SAFE_BUILTINS}, {"df": df, "pd": pd})
                return _to_frame(result).head(row_cap)
            except Exception as exc:
                error_note = f"Your previous expression failed.\nExpression: {expr}\nError: {exc}\nFix it."
        raise RetrievalError(f"Could not produce a working pandas expression. {error_note}")

    return run
```
Note: `eval` of model-written code is acceptable here only because the app is a local single-user tool and the forbidden-token filter and empty builtins limit it; do not expose this app to untrusted users without a real sandbox.

- [ ] **Step 4: Implement `graph/nodes/retrieve.py`**

```python
from config import ROW_CAP
from graph.router import choose_tool


def make_retrieve_node(sql_tool, pandas_tool):
    def retrieve(state):
        df = state.get("df")
        question = state["question"]
        tool = choose_tool(question, None if df is None else list(df.columns))
        if tool == "pandas":
            data_slice = pandas_tool(question, df)
        else:
            data_slice = sql_tool(question, state["schema"])
        return {"query_type": tool, "data_slice": data_slice.head(ROW_CAP)}

    return retrieve
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_retrieval.py -v`
Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add graph/tools.py graph/nodes/retrieve.py tests/test_retrieval.py
git commit -m "feat: add SQL and pandas retrieval tools and retrieve node"
```

---

### Task 9: Analyst node

**Files:**
- Create: `graph/nodes/analyst.py`
- Test: `tests/test_analyst.py`

**Interfaces:**
- Consumes: `Insight`, `render`, `extract_json`, `detect_domain`, `load_skill`.
- Produces: `class AnalysisError(RuntimeError)`; `format_history(history: list[dict]) -> str` (`"(none)"` when empty, else `Q: …\nA: …` pairs); `make_analyst_node(llm) -> Callable[[AnalyzerState], dict]` returning `{"insight": Insight, "prompts": [...]}`. Empty `data_slice` returns a fixed "No data matched" insight with no LLM call. Invalid JSON or schema gets one retry with `error_note`; a second failure raises `AnalysisError`.

- [ ] **Step 1: Write the failing tests**

`tests/test_analyst.py`:
```python
import json

import pandas as pd
import pytest

from graph.nodes.analyst import AnalysisError, format_history, make_analyst_node
from tests.fakes import FakeLLM

GOOD = json.dumps({"finding": "Engineering converts best.", "evidence": ["Eng 20% vs Sales 10%"],
                   "recommendation": "Shift budget to Engineering."})
SLICE = pd.DataFrame({"category": ["Eng", "Sales"], "views": [100, 100], "applications": [20, 10]})


def base_state(**over):
    state = {"question": "which converts best?", "data_summary": "SUMMARY", "data_slice": SLICE,
             "chat_history": [], "prompts": []}
    state.update(over)
    return state


def test_format_history():
    assert format_history([]) == "(none)"
    text = format_history([{"question": "q1", "finding": "f1"}])
    assert "Q: q1" in text and "A: f1" in text


def test_analyst_happy_path_injects_context():
    llm = FakeLLM([GOOD])
    out = make_analyst_node(llm)(base_state(chat_history=[{"question": "old", "finding": "older"}]))
    assert out["insight"].finding == "Engineering converts best."
    prompt = llm.prompts[0]
    assert "SUMMARY" in prompt and "category,views,applications" in prompt
    assert "Job Posting Analytics" in prompt          # domain skill injected
    assert "Q: old" in prompt
    assert out["prompts"][-1]["node"] == "analyst"


def test_analyst_retries_once_on_bad_json():
    llm = FakeLLM(["not json", GOOD])
    out = make_analyst_node(llm)(base_state())
    assert out["insight"].recommendation.startswith("Shift")
    assert "invalid" in llm.prompts[1].lower()


def test_analyst_gives_up_after_second_failure():
    with pytest.raises(AnalysisError):
        make_analyst_node(FakeLLM(["bad", "worse"]))(base_state())


def test_analyst_empty_slice_skips_llm():
    llm = FakeLLM([])
    out = make_analyst_node(llm)(base_state(data_slice=SLICE.iloc[0:0]))
    assert "No data matched" in out["insight"].finding
    assert llm.prompts == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_analyst.py -v`
Expected: FAIL (`ModuleNotFoundError: graph.nodes.analyst`).

- [ ] **Step 3: Implement `graph/nodes/analyst.py`**

```python
from graph.models import Insight
from graph.parsing import extract_json
from graph.prompts import render
from graph.skills import detect_domain, load_skill


class AnalysisError(RuntimeError):
    pass


def format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    return "\n".join(f"Q: {h['question']}\nA: {h['finding']}" for h in history)


def make_analyst_node(llm):
    def analyst(state):
        data_slice = state["data_slice"]
        if data_slice.empty:
            return {"insight": Insight(
                finding="No data matched your question.",
                evidence=[],
                recommendation="Try rephrasing the question or check that the relevant data is loaded.",
            )}

        domain = detect_domain(list(data_slice.columns))
        skill = load_skill(domain) if domain else "(none)"
        prompts = list(state.get("prompts", []))
        error_note = ""
        for _ in range(2):
            prompt = render(
                "query",
                summary=state.get("data_summary", ""),
                skill=skill,
                history=format_history(state.get("chat_history", [])),
                slice=data_slice.to_csv(index=False),
                question=state["question"],
                error_note=error_note,
            )
            prompts.append({"node": "analyst", "prompt": prompt})
            try:
                insight = Insight(**extract_json(llm.invoke(prompt).content))
                return {"insight": insight, "prompts": prompts}
            except ValueError as exc:
                error_note = f"Your previous answer was invalid ({exc}). Return only the JSON object described above."
        raise AnalysisError("The analyst could not produce a valid insight.")

    return analyst
```
(`pydantic.ValidationError` subclasses `ValueError`, so one `except ValueError` covers bad JSON and bad schema.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_analyst.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/analyst.py tests/test_analyst.py
git commit -m "feat: add analyst node with structured insight and retry"
```

---

### Task 10: Output node and chart building

**Files:**
- Create: `graph/viz.py`, `graph/nodes/output.py`
- Test: `tests/test_output.py`

**Interfaces:**
- Consumes: `ChartConfig`, `Insight`, `extract_json`, `render`, `config.HISTORY_TURNS`.
- Produces:
  - `validate_chart(raw: dict, df: pd.DataFrame) -> ChartConfig` — raises `ValueError` if `x`/`y` not in `df.columns` or `y` not numeric.
  - `build_figure(cfg: ChartConfig, df: pd.DataFrame) -> plotly.graph_objects.Figure` for `bar`, `line`, `scatter`, `pie` (pie uses `names=x, values=y`).
  - `make_output_node(llm) -> Callable[[AnalyzerState], dict]` returning `chart_config` (`ChartConfig | None`), `chat_history` (appended `{"question","finding"}`, trimmed to last `HISTORY_TURNS`), `insight_memory` (appended `insight.model_dump()`), `prompts` (appended `{"node": "visualization", ...}` per attempt), `errors` (appends `"Chart could not be generated."` when both attempts fail on a non-empty slice). Empty slice → no LLM call, `chart_config=None`, no error.

- [ ] **Step 1: Write the failing tests**

`tests/test_output.py`:
```python
import json

import pandas as pd
import pytest

from graph.models import ChartConfig, Insight
from graph.nodes.output import make_output_node
from graph.viz import build_figure, validate_chart
from tests.fakes import FakeLLM

SLICE = pd.DataFrame({"category": ["Eng", "Sales"], "rate": [0.2, 0.1]})
INSIGHT = Insight(finding="Eng leads.", evidence=["20% vs 10%"], recommendation="Invest.")
GOOD = json.dumps({"type": "bar", "x": "category", "y": "rate", "title": "Rate by category"})


def test_validate_chart_rules():
    assert validate_chart(json.loads(GOOD), SLICE).type == "bar"
    with pytest.raises(ValueError):
        validate_chart({"type": "bar", "x": "nope", "y": "rate", "title": "t"}, SLICE)
    with pytest.raises(ValueError):
        validate_chart({"type": "bar", "x": "rate", "y": "category", "title": "t"}, SLICE)


@pytest.mark.parametrize("kind", ["bar", "line", "scatter", "pie"])
def test_build_figure_types(kind):
    fig = build_figure(ChartConfig(type=kind, x="category", y="rate", title="T"), SLICE)
    assert len(fig.data) == 1


def state(**over):
    s = {"question": "q", "insight": INSIGHT, "data_slice": SLICE, "chat_history": [],
         "insight_memory": [], "prompts": [], "errors": []}
    s.update(over)
    return s


def test_output_happy_path():
    out = make_output_node(FakeLLM([GOOD]))(state())
    assert out["chart_config"].y == "rate"
    assert out["chat_history"] == [{"question": "q", "finding": "Eng leads."}]
    assert out["insight_memory"][0]["finding"] == "Eng leads."
    assert out["prompts"][-1]["node"] == "visualization"
    assert out["errors"] == []


def test_output_retries_then_succeeds():
    llm = FakeLLM(["nonsense", GOOD])
    out = make_output_node(llm)(state())
    assert out["chart_config"] is not None
    assert "invalid" in llm.prompts[1].lower()


def test_output_falls_back_after_two_failures():
    out = make_output_node(FakeLLM(["bad", "bad"]))(state())
    assert out["chart_config"] is None
    assert out["errors"] == ["Chart could not be generated."]


def test_output_empty_slice_no_llm():
    llm = FakeLLM([])
    out = make_output_node(llm)(state(data_slice=SLICE.iloc[0:0]))
    assert out["chart_config"] is None and out["errors"] == [] and llm.prompts == []


def test_history_trimmed_to_six():
    hist = [{"question": f"q{i}", "finding": f"f{i}"} for i in range(6)]
    out = make_output_node(FakeLLM([GOOD]))(state(chat_history=hist))
    assert len(out["chat_history"]) == 6
    assert out["chat_history"][-1]["question"] == "q"
    assert out["chat_history"][0]["question"] == "q1"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_output.py -v`
Expected: FAIL (`ModuleNotFoundError: graph.viz`).

- [ ] **Step 3: Implement `graph/viz.py`**

```python
import pandas as pd
import plotly.express as px

from graph.models import ChartConfig


def validate_chart(raw: dict, df: pd.DataFrame) -> ChartConfig:
    cfg = ChartConfig(**raw)
    for col in (cfg.x, cfg.y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is not in the data: {list(df.columns)}")
    if not pd.api.types.is_numeric_dtype(df[cfg.y]):
        raise ValueError(f"y column '{cfg.y}' must be numeric")
    return cfg


def build_figure(cfg: ChartConfig, df: pd.DataFrame):
    if cfg.type == "bar":
        return px.bar(df, x=cfg.x, y=cfg.y, title=cfg.title)
    if cfg.type == "line":
        return px.line(df, x=cfg.x, y=cfg.y, title=cfg.title)
    if cfg.type == "scatter":
        return px.scatter(df, x=cfg.x, y=cfg.y, title=cfg.title)
    return px.pie(df, names=cfg.x, values=cfg.y, title=cfg.title)
```

- [ ] **Step 4: Implement `graph/nodes/output.py`**

```python
from config import HISTORY_TURNS
from graph.parsing import extract_json
from graph.prompts import render
from graph.viz import validate_chart


def make_output_node(llm):
    def output(state):
        data_slice = state["data_slice"]
        insight = state["insight"]
        prompts = list(state.get("prompts", []))
        errors = list(state.get("errors", []))
        chart = None

        if not data_slice.empty:
            columns = ", ".join(f"{c} ({data_slice[c].dtype})" for c in data_slice.columns)
            insight_text = " ".join([insight.finding, *insight.evidence])
            error_note = ""
            for _ in range(2):
                prompt = render("visualization", insight=insight_text, columns=columns, error_note=error_note)
                prompts.append({"node": "visualization", "prompt": prompt})
                try:
                    chart = validate_chart(extract_json(llm.invoke(prompt).content), data_slice)
                    break
                except ValueError as exc:
                    error_note = f"Your previous answer was invalid ({exc}). Return only the JSON object described above."
            if chart is None:
                errors.append("Chart could not be generated.")

        history = (state.get("chat_history", []) + [
            {"question": state["question"], "finding": insight.finding}
        ])[-HISTORY_TURNS:]
        memory = list(state.get("insight_memory", [])) + [insight.model_dump()]
        return {"chart_config": chart, "chat_history": history,
                "insight_memory": memory, "prompts": prompts, "errors": errors}

    return output
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_output.py -v`
Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add graph/viz.py graph/nodes/output.py tests/test_output.py
git commit -m "feat: add output node, chart validation and Plotly figure builder"
```

---

### Task 11: Graph wiring

**Files:**
- Create: `graph/build_graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: all four `make_*_node` factories, `AnalyzerState`.
- Produces: `route_entry(state) -> str` (`"ingest"` when `state.get("data_summary")` is falsy, else `"retrieve"`); `build_graph(store, fast_llm, smart_llm, sql_tool, pandas_tool)` returning a compiled LangGraph (`.invoke(state_dict) -> dict`). Wiring: `START →(route_entry)→ ingest|retrieve`, `ingest → retrieve → analyst → output → END`. `ingest` and `retrieve`/`output` use `fast_llm`/tools; `analyst` uses `smart_llm`.

- [ ] **Step 1: Write the failing tests**

`tests/test_graph.py`:
```python
import json

import pandas as pd

from graph.build_graph import build_graph, route_entry
from tests.fakes import FakeLLM

INSIGHT = json.dumps({"finding": "Eng leads.", "evidence": ["20% vs 10%"], "recommendation": "Invest."})
CHART = json.dumps({"type": "bar", "x": "category", "y": "conversion", "title": "Conversion"})


def sql_tool(question, schema):
    return pd.DataFrame({"category": ["Eng", "Sales"], "conversion": [0.2, 0.1]})


def no_pandas(question, df):
    raise AssertionError("pandas tool should not be used")


def make(store, fast, smart, sql=sql_tool):
    store.replace_table(pd.DataFrame({"a": [1]}), "t")
    return build_graph(store, fast, smart, sql, no_pandas)


def test_route_entry():
    assert route_entry({}) == "ingest"
    assert route_entry({"data_summary": ""}) == "ingest"
    assert route_entry({"data_summary": "x"}) == "retrieve"


def test_first_run_ingests_then_answers(store):
    fast = FakeLLM(["DB summary", CHART])
    smart = FakeLLM([INSIGHT])
    result = make(store, fast, smart).invoke({"question": "q1", "prompts": [], "errors": []})
    assert result["data_summary"] == "DB summary"
    assert result["insight"].finding == "Eng leads."
    assert result["chart_config"].type == "bar"
    assert [p["node"] for p in result["prompts"]] == ["ingest", "analyst", "visualization"]
    assert len(result["chat_history"]) == 1


def test_follow_up_skips_ingest_and_uses_history(store):
    fast = FakeLLM(["DB summary", CHART, CHART])
    smart = FakeLLM([INSIGHT, INSIGHT])
    graph = make(store, fast, smart)
    first = graph.invoke({"question": "q1", "prompts": [], "errors": []})
    second = graph.invoke({**first, "question": "q2", "prompts": [], "errors": []})
    assert [p["node"] for p in second["prompts"]] == ["analyst", "visualization"]
    assert len(second["chat_history"]) == 2
    assert "Q: q1" in smart.prompts[1]
    assert len(second["insight_memory"]) == 2


def test_empty_slice_skips_llm_calls(store):
    fast = FakeLLM(["DB summary"])
    smart = FakeLLM([])
    graph = make(store, fast, smart, sql=lambda q, s: pd.DataFrame({"a": []}))
    result = graph.invoke({"question": "q", "prompts": [], "errors": []})
    assert "No data matched" in result["insight"].finding
    assert result["chart_config"] is None
    assert smart.prompts == []


def test_chart_failure_is_reported_not_raised(store):
    fast = FakeLLM(["DB summary", "bad", "bad"])
    result = make(store, fast, FakeLLM([INSIGHT])).invoke({"question": "q", "prompts": [], "errors": []})
    assert result["chart_config"] is None
    assert result["errors"] == ["Chart could not be generated."]
    assert result["insight"].finding == "Eng leads."
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL (`ModuleNotFoundError: graph.build_graph`).

- [ ] **Step 3: Implement `graph/build_graph.py`**

```python
from langgraph.graph import END, START, StateGraph

from graph.nodes.analyst import make_analyst_node
from graph.nodes.ingest import make_ingest_node
from graph.nodes.output import make_output_node
from graph.nodes.retrieve import make_retrieve_node
from graph.state import AnalyzerState


def route_entry(state) -> str:
    return "retrieve" if state.get("data_summary") else "ingest"


def build_graph(store, fast_llm, smart_llm, sql_tool, pandas_tool):
    g = StateGraph(AnalyzerState)
    g.add_node("ingest", make_ingest_node(store, fast_llm))
    g.add_node("retrieve", make_retrieve_node(sql_tool, pandas_tool))
    g.add_node("analyst", make_analyst_node(smart_llm))
    g.add_node("output", make_output_node(fast_llm))
    g.add_conditional_edges(START, route_entry, {"ingest": "ingest", "retrieve": "retrieve"})
    g.add_edge("ingest", "retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_edge("analyst", "output")
    g.add_edge("output", END)
    return g.compile()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_graph.py -v`
Expected: 5 passed. If LangGraph rejects DataFrame/pydantic values in state, report the error instead of changing the state shape.

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: all tests so far pass.

- [ ] **Step 6: Commit**

```bash
git add graph/build_graph.py tests/test_graph.py
git commit -m "feat: wire ingest/retrieve/analyst/output into a LangGraph"
```

---

### Task 12: Streamlit app

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `build_graph`, `make_ingest_node`, `make_sql_tool`, `make_pandas_tool`, `get_llm`, `friendly_error`, `build_figure`, `SQLiteStore`, `seed_database`, `config.*`.
- Produces: runnable `streamlit run app.py`. Behaviour: missing key → `st.error` mentioning `GROQ_API_KEY` and `st.stop()`; empty DB is auto-seeded; sidebar upload + "Load file" button calls the ingest node directly; chat input runs the graph; each assistant message shows finding, evidence bullets, recommendation, the Plotly chart (or a caption when the chart failed), a "Data used" expander, and a "Prompts sent to Groq" expander; Groq/other exceptions show `friendly_error` text via `st.error` without crashing.

- [ ] **Step 1: Write the failing test**

`tests/test_app.py`:
```python
from streamlit.testing.v1 import AppTest


def test_app_shows_setup_error_without_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file("app.py").run(timeout=30)
    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_app.py -v`
Expected: FAIL (`app.py` not found).

- [ ] **Step 3: Implement `app.py`**

```python
import streamlit as st

import config
from data.seed import seed_database
from data.store import SQLiteStore
from graph.build_graph import build_graph
from graph.llm import friendly_error, get_llm
from graph.nodes.ingest import make_ingest_node
from graph.tools import make_pandas_tool, make_sql_tool
from graph.viz import build_figure

st.set_page_config(page_title="Naukri Personal Data Analyzer", layout="wide")
st.title("Naukri Personal Data Analyzer")

try:
    config.get_api_key()
except config.MissingKeyError as exc:
    st.error(str(exc))
    st.stop()

PERSISTED = ("df", "table_name", "schema", "data_summary", "chat_history", "insight_memory")


@st.cache_resource
def get_runtime():
    store = SQLiteStore(config.DB_PATH)
    if not store.list_tables():
        seed_database(store)
    fast = get_llm(config.MODEL_FAST)
    smart = get_llm(config.MODEL_SMART)
    graph = build_graph(store, fast, smart, make_sql_tool(store, fast), make_pandas_tool(fast))
    return store, make_ingest_node(store, fast), graph


store, ingest_node, graph = get_runtime()
shared = st.session_state.setdefault("shared", {})
messages = st.session_state.setdefault("messages", [])


def render_prompts(prompts):
    with st.expander("Prompts sent to Groq"):
        for p in prompts:
            st.caption(p["node"])
            st.code(p["prompt"], language="text")


def render_assistant(m):
    ins = m["insight"]
    st.markdown(f"**{ins['finding']}**")
    for e in ins["evidence"]:
        st.markdown(f"- {e}")
    st.markdown(f"*Recommendation:* {ins['recommendation']}")
    if m["chart"] is not None:
        st.plotly_chart(build_figure(m["chart"], m["slice"]), use_container_width=True)
    for err in m["errors"]:
        st.caption(err)
    with st.expander("Data used"):
        st.dataframe(m["slice"])
    render_prompts(m["prompts"])


with st.sidebar:
    st.header("Data")
    st.caption("Tables: " + ", ".join(store.list_tables()))
    up = st.file_uploader("Upload Excel / CSV / JSON", type=["xlsx", "csv", "json"])
    if up is not None and st.button("Load file"):
        try:
            with st.spinner("Ingesting..."):
                update = ingest_node({**shared, "upload": {"name": up.name, "bytes": up.getvalue()}, "prompts": []})
            shared.update({k: update[k] for k in ("df", "table_name", "schema", "data_summary")})
            st.success(f"{update['ingest_action'].title()} table `{update['table_name']}` ({len(update['df'])} rows)")
        except Exception as exc:
            st.error(friendly_error(exc))
    if shared.get("data_summary"):
        with st.expander("Data summary"):
            st.write(shared["data_summary"])

for m in messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
        else:
            render_assistant(m)

question = st.chat_input("Ask about your talent data")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analysing..."):
                result = graph.invoke({**shared, "question": question, "prompts": [], "errors": []})
            shared.update({k: result[k] for k in PERSISTED if k in result})
            msg = {"role": "assistant", "insight": result["insight"].model_dump(),
                   "chart": result.get("chart_config"), "slice": result["data_slice"],
                   "prompts": result["prompts"], "errors": result.get("errors", [])}
            messages.append(msg)
            render_assistant(msg)
        except Exception as exc:
            st.error(friendly_error(exc))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_app.py -v`
Expected: 1 passed.

- [ ] **Step 5: Manual check with the real key**

Ask the user to copy `.env.example` to `.env`, paste their **new** Groq key into `.env` themselves, then run:

```bash
streamlit run app.py
```
Expected: the app opens; asking "Which job category has the highest application to view conversion rate?" returns a finding, evidence, recommendation, a chart, and populated expanders. Do not read or echo `.env`.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add Streamlit UI with upload, chat, chart and prompt expander"
```

---

### Task 13: Eval questions, live smoke test, README

**Files:**
- Create: `tests/eval_questions.json`, `tests/test_live_smoke.py`, `README.md`

**Interfaces:**
- Consumes: the full stack with real Groq.
- Produces: 6 eval questions with an `expect_tool` field; a live test skipped when `GROQ_API_KEY` is empty; setup docs.

- [ ] **Step 1: Write the eval set**

`tests/eval_questions.json`:
```json
[
  {"question": "Which job category has the highest application-to-view conversion rate?", "expect_tool": "sql"},
  {"question": "What is the drop-off rate at each hiring funnel stage?", "expect_tool": "sql"},
  {"question": "Which traffic source has the highest average bounce rate?", "expect_tool": "sql"},
  {"question": "How many open job postings are there in each location?", "expect_tool": "sql"},
  {"question": "How do average session durations compare across traffic sources?", "expect_tool": "sql"},
  {"question": "Which recruiters have the lowest average response rate?", "expect_tool": "sql"}
]
```

- [ ] **Step 2: Write the live smoke test**

`tests/test_live_smoke.py`:
```python
import json
import os
from pathlib import Path

import pytest

import config
from data.seed import seed_database
from graph.build_graph import build_graph
from graph.llm import get_llm
from graph.tools import make_pandas_tool, make_sql_tool

QUESTIONS = json.loads((Path(__file__).parent / "eval_questions.json").read_text())

pytestmark = pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    from data.store import SQLiteStore

    store = SQLiteStore(tmp_path_factory.mktemp("live") / "live.db")
    seed_database(store, scale=0.2)
    fast, smart = get_llm(config.MODEL_FAST), get_llm(config.MODEL_SMART)
    return build_graph(store, fast, smart, make_sql_tool(store, fast), make_pandas_tool(fast))


@pytest.mark.parametrize("item", QUESTIONS, ids=[q["question"][:40] for q in QUESTIONS])
def test_live_question(graph, item):
    result = graph.invoke({"question": item["question"], "prompts": [], "errors": []})
    assert result["query_type"] == item["expect_tool"]
    assert len(result["data_slice"]) > 0
    assert result["insight"].finding.strip()
    assert result["insight"].recommendation.strip()
```
Note: these calls use Groq free-tier tokens. Each question makes about 3 LLM calls.

- [ ] **Step 3: Write `README.md`**

```markdown
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
```

- [ ] **Step 4: Run the offline suite and the live smoke test**

Run: `pytest -v`
Expected: everything passes; `test_live_smoke.py` is skipped if no key is in the environment. Because `config.py` loads `.env`, the live tests run automatically when the user has created `.env`.

If live tests run and fail, report which question and why. Do not weaken assertions to make them pass. An LLM-quality failure (for example `query_type` or empty slice) means a prompt needs tuning; propose the prompt change to the user.

- [ ] **Step 5: Commit**

```bash
git add tests/eval_questions.json tests/test_live_smoke.py README.md
git commit -m "test: add eval questions, live smoke test and README"
```

---

## Self-Review (spec coverage)

| Spec requirement | Task |
|---|---|
| SQLite via SQLAlchemy, `DataStore` interface, read-only SQL, table-name sanitising | 2 |
| Faker seed of 5 tables, reproducible | 3 |
| Versioned prompt files, 4 SKILL.md files | 4 |
| Groq models, friendly errors, missing-key message | 1, 5, 12 |
| Shared state, models, router | 6 |
| Excel/CSV/JSON upload → append or create → summary | 7 |
| SQL + Pandas retrieval, 200-row cap, retry | 8 |
| Analyst structured insight, skill + history injection | 9 |
| Chart JSON validated, one retry, table fallback, Plotly | 10, 12 |
| History window of 6, insight memory pinned | 10 |
| LangGraph wiring, follow-ups skip ingest | 11 |
| Streamlit upload/chat/chart/prompt expander | 12 |
| Eval questions, live smoke test | 13 |
| Key only in `.env`, not rendered | 1, 12 (prompt expander shows prompts only) |

Type names checked across tasks: `Insight`, `ChartConfig`, `make_*_node`, `render`, `extract_json`, `run_sql`, `schema_text`, `append_or_create` are used identically everywhere. Prompt-node labels are `ingest`, `analyst`, `visualization`.
