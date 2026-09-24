# Naukri Personal Data Analyzer — Phase 2 Design

Builds on `2026-09-24-phase1-design.md` (implemented; 126 offline tests). Source of intent: `Naukri_Architecture_Document.docx` sections 4, 8, 9, 11.

## Goals
- Add the Judge Agent (orchestrator + quality judge), Guardrails AI input/output validation with a self-correction loop, Tiktoken token budgeting, a Groq-only fallback chain with a rate-limit manager, and .pptx export.
- Keep every node testable offline with the scripted `FakeLLM`. Nodes still communicate only through shared LangGraph state.

## Non-goals (unchanged or deferred)
- Together AI fallback: **dropped for now**. The chain is built so a third level can be added by config plus an API key later.
- The pandas route stays **disabled** (phase 1 security ruling). Questions about uploads run as read-only SQL.
- Phase 3: Tavily web RAG, embeddings/pgvector/FlashRank, real Postgres/Mongo, MLflow, LangSmith, auth/roles/audit trail, cross-session memory.
- Escalating a rejected spoke output to a different model. After the maximum rejections the best attempt is accepted with a warning.

## Assumptions to verify (Groq free tier, configurable in `config.py`)
| Model | Per-request budget (tokens per minute) | Per day |
|---|---|---|
| `llama-3.3-70b-versatile` | 12,000 | 100,000 |
| `llama-3.1-8b-instant` | 6,000 | 500,000 |
The effective per-request prompt budget is `min(context window, TPM limit)`, not the 131k window. Check current values in the Groq console before relying on them; they change.

## Graph (phase 2)
```
START → guard_input ──fail──→ END (structured error)
          │ pass
          ├─ no data_summary → ingest ─┐
          ├─ revision_note  → analyst   │
          └─ else ──────────────────────┴→ orchestrate → retrieve → judge_retrieval
judge_retrieval ──reject (≤2)──→ retrieve (with correction)
judge_retrieval ──accept──→ analyst → guard_analyst
guard_analyst ──fail (<3)──→ analyst (with validation error)
guard_analyst ──fail (=3)──→ degraded → END
guard_analyst ──pass──→ judge_analyst
judge_analyst ──reject (≤2)──→ analyst (with correction)
judge_analyst ──accept──→ output → END
```
New state fields: `guard_error`, `analyst_attempts`, `retrieval_rejections`, `analyst_rejections`, `judge_scores` (list), `revision_note`, `orchestration` (dict: `intent`, `retrieval_instruction`), `degraded` (bool).

## Components
### Token budget (`graph/budget.py`)
- `count_tokens(text) -> int` using tiktoken `cl100k_base` (approximate for Llama) with a 10% safety margin.
- `available_tokens(model, system, history, question) -> int` = effective limit − system − history − question.
- Analyst slice and ingest sample are fitted to the budget by dropping trailing rows/columns and appending a `# truncated` note. This replaces the phase 1 character caps (`MAX_PROMPT_CHARS`).

### Rate-limit manager (`graph/ratelimit.py`)
- SQLite table `llm_usage(ts_utc, model, tokens)` in the same DB file via `SQLiteStore`.
- `usage(model, window)` for the last 60 s and since UTC midnight; `can_use(model, est_tokens)` is false above 90% of TPM or TPD; `record(model, tokens)`.

### Fallback LLM (`graph/llm.py`)
- `FallbackLLM(chain, manager)` exposes `.invoke(prompt) -> response` (same `.content`) so existing nodes are unchanged. It estimates tokens, skips models the manager refuses, catches Groq rate-limit errors and tries the next model, records usage (provider `usage_metadata` when present, else the estimate), and exposes `.last_model`.
- Chains: smart = [70B, 8B]; fast = [8B]. If the whole chain is exhausted, raise a rate-limit error that `friendly_error` already maps.

### Guardrails (`graph/guards.py`)
- Uses the `guardrails-ai` library with custom validators; if it will not install cleanly on Python 3.13, plain-Python validators with the same `Guard.validate` shape are used and the fallback is stated in the commit.
- Input validators: `HRRelevance` (keyword/heuristic check, no extra LLM call), `NoInjection` (pattern list), `ColumnsExist` (identifiers the question names that look like columns but are absent from the schema), `Coherence` (minimum length/word count).
- Output validators: `NonEmptyInsight`, `NoPlaceholder` ("TBD", "lorem", "N/A", "[…]", "XX%"), `NumbersTraceable` (every number in finding/evidence appears in the slice CSV or derives trivially, e.g. percent of a slice value; tolerance for rounding).
- Chart JSON validation stays in `graph/viz.py` (phase 1).
- Failed input → structured error insight, no LLM calls. Failed analyst output → `guard_error` appended to the next analyst prompt via a new `$guard_error` variable; the third consecutive failure routes to `degraded`, which returns an insight explaining the answer could not be validated and shows the raw slice.

### Judge (`graph/nodes/orchestrate.py`, `judge.py`)
- **Orchestrator (smart model):** given schema, data summary and question, returns JSON `{"intent": ..., "retrieval_instruction": ...}`; `query_type` is always `sql`. Prompt `prompts/orchestrator.v1.txt`.
- **Judge (smart model):** prompt `prompts/judge.v1.txt`, input = stage name, question, and the output under review (retrieval: SQL, row count, first 10 rows; analyst: insight plus first 20 rows). Returns JSON scores 1–5 for `relevance`, `specificity`, `actionability` and a `correction` string. Accept iff all three ≥ 3, computed in code, not from a model-supplied verdict.
- Rejections re-dispatch the same spoke with the correction (`$correction` in `sql.v1.txt`, `$judge_correction` in `query.v1.txt`). At most 2 rejections per spoke; then accept and append `"Low-confidence answer: <reason>"` to `errors`. Scores are recorded in `judge_scores`.
- Config switches: `JUDGE_ENABLED` (default true), `JUDGE_RETRIEVAL` (default true). When disabled, the node is skipped through a conditional edge. A judge that itself fails to parse counts as accept with a warning.
- **User feedback:** each assistant message gets "Revise" (text box) and "Approve". Approve marks the insight as approved in `insight_memory`. Revise sets `revision_note` and re-enters at `analyst` with the existing `data_slice`, skipping retrieval.

### Insight memory and .pptx export (`export/deck.py`)
- `insight_memory` entries become `{"question", "insight", "chart", "slice"}` with `slice` as records capped at 50 rows, so a deck can be built without re-querying. (Phase 1 entries held only the insight dict; tests are updated.)
- `build_deck(entries) -> bytes` with python-pptx following `skills/slide-gen/SKILL.md`: one insight per slide; title = finding (≤ 12 words, truncated with an ellipsis); body = up to 3 evidence bullets (≤ 40 words total) and one recommendation line; chart is a **native PowerPoint chart** (bar, line, pie, scatter → `XL_CHART_TYPE`), not an image. Entries whose chart config is null get a slide with a small table instead.
- Sidebar: multiselect of insights, "Export slide deck" via `st.download_button`.

### UI additions (`app.py`)
Judge scores expander per message, revise/approve controls, degraded and low-confidence warnings, model-used caption (`last_model`), deck export.

## Prompts
New: `orchestrator.v1.txt`, `judge.v1.txt`. Changed (new variables, previous version files kept): `sql.v2.txt` (`$correction`), `query.v2.txt` (`$guard_error`, `$judge_correction`, `$revision_note`). Loader supports selecting a version per call; nodes reference `v2`.

## Errors
Groq errors keep phase 1 mapping. New: input guard rejection (message shown in chat), degraded response (shown as a warning insight), fallback exhaustion (rate-limit message), judge parse failure (accept with warning).

## Security
Unchanged from phase 1 (read-only SQL, keys only in `.env`, key redaction). `NoInjection` also runs on the question only, not on uploaded data (uploaded content reaches prompts only through the capped sample and slice, as before).

## Testing
- Unit: budget math and truncation, rate-limit manager (clock injected), FallbackLLM (rate-limit error on first model → second used; pre-emptive skip), each validator with pass/fail cases, judge scoring/accept logic, deck builder (slide count, title truncation, native chart present, table fallback).
- Graph tests with `FakeLLM`: input rejection short-circuits with zero LLM calls; guard failure loop (two failures then pass; three failures → degraded); judge reject-then-accept; two rejections → accepted with warning; revision skips retrieval and ingest; disabled judge skipped.
- Live (opt-in `-m live`, not run without the user's OK): extended eval questions, plus a rate-limit fallback smoke check.

## Success criteria
1. An off-topic or injection-style question is rejected without any LLM call and with a clear message.
2. Bad analyst output triggers up to 3 self-correction loops, then a degraded answer; no crash.
3. The Judge can reject an output and the corrected re-dispatch is visible in the prompts expander and `judge_scores`.
4. With the 70B model rate-limited (simulated), the same question completes on the 8B model and the UI shows which model answered.
5. "Revise" re-runs only the analyst on the existing slice.
6. Exporting selected insights downloads a valid .pptx (opens with python-pptx, one slide per insight, native charts).
7. All offline tests pass; live tests remain opt-in.

## Risks
- Judge and orchestrator add up to 4 smart-model calls per question; on the free tier this will often trip the pre-emptive 70B limit and run on 8B. `JUDGE_*` switches and the rate-limit manager exist for this reason.
- `guardrails-ai` is a heavy dependency with a changing API; plain-Python fallback is acceptable.
- Tiktoken counts are approximate for Llama; the 10% margin and the provider's `usage_metadata` correct for drift.
