import logging

import pandas as pd
import streamlit as st

import export.deck as deck
from export.deck import entry_label, select_entries
from graph.llm import friendly_error
from graph.state import new_turn
from graph.textsafe import safe_text
from graph.trace import generated_sql
from graph.viz import build_figure
from ui.auth_ui import flash
from ui.permissions_ui import guard

logger = logging.getLogger(__name__)
RECORD_WARNING = "Could not record this answer in the history."

PERSISTED = ("df", "table_name", "schema", "data_summary", "chat_history", "insight_memory")


def _safely(step) -> bool:
    """Run a bookkeeping write; never let it break the user's answer. Logs only the exception type."""
    try:
        step()
        return True
    except Exception as exc:
        logger.warning("Recording failed: %s", type(exc).__name__)
        return False


def record_turn(ctx, m, result, question: str, action: str) -> bool:
    """Audit, history and session bookkeeping for a finished turn. Returns False if any write failed."""
    s, user, sid = ctx.services, ctx.user, ctx.session_id
    rejected = bool(result.get("guard_rejected"))
    degraded = bool(result.get("degraded"))

    def audit():
        s.audit.record(user, action, session_id=sid, question=question, query_type=result.get("query_type"),
                       sql=generated_sql(result.get("prompts", [])), models=m["models"],
                       guard_rejected=rejected, degraded=degraded,
                       judge=[j["accepted"] for j in result.get("judge_scores", [])])

    def history():
        if not rejected:
            m["history_id"] = s.history.add(user["id"], sid, question, m["insight"], m["chart"],
                                            m["slice"].to_dict("records") if hasattr(m["slice"], "to_dict")
                                            else m["slice"], degraded=degraded)

    def touch():
        if action == "question":
            s.sessions.touch(sid, questions=1)

    return all([_safely(audit), _safely(history), _safely(touch)])


def audit_export(ctx, count: int, source: str = "analyze") -> None:
    """on_click callback for download buttons: audits the export. Must not rerun; swallows its own errors."""
    _safely(lambda: ctx.services.audit.record(ctx.user, "export", session_id=ctx.session_id, insights=count, source=source))


def audit_upload(ctx, name: str, table: str, action: str, rows: int) -> None:
    ctx.services.audit.record(ctx.user, "upload", session_id=ctx.session_id, file=name, table=table, action=action, rows=rows)


def run_turn(ctx, question: str, action: str = "question", **extra) -> None:
    graph, llms, shared, messages = ctx.graph, ctx.llms, ctx.shared, ctx.messages
    try:
        for llm in llms:
            llm.reset()
        with st.spinner("Analysing..."):
            result = graph.invoke(new_turn(shared, question, **extra), config={"recursion_limit": 60})
        shared.update({k: result[k] for k in PERSISTED if k in result})
        keep = not result.get("degraded") and not result.get("guard_rejected")
        m = {
            "role": "assistant", "question": question, "insight": result["insight"].model_dump(),
            "chart": result.get("chart_config"), "slice": result["data_slice"],
            "prompts": result["prompts"], "errors": result.get("errors", []),
            "memory_index": result.get("memory_index") if keep else None,
            "judge_scores": result.get("judge_scores", []), "degraded": bool(result.get("degraded")),
            "models": sorted({mod for llm in llms for mod in llm.used_models()}),
        }
        messages.append(m)
        if not record_turn(ctx, m, result, question, action):
            flash("warning", RECORD_WARNING)
    except Exception as exc:
        messages.append({"role": "assistant", "error": friendly_error(exc)})
    st.rerun()


def ask(ctx, question: str, action: str = "question", shown: str | None = None, **extra) -> bool:
    """Permission-checked entry point for a question or revision. A refused question never enters the chat."""
    if not guard(ctx, "ask"):
        return False
    ctx.messages.append({"role": "user", "content": shown or question})
    run_turn(ctx, question, action, **extra)
    return True


def render_prompts(prompts):
    with st.expander("Prompts sent to Groq"):
        for p in prompts:
            st.caption(p["node"])
            st.code(p["prompt"], language="text")


def render_feedback(ctx, m, i) -> None:
    shared = ctx.shared
    idx = m.get("memory_index")
    if idx is None:
        return
    entry = shared["insight_memory"][idx]
    left, right = st.columns([1, 4])
    with left:
        if entry.get("approved"):
            st.caption("✅ Approved")
        elif st.button("Approve", key=f"approve_{i}"):
            entry["approved"] = True
            hid = m.get("history_id")
            svc = ctx.services

            def record():
                if hid is not None:
                    svc.history.mark_approved(hid)
                svc.audit.record(ctx.user, "approve", session_id=ctx.session_id, history_id=hid)

            if not _safely(record):
                flash("warning", RECORD_WARNING)
            st.rerun()
    with right:
        note = st.text_input("Ask for a revision", key=f"note_{i}", label_visibility="collapsed",
                             placeholder="Ask for a revision, e.g. focus on Mumbai")
        if st.button("Revise", key=f"revise_{i}") and note.strip():
            st.session_state["pending_revision"] = {
                "question": m["question"], "slice": m["slice"], "note": note.strip(),
                "finding": m["insight"]["finding"]}
            st.rerun()


def render_assistant(ctx, m, i):
    ins = m["insight"]
    st.markdown(f"**{safe_text(ins['finding'])}**")
    for e in ins["evidence"]:
        st.markdown(f"- {safe_text(e)}")
    st.markdown(f"*Recommendation:* {safe_text(ins['recommendation'])}")
    if m.get("degraded"):
        st.warning("Answer failed validation — showing the raw data instead.")
    if m["chart"] is not None:
        st.plotly_chart(build_figure(m["chart"], m["slice"]), width="stretch")
    for err in m["errors"]:
        st.caption(safe_text(err))
    if m.get("models"):
        st.caption("Answered by: " + ", ".join(m["models"]))
    with st.expander("Data used"):
        st.dataframe(m["slice"])
    if m.get("judge_scores"):
        with st.expander("Judge scores"):
            st.dataframe(pd.DataFrame(m["judge_scores"]))
    render_prompts(m["prompts"])
    render_feedback(ctx, m, i)


def render_analyze(ctx) -> None:
    store, ingest_node, shared, messages = ctx.store, ctx.ingest_node, ctx.shared, ctx.messages
    with st.sidebar:
        st.header("Data")
        st.caption("Tables: " + ", ".join(safe_text(t) for t in store.list_tables()))
        up = st.file_uploader("Upload Excel / CSV / JSON", type=["xlsx", "csv", "json"])
        if up is not None and st.button("Load file") and guard(ctx, "upload"):
            try:
                with st.spinner("Ingesting..."):
                    update = ingest_node({**shared, "upload": {"name": up.name, "bytes": up.getvalue()}, "prompts": []})
                shared.update({k: update[k] for k in ("df", "table_name", "schema", "data_summary")})
                st.session_state["ingest_prompts"] = update["prompts"]
                if not _safely(lambda: audit_upload(ctx, up.name, update["table_name"],
                                                    update["ingest_action"], len(update["df"]))):
                    st.warning("Could not record this upload in the audit log.")
                st.success(f"{update['ingest_action'].title()} table `{update['table_name']}` ({len(update['df'])} rows)")
            except Exception as exc:
                st.error(friendly_error(exc))
        if shared.get("data_summary"):
            with st.expander("Data summary"):
                st.markdown(safe_text(shared["data_summary"]))
                for p in st.session_state.get("ingest_prompts", []):
                    st.caption(p["node"])
                    st.code(p["prompt"], language="text")

        memory = shared.get("insight_memory", [])
        if memory and guard(ctx, "export"):
            st.header("Slide deck")
            labels = [entry_label(i, e) for i, e in enumerate(memory)]
            approved = [entry_label(i, e) for i, e in enumerate(memory) if e.get("approved")]
            picked = st.multiselect("Insights to export", labels, default=approved or labels)
            chosen = select_entries(memory, picked)
            if chosen:
                try:
                    deck_bytes = deck.build_deck(chosen)
                except Exception as exc:
                    st.warning("Could not build the slide deck: " + friendly_error(exc))
                else:
                    st.download_button("Export slide deck", data=deck_bytes, file_name="naukri_insights.pptx",
                                       mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                                       on_click=audit_export, args=(ctx, len(chosen)))

    for i, m in enumerate(messages):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["content"])
            elif "error" in m:
                st.error(m["error"])
            else:
                render_assistant(ctx, m, i)

    pending = st.session_state.pop("pending_revision", None)
    question = st.chat_input("Ask about your talent data")
    if pending:
        ask(ctx, pending["question"], "revise", shown=f"Revise: {pending['note']}", data_slice=pending["slice"],
            revision_note=f"{pending['note']} (previous finding: {pending['finding']})")
    elif question:
        ask(ctx, question)
