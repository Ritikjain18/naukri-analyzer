import logging
from datetime import UTC, date, datetime, time

import pandas as pd
import streamlit as st

import export.deck as deck
from accounts.permissions import can
from graph.textsafe import safe_text
from graph.viz import build_figure
from ui.analyze import audit_export, deck_failure_message
from ui.permissions_ui import guard

logger = logging.getLogger(__name__)

LIST_LIMIT = 200
EXPANDER_LIMIT = 25
MAX_VERSIONS = 4
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def _bound(d, end: bool):
    if not isinstance(d, date):
        return None
    return datetime.combine(d, time.max if end else time.min, tzinfo=UTC)


def _audit_view(ctx) -> None:
    if st.session_state.get("history_viewed"):
        return
    try:
        ctx.services.audit.record(ctx.user, "history_view", session_id=ctx.session_id)
        st.session_state["history_viewed"] = True
    except Exception as exc:
        logger.warning("History view audit failed: %s", type(exc).__name__)


def _pick_label(row: dict) -> str:
    return f"{row['id']}. {str(row['insight'].get('finding') or '(no finding)')[:60]}"


def _render_row(row: dict) -> None:
    ins = row["insight"]
    with st.expander(f"#{row['id']} {safe_text(row['question'])[:80]}"):
        st.markdown(f"**{safe_text(ins.get('finding', ''))}**")
        for e in ins.get("evidence", []):
            st.markdown(f"- {safe_text(e)}")
        st.markdown(f"*Recommendation:* {safe_text(ins.get('recommendation', ''))}")
        if row["chart"] is not None:
            try:
                st.plotly_chart(build_figure(row["chart"], row["slice"]), width="stretch")
            except Exception as exc:
                logger.warning("History chart failed: %s", type(exc).__name__)
                st.caption("Chart could not be drawn.")
        if row["slice"]:
            st.dataframe(pd.DataFrame(row["slice"]))


def _render_export(ctx, rows: list[dict]) -> None:
    if not rows or not guard(ctx, "export"):
        return
    labels = [_pick_label(r) for r in rows]
    picked = st.multiselect("Insights to export", labels, key="hist_deck_pick")
    chosen = [r for r, label in zip(rows, labels) if label in set(picked)]
    if not chosen:
        return
    try:
        data = deck.build_deck(chosen)
    except Exception as exc:
        st.warning(deck_failure_message(exc))
        return
    st.download_button("Export selected as slide deck", data=data, file_name="naukri_history_insights.pptx",
                       mime=PPTX_MIME, on_click=audit_export, args=(ctx, len(chosen), "history"))


def _history_tab(ctx) -> None:
    users = ctx.services.auth.list_users()
    by_name = {u.username: u.id for u in users}
    c1, c2, c3, c4 = st.columns(4)
    who = c1.selectbox("User", [None, *by_name], format_func=lambda n: "All users" if n is None else n,
                       key="hist_user")
    since = _bound(c2.date_input("From", value=None, key="hist_from"), False)
    until = _bound(c3.date_input("To", value=None, key="hist_to"), True)
    text = c4.text_input("Question contains", key="hist_text")
    rows = ctx.services.history.list(user_id=by_name.get(who), text=text or None, since=since, until=until,
                                     limit=LIST_LIMIT)
    st.dataframe(pd.DataFrame(
        [{"id": r["id"], "when": r["ts_utc"][:16], "user": r["username"], "question": r["question"],
          "finding": r["insight"].get("finding", ""), "approved": r["approved"], "degraded": r["degraded"]}
         for r in rows],
        columns=["id", "when", "user", "question", "finding", "approved", "degraded"]), hide_index=True)
    if not rows:
        st.info("No insights match these filters.")
        return
    if len(rows) > EXPANDER_LIMIT:
        st.caption(f"Showing details for the newest {EXPANDER_LIMIT} of {len(rows)} rows.")
    for row in rows[:EXPANDER_LIMIT]:
        _render_row(row)
    _render_export(ctx, rows)


def _comparative_tab(ctx) -> None:
    repeated = ctx.services.history.repeated_questions()
    if not repeated:
        st.info("No question has been asked in more than one session yet.")
        return
    labels = {f"{safe_text(g['sample'])} ({g['sessions']} sessions)": g for g in repeated}
    choice = st.selectbox("Question", list(labels), key="cmp_question")
    versions = ctx.services.history.by_question(labels[choice]["question_norm"])
    shown = versions[:MAX_VERSIONS]
    for col, v in zip(st.columns(len(shown)), shown):
        ins = v["insight"]
        with col:
            st.caption(f"{v['ts_utc'][:16]} | {safe_text(v['username'])} | session {(v.get('session_id') or '')[:8]}")
            st.markdown(f"**{safe_text(ins.get('finding', ''))}**")
            for e in ins.get("evidence", []):
                st.markdown(f"- {safe_text(e)}")
            st.markdown(f"*Recommendation:* {safe_text(ins.get('recommendation', ''))}")
    st.dataframe(pd.DataFrame(
        [{"when": v["ts_utc"][:16], "user": v["username"], "session": (v.get("session_id") or "")[:8],
          "finding": v["insight"].get("finding", ""), "recommendation": v["insight"].get("recommendation", "")}
         for v in versions]), hide_index=True)


def render_history(ctx) -> None:
    if not guard(ctx, "view_history"):
        return
    _audit_view(ctx)
    if can(ctx.user["role"], "view_comparative"):
        tab_hist, tab_cmp = st.tabs(["History", "Comparative report"])
        with tab_hist:
            _history_tab(ctx)
        with tab_cmp:
            _comparative_tab(ctx)
    else:
        _history_tab(ctx)
