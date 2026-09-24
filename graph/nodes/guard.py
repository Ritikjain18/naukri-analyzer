import pandas as pd

from config import MAX_GUARD_FAILURES
from graph.guardrails_adapter import run_input_guard, run_output_guard
from graph.models import Insight


def make_guard_input_node(store):
    def guard_input(state):
        revision = state.get("revision_note", "")
        text = revision or state["question"]
        schema = state.get("schema") or store.schema_text()
        result = run_input_guard(text, schema, revision=bool(revision))
        if result.ok:
            return {"guard_rejected": False}
        return {
            "guard_rejected": True,
            "insight": Insight(
                finding="I can't answer that yet: " + " ".join(result.reasons),
                evidence=[],
                recommendation=("Ask about job postings, the hiring funnel, traffic or recruiter activity, "
                                "using the table and column names shown in the sidebar."),
            ),
            "data_slice": pd.DataFrame(),
            "chart_config": None,
            "errors": list(state.get("errors", [])) + ["Question rejected by input guardrails."],
        }

    return guard_input


def route_after_input(state) -> str:
    from graph.build_graph import route_entry  # lazy: build_graph imports this module

    if state.get("guard_rejected"):
        return "end"
    return route_entry(state)


def make_guard_analyst_node():
    def guard_analyst(state):
        data_slice = state["data_slice"]
        if data_slice.empty:
            return {"guard_error": "", "guard_failures": 0}
        result = run_output_guard(state["insight"], data_slice, state.get("question", ""))
        if result.ok:
            return {"guard_error": "", "guard_failures": 0}
        return {"guard_error": "; ".join(result.reasons),
                "guard_failures": state.get("guard_failures", 0) + 1}

    return guard_analyst


def route_after_guard(state) -> str:
    if not state.get("guard_error"):
        return "pass"
    return "retry" if state.get("guard_failures", 0) < MAX_GUARD_FAILURES else "degraded"


def degraded_node(state):
    data_slice = state["data_slice"]
    rows = [", ".join(f"{c}={v}" for c, v in row.items()) for row in data_slice.head(3).to_dict("records")]
    return {
        "degraded": True,
        "chart_config": None,
        "insight": Insight(
            finding="I couldn't produce a validated answer for this question.",
            evidence=rows,
            recommendation="Try a narrower question or check the data slice below.",
        ),
        "errors": list(state.get("errors", [])) + [f"Answer failed validation: {state.get('guard_error', '')}"],
    }
