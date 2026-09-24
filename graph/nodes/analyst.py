from config import MODEL_SMART, OUTPUT_RESERVE
from graph.budget import count_tokens, effective_limit, fit_rows
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
        base = dict(summary=state.get("data_summary", ""), skill=skill,
                    history=format_history(state.get("chat_history", [])), question=state["question"])
        overhead = count_tokens(render("query", slice="", error_note="", **base))
        slice_budget = max(200, effective_limit(MODEL_SMART) - OUTPUT_RESERVE - overhead)
        slice_text = fit_rows(data_slice, slice_budget)
        error_note = ""
        for _ in range(2):
            prompt = render("query", slice=slice_text, error_note=error_note, **base)
            prompts.append({"node": "analyst", "prompt": prompt})
            try:
                insight = Insight(**extract_json(llm.invoke(prompt).content))
                return {"insight": insight, "prompts": prompts}
            except ValueError as exc:
                error_note = f"Your previous answer was invalid ({exc}). Return only the JSON object described above."
        raise AnalysisError("The analyst could not produce a valid insight.")

    return analyst
