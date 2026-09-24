from config import MODEL_SMART, OUTPUT_RESERVE, PRIOR_CALLS_RESERVE
from graph.budget import count_tokens, fit_rows, request_budget
from graph.models import Insight
from graph.parsing import extract_json
from graph.prompts import render
from graph.skills import detect_domain, load_skill


ERROR_DETAIL_MAX = 300
ERROR_NOTE = "Your previous answer was invalid ({detail}). Return only the JSON object described above."


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
                    history=format_history(state.get("chat_history", [])), question=state["question"],
                    guard_error=(f"Your previous answer failed validation: {state['guard_error']}. "
                                 "Correct it using only numbers from the data slice.")
                    if state.get("guard_error") else "",
                    judge_correction=(f"A reviewer asked you to improve the previous answer: "
                                      f"{state['judge_correction']}") if state.get("judge_correction") else "",
                    revision_note=(f"The user asked for a revision: {state['revision_note']}")
                    if state.get("revision_note") else "")
        worst_note = ERROR_NOTE.format(detail="x" * ERROR_DETAIL_MAX)  # reserve the retry note
        overhead = count_tokens(render("query", "v2", slice="", error_note=worst_note, **base))
        # Size against the admissible request budget (rate-limit headroom), not the raw TPM, minus what the
        # orchestrator/judge calls of the same turn already used this minute. The 200-token floor is kept: if
        # even that overflows the budget the call goes ahead and FallbackLLM decides.
        slice_budget = max(200, request_budget(MODEL_SMART) - OUTPUT_RESERVE - PRIOR_CALLS_RESERVE - overhead)
        slice_text = fit_rows(data_slice, slice_budget)
        error_note = ""
        for _ in range(2):
            prompt = render("query", "v2", slice=slice_text, error_note=error_note, **base)
            prompts.append({"node": "analyst", "prompt": prompt})
            try:
                insight = Insight(**extract_json(llm.invoke(prompt).content))
                return {"insight": insight, "prompts": prompts}
            except ValueError as exc:
                error_note = ERROR_NOTE.format(detail=str(exc)[:ERROR_DETAIL_MAX])
        raise AnalysisError("The analyst could not produce a valid insight.")

    return analyst
