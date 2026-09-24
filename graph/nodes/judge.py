import config
from graph.llm import RateLimitExhausted, is_rate_limit
from graph.parsing import extract_json
from graph.prompts import render

CRITERIA = ("relevance", "specificity", "actionability")
DEFAULT_CORRECTION = "Improve relevance, specificity and actionability."


def _generated_sql(state) -> str:
    for entry in reversed(state.get("prompts", [])):
        if entry.get("node") == "retrieve-sql" and "--- model output ---\n" in entry["prompt"]:
            return entry["prompt"].split("--- model output ---\n", 1)[1]
    return "(unknown)"


def _output_under_review(state, stage: str) -> str:
    data_slice = state["data_slice"]
    if stage == "retrieval":
        return (f"SQL:\n{_generated_sql(state)}\nRows returned: {len(data_slice)}\n"
                f"First rows:\n{data_slice.head(10).to_csv(index=False)}")
    ins = state["insight"]
    evidence = "\n".join(f"- {e}" for e in ins.evidence)
    return (f"Finding: {ins.finding}\nEvidence:\n{evidence}\nRecommendation: {ins.recommendation}\n\n"
            f"Data (first 20 rows):\n{data_slice.head(20).to_csv(index=False)}")


def make_judge_node(llm, stage: str):
    counter = "retrieval_rejections" if stage == "retrieval" else "analyst_rejections"
    correction_key = "retrieval_correction" if stage == "retrieval" else "judge_correction"

    def judge(state):
        if (not config.JUDGE_ENABLED or (stage == "retrieval" and not config.JUDGE_RETRIEVAL)
                or state["data_slice"].empty):
            return {"judge_verdict": "accept"}
        prompt = render("judge", stage=stage, question=state["question"],
                        output=_output_under_review(state, stage))
        prompts = list(state.get("prompts", [])) + [{"node": f"judge-{stage}", "prompt": prompt}]
        errors = list(state.get("errors", []))
        try:
            data = extract_json(llm.invoke(prompt).content)
            scores = {k: max(1, min(5, int(data[k]))) for k in CRITERIA}
            correction = str(data.get("correction", "")).strip()
        except (ValueError, KeyError, TypeError):
            errors.append("Judge could not score this answer.")
            return {"judge_verdict": "accept", "prompts": prompts, "errors": errors}
        except Exception as exc:  # optional step: a rate limit must not kill the turn
            if not (isinstance(exc, RateLimitExhausted) or is_rate_limit(exc)):
                raise
            errors.append("Judge skipped: rate limit.")
            return {"judge_verdict": "accept", "prompts": prompts, "errors": errors}

        accepted = min(scores.values()) >= config.JUDGE_MIN_SCORE
        rejections = state.get(counter, 0)
        update = {"judge_scores": list(state.get("judge_scores", [])) + [{"stage": stage, **scores, "accepted": accepted}],
                  "prompts": prompts}
        if accepted:
            update.update(judge_verdict="accept", errors=errors, **{correction_key: ""})
        elif rejections < config.MAX_REJECTIONS:
            update.update(judge_verdict="reject", errors=errors,
                          **{counter: rejections + 1, correction_key: correction or DEFAULT_CORRECTION})
        else:
            errors.append(f"Low-confidence answer: {correction or 'the judge scored it below the threshold'}")
            update.update(judge_verdict="accept", errors=errors, **{correction_key: ""})
        return update

    return judge


def route_after_judge(state) -> str:
    return "reject" if state.get("judge_verdict") == "reject" else "accept"
