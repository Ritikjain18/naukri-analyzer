import config
from graph.llm import RateLimitExhausted, is_rate_limit
from graph.nodes.analyst import format_history, prior_context_block
from graph.parsing import extract_json
from graph.prompts import render


def make_orchestrate_node(llm):
    def orchestrate(state):
        question = state["question"]
        fallback = {"intent": question, "retrieval_instruction": question}
        if not config.JUDGE_ENABLED:
            return {"orchestration": fallback, "query_type": "sql"}
        prompt = render("orchestrator", "v2", schema=state.get("schema", ""), summary=state.get("data_summary", ""),
                        history=format_history(state.get("chat_history", [])), question=question,
                        prior_context=prior_context_block(state))
        prompts = list(state.get("prompts", [])) + [{"node": "orchestrate", "prompt": prompt}]
        errors = list(state.get("errors", []))
        try:
            data = extract_json(llm.invoke(prompt).content)
            plan = {"intent": str(data.get("intent") or question),
                    "retrieval_instruction": str(data.get("retrieval_instruction") or question)}
        except ValueError:
            plan = fallback
        except Exception as exc:  # optional step: a rate limit must not kill the turn
            if not (isinstance(exc, RateLimitExhausted) or is_rate_limit(exc)):
                raise
            plan = fallback
            errors.append("Orchestrator skipped: rate limit.")
        return {"orchestration": plan, "query_type": "sql", "prompts": prompts, "errors": errors}

    return orchestrate
