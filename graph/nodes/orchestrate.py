import config
from graph.nodes.analyst import format_history
from graph.parsing import extract_json
from graph.prompts import render


def make_orchestrate_node(llm):
    def orchestrate(state):
        question = state["question"]
        fallback = {"intent": question, "retrieval_instruction": question}
        if not config.JUDGE_ENABLED:
            return {"orchestration": fallback, "query_type": "sql"}
        prompt = render("orchestrator", schema=state.get("schema", ""), summary=state.get("data_summary", ""),
                        history=format_history(state.get("chat_history", [])), question=question)
        prompts = list(state.get("prompts", [])) + [{"node": "orchestrate", "prompt": prompt}]
        try:
            data = extract_json(llm.invoke(prompt).content)
            plan = {"intent": str(data.get("intent") or question),
                    "retrieval_instruction": str(data.get("retrieval_instruction") or question)}
        except ValueError:
            plan = fallback
        return {"orchestration": plan, "query_type": "sql", "prompts": prompts}

    return orchestrate
