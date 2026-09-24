from config import HISTORY_TURNS, MEMORY_SLICE_ROWS
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
                    error_note = f"Your previous answer was invalid ({str(exc)[:300]}). Return only the JSON object described above."
            if chart is None:
                errors.append("Chart could not be generated.")

        history = (state.get("chat_history", []) + [
            {"question": state["question"], "finding": insight.finding}
        ])[-HISTORY_TURNS:]
        entry = {"question": state["question"], "insight": insight.model_dump(),
                 "chart": chart.model_dump() if chart else None,
                 "slice": data_slice.head(MEMORY_SLICE_ROWS).to_dict("records"), "approved": False}
        memory = list(state.get("insight_memory", [])) + [entry]
        return {"chart_config": chart, "chat_history": history, "insight_memory": memory,
                "memory_index": len(memory) - 1, "prompts": prompts, "errors": errors}

    return output
