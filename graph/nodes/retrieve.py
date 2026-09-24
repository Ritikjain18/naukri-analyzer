from config import ROW_CAP
from graph.nodes.analyst import format_history
from graph.router import choose_tool


def make_retrieve_node(sql_tool, pandas_tool):
    def retrieve(state):
        df = state.get("df")
        question = state["question"]
        history = format_history(state.get("chat_history", []))
        trace: list = []
        tool = choose_tool(question, None if df is None else list(df.columns))
        if tool == "pandas":
            data_slice = pandas_tool(question, df, history=history, trace=trace)
        else:
            data_slice = sql_tool(question, state["schema"], history=history, trace=trace)
        return {"query_type": tool, "data_slice": data_slice.head(ROW_CAP),
                "prompts": list(state.get("prompts", [])) + trace}

    return retrieve
