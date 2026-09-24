from config import ROW_CAP
from graph.router import choose_tool


def make_retrieve_node(sql_tool, pandas_tool):
    def retrieve(state):
        df = state.get("df")
        question = state["question"]
        tool = choose_tool(question, None if df is None else list(df.columns))
        if tool == "pandas":
            data_slice = pandas_tool(question, df)
        else:
            data_slice = sql_tool(question, state["schema"])
        return {"query_type": tool, "data_slice": data_slice.head(ROW_CAP)}

    return retrieve
