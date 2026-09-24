from langgraph.graph import END, START, StateGraph

from graph.nodes.analyst import make_analyst_node
from graph.nodes.ingest import make_ingest_node
from graph.nodes.output import make_output_node
from graph.nodes.retrieve import make_retrieve_node
from graph.state import AnalyzerState


def route_entry(state) -> str:
    return "retrieve" if state.get("data_summary") else "ingest"


def build_graph(store, fast_llm, smart_llm, sql_tool, pandas_tool):
    g = StateGraph(AnalyzerState)
    g.add_node("ingest", make_ingest_node(store, fast_llm))
    g.add_node("retrieve", make_retrieve_node(sql_tool, pandas_tool))
    g.add_node("analyst", make_analyst_node(smart_llm))
    g.add_node("output", make_output_node(fast_llm))
    g.add_conditional_edges(START, route_entry, {"ingest": "ingest", "retrieve": "retrieve"})
    g.add_edge("ingest", "retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_edge("analyst", "output")
    g.add_edge("output", END)
    return g.compile()
