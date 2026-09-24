from langgraph.graph import END, START, StateGraph

from graph.nodes.analyst import make_analyst_node
from graph.nodes.guard import (degraded_node, make_guard_analyst_node, make_guard_input_node,
                               route_after_guard, route_after_input)
from graph.nodes.ingest import make_ingest_node
from graph.nodes.output import make_output_node
from graph.nodes.retrieve import make_retrieve_node
from graph.state import AnalyzerState


def route_entry(state) -> str:
    return "retrieve" if state.get("data_summary") else "ingest"


def build_graph(store, fast_llm, smart_llm, sql_tool, pandas_tool=None):
    g = StateGraph(AnalyzerState)
    g.add_node("guard_input", make_guard_input_node(store))
    g.add_node("ingest", make_ingest_node(store, fast_llm))
    g.add_node("retrieve", make_retrieve_node(sql_tool, pandas_tool))
    g.add_node("analyst", make_analyst_node(smart_llm))
    g.add_node("guard_analyst", make_guard_analyst_node())
    g.add_node("degraded", degraded_node)
    g.add_node("output", make_output_node(fast_llm))
    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", route_after_input,
                            {"end": END, "ingest": "ingest", "retrieve": "retrieve"})
    g.add_edge("ingest", "retrieve")
    g.add_edge("retrieve", "analyst")
    g.add_edge("analyst", "guard_analyst")
    g.add_conditional_edges("guard_analyst", route_after_guard,
                            {"pass": "output", "retry": "analyst", "degraded": "degraded"})
    g.add_edge("degraded", END)
    g.add_edge("output", END)
    return g.compile()
