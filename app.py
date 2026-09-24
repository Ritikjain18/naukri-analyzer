import streamlit as st

import config
from data.seed import seed_database
from data.store import SQLiteStore
from graph.build_graph import build_graph
from graph.llm import friendly_error, get_llm
from graph.nodes.ingest import make_ingest_node
from graph.tools import make_sql_tool
from graph.viz import build_figure

st.set_page_config(page_title="Naukri Personal Data Analyzer", layout="wide")
st.title("Naukri Personal Data Analyzer")

try:
    config.get_api_key()
except config.MissingKeyError as exc:
    st.error(str(exc))
    st.stop()

PERSISTED = ("df", "table_name", "schema", "data_summary", "chat_history", "insight_memory")


@st.cache_resource
def get_runtime():
    store = SQLiteStore(config.DB_PATH)
    if not store.list_tables():
        seed_database(store)
    fast = get_llm(config.MODEL_FAST)
    smart = get_llm(config.MODEL_SMART)
    graph = build_graph(store, fast, smart, make_sql_tool(store, fast))
    return store, make_ingest_node(store, fast), graph


store, ingest_node, graph = get_runtime()
shared = st.session_state.setdefault("shared", {})
messages = st.session_state.setdefault("messages", [])


def render_prompts(prompts):
    with st.expander("Prompts sent to Groq"):
        for p in prompts:
            st.caption(p["node"])
            st.code(p["prompt"], language="text")


def render_assistant(m):
    ins = m["insight"]
    st.markdown(f"**{ins['finding']}**")
    for e in ins["evidence"]:
        st.markdown(f"- {e}")
    st.markdown(f"*Recommendation:* {ins['recommendation']}")
    if m["chart"] is not None:
        st.plotly_chart(build_figure(m["chart"], m["slice"]), width="stretch")
    for err in m["errors"]:
        st.caption(err)
    with st.expander("Data used"):
        st.dataframe(m["slice"])
    render_prompts(m["prompts"])


with st.sidebar:
    st.header("Data")
    st.caption("Tables: " + ", ".join(store.list_tables()))
    up = st.file_uploader("Upload Excel / CSV / JSON", type=["xlsx", "csv", "json"])
    if up is not None and st.button("Load file"):
        try:
            with st.spinner("Ingesting..."):
                update = ingest_node({**shared, "upload": {"name": up.name, "bytes": up.getvalue()}, "prompts": []})
            shared.update({k: update[k] for k in ("df", "table_name", "schema", "data_summary")})
            st.session_state["ingest_prompts"] = update["prompts"]
            st.success(f"{update['ingest_action'].title()} table `{update['table_name']}` ({len(update['df'])} rows)")
        except Exception as exc:
            st.error(friendly_error(exc))
    if shared.get("data_summary"):
        with st.expander("Data summary"):
            st.write(shared["data_summary"])
            for p in st.session_state.get("ingest_prompts", []):
                st.caption(p["node"])
                st.code(p["prompt"], language="text")

for m in messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
        elif "error" in m:
            st.error(m["error"])
        else:
            render_assistant(m)

question = st.chat_input("Ask about your talent data")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Analysing..."):
                result = graph.invoke({**shared, "question": question, "prompts": [], "errors": []})
            shared.update({k: result[k] for k in PERSISTED if k in result})
            msg = {"role": "assistant", "insight": result["insight"].model_dump(),
                   "chart": result.get("chart_config"), "slice": result["data_slice"],
                   "prompts": result["prompts"], "errors": result.get("errors", [])}
            messages.append(msg)
            render_assistant(msg)
        except Exception as exc:
            err = {"role": "assistant", "error": friendly_error(exc)}
            messages.append(err)
            st.error(err["error"])
