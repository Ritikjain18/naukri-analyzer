import streamlit as st

import config
from accounts.services import build_services
from data.seed import seed_database
from data.store import SQLiteStore
from graph.build_graph import build_graph
from graph.llm import build_llm
from graph.nodes.ingest import make_ingest_node
from graph.ratelimit import RateLimitManager
from graph.tools import make_sql_tool
from ui import router
from ui.auth_ui import begin_session, render_account_box, render_bootstrap, render_login
from ui.context import Ctx

st.set_page_config(page_title="Naukri Personal Data Analyzer", layout="wide")
st.title("Naukri Personal Data Analyzer")

try:
    config.get_api_key()
except config.MissingKeyError as exc:
    st.error(str(exc))
    st.stop()


@st.cache_resource
def get_runtime():
    store = SQLiteStore(config.DB_PATH)
    if not store.list_tables():
        seed_database(store)
    manager = RateLimitManager(config.USAGE_DB_PATH)
    fast = build_llm([config.MODEL_FAST], manager)
    smart = build_llm([config.MODEL_SMART, config.MODEL_FAST], manager)
    graph = build_graph(store, fast, smart, make_sql_tool(store, fast))
    services = build_services(config.APP_DB_PATH, fast)
    return store, make_ingest_node(store, fast), graph, (fast, smart), services


store, ingest_node, graph, llms, services = get_runtime()

if services.auth.needs_bootstrap():
    render_bootstrap(services)
    st.stop()
auth = st.session_state.get("auth")
if not auth:
    render_login(services)
    st.stop()
if "session_id" not in st.session_state:
    begin_session(services, auth)

shared = st.session_state.setdefault("shared", {})
messages = st.session_state.setdefault("messages", [])
ctx = Ctx(store, ingest_node, graph, llms, services,
          {"id": auth["user_id"], "username": auth["username"], "role": auth["role"]},
          st.session_state["session_id"], shared, messages)
render_account_box(ctx)
page = st.sidebar.radio("Page", router.allowed_pages(ctx.user["role"]), key="page")
router.render_page(ctx, page)
