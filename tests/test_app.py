from streamlit.testing.v1 import AppTest


def test_app_shows_setup_error_without_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)


def test_graph_error_is_persisted_across_reruns(monkeypatch, tmp_path):
    import streamlit as st

    import config
    import graph.build_graph as bg

    class BoomGraph:
        def invoke(self, state):
            raise RuntimeError("429 rate limit")

    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(bg, "build_graph", lambda *a, **k: BoomGraph())
    st.cache_resource.clear()
    try:
        at = AppTest.from_file("../app.py").run(timeout=60)
        assert not at.exception
        at.chat_input[0].set_value("q").run(timeout=60)
        assert not at.exception
        expected = "Groq rate limit reached"
        assert sum(expected in e.value for e in at.error) == 1
        at.run(timeout=60)
        assert not at.exception
        assert sum(expected in e.value for e in at.error) == 1
        assert [m["role"] for m in at.session_state["messages"]] == ["user", "assistant"]
    finally:
        st.cache_resource.clear()


def test_app_does_not_construct_pandas_tool():
    from pathlib import Path

    assert "make_pandas_tool" not in (Path(__file__).parent.parent / "app.py").read_text()
