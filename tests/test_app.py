from streamlit.testing.v1 import AppTest


def test_app_shows_setup_error_without_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)
