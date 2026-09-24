import pytest

from data.store import SQLiteStore
from graph import budget


@pytest.fixture(autouse=True)
def _offline_token_estimator():
    budget.use_estimator(lambda text: len(text) // 4)
    yield
    budget.use_estimator(None)


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "test.db")


@pytest.fixture(autouse=True)
def _fast_scrypt(monkeypatch):
    from accounts import auth

    monkeypatch.setattr(auth, "SCRYPT_N", 16)


@pytest.fixture(autouse=True)
def _reset_prompt_overrides():
    from graph import prompts

    prompts.set_overrides({})
    yield
    prompts.set_overrides({})
