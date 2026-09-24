import pytest

from data.store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(tmp_path / "test.db")
