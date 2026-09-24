import json
from pathlib import Path

import pytest

from data.seed import seed_database
from graph.guards import check_input

CASES = json.loads((Path(__file__).parent / "guard_questions.json").read_text())


@pytest.fixture(scope="module")
def schema(tmp_path_factory):
    from data.store import SQLiteStore

    store = SQLiteStore(tmp_path_factory.mktemp("g") / "g.db")
    seed_database(store, scale=0.05)
    return store.schema_text()


@pytest.mark.parametrize("q", CASES["accept"])
def test_accepts_hr_questions(schema, q):
    assert check_input(q, schema).ok, q


@pytest.mark.parametrize("q", CASES["reject"])
def test_rejects_off_topic_and_injection(schema, q):
    assert not check_input(q, schema).ok, q
