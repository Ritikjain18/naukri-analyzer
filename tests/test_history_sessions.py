from datetime import datetime, timedelta, timezone

import pytest

from accounts.auth import AuthService
from accounts.db import AppDB
from accounts.history import InsightHistory, normalise_question
from accounts.sessions import SessionTracker

INSIGHT = {"finding": "Eng leads.", "evidence": ["20% vs 10%"], "recommendation": "Invest."}


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    db = AppDB(tmp_path / "a.db", clock=clock)
    auth = AuthService(db)
    return {"clock": clock, "db": db, "hist": InsightHistory(db), "sess": SessionTracker(db),
            "alice": auth.create_user("alice", "correct horse", "analyst"),
            "bob": auth.create_user("bob", "correct horse", "manager")}


def test_normalise_question():
    assert normalise_question("  Which CATEGORY, has the best rate?? ") == "which category has the best rate"


def test_add_get_list_with_username_and_cap(env):
    rows = [{"a": i} for i in range(80)]
    hid = env["hist"].add(env["alice"], "s1", "Which category?", INSIGHT, {"type": "bar"}, rows)
    got = env["hist"].get(hid)
    assert got["insight"] == INSIGHT and got["chart"] == {"type": "bar"} and len(got["slice"]) == 50
    assert got["approved"] is False and got["degraded"] is False
    assert env["hist"].list()[0]["username"] == "alice"


def test_list_filters_and_order(env):
    h = env["hist"]
    h.add(env["alice"], "s1", "Bounce by source?", INSIGHT, None, [])
    env["clock"].now += timedelta(days=2)
    h.add(env["bob"], "s2", "Views by category?", INSIGHT, None, [], degraded=True)
    assert [r["question"] for r in h.list()] == ["Views by category?", "Bounce by source?"]
    assert [r["username"] for r in h.list(user_id=env["alice"])] == ["alice"]
    assert [r["question"] for r in h.list(text="BOUNCE")] == ["Bounce by source?"]
    cutoff = env["clock"].now - timedelta(days=1)
    assert len(h.list(since=cutoff)) == 1 and len(h.list(until=cutoff)) == 1
    assert h.list()[0]["degraded"] is True


def test_mark_approved(env):
    hid = env["hist"].add(env["alice"], "s1", "q one two", INSIGHT, None, [])
    env["hist"].mark_approved(hid)
    assert env["hist"].get(hid)["approved"] is True


def test_by_question_and_repeated_questions(env):
    h = env["hist"]
    h.add(env["alice"], "s1", "Which category is best?", INSIGHT, None, [])
    env["clock"].now += timedelta(hours=1)
    h.add(env["bob"], "s2", "which category is BEST", INSIGHT, None, [])
    h.add(env["alice"], "s2", "Only once here", INSIGHT, None, [])
    norm = normalise_question("Which category is best")
    assert len(h.by_question(norm)) == 2 and h.by_question(norm)[0]["username"] == "bob"
    rep = h.repeated_questions()
    assert [(r["question_norm"], r["sessions"], r["count"]) for r in rep] == [(norm, 2, 2)]
    assert rep[0]["sample"] == "which category is BEST"


def test_session_lifecycle_and_pending(env):
    s, hist = env["sess"], env["hist"]
    sid = s.start(env["alice"])
    assert len(sid) == 32 and s.get(sid)["question_count"] == 0
    hist.add(env["alice"], sid, "q one two", INSIGHT, None, [])
    s.touch(sid, questions=1)
    assert s.get(sid)["question_count"] == 1
    assert s.pending_for_user(env["alice"], 30) == []                 # active recently
    env["clock"].now += timedelta(minutes=31)
    assert s.pending_for_user(env["alice"], 30) == [sid]              # abandoned
    s.mark_summarised(sid)
    assert s.pending_for_user(env["alice"], 30) == []
    assert [e["question"] for e in s.entries_for(sid)] == ["q one two"]


def test_ended_session_is_pending_immediately_and_empty_ones_never(env):
    s = env["sess"]
    a, empty = s.start(env["alice"]), s.start(env["alice"])
    s.touch(a, questions=2)
    s.end(a)
    s.end(empty)
    assert s.pending_for_user(env["alice"], 30) == [a]
    first_end = s.get(a)["ended_at"]
    env["clock"].now += timedelta(minutes=5)
    s.end(a)
    assert s.get(a)["ended_at"] == first_end                          # ended_at is set once


def test_question_and_insight_are_redacted_before_storage(env):
    h = env["hist"]
    q = "Why is gsk_abcdef123456 used with password=hunter2 here?"
    insight = {"finding": "Key gsk_abcdef123456 leaked.", "evidence": ["password=hunter2 seen"],
               "recommendation": "Rotate."}
    hid = h.add(env["alice"], "s1", q, insight, {"title": "t gsk_abcdef123456"}, [{"c": "gsk_abcdef123456", "n": 1}])
    got = h.get(hid)
    for text in (got["question"], got["question_norm"], str(got["insight"]), str(got["chart"]), str(got["slice"])):
        assert "gsk_abcdef123456" not in text and "hunter2" not in text
    assert "[REDACTED]" in got["question"]
    assert got["slice"][0]["n"] == 1 and "c" in got["slice"][0]


def test_ordinary_text_is_unchanged(env):
    hid = env["hist"].add(env["alice"], "s1", "Which category?", INSIGHT, {"type": "bar"}, [{"a": 1}])
    got = env["hist"].get(hid)
    assert got["question"] == "Which category?" and got["insight"] == INSIGHT and got["slice"] == [{"a": 1}]


def test_list_text_escapes_like_wildcards(env):
    h = env["hist"]
    h.add(env["alice"], "s1", "Is 50% the rate?", INSIGHT, None, [])
    h.add(env["alice"], "s1", "Is 500 the rate?", INSIGHT, None, [])
    h.add(env["alice"], "s1", "a_b question", INSIGHT, None, [])
    h.add(env["alice"], "s1", "axb question", INSIGHT, None, [])
    assert [r["question"] for r in h.list(text="50%")] == ["Is 50% the rate?"]
    assert [r["question"] for r in h.list(text="a_b")] == ["a_b question"]


def test_unknown_session_raises(env):
    from accounts.auth import AccountError
    s = env["sess"]
    for call in (lambda: s.touch("nope", 1), lambda: s.end("nope"), lambda: s.mark_summarised("nope")):
        with pytest.raises(AccountError) as e:
            call()
        assert str(e.value) == "Unknown session."
