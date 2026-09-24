from datetime import datetime, timedelta, timezone

import pytest

from graph.ratelimit import RateLimitManager

LIMITS = {"m": {"tpm": 1000, "tpd": 5000}}


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock(datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc))


@pytest.fixture
def mgr(tmp_path, clock):
    return RateLimitManager(tmp_path / "usage.db", limits=LIMITS, clock=clock)


def test_unknown_model_always_usable(mgr):
    assert mgr.can_use("other", 10**9)


def test_single_request_over_tpm_refused(mgr):
    assert not mgr.can_use("m", 1001)


def test_minute_window_and_headroom(mgr, clock):
    mgr.record("m", 800)
    assert mgr.used_last_minute("m") == 800
    assert not mgr.can_use("m", 101)   # 901 > 0.9 * 1000
    assert mgr.can_use("m", 100)       # 900 <= 900
    clock.now += timedelta(seconds=61)
    assert mgr.used_last_minute("m") == 0
    assert mgr.can_use("m", 900)


def test_daily_window_resets_at_utc_midnight(mgr, clock):
    for _ in range(4):
        mgr.record("m", 1000)
        clock.now += timedelta(minutes=5)
    assert mgr.used_today("m") == 4000
    assert not mgr.can_use("m", 600)   # 4600 > 4500
    clock.now = datetime(2026, 9, 25, 0, 0, 1, tzinfo=timezone.utc)
    assert mgr.used_today("m") == 0
    assert mgr.can_use("m", 600)


def test_usage_is_per_model_and_persists(tmp_path, clock):
    limits = {"a": {"tpm": 1000, "tpd": 5000}, "b": {"tpm": 1000, "tpd": 5000}}
    m1 = RateLimitManager(tmp_path / "u.db", limits=limits, clock=clock)
    m1.record("a", 500)
    m2 = RateLimitManager(tmp_path / "u.db", limits=limits, clock=clock)
    assert m2.used_last_minute("a") == 500 and m2.used_last_minute("b") == 0
