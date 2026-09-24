from types import SimpleNamespace

import pytest

from graph.llm import FallbackLLM, RateLimitExhausted, build_llm, friendly_error, is_rate_limit
from graph.ratelimit import RateLimitManager

LIMITS = {"big": {"tpm": 10000, "tpd": 100000}, "small": {"tpm": 10000, "tpd": 100000}}


class RateLimitError(Exception):
    pass


class Provider:
    def __init__(self, reply="ok", raises=None, total_tokens=None):
        self.reply, self.raises, self.total_tokens, self.prompts = reply, raises, total_tokens, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.raises:
            raise self.raises
        resp = SimpleNamespace(content=self.reply)
        if self.total_tokens is not None:
            resp.usage_metadata = {"total_tokens": self.total_tokens}
        return resp


@pytest.fixture
def mgr(tmp_path):
    return RateLimitManager(tmp_path / "u.db", limits=LIMITS)


def test_primary_used_and_usage_recorded(mgr):
    big, small = Provider("A", total_tokens=321), Provider("B")
    llm = FallbackLLM([("big", big), ("small", small)], mgr)
    assert llm.invoke("hello").content == "A"
    assert llm.last_model == "big" and llm.used_models() == ["big"]
    assert mgr.used_last_minute("big") == 321 and small.prompts == []


def test_rate_limit_error_falls_back(mgr):
    big = Provider(raises=RateLimitError("429 rate limit"))
    small = Provider("B")
    llm = FallbackLLM([("big", big), ("small", small)], mgr)
    assert llm.invoke("hello").content == "B"
    assert llm.used_models() == ["small"]


def test_preemptive_skip_when_manager_refuses(mgr):
    mgr.record("big", 9500)
    big, small = Provider("A"), Provider("B")
    llm = FallbackLLM([("big", big), ("small", small)], mgr)
    assert llm.invoke("hello").content == "B"
    assert big.prompts == []


def test_all_exhausted_raises(mgr):
    mgr.record("big", 9500)
    mgr.record("small", 9500)
    llm = FallbackLLM([("big", Provider()), ("small", Provider())], mgr)
    with pytest.raises(RateLimitExhausted):
        llm.invoke("hello")


def test_all_providers_rate_limited_raises_last_error(mgr):
    llm = FallbackLLM([("big", Provider(raises=RateLimitError("429"))),
                       ("small", Provider(raises=RateLimitError("429")))], mgr)
    with pytest.raises(RateLimitError):
        llm.invoke("hello")


def test_other_errors_propagate_without_fallback(mgr):
    small = Provider("B")
    llm = FallbackLLM([("big", Provider(raises=ValueError("boom"))), ("small", small)], mgr)
    with pytest.raises(ValueError):
        llm.invoke("hello")
    assert small.prompts == []


def test_reset_clears_used_models(mgr):
    llm = FallbackLLM([("big", Provider())], mgr)
    llm.invoke("x")
    llm.reset()
    assert llm.used_models() == []


def test_is_rate_limit_and_friendly_error():
    assert is_rate_limit(RateLimitError("x"))
    assert is_rate_limit(RuntimeError("Rate limit reached"))
    assert not is_rate_limit(ValueError("nope"))
    assert "rate limit" in friendly_error(RateLimitExhausted("all models limited")).lower()


def test_build_llm_makes_chain(monkeypatch, mgr):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    llm = build_llm(["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], mgr)
    assert [m for m, _ in llm.chain] == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def test_prompt_too_large_when_no_model_could_ever_take_it(mgr):
    from graph.llm import TOO_LARGE_MSG, PromptTooLarge

    llm = FallbackLLM([("big", Provider()), ("small", Provider())], mgr)
    with pytest.raises(PromptTooLarge) as info:
        llm.invoke("x" * 4 * 9500)   # ~9500 tokens + reserve > 0.9 * 10000 for every model
    assert isinstance(info.value, RateLimitExhausted)
    assert friendly_error(info.value) == TOO_LARGE_MSG


def test_usage_block_is_not_prompt_too_large(mgr):
    from graph.llm import PromptTooLarge

    mgr.record("big", 9500)
    mgr.record("small", 9500)
    llm = FallbackLLM([("big", Provider()), ("small", Provider())], mgr)
    with pytest.raises(RateLimitExhausted) as info:
        llm.invoke("hello")
    assert not isinstance(info.value, PromptTooLarge) and info.value.daily is False
    assert "wait a minute" in friendly_error(info.value).lower()


def test_daily_exhaustion_reports_daily_message(tmp_path):
    limits = {"big": {"tpm": 10000, "tpd": 3000}, "small": {"tpm": 10000, "tpd": 3000}}
    mgr = RateLimitManager(tmp_path / "d.db", limits=limits)
    mgr.record("big", 2000)
    mgr.record("small", 2000)
    llm = FallbackLLM([("big", Provider()), ("small", Provider())], mgr)
    with pytest.raises(RateLimitExhausted) as info:
        llm.invoke("hello")
    assert info.value.daily is True and "daily" in str(info.value)
    assert friendly_error(info.value) == "Groq daily token limit reached for today. Try again after 00:00 UTC."
