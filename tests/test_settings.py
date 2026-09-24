import logging

import pytest

import config
from accounts.auth import AccountError
from accounts.db import AppDB
from accounts.settings import REQUIRED_VARS, AppSettings, PromptSettings, placeholders
from graph import prompts
from graph.nodes.judge import make_judge_node
from tests.fakes import FakeLLM


def write(dirpath, name, version, text):
    (dirpath / f"{name}.{version}.txt").write_text(text)


@pytest.fixture
def pdir(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    write(d, "sql", "v1", "old $schema $question")
    write(d, "sql", "v2", "$schema $limit $history $question $correction $error_note")
    write(d, "sql", "v10", "$schema $limit $history $question $correction $error_note more")
    write(d, "sql", "v3", "$schema $limit $history $question $correction $error_note")
    return d


@pytest.fixture
def ps(tmp_path, pdir):
    return PromptSettings(AppDB(tmp_path / "a.db"), prompts_dir=pdir)


def test_placeholders():
    assert placeholders("a $x and ${y} and $z_1 $$") == {"x", "y", "z_1"}
    assert placeholders("$$x and $$$y") == {"y"}
    assert placeholders("cost $5 and ${y}") == {"y"}


def test_versions_sorted_numerically_and_compatibility(ps):
    assert ps.versions("sql") == ["v1", "v2", "v3", "v10"]
    assert ps.compatible_versions("sql") == ["v2", "v3", "v10"]
    assert ps.versions("nothing") == []


def test_set_active_validates(ps):
    ps.set_active("sql", "v2", user_id=1)
    assert ps.active("sql") == "v2" and ps.overrides() == {"sql": "v2"}
    with pytest.raises(AccountError, match="missing"):
        ps.set_active("sql", "v1", user_id=1)          # lacks $limit etc
    with pytest.raises(AccountError, match="not found"):
        ps.set_active("sql", "v99", user_id=1)
    with pytest.raises(AccountError, match="Unknown prompt"):
        ps.set_active("bogus", "v1", user_id=1)
    assert ps.active("sql") == "v2"


@pytest.mark.parametrize("name,version", [("../x", "v1"), ("sql", "v1/../../x"), ("sql", "../v1"),
                                          ("sql", "v1\n"), ("sql", ""), ("sql", "V1")])
def test_set_active_rejects_traversal_names_and_versions(ps, name, version):
    with pytest.raises(AccountError):
        ps.set_active(name, version, user_id=1)
    assert ps.overrides() == {}


def test_clear_and_apply_sets_render_override(ps, pdir, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    ps.set_active("sql", "v3", user_id=1)
    ps.apply()
    assert prompts.get_overrides() == {"sql": "v3"}
    assert prompts.effective_version("sql", "v2") == "v3"
    out = prompts.render("sql", "v2", schema="S", limit=5, history="h", question="Q", correction="", error_note="")
    assert out.startswith("S 5")
    ps.clear("sql")
    ps.apply()
    assert prompts.effective_version("sql", "v2") == "v2"


def test_effective_version_ignores_override_without_file(monkeypatch, pdir):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    prompts.set_overrides({"sql": "v77"})
    assert prompts.effective_version("sql", "v2") == "v2"


def test_required_vars_cover_real_prompt_files(tmp_path):
    real = PromptSettings(AppDB(tmp_path / "a.db"))
    for name, needed in REQUIRED_VARS.items():
        if name == "session_summary":
            pytest.skip("session_summary.v1 arrives in Task 7")
        assert real.compatible_versions(name), f"no compatible version for {name}"
        assert needed


def test_app_settings_validate_store_and_apply(tmp_path, monkeypatch):
    s = AppSettings(AppDB(tmp_path / "a.db"))
    assert s.get("JUDGE_ENABLED") is config.JUDGE_ENABLED
    s.set("JUDGE_ENABLED", False, user_id=1)
    s.set("JUDGE_MIN_SCORE", 4, user_id=1)
    with pytest.raises(AccountError):
        s.set("JUDGE_MIN_SCORE", 9, user_id=1)
    with pytest.raises(AccountError):
        s.set("JUDGE_ENABLED", "yes", user_id=1)
    with pytest.raises(AccountError, match="Unknown setting"):
        s.set("MODEL_SMART", "x", user_id=1)
    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    monkeypatch.setattr(config, "JUDGE_MIN_SCORE", 3)
    s.apply(config)
    assert config.JUDGE_ENABLED is False and config.JUDGE_MIN_SCORE == 4
    assert s.all() == {"JUDGE_ENABLED": False, "JUDGE_RETRIEVAL": config.JUDGE_RETRIEVAL, "JUDGE_MIN_SCORE": 4}


def test_apply_takes_effect_in_judge_node(tmp_path, monkeypatch):
    import pandas as pd
    from graph.models import Insight

    monkeypatch.setattr(config, "JUDGE_ENABLED", True)
    monkeypatch.setattr(config, "JUDGE_MIN_SCORE", 3)
    state = {"question": "q", "data_slice": pd.DataFrame({"a": [1]}),
             "insight": Insight(finding="f", evidence=["e"], recommendation="r"),
             "prompts": [], "errors": [], "judge_scores": []}
    llm = FakeLLM(['{"relevance":4,"specificity":4,"actionability":4,"correction":""}'])
    node = make_judge_node(llm, "analyst")
    s = AppSettings(AppDB(tmp_path / "a.db"))
    s.set("JUDGE_ENABLED", False, user_id=1)
    s.apply()
    assert node(state) == {"judge_verdict": "accept"} and llm.prompts == []   # disabled: no LLM call
    s.set("JUDGE_ENABLED", True, user_id=1)
    s.apply()
    out = node(state)
    assert len(llm.prompts) == 1 and out["judge_scores"][-1]["accepted"] is True


def test_unicode_digit_versions_rejected(ps, pdir):
    write(pdir, "sql", "v\u0661", "$schema $limit $history $question $correction $error_note")
    assert "v\u0661" not in ps.versions("sql")
    with pytest.raises(AccountError):
        ps.set_active("sql", "v\u0661", user_id=1)


def _render_sql():
    return prompts.render("sql", "v3", schema="S", limit=77, history="h", question="Q",
                          correction="C", error_note="E")


def test_override_edited_incompatible_is_not_applied(ps, pdir, monkeypatch, caplog):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    ps.set_active("sql", "v2", user_id=1)
    write(pdir, "sql", "v2", "$schema $history $question $correction $error_note")   # dropped $limit
    with caplog.at_level(logging.WARNING):
        ps.apply()
    assert prompts.get_overrides() == {}
    assert "sql" in caplog.text and "v2" in caplog.text
    out = _render_sql()
    assert "77" in out and "$limit" not in out


def test_effective_version_guards_incompatible_file_directly(pdir, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    write(pdir, "sql", "v2", "$schema $question")
    prompts.set_overrides({"sql": "v2"}, REQUIRED_VARS)
    assert prompts.effective_version("sql", "v3") == "v3"
    assert "77" in _render_sql()


def test_override_with_deleted_file_ignored(ps, pdir, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    ps.set_active("sql", "v2", user_id=1)
    (pdir / "sql.v2.txt").unlink()
    ps.apply()
    assert prompts.get_overrides() == {}
    assert prompts.effective_version("sql", "v3") == "v3"


def test_compatible_override_applies(ps, pdir, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPTS_DIR", pdir)
    write(pdir, "sql", "v2", "MARK $schema $limit $history $question $correction $error_note")
    ps.set_active("sql", "v2", user_id=1)
    ps.apply()
    assert prompts.effective_version("sql", "v3") == "v2"
    assert _render_sql().startswith("MARK")


@pytest.mark.parametrize("key,raw", [
    ("JUDGE_MIN_SCORE", "not json{"), ("JUDGE_MIN_SCORE", '"x"'), ("JUDGE_MIN_SCORE", "9"),
    ("JUDGE_MIN_SCORE", "0"), ("JUDGE_MIN_SCORE", "true"), ("JUDGE_ENABLED", '"yes"'),
    ("JUDGE_ENABLED", "1"), ("JUDGE_RETRIEVAL", "null"),
])
def test_corrupt_stored_setting_falls_back(tmp_path, monkeypatch, caplog, key, raw):
    db = AppDB(tmp_path / "a.db")
    db.execute("INSERT INTO app_settings (key, value_json, updated_by, ts_utc) VALUES (?,?,?,?)",
               (key, raw, 1, "2026-01-01T00:00:00Z"))
    s = AppSettings(db)
    default = getattr(config, key)
    with caplog.at_level(logging.WARNING):
        assert s.get(key) == default
    assert key in caplog.text
    monkeypatch.setattr(config, key, default)
    s.apply(config)
    assert getattr(config, key) == default
