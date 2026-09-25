import json

import pytest

import config
from accounts.permissions import PermissionDenied
from accounts.services import build_services
from graph import prompts
from tests.fakes import FakeLLM
from ui import admin_page
from ui.context import Ctx


class Store:
    def list_tables(self):
        return ["candidates"]


@pytest.fixture
def env(tmp_path):
    services = build_services(tmp_path / "app.db", FakeLLM([]))
    aid = services.auth.create_user("root", "root password 1", "admin")
    mid = services.auth.create_user("mona", "mona password 1", "manager")

    def ctx_for(uid, name, role):
        return Ctx(store=Store(), ingest_node=None, graph=None, llms=(), services=services,
                   user={"id": uid, "username": name, "role": role}, session_id="s1", shared={}, messages=[])

    yield services, ctx_for(aid, "root", "admin"), ctx_for(mid, "mona", "manager"), aid, mid
    prompts.set_overrides({})


def actions(services, action):
    return services.audit.query(action=action)


def test_create_user_audits_without_password(env):
    services, admin, _, _, _ = env
    msg = admin_page.admin_create_user(admin, "carol", "carol password 1", "manager")
    assert msg.ok and services.auth.get_by_username("carol").role == "manager"
    row = actions(services, "user_created")[0]
    assert row["detail"]["role"] == "manager" and row["detail"]["target_username"] == "carol"
    assert "carol password 1" not in json.dumps([r["detail"] for r in services.audit.query()])


def test_create_user_errors_are_messages(env):
    services, admin, _, _, _ = env
    bad = admin_page.admin_create_user(admin, "x", "short", "manager")
    assert not bad.ok and bad
    dup = admin_page.admin_create_user(admin, "root", "another password 1", "analyst")
    assert not dup.ok and "taken" in dup
    assert actions(services, "user_created") == []


def test_non_admin_is_refused_everywhere(env, monkeypatch, tmp_path):
    services, _, mgr, aid, _ = env
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "mem")
    calls = [lambda: admin_page.admin_create_user(mgr, "carol", "carol password 1", "analyst"),
             lambda: admin_page.admin_set_role(mgr, aid, "analyst"),
             lambda: admin_page.admin_set_active(mgr, aid, False),
             lambda: admin_page.admin_reset_password(mgr, aid, "new password 12"),
             lambda: admin_page.admin_set_prompt(mgr, "sql", "v2"),
             lambda: admin_page.admin_clear_prompt(mgr, "sql"),
             lambda: admin_page.admin_set_config(mgr, "JUDGE_MIN_SCORE", 4),
             lambda: admin_page.admin_write_memory(mgr)]
    for call in calls:
        with pytest.raises(PermissionDenied):
            call()
    assert not (tmp_path / "mem").exists()


def test_last_admin_protection_and_self_disable(env):
    services, admin, _, aid, mid = env
    demote = admin_page.admin_set_role(admin, aid, "manager")
    assert not demote.ok and "At least one active admin" in demote
    off = admin_page.admin_set_active(admin, aid, False)
    assert not off.ok and off == "You cannot disable your own account."
    assert actions(services, "user_updated") == []


def test_role_and_active_changes_are_audited(env):
    services, admin, _, aid, mid = env
    assert admin_page.admin_set_role(admin, mid, "analyst").ok
    assert admin_page.admin_set_active(admin, mid, False).ok
    rows = actions(services, "user_updated")
    details = [r["detail"] for r in rows]
    assert {"target_id": mid, "target_username": "mona", "field": "role", "old": "manager", "new": "analyst"} in details
    assert any(d["field"] == "active" and d["old"] is True and d["new"] is False for d in details)


def test_own_role_change_notes_next_run(env):
    services, admin, _, aid, _ = env
    services.auth.create_user("root2", "root2 password 1", "admin")
    msg = admin_page.admin_set_role(admin, aid, "manager")
    assert msg.ok and "next run" in msg


def test_reset_password_audit_has_no_secret(env):
    services, admin, _, _, mid = env
    assert admin_page.admin_reset_password(admin, mid, "brand new pass 1").ok
    assert services.auth.authenticate("mona", "brand new pass 1").ok
    d = actions(services, "password_changed")[0]["detail"]
    assert d == {"target_id": mid, "target_username": "mona", "by_admin": True}
    assert not admin_page.admin_reset_password(admin, mid, "x").ok


def test_prompt_set_and_clear(env):
    services, admin, _, _, _ = env
    bad = admin_page.admin_set_prompt(admin, "sql", "v1")
    assert not bad.ok and prompts.get_overrides() == {}
    assert admin_page.admin_set_prompt(admin, "sql", "v2").ok
    assert prompts.get_overrides()["sql"] == "v2"
    assert admin_page.admin_set_prompt(admin, "sql", "v3").ok
    d = [r["detail"] for r in actions(services, "prompt_version_changed")]
    assert {"name": "sql", "version": "v2", "previous": "default"} in d
    assert {"name": "sql", "version": "v3", "previous": "v2"} in d
    assert admin_page.admin_clear_prompt(admin, "sql").ok
    assert "sql" not in prompts.get_overrides()
    assert actions(services, "prompt_version_changed")[0]["detail"]["version"] == "default"


def test_config_change_validates_applies_and_audits(env, monkeypatch):
    services, admin, _, _, _ = env
    monkeypatch.setattr(config, "JUDGE_MIN_SCORE", config.JUDGE_MIN_SCORE)
    old = config.JUDGE_MIN_SCORE
    assert not admin_page.admin_set_config(admin, "JUDGE_MIN_SCORE", 9).ok
    assert config.JUDGE_MIN_SCORE == old and actions(services, "config_changed") == []
    assert admin_page.admin_set_config(admin, "JUDGE_MIN_SCORE", 4).ok
    assert config.JUDGE_MIN_SCORE == 4
    assert actions(services, "config_changed")[0]["detail"] == {"key": "JUDGE_MIN_SCORE", "old": old, "new": 4}


def test_write_memory_under_memory_dir(env, monkeypatch, tmp_path):
    services, admin, _, _, _ = env
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "mem")
    path = admin_page.admin_write_memory(admin)
    assert path == tmp_path / "mem" / "PROJECT_MEMORY.md" and "candidates" in path.read_text()
    assert not list(tmp_path.rglob("CLAUDE.md"))
    assert actions(services, "config_changed")[0]["detail"]["key"] == "project_memory"


def test_audit_failure_does_not_break_helper(env, monkeypatch):
    services, admin, _, _, _ = env

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(type(services.audit), "record", boom)
    assert admin_page.admin_create_user(admin, "dave", "dave password 1", "analyst").ok
