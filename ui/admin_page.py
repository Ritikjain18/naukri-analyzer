import json
import logging
from datetime import UTC, date, datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from accounts.auth import AccountError
from accounts.memory import write_project_memory
from accounts.permissions import ROLES, PermissionDenied, can, require
from accounts.settings import AppSettings
from graph.textsafe import safe_text
from ui.analyze import audit_export
from ui.permissions_ui import guard

logger = logging.getLogger(__name__)

ADMIN_PERMISSIONS = ("manage_users", "manage_prompts", "manage_config", "view_audit")
AUDIT_LIMIT = 500
GENERIC_ERROR = "Something went wrong. The change was not applied."
KNOWN_ACTIONS = ("login", "login_failed", "account_locked", "logout", "session_end", "session_revoked",
                 "question", "revise", "approve", "upload", "export", "history_view", "user_created",
                 "user_updated", "password_changed", "prompt_version_changed", "config_changed")
CONFIG_KEYS = ("JUDGE_ENABLED", "JUDGE_RETRIEVAL", "JUDGE_MIN_SCORE")


class Outcome(str):
    """A status message that also says whether the action succeeded."""

    ok: bool

    def __new__(cls, text: str, ok: bool):
        obj = super().__new__(cls, text)
        obj.ok = ok
        return obj


def _ok(text: str) -> Outcome:
    return Outcome(text, True)


def _fail(text: str) -> Outcome:
    return Outcome(text, False)


def _audit(ctx, action: str, **detail) -> None:
    """Record an audit row; a failing audit write must never break the admin action or the page."""
    try:
        ctx.services.audit.record(ctx.user, action, session_id=ctx.session_id, **detail)
    except Exception as exc:
        logger.warning("Admin audit write failed: %s", type(exc).__name__)


def _target(ctx, user_id: int):
    return ctx.services.auth.get_user(user_id)


# ---- pure action helpers -------------------------------------------------------------------------------------------

def admin_create_user(ctx, username: str, password: str, role: str) -> Outcome:
    require(ctx.user["role"], "manage_users")
    try:
        uid = ctx.services.auth.create_user(username, password, role)
    except AccountError as exc:
        return _fail(str(exc))
    _audit(ctx, "user_created", role=role, target_id=uid, target_username=username.strip())
    return _ok(f"Created user {username.strip()} ({role}).")


def admin_set_role(ctx, user_id: int, role: str) -> Outcome:
    require(ctx.user["role"], "manage_users")
    target = _target(ctx, user_id)
    if target is None:
        return _fail("Unknown user.")
    if target.role == role:
        return _ok("Role unchanged.")
    try:
        ctx.services.auth.set_role(user_id, role)
    except AccountError as exc:
        return _fail(str(exc))
    _audit(ctx, "user_updated", target_id=user_id, target_username=target.username, field="role",
           old=target.role, new=role)
    note = " The change applies on your next run." if user_id == ctx.user["id"] else ""
    return _ok(f"{target.username} is now {role}.{note}")


def admin_set_active(ctx, user_id: int, active: bool) -> Outcome:
    require(ctx.user["role"], "manage_users")
    if not active and user_id == ctx.user["id"]:
        return _fail("You cannot disable your own account.")
    target = _target(ctx, user_id)
    if target is None:
        return _fail("Unknown user.")
    try:
        ctx.services.auth.set_active(user_id, active)
    except AccountError as exc:
        return _fail(str(exc))
    _audit(ctx, "user_updated", target_id=user_id, target_username=target.username, field="active",
           old=target.active, new=active)
    return _ok(f"{target.username} is now {'enabled' if active else 'disabled'}.")


def admin_reset_password(ctx, user_id: int, new: str) -> Outcome:
    require(ctx.user["role"], "manage_users")
    target = _target(ctx, user_id)
    if target is None:
        return _fail("Unknown user.")
    try:
        ctx.services.auth.reset_password(user_id, new)
    except AccountError as exc:
        return _fail(str(exc))
    _audit(ctx, "password_changed", target_id=user_id, target_username=target.username, by_admin=True)
    return _ok(f"Password reset for {target.username}.")


def _apply_prompts(ctx) -> None:
    ctx.services.prompts.apply()


def admin_set_prompt(ctx, name: str, version: str) -> Outcome:
    require(ctx.user["role"], "manage_prompts")
    previous = ctx.services.prompts.active(name) or "default"
    try:
        ctx.services.prompts.set_active(name, version, ctx.user["id"])
    except AccountError as exc:
        return _fail(str(exc))
    _apply_prompts(ctx)
    _audit(ctx, "prompt_version_changed", name=name, version=version, previous=previous)
    return _ok(f"{name} now uses {version}.")


def admin_clear_prompt(ctx, name: str) -> Outcome:
    require(ctx.user["role"], "manage_prompts")
    if name not in ctx.services.prompts.names():
        return _fail("Unknown prompt.")
    previous = ctx.services.prompts.active(name) or "default"
    ctx.services.prompts.clear(name)
    _apply_prompts(ctx)
    _audit(ctx, "prompt_version_changed", name=name, version="default", previous=previous)
    return _ok(f"{name} now uses the default version.")


def admin_set_config(ctx, key: str, value) -> Outcome:
    require(ctx.user["role"], "manage_config")
    if key not in AppSettings.ALLOWED:
        return _fail("Unknown setting.")
    old = ctx.services.settings.get(key)
    try:
        ctx.services.settings.set(key, value, ctx.user["id"])
    except AccountError as exc:
        return _fail(str(exc))
    ctx.services.settings.apply(config)
    _audit(ctx, "config_changed", key=key, old=old, new=value)
    return _ok(f"{key} set to {value}.")


def admin_write_memory(ctx) -> Path:
    require(ctx.user["role"], "manage_config")
    path = Path(config.MEMORY_DIR) / "PROJECT_MEMORY.md"
    write_project_memory(path, ctx.store, ctx.services.prompts, ctx.services.history,
                         {"smart": config.MODEL_SMART, "fast": config.MODEL_FAST})
    _audit(ctx, "config_changed", key="project_memory")
    return path


# ---- UI plumbing ----------------------------------------------------------------------------------------------------

def _run(ctx, permission: str, fn, *args):
    """Run a helper; PermissionDenied and unexpected errors become messages (denials are audited)."""
    try:
        return fn(ctx, *args)
    except PermissionDenied as exc:
        _audit(ctx, f"denied:{permission}")
        return _fail(str(exc))
    except Exception as exc:
        logger.warning("Admin action failed: %s", type(exc).__name__)
        return _fail(GENERIC_ERROR)


def _finish(outcome, clear: tuple = ()) -> None:
    """Remember the message (and inputs to clear), then rerun so every table shows the new state."""
    st.session_state["admin_flash"] = (bool(getattr(outcome, "ok", False)), str(outcome))
    st.session_state["admin_clear"] = list(clear)
    st.rerun()


def _show_flash() -> None:
    for key in st.session_state.pop("admin_clear", []):
        st.session_state[key] = ""        # before the widget is created, so this is allowed
    flash = st.session_state.pop("admin_flash", None)
    if flash:
        (st.success if flash[0] else st.error)(safe_text(flash[1]))


def _bound(d, end: bool):
    if not isinstance(d, date):
        return None
    return datetime.combine(d, time.max if end else time.min, tzinfo=UTC)


# ---- tabs -----------------------------------------------------------------------------------------------------------

def _users_tab(ctx) -> None:
    if not guard(ctx, "manage_users"):
        return
    users = ctx.services.auth.list_users()
    st.dataframe(pd.DataFrame([{"id": u.id, "username": u.username, "role": u.role, "active": u.active}
                               for u in users], columns=["id", "username", "role", "active"]), hide_index=True)
    st.subheader("Create user")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Username", key="new_username")
    pw = c2.text_input("Password", type="password", key="new_password")
    role = c3.selectbox("Role", list(ROLES), key="new_role")
    if st.button("Create user", key="create_user"):
        _finish(_run(ctx, "manage_users", admin_create_user, name, pw, role),
                clear=("new_password",))
    st.subheader("Manage users")
    for u in users:
        with st.expander(f"{safe_text(u.username)} ({u.role}, {'active' if u.active else 'disabled'})"):
            r1, r2, r3 = st.columns(3)
            new_role = r1.selectbox("Role", list(ROLES), index=list(ROLES).index(u.role), key=f"role_{u.id}")
            if r1.button("Apply role", key=f"apply_role_{u.id}"):
                _finish(_run(ctx, "manage_users", admin_set_role, u.id, new_role))
            if r2.button("Disable" if u.active else "Enable", key=f"toggle_active_{u.id}"):
                _finish(_run(ctx, "manage_users", admin_set_active, u.id, not u.active))
            pw_new = r3.text_input("New password", type="password", key=f"reset_pw_{u.id}")
            if r3.button("Reset password", key=f"reset_pw_apply_{u.id}"):
                _finish(_run(ctx, "manage_users", admin_reset_password, u.id, pw_new),
                        clear=(f"reset_pw_{u.id}",))


def _prompt_text(ctx, name: str, version: str) -> str:
    """Read only a validated (name, version) pair from the prompts directory."""
    version = "v1" if version == "default" else version
    if name not in ctx.services.prompts.names() or version not in ctx.services.prompts.versions(name):
        return ""
    try:
        return (ctx.services.prompts.dir / f"{name}.{version}.txt").read_text()
    except OSError:
        return ""


def _prompts_tab(ctx) -> None:
    if not guard(ctx, "manage_prompts"):
        return
    st.caption("Prompt overrides are process-wide: a change applies to every user of this app immediately.")
    for name in ctx.services.prompts.names():
        options = ["default"] + ctx.services.prompts.compatible_versions(name)
        active = ctx.services.prompts.active(name)
        with st.expander(f"{name} (active: {safe_text(active or 'default')})"):
            choice = st.selectbox("Version", options, index=options.index(active) if active in options else 0,
                                  key=f"prompt_{name}")
            st.code(_prompt_text(ctx, name, choice) or "(no file)", language=None)
            a, b = st.columns(2)
            if a.button("Apply", key=f"prompt_apply_{name}"):
                if choice == "default":
                    _finish(_run(ctx, "manage_prompts", admin_clear_prompt, name))
                _finish(_run(ctx, "manage_prompts", admin_set_prompt, name, choice))
            if b.button("Use default", key=f"prompt_clear_{name}"):
                _finish(_run(ctx, "manage_prompts", admin_clear_prompt, name))


def _config_tab(ctx) -> None:
    if not guard(ctx, "manage_config"):
        return
    current = ctx.services.settings.all()
    new = {
        "JUDGE_ENABLED": st.checkbox("Judge enabled", value=current["JUDGE_ENABLED"], key="cfg_JUDGE_ENABLED"),
        "JUDGE_RETRIEVAL": st.checkbox("Judge retrieval", value=current["JUDGE_RETRIEVAL"],
                                       key="cfg_JUDGE_RETRIEVAL"),
        "JUDGE_MIN_SCORE": int(st.number_input("Judge minimum score", min_value=1, max_value=5, step=1,
                                               value=int(current["JUDGE_MIN_SCORE"]), key="cfg_JUDGE_MIN_SCORE")),
    }
    if st.button("Save", key="cfg_save"):
        results = [_run(ctx, "manage_config", admin_set_config, k, v) for k, v in new.items() if v != current[k]]
        if not results:
            _finish(_ok("No changes."))
        failed = [r for r in results if not r.ok]
        _finish(failed[0] if failed else _ok("; ".join(results)))


def _audit_tab(ctx) -> None:
    if not guard(ctx, "view_audit"):
        return
    audit = ctx.services.audit
    known = sorted(set(KNOWN_ACTIONS) | {r["action"] for r in ctx.services.db.query(
        "SELECT DISTINCT action FROM audit_log")})
    c1, c2, c3, c4 = st.columns(4)
    who = c1.text_input("User contains", key="audit_user")
    action = c2.selectbox("Action", ["All", *known], key="audit_action")
    since = _bound(c3.date_input("From", value=None, key="audit_from"), False)
    until = _bound(c4.date_input("To", value=None, key="audit_to"), True)
    entries = audit.query(user=who or None, action=None if action == "All" else action, since=since, until=until,
                          limit=AUDIT_LIMIT)
    st.dataframe(pd.DataFrame(
        [{"ts": r["ts_utc"], "user": r["username"] or "", "action": r["action"],
          "session": (r["session_id"] or "")[:8], "detail": json.dumps(r["detail"], separators=(",", ":"))}
         for r in entries], columns=["ts", "user", "action", "session", "detail"]), hide_index=True)
    if len(entries) >= AUDIT_LIMIT:
        st.caption(f"Showing the newest {AUDIT_LIMIT} rows.")
    if entries:
        st.download_button("Download CSV", data=audit.to_csv(entries), file_name="audit_log.csv", mime="text/csv",
                           on_click=audit_export, args=(ctx, len(entries), "audit"), key="audit_csv")


def _memory_tab(ctx) -> None:
    if not guard(ctx, "manage_config"):
        return
    if st.button("Write project memory", key="write_memory"):
        outcome = _run(ctx, "manage_config", _memory_outcome)
        _finish(outcome)
    path = Path(config.MEMORY_DIR) / "PROJECT_MEMORY.md"
    if path.is_file() and not path.is_symlink():
        with st.expander("PROJECT_MEMORY.md"):
            try:
                st.code(path.read_text(), language=None)
            except OSError:
                st.caption("The file could not be read.")
    else:
        st.caption("No project memory file has been written yet.")


def _memory_outcome(ctx) -> Outcome:
    admin_write_memory(ctx)
    return _ok("Project memory written.")


def render_admin(ctx) -> None:
    role = ctx.user["role"]
    if not any(can(role, p) for p in ADMIN_PERMISSIONS):
        try:
            require(role, "manage_users")
        except PermissionDenied as exc:
            st.error(str(exc))
            _audit(ctx, "denied:manage_users")
        return
    _show_flash()
    tabs = [(label, perm, fn) for label, perm, fn in (
        ("Users", "manage_users", _users_tab), ("Prompts", "manage_prompts", _prompts_tab),
        ("Config", "manage_config", _config_tab), ("Audit", "view_audit", _audit_tab),
        ("Memory", "manage_config", _memory_tab)) if can(role, perm)]
    for container, (_, _, fn) in zip(st.tabs([t[0] for t in tabs]), tabs):
        with container:
            fn(ctx)
