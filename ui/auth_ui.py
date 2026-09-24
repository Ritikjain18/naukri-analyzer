import logging

import streamlit as st

from accounts.auth import AccountError

log = logging.getLogger(__name__)
PASSWORD_KEYS = ("login_password", "bootstrap_password", "bootstrap_confirm", "pw_old", "pw_new")
SESSION_KEYS = ("auth", "session_id", "shared", "messages", "pending_revision", "ingest_prompts", "history_viewed")


def _clear_passwords() -> None:
    for key in PASSWORD_KEYS:
        st.session_state.pop(key, None)


def begin_session(services, auth: dict, summarise: bool = False) -> None:
    """Start a tracked session for a logged-in user and load their prior-session context."""
    uid = auth["user_id"]
    if summarise:
        try:
            services.memory.summarise_pending(uid)
        except Exception as exc:
            log.warning("summarise_pending failed: %s", type(exc).__name__)
            st.warning("Earlier sessions could not be summarised; continuing without them.")
    st.session_state["session_id"] = services.sessions.start(uid)
    shared = st.session_state.setdefault("shared", {})
    try:
        shared["prior_context"] = services.memory.prior_context(uid)
    except Exception as exc:
        log.warning("prior_context failed: %s", type(exc).__name__)
        shared["prior_context"] = ""


def _log_in(services, user) -> None:
    auth = {"user_id": user.id, "username": user.username, "role": user.role}
    st.session_state["auth"] = auth
    begin_session(services, auth, summarise=True)
    services.audit.record({"id": user.id, "username": user.username}, "login", st.session_state["session_id"])
    _clear_passwords()
    st.rerun()


def render_bootstrap(services) -> None:
    st.subheader("Create the first admin account")
    username = st.text_input("Username", key="bootstrap_username")
    password = st.text_input("Password", type="password", key="bootstrap_password")
    confirm = st.text_input("Confirm password", type="password", key="bootstrap_confirm")
    if not st.button("Create admin", key="bootstrap_create"):
        return
    if password != confirm:
        st.error("Passwords do not match.")
        return
    try:
        uid = services.auth.create_user(username, password, "admin")
    except AccountError as exc:
        st.error(str(exc))
        return
    user = services.auth.get_user(uid)
    services.audit.record({"id": user.id, "username": user.username}, "user_created",
                          role="admin", username=user.username, bootstrap=True)
    _log_in(services, user)


def render_login(services) -> None:
    st.subheader("Sign in")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if not st.button("Sign in", key="login_submit"):
        return
    result = services.auth.authenticate(username, password)
    if result.ok:
        _log_in(services, result.user)
        return
    tried = {"username_tried": username.strip()[:64]}
    if result.reason == "locked":
        services.audit.record(None, "account_locked", **tried)
        st.error("Too many failed attempts. Try again later.")
    else:
        services.audit.record(None, "login_failed", reason=result.reason, **tried)
        st.error("Invalid username or password.")


def _finish_session(ctx, action: str) -> None:
    try:
        ctx.services.memory.end_session(ctx.session_id, ctx.user["id"])
    except Exception as exc:
        log.warning("end_session failed: %s", type(exc).__name__)
        st.warning("Could not summarise this session; it will be retried later.")
    ctx.services.audit.record(ctx.user, "session_end", ctx.session_id)
    if action == "logout":
        ctx.services.audit.record(ctx.user, "logout", ctx.session_id)


def _reset_state() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def render_account_box(ctx) -> None:
    st.sidebar.caption(f"Signed in as {ctx.user['username']} ({ctx.user['role']})")
    if st.sidebar.button("End session", key="end_session"):
        auth = st.session_state["auth"]
        _finish_session(ctx, "end")
        _reset_state()
        st.session_state["auth"] = auth
        begin_session(ctx.services, auth, summarise=False)
        st.rerun()
    if st.sidebar.button("Log out", key="logout"):
        _finish_session(ctx, "logout")
        _reset_state()
        st.rerun()
    with st.sidebar.expander("Change password"):
        old = st.text_input("Current password", type="password", key="pw_old")
        new = st.text_input("New password", type="password", key="pw_new")
        if st.button("Save password", key="pw_save"):
            try:
                ctx.services.auth.change_password(ctx.user["id"], old, new)
            except AccountError as exc:
                st.error(str(exc))
            else:
                ctx.services.audit.record(ctx.user, "password_changed", ctx.session_id)
                _clear_passwords()
                st.success("Password changed.")
