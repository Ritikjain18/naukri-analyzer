import streamlit as st

from accounts.permissions import PermissionDenied, require


def guard(ctx, permission: str) -> bool:
    """True when the user may perform `permission`; otherwise show the error and audit the denial."""
    try:
        require(ctx.user["role"], permission)
    except PermissionDenied as exc:
        st.error(str(exc))
        ctx.services.audit.record(ctx.user, f"denied:{permission}", session_id=ctx.session_id)
        return False
    return True
