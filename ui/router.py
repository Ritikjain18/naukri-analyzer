import streamlit as st

from accounts.permissions import can
from ui.analyze import render_analyze

ADMIN_PERMISSIONS = ("manage_users", "manage_prompts", "manage_config", "view_audit")


def allowed_pages(role) -> list[str]:
    pages = ["Analyze"]
    if can(role, "view_history"):
        pages.append("History")
    if any(can(role, p) for p in ADMIN_PERMISSIONS):
        pages.append("Admin")
    return pages


def render_history(ctx) -> None:
    st.info("Coming soon")


def render_admin(ctx) -> None:
    st.info("Coming soon")


def render_page(ctx, page: str) -> None:
    if page not in allowed_pages(ctx.user["role"]):
        page = "Analyze"
    {"Analyze": render_analyze, "History": render_history, "Admin": render_admin}[page](ctx)
