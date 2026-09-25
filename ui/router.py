from accounts.permissions import can
from ui import admin_page, analyze, history_page

ADMIN_PERMISSIONS = ("manage_users", "manage_prompts", "manage_config", "view_audit")


def allowed_pages(role) -> list[str]:
    pages = ["Analyze"]
    if can(role, "view_history"):
        pages.append("History")
    if any(can(role, p) for p in ADMIN_PERMISSIONS):
        pages.append("Admin")
    return pages


def render_page(ctx, page: str) -> None:
    if page not in allowed_pages(ctx.user["role"]):
        page = "Analyze"
    if page == "Analyze":
        analyze.render_analyze(ctx)
    elif page == "History":
        history_page.render_history(ctx)
    else:
        admin_page.render_admin(ctx)
