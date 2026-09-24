ANALYST = frozenset({"upload", "ask", "export"})
MANAGER = ANALYST | {"view_history", "view_comparative"}
ADMIN = MANAGER | {"manage_users", "manage_prompts", "manage_config", "view_audit"}
PERMISSIONS = {"analyst": ANALYST, "manager": MANAGER, "admin": ADMIN}


class PermissionDenied(PermissionError):
    pass


def can(role, permission: str) -> bool:
    return permission in PERMISSIONS.get(role, frozenset())


def require(role, permission: str) -> None:
    if not can(role, permission):
        raise PermissionDenied(f"Your role does not allow: {permission}.")
