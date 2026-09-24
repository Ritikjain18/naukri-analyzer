ANALYST = frozenset({"upload", "ask", "export"})
MANAGER = ANALYST | {"view_history", "view_comparative"}
ADMIN = MANAGER | {"manage_users", "manage_prompts", "manage_config", "view_audit"}
PERMISSIONS = {"analyst": ANALYST, "manager": MANAGER, "admin": ADMIN}
ROLES = tuple(PERMISSIONS)


class PermissionDenied(PermissionError):
    pass


def can(role, permission: str) -> bool:
    # Return False if role is not a string
    if not isinstance(role, str):
        return False
    return permission in PERMISSIONS.get(role, frozenset())


def require(role, permission: str) -> None:
    if not can(role, permission):
        raise PermissionDenied(f"Your role does not allow: {permission}.")
