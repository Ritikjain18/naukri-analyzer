import re
from string import Template

from config import ROOT

PROMPTS_DIR = ROOT / "prompts"
_OVERRIDES: dict[str, str] = {}
_REQUIRED: dict[str, set[str]] = {}
_PLACEHOLDER = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)|\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def placeholders(text: str) -> set[str]:
    # "$$" is an escaped literal dollar in string.Template, never a placeholder.
    return {a or b for a, b in _PLACEHOLDER.findall(text.replace("$$", ""))}


def set_overrides(overrides: dict[str, str], required: dict[str, set[str]] | None = None) -> None:
    global _OVERRIDES, _REQUIRED
    _OVERRIDES = dict(overrides)
    _REQUIRED = {k: set(v) for k, v in (required or {}).items()}


def get_overrides() -> dict[str, str]:
    return dict(_OVERRIDES)


def effective_version(name: str, requested: str) -> str:
    override = _OVERRIDES.get(name)
    if not override:
        return requested
    path = PROMPTS_DIR / f"{name}.{override}.txt"
    try:
        text = path.read_text()
    except OSError:
        return requested
    if _REQUIRED.get(name, set()) - placeholders(text):
        return requested
    return override


def load_prompt(name: str, version: str = "v1") -> str:
    return (PROMPTS_DIR / f"{name}.{version}.txt").read_text()


def render(name: str, version: str = "v1", **values) -> str:
    return Template(load_prompt(name, effective_version(name, version))).safe_substitute(
        {k: str(v) for k, v in values.items()}
    )
