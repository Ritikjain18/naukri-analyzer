from string import Template

from config import ROOT

PROMPTS_DIR = ROOT / "prompts"
_OVERRIDES: dict[str, str] = {}


def set_overrides(overrides: dict[str, str]) -> None:
    global _OVERRIDES
    _OVERRIDES = dict(overrides)


def get_overrides() -> dict[str, str]:
    return dict(_OVERRIDES)


def effective_version(name: str, requested: str) -> str:
    override = _OVERRIDES.get(name)
    if override and (PROMPTS_DIR / f"{name}.{override}.txt").exists():
        return override
    return requested


def load_prompt(name: str, version: str = "v1") -> str:
    return (PROMPTS_DIR / f"{name}.{version}.txt").read_text()


def render(name: str, version: str = "v1", **values) -> str:
    return Template(load_prompt(name, effective_version(name, version))).safe_substitute(
        {k: str(v) for k, v in values.items()}
    )
