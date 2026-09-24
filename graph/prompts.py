from string import Template

from config import ROOT

PROMPTS_DIR = ROOT / "prompts"


def load_prompt(name: str, version: str = "v1") -> str:
    return (PROMPTS_DIR / f"{name}.{version}.txt").read_text()


def render(name: str, version: str = "v1", **values) -> str:
    return Template(load_prompt(name, version)).safe_substitute(
        {k: str(v) for k, v in values.items()}
    )
