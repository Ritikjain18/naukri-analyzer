import re

_SPECIAL = re.compile(r"([\\*_`\[\]()!<>#|~$])")
_BULLET = re.compile(r"^(\s*)([-+])", re.MULTILINE)
_NUMBERED = re.compile(r"^(\s*\d+)\.", re.MULTILINE)


def safe_text(text) -> str:
    """Escape markdown control characters so model/data text renders literally (no images, links, HTML)."""
    if text is None:
        return ""
    escaped = _SPECIAL.sub(r"\\\1", str(text))
    escaped = _BULLET.sub(r"\1\\\2", escaped)
    return _NUMBERED.sub(r"\1\\.", escaped)
