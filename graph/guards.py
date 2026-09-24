import math
import re
from dataclasses import dataclass, field

import pandas as pd

from graph.models import Insight


@dataclass
class GuardResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)


INJECTION_PATTERNS = [re.compile(p, re.I) for p in (
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules)",
    r"disregard\s+.{0,30}(instructions?|rules|prompt)",
    r"(reveal|show|print|repeat)\s+.{0,30}(system\s+prompt|instructions|api\s*key|password|secret)",
    r"you\s+are\s+now\b",
    r"\bjailbreak\b",
    r"\b(drop|truncate)\s+table\b",
    r"\bdelete\s+from\b",
    r";\s*(drop|delete|update|insert|attach|pragma)\b",
    r"<\s*/?\s*(system|assistant|script)\b",
)]

HR_STEMS = (
    "job", "post", "applic", "candid", "hire", "hiri", "funnel", "stage", "recruit", "traffic", "visit",
    "session", "bounce", "source", "categor", "locat", "convers", "view", "skill", "experien", "educat",
    "interview", "offer", "drop", "rate", "metric", "trend", "compar", "average", "total", "count",
    "month", "week", "salary", "screen", "response", "talent", "profile", "engag",
)


def schema_identifiers(schema_text: str) -> set[str]:
    ids: set[str] = set()
    for line in schema_text.splitlines():
        m = re.match(r"\s*(\w+)\((.*)\)\s*$", line)
        if not m:
            continue
        ids.add(m.group(1).lower())
        for part in m.group(2).split(","):
            name = part.strip().split(" ")[0].lower()
            if name:
                ids.add(name)
    return ids


def check_input(question: str, schema_text: str = "", revision: bool = False) -> GuardResult:
    reasons: list[str] = []
    q = question.strip()
    words = re.findall(r"[A-Za-z0-9_]+", q)
    if len(q) < 10 or len(words) < 3:
        reasons.append("The question is too short or unclear.")
    if any(p.search(q) for p in INJECTION_PATTERNS):
        reasons.append("The question looks like an attempt to override instructions.")
    if not revision:
        lower = {w.lower() for w in words}
        ids = schema_identifiers(schema_text)
        id_parts = {part for i in ids for part in i.split("_")}
        related = (
            any(w.startswith(stem) for w in lower for stem in HR_STEMS)
            or bool(lower & ids)
            or bool(lower & id_parts)
        )
        if not related:
            reasons.append("The question doesn't look related to HR / talent analytics data.")
        unknown = [w for w in words if "_" in w and w.lower() not in ids]
        if unknown:
            reasons.append(f"Unknown column(s): {', '.join(unknown)}.")
    return GuardResult(not reasons, reasons)


PLACEHOLDER = re.compile(
    r"\bTBD\b|\bTODO\b|lorem ipsum|\bplaceholder\b|\bX{2,}%?|"
    r"\[(insert|value|number|x)[^\]]*\]|<[^>]*(value|number|insert)[^>]*>",
    re.I,
)
NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for m in NUMBER.finditer(text):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            continue
    return out


def _candidate_values(df: pd.DataFrame) -> set[float]:
    vals: set[float] = {float(len(df)), float(len(df.columns))}
    nums = df.select_dtypes("number")
    for col in nums.columns:
        s = nums[col].dropna().astype(float)
        vals.update(s.head(200).tolist())
        if len(s):
            vals.update([s.sum(), s.mean(), s.max(), s.min(), s.median()])
    text_cells = df.select_dtypes(exclude="number").head(200).to_numpy().ravel()
    vals.update(_numbers(" ".join(map(str, text_cells))))
    base = sorted(vals)[:40]
    derived: set[float] = {v * 100 for v in base}
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            derived.update({abs(a - b), a + b})
            if b:
                derived.add(a / b * 100)
            if a:
                derived.add(b / a * 100)
    for col in nums.columns:
        s = nums[col].dropna().astype(float)
        total = s.sum()
        if total:
            derived.update(float(v) / total * 100 for v in s.head(50))
    return vals | derived


def untraceable_numbers(text: str, data_slice: pd.DataFrame, question: str = "") -> list[str]:
    allowed = set(_numbers(question))
    candidates = _candidate_values(data_slice)
    bad = []
    for n in _numbers(text):
        if n in allowed or (abs(n) <= 10 and n == int(n)) or (1990 <= n <= 2100 and n == int(n)):
            continue
        if not any(math.isclose(n, c, rel_tol=0.02, abs_tol=0.06) for c in candidates):
            bad.append(f"{n:g}")
    return bad


def check_insight(insight: Insight, data_slice: pd.DataFrame, question: str = "") -> GuardResult:
    reasons: list[str] = []
    text = " ".join([insight.finding, *insight.evidence, insight.recommendation])
    if len(insight.finding.strip()) < 15:
        reasons.append("The finding is too short.")
    if not data_slice.empty and not insight.evidence:
        reasons.append("Evidence is missing.")
    if PLACEHOLDER.search(text):
        reasons.append("The answer contains placeholder text.")
    bad = untraceable_numbers(" ".join([insight.finding, *insight.evidence]), data_slice, question)
    if bad:
        reasons.append("Numbers not found in the data: " + ", ".join(bad[:5]) + ".")
    return GuardResult(not reasons, reasons)
