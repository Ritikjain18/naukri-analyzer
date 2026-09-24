from config import ROOT

SKILLS_DIR = ROOT / "skills"
DOMAINS = ["job-posting", "hiring-funnel", "traffic"]
DOMAIN_KEYWORDS = {
    "job-posting": ["job_id", "views", "applications", "category", "location", "date_posted", "title"],
    "hiring-funnel": ["stage", "drop_off", "candidate_id", "time_to_hire"],
    "traffic": ["visits", "bounce", "session", "source"],
}


def load_skill(domain: str) -> str:
    return (SKILLS_DIR / domain / "SKILL.md").read_text()


def load_domain_skills() -> str:
    return "\n\n".join(f"## {d}\n{load_skill(d)}" for d in DOMAINS)


def detect_domain(columns: list[str]) -> str | None:
    cols = [c.lower() for c in columns]
    scores = {
        d: sum(any(k in c for c in cols) for k in kws)
        for d, kws in DOMAIN_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None
