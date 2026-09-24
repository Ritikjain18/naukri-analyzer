import random
from datetime import date, timedelta

import pandas as pd
from faker import Faker

import config
from data.store import SQLiteStore

CATEGORIES = ["Engineering", "Sales", "Marketing", "Finance", "Operations", "Design", "HR", "Customer Support"]
LOCATIONS = ["Bengaluru", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Remote"]
SOURCES = ["organic_search", "direct", "referral", "social", "email", "paid_ads"]
STAGES = ["applied", "screened", "interviewed", "offered", "hired"]
DROP_PROB = [0.35, 0.40, 0.50, 0.20]  # drop chance after each non-final stage
SKILLS = ["Python", "SQL", "Excel", "Java", "Sales", "Communication", "Figma", "Accounting", "React", "Recruiting"]
EDUCATION = ["B.Tech", "B.Com", "MBA", "B.A.", "M.Tech", "BBA"]
END = date(2026, 9, 1)  # fixed so seeded data is reproducible


def seed_database(store: SQLiteStore, seed: int = 42, scale: float = 1.0) -> dict[str, int]:
    rng = random.Random(seed)
    fake = Faker()
    fake.seed_instance(seed)

    def n(base: int) -> int:
        return max(10, int(base * scale))

    n_jobs, n_cand, n_rec = n(500), n(2000), n(100)

    jobs = []
    for job_id in range(1, n_jobs + 1):
        views = rng.randint(50, 5000)
        jobs.append({
            "job_id": job_id,
            "title": fake.job(),
            "category": rng.choice(CATEGORIES),
            "location": rng.choice(LOCATIONS),
            "date_posted": (END - timedelta(days=rng.randint(0, 364))).isoformat(),
            "views": views,
            "applications": int(views * rng.uniform(0.02, 0.25)),
            "status": rng.choice(["open", "closed", "paused"]),
        })

    funnel = []
    for cid in range(1, n_cand + 1):
        job_id = rng.randint(1, n_jobs)
        day = END - timedelta(days=rng.randint(30, 364))
        for i, stage in enumerate(STAGES):
            row = {"candidate_id": cid, "job_id": job_id, "stage": stage,
                   "date": day.isoformat(), "drop_off_flag": 0}
            funnel.append(row)
            if stage == STAGES[-1]:
                break
            if rng.random() < DROP_PROB[i]:
                row["drop_off_flag"] = 1
                break
            day += timedelta(days=rng.randint(1, 10))

    traffic = []
    for offset in range(365):
        day = END - timedelta(days=offset)
        for source in SOURCES:
            traffic.append({
                "date": day.isoformat(),
                "source": source,
                "visits": rng.randint(200, 6000),
                "session_duration": round(rng.uniform(30, 420), 1),
                "bounce_rate": round(rng.uniform(0.25, 0.85), 3),
            })

    recruiters = [{
        "recruiter_id": rng.randint(1, n_rec),
        "job_id": rng.randint(1, n_jobs),
        "actions": rng.randint(1, 200),
        "response_rate": round(rng.uniform(0.05, 0.95), 3),
    } for _ in range(n(800))]

    candidates = [{
        "candidate_id": cid,
        "skills": ", ".join(rng.sample(SKILLS, 3)),
        "experience_years": rng.randint(0, 20),
        "education": rng.choice(EDUCATION),
    } for cid in range(1, n_cand + 1)]

    tables = {
        "job_postings": jobs,
        "hiring_funnel": funnel,
        "traffic_metrics": traffic,
        "recruiter_activity": recruiters,
        "candidate_profiles": candidates,
    }
    for name, rows in tables.items():
        store.replace_table(pd.DataFrame(rows), name)
    return {name: len(rows) for name, rows in tables.items()}


if __name__ == "__main__":
    print(seed_database(SQLiteStore(config.DB_PATH)))
