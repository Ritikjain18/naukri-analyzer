from data.seed import STAGES, seed_database

DOC_COLUMNS = {
    "job_postings": {"job_id", "title", "category", "location", "date_posted", "views", "applications", "status"},
    "hiring_funnel": {"candidate_id", "job_id", "stage", "date", "drop_off_flag"},
    "traffic_metrics": {"date", "source", "visits", "session_duration", "bounce_rate"},
    "recruiter_activity": {"recruiter_id", "job_id", "actions", "response_rate"},
    "candidate_profiles": {"candidate_id", "skills", "experience_years", "education"},
}


def test_seed_creates_doc_tables(store):
    counts = seed_database(store, scale=0.05)
    assert set(counts) == set(DOC_COLUMNS)
    for table, cols in DOC_COLUMNS.items():
        assert {c for c, _ in store.get_schema()[table]} == cols
        assert counts[table] > 0


def test_seed_is_reproducible(store, tmp_path):
    from data.store import SQLiteStore

    seed_database(store, seed=7, scale=0.05)
    other = SQLiteStore(tmp_path / "other.db")
    seed_database(other, seed=7, scale=0.05)
    q = "SELECT * FROM job_postings ORDER BY job_id"
    assert store.run_sql(q).equals(other.run_sql(q))


def test_funnel_stages_are_ordered_prefix(store):
    seed_database(store, scale=0.05)
    df = store.run_sql("SELECT * FROM hiring_funnel ORDER BY candidate_id, date")
    for _, group in df.groupby("candidate_id"):
        stages = list(group["stage"])
        assert stages == STAGES[: len(stages)]
        assert list(group["drop_off_flag"])[:-1] == [0] * (len(stages) - 1)


def test_applications_never_exceed_views(store):
    seed_database(store, scale=0.05)
    bad = store.run_sql("SELECT COUNT(*) AS n FROM job_postings WHERE applications > views")
    assert bad["n"][0] == 0
