import re

import pytest

from graph.prompts import load_prompt, render
from graph.skills import detect_domain, load_domain_skills, load_skill

VARS = {
    "data_understanding": dict(skills="s", schema="sc", sample="sm"),
    "query": dict(summary="a", skill="b", history="c", slice="d", question="e", error_note=""),
    "visualization": dict(insight="i", columns="c", error_note=""),
    "sql": dict(schema="s", limit=200, history="h", question="q", error_note=""),
    "orchestrator": dict(schema="s", summary="m", history="h", question="q"),
    "judge": dict(stage="analyst", question="q", output="o"),
    "pandas": dict(columns="c", sample="s", history="h", question="q", error_note=""),
}


@pytest.mark.parametrize("name", VARS)
def test_every_prompt_renders_fully(name):
    out = render(name, **VARS[name])
    assert not re.search(r"\$[A-Za-z_]+", out)


def test_render_keeps_json_braces():
    out = render("visualization", **VARS["visualization"])
    assert '{"type"' in out


def test_load_prompt_missing_version_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("query", "v99")


@pytest.mark.parametrize("cols,expected", [
    (["job_id", "title", "category", "location", "views", "applications"], "job-posting"),
    (["candidate_id", "job_id", "stage", "date", "drop_off_flag"], "hiring-funnel"),
    (["date", "source", "visits", "session_duration", "bounce_rate"], "traffic"),
    (["foo", "bar"], None),
])
def test_detect_domain(cols, expected):
    assert detect_domain(cols) == expected


def test_skills_load():
    for d in ["job-posting", "hiring-funnel", "traffic", "slide-gen"]:
        assert len(load_skill(d)) > 100
    combined = load_domain_skills()
    assert "job-posting" in combined and "traffic" in combined
    assert "slide-gen" not in combined


def test_query_v2_leaves_no_placeholders():
    out = render("query", "v2", summary="a", skill="b", history="c", slice="d", question="e",
                 error_note="", guard_error="", judge_correction="", revision_note="")
    assert "$" not in out


def test_sql_v2_renders_with_correction():
    out = render("sql", "v2", schema="s", limit=200, history="h", question="q", correction="Fix X.", error_note="")
    assert "Fix X." in out and not re.search(r"\$[A-Za-z_]+", out)


def test_sql_v3_has_real_division_guidance_and_renders_fully():
    raw = load_prompt("sql", "v3")
    assert "CAST" in raw and "NULLIF" in raw and "integer" in raw.lower()
    out = render("sql", "v3", schema="s", limit=200, history="h", question="q", correction="", error_note="")
    assert not re.search(r"\$[A-Za-z_]+", out)
