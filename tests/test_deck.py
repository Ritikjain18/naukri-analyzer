from io import BytesIO

import pytest
from pptx import Presentation

from export.deck import build_deck, fit_body, truncate_title


def entry(chart_type="bar", finding="Engineering has the highest conversion rate.", chart=True, slice_rows=None):
    rows = slice_rows if slice_rows is not None else [{"category": "Eng", "rate": 0.2}, {"category": "Sales", "rate": 0.1}]
    return {"question": "q", "approved": False,
            "insight": {"finding": finding, "evidence": ["Eng 20%", "Sales 10%", "Gap 10 points", "Extra bullet"],
                        "recommendation": "Shift budget to Engineering."},
            "chart": {"type": chart_type, "x": "category", "y": "rate", "title": "Conversion"} if chart else None,
            "slice": rows}


def load(entries):
    return Presentation(BytesIO(build_deck(entries)))


def shapes_of(slide):
    return list(slide.shapes)


def test_truncate_title():
    assert truncate_title("one two three") == "one two three"
    long = " ".join(f"w{i}" for i in range(20))
    out = truncate_title(long)
    assert out.endswith("…") and len(out.rstrip("…").split()) == 12


def test_fit_body_limits_words_and_bullets():
    ev = ["word " * 20, "two", "three", "four"]
    bullets, rec = fit_body(ev, "Do the thing now.", budget=40)
    assert len(bullets) <= 3 and rec.startswith("Do the thing")
    assert sum(len(b.split()) for b in bullets) + len(rec.split()) <= 40


def test_fit_body_cuts_recommendation_when_alone_too_long():
    bullets, rec = fit_body(["x"], "word " * 60, budget=40)
    assert len(rec.split()) <= 40 and bullets == []


def test_one_slide_per_entry_with_title():
    prs = load([entry(), entry(finding="Sales is lowest.")])
    assert len(prs.slides) == 2
    assert prs.slides[0].shapes.title.text == "Engineering has the highest conversion rate."


@pytest.mark.parametrize("kind", ["bar", "line", "pie"])
def test_native_chart_present(kind):
    slide = load([entry(chart_type=kind)]).slides[0]
    assert any(s.has_chart for s in shapes_of(slide))


def test_scatter_with_numeric_x_and_fallback_to_table():
    rows = [{"category": 1.0, "rate": 0.2}, {"category": 2.0, "rate": 0.1}]
    assert any(s.has_chart for s in shapes_of(load([entry("scatter", slice_rows=rows)]).slides[0]))
    assert any(s.has_table for s in shapes_of(load([entry("scatter")]).slides[0]))   # x non-numeric


def test_table_fallback_when_no_chart():
    slide = load([entry(chart=False)]).slides[0]
    assert any(s.has_table for s in shapes_of(slide)) and not any(s.has_chart for s in shapes_of(slide))


def test_no_visual_when_slice_empty():
    slide = load([entry(chart=False, slice_rows=[])]).slides[0]
    assert not any(s.has_table or s.has_chart for s in shapes_of(slide))


def test_body_text_contains_recommendation_and_bullet_limit():
    slide = load([entry()]).slides[0]
    text = " ".join(s.text_frame.text for s in shapes_of(slide) if s.has_text_frame)
    assert "Recommendation: Shift budget to Engineering." in text and "Extra bullet" not in text
