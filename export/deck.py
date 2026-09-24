from io import BytesIO

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches, Pt

MAX_TITLE_WORDS = 12
BODY_WORD_BUDGET = 40
MAX_BULLETS = 3
TABLE_ROWS, TABLE_COLS = 8, 5
CHART_TYPES = {"bar": XL_CHART_TYPE.COLUMN_CLUSTERED, "line": XL_CHART_TYPE.LINE_MARKERS,
               "pie": XL_CHART_TYPE.PIE, "scatter": XL_CHART_TYPE.XY_SCATTER}


def truncate_title(text: str, max_words: int = MAX_TITLE_WORDS) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + "…"


def _cut(text: str, words: int) -> str:
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words])


def fit_body(evidence: list[str], recommendation: str, budget: int = BODY_WORD_BUDGET) -> tuple[list[str], str]:
    rec = _cut(recommendation, budget)
    remaining = budget - len(rec.split())
    bullets: list[str] = []
    for item in evidence[:MAX_BULLETS]:
        if remaining <= 0:
            break
        cut = _cut(item, remaining)
        bullets.append(cut)
        remaining -= len(cut.split())
    return bullets, rec


def _add_table(slide, df: pd.DataFrame) -> None:
    df = df.iloc[:TABLE_ROWS, :TABLE_COLS]
    shape = slide.shapes.add_table(len(df) + 1, len(df.columns), Inches(6.6), Inches(1.7), Inches(6.2), Inches(0.4) * (len(df) + 1))
    table = shape.table
    for j, col in enumerate(df.columns):
        table.cell(0, j).text = str(col)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, value in enumerate(row):
            table.cell(i, j).text = f"{value:.4g}" if isinstance(value, float) else str(value)


def _add_chart(slide, chart: dict, df: pd.DataFrame) -> bool:
    x, y, kind = chart["x"], chart["y"], chart["type"]
    if kind not in CHART_TYPES or x not in df.columns or y not in df.columns:
        return False
    if not pd.api.types.is_numeric_dtype(df[y]):
        return False
    if kind == "scatter":
        if not pd.api.types.is_numeric_dtype(df[x]):
            return False
        data = XyChartData()
        series = data.add_series(chart["title"])
        for xv, yv in zip(df[x], df[y]):
            series.add_data_point(float(xv), float(yv))
    else:
        data = CategoryChartData()
        data.categories = [str(v) for v in df[x]]
        data.add_series(y, [float(v) for v in df[y]])
    frame = slide.shapes.add_chart(CHART_TYPES[kind], Inches(6.6), Inches(1.7), Inches(6.2), Inches(5), data)
    frame.chart.has_title = True
    frame.chart.chart_title.text_frame.text = chart["title"]
    return True


def build_deck(entries: list[dict]) -> bytes:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for e in entries:
        slide = prs.slides.add_slide(prs.slide_layouts[5])  # Title Only
        ins = e["insight"]
        slide.shapes.title.text = truncate_title(ins["finding"])
        bullets, rec = fit_body(ins["evidence"], ins["recommendation"])
        box = slide.shapes.add_textbox(Inches(0.6), Inches(1.7), Inches(5.7), Inches(5))
        frame = box.text_frame
        frame.word_wrap = True
        for i, text in enumerate(bullets):
            p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            p.text = f"• {text}"
            p.font.size = Pt(18)
        p = frame.paragraphs[0] if not bullets else frame.add_paragraph()
        p.text = f"Recommendation: {rec}"
        p.font.size = Pt(18)
        p.font.bold = True
        df = pd.DataFrame(e.get("slice") or [])
        if df.empty:
            continue
        if not (e.get("chart") and _add_chart(slide, e["chart"], df)):
            _add_table(slide, df)
    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()
