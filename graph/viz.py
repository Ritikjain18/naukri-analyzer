import pandas as pd
import plotly.express as px

from graph.models import ChartConfig


def validate_chart(raw: dict, df: pd.DataFrame) -> ChartConfig:
    cfg = ChartConfig(**raw)
    for col in (cfg.x, cfg.y):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is not in the data: {list(df.columns)}")
    if not pd.api.types.is_numeric_dtype(df[cfg.y]):
        raise ValueError(f"y column '{cfg.y}' must be numeric")
    return cfg


def build_figure(cfg: ChartConfig, df: pd.DataFrame):
    if cfg.type == "bar":
        return px.bar(df, x=cfg.x, y=cfg.y, title=cfg.title)
    if cfg.type == "line":
        return px.line(df, x=cfg.x, y=cfg.y, title=cfg.title)
    if cfg.type == "scatter":
        return px.scatter(df, x=cfg.x, y=cfg.y, title=cfg.title)
    return px.pie(df, names=cfg.x, values=cfg.y, title=cfg.title)
