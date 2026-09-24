from typing import Literal

from pydantic import BaseModel, Field


class Insight(BaseModel):
    finding: str = Field(min_length=1)
    evidence: list[str]
    recommendation: str = Field(min_length=1)


class ChartConfig(BaseModel):
    type: Literal["bar", "line", "scatter", "pie"]
    x: str
    y: str
    title: str
