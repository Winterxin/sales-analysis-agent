from __future__ import annotations

from pydantic import BaseModel, Field


class AnalysisPlan(BaseModel):
    analysis_plan: list[str] = Field(default_factory=list)
    chart_preferences: dict[str, str] = Field(default_factory=dict)
    reasoning_summary: list[str] = Field(default_factory=list)
