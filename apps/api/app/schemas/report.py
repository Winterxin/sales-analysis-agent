from __future__ import annotations

from pydantic import BaseModel, Field


class ModuleReport(BaseModel):
    module_id: str
    title: str
    chart_type: str
    summary_metrics: dict[str, object] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, object]]] = Field(default_factory=dict)
    chart_payload: dict[str, object] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    task_id: str
    dataset_type: str
    module_count: int
    summary: list[str] = Field(default_factory=list)
    modules: list[ModuleReport] = Field(default_factory=list)
