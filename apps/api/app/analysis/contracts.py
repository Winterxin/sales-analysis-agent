from __future__ import annotations

from pydantic import BaseModel, Field


class ModuleResult(BaseModel):
    module_id: str
    title: str
    chart_type: str
    summary_metrics: dict[str, object] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, object]]] = Field(default_factory=dict)
    chart_payload: dict[str, object] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
