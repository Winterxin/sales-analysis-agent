from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelingTarget(BaseModel):
    name: str
    source_field: str
    expression: str
    positive_class: str


class ModelingFeaturePlan(BaseModel):
    name: str
    feature_type: Literal["numeric", "categorical", "date", "unknown"] = "unknown"
    reason: str = ""


class ModelingPlan(BaseModel):
    status: Literal["runnable", "skipped"]
    task_type: str = "loss_risk_classification"
    business_goal: str = "Identify records at risk of negative profit for earlier intervention."
    target: ModelingTarget | None = None
    allowed_features: list[ModelingFeaturePlan] = Field(default_factory=list)
    blocked_features: list[ModelingFeaturePlan] = Field(default_factory=list)
    feature_engineering_steps: list[dict[str, Any]] = Field(default_factory=list)
    selected_models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    business_priority_metric: str = "recall"
    business_priority_reason: str = "Loss risk identification prioritizes reducing missed loss cases."
    skip_reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
