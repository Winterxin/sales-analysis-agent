from __future__ import annotations

import time

from app.schemas.llm_trace import LLMStageTrace
from app.services.llm_trace_utils import build_llm_stage_trace

DEMO_SAFE_LLM_STAGE_ALLOWLIST = {
    "analysis_agent_plan",
    "analysis_agent_inspect",
    "notebook_outline",
    "notebook_content",
    "client_report",
    "modeling_interpretation",
    "modeling_opportunity_decision",
    "modeling_outcome_interpretation",
    "final_synthesis",
    "notebook_revision_decision",
    "postrun_chart_reflection",
}


class RunBudget:
    def __init__(
        self,
        *,
        demo_safe: bool,
        total_seconds: float | None,
        min_stage_seconds: float,
    ) -> None:
        self.demo_safe = demo_safe
        self.total_seconds = total_seconds if total_seconds and total_seconds > 0 else None
        self.min_stage_seconds = max(0.0, min_stage_seconds)
        self._started_at = time.monotonic()

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at

    def remaining_seconds(self) -> float | None:
        if self.total_seconds is None:
            return None
        return max(0.0, self.total_seconds - self.elapsed_seconds())

    def remaining_ms(self) -> float | None:
        remaining = self.remaining_seconds()
        if remaining is None:
            return None
        return round(remaining * 1000, 3)

    def stage_allowed(self, stage: str) -> bool:
        if not self.demo_safe:
            return True
        return stage in DEMO_SAFE_LLM_STAGE_ALLOWLIST

    def has_budget_for_stage(self, stage: str) -> bool:
        if not self.stage_allowed(stage):
            return False
        remaining = self.remaining_seconds()
        if remaining is None:
            return True
        return remaining >= self.min_stage_seconds


def build_skipped_stage_trace(
    *,
    stage: str,
    llm_client,
    budget: RunBudget,
    status: str,
    reason: str,
) -> LLMStageTrace:
    trace = build_llm_stage_trace(
        stage=stage,
        llm_client=llm_client,
        status=status,
        reason=reason,
        attempted=False,
        applied=False,
    )
    return trace.model_copy(
        update={
            "elapsed_ms": None,
            "prompt_chars": None,
            "response_chars": None,
            "attempt_count": None,
            "error_type": None,
            "cache_status": "bypassed",
            "cache_key": None,
            "remote_elapsed_ms": None,
            "saved_ms": 0.0,
            "budget_remaining_ms": budget.remaining_ms(),
        }
    )


class DemoSafeChartDecisionLLMClient:
    """Expose only the fast chart-decision surface needed for demo-safe notebooks."""

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._wrapped, "enabled", False))

    @property
    def source(self) -> str:
        return str(getattr(self._wrapped, "source", "unknown"))

    @property
    def configured_model(self) -> str | None:
        return getattr(self._wrapped, "configured_model", None)

    @property
    def last_completion_metrics(self) -> dict:
        return getattr(self._wrapped, "last_completion_metrics", {})

    def suggest_sales_trend_strategy(self, payload):
        return self._wrapped.suggest_sales_trend_strategy(payload)

    def suggest_metric_distribution_strategy(self, payload):
        return self._wrapped.suggest_metric_distribution_strategy(payload)

    def suggest_product_category_strategy(self, payload):
        return self._wrapped.suggest_product_category_strategy(payload)

    def suggest_segment_region_strategy(self, payload):
        return self._wrapped.suggest_segment_region_strategy(payload)

    def suggest_discount_profit_strategy(self, payload):
        return self._wrapped.suggest_discount_profit_strategy(payload)
