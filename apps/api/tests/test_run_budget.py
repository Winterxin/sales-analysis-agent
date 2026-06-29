from __future__ import annotations

from app.services.run_budget import (
    DemoSafeChartDecisionLLMClient,
    RunBudget,
    build_skipped_stage_trace,
)
from app.application.analysis_run_service import _content_has_llm_chart_decision


def test_demo_safe_stage_allowlist_keeps_agent_critical_stages() -> None:
    budget = RunBudget(demo_safe=True, total_seconds=90, min_stage_seconds=5)

    assert budget.stage_allowed("notebook_outline")
    assert budget.stage_allowed("notebook_content")
    assert budget.stage_allowed("postrun_chart_reflection")
    assert budget.stage_allowed("notebook_revision_decision")
    assert not budget.stage_allowed("notebook_narrative")
    assert not budget.stage_allowed("report_summary")


def test_budget_blocks_allowed_stage_when_remaining_time_is_too_low() -> None:
    budget = RunBudget(demo_safe=True, total_seconds=1, min_stage_seconds=5)

    assert not budget.has_budget_for_stage("notebook_content")


def test_skipped_budget_trace_records_remaining_budget() -> None:
    budget = RunBudget(demo_safe=True, total_seconds=30, min_stage_seconds=5)

    trace = build_skipped_stage_trace(
        stage="notebook_content",
        llm_client=None,
        budget=budget,
        status="skipped_by_budget",
        reason="Not enough budget.",
    )

    assert trace.stage == "notebook_content"
    assert trace.status == "skipped_by_budget"
    assert trace.fallback_type == "budget"
    assert trace.elapsed_ms is None
    assert trace.prompt_chars is None
    assert trace.budget_remaining_ms is not None
    assert trace.budget_remaining_ms <= 30000


class FullLLMClient:
    enabled = True
    source = "test"
    configured_model = "fake-model"
    last_completion_metrics = {}

    def suggest_sales_trend_strategy(self, payload):
        return {"views": [{"chart": "daily_rolling_line"}]}

    def suggest_metric_distribution_strategy(self, payload):
        return {"decisions": [{"metric": "Sales", "charts": ["histogram"]}]}

    def suggest_product_category_strategy(self, payload):
        return {"views": [{"chart": "top_product_bar"}]}

    def suggest_segment_region_strategy(self, payload):
        return {"views": [{"chart": "segment_region_sales_heatmap"}]}

    def suggest_discount_profit_strategy(self, payload):
        return {"views": [{"chart": "discount_vs_profit"}]}

    def suggest_notebook_section_content(self, *args, **kwargs):
        return {"section_id": "sales_trends", "markdown_blocks": [], "code_cells": []}


def test_demo_safe_chart_decision_client_hides_section_content_generation() -> None:
    client = DemoSafeChartDecisionLLMClient(FullLLMClient())

    assert client.enabled
    assert client.suggest_sales_trend_strategy({}) == {
        "views": [{"chart": "daily_rolling_line"}]
    }
    assert client.suggest_metric_distribution_strategy({}) == {
        "decisions": [{"metric": "Sales", "charts": ["histogram"]}]
    }
    assert client.suggest_product_category_strategy({}) == {
        "views": [{"chart": "top_product_bar"}]
    }
    assert client.suggest_segment_region_strategy({}) == {
        "views": [{"chart": "segment_region_sales_heatmap"}]
    }
    assert client.suggest_discount_profit_strategy({}) == {
        "views": [{"chart": "discount_vs_profit"}]
    }
    assert not hasattr(client, "suggest_notebook_section_content")


class Section:
    def __init__(self, code_cells: list[str]) -> None:
        self.code_cells = code_cells


class Content:
    def __init__(self, sections: list[Section]) -> None:
        self.sections = sections


def test_content_chart_decision_detection_requires_llm_sanitized_code() -> None:
    assert _content_has_llm_chart_decision(
        Content([Section(["trend_strategy = {'source': 'llm_sanitized'}"])])
    )
    assert not _content_has_llm_chart_decision(
        Content([Section(["trend_strategy = {'source': 'deterministic_fallback'}"])])
    )
