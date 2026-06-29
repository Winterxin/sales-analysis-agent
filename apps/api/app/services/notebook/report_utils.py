from __future__ import annotations

from typing import Any

from app.schemas.notebook_narrative import NotebookNarrative, NotebookSectionNarrative
from app.schemas.report import AnalysisReport


SECTION_TO_MODULE = {
    "sales_trends": "sales_trend_analysis",
    "product_and_category": "product_contribution_analysis",
    "segment_and_region": "dimension_breakdown_analysis",
    "country_market": "country_market_analysis",
    "order_structure": "order_structure_analysis",
    "discount_and_profit": "discount_profit_analysis",
    "modeling": "loss_risk_modeling",
    "forecast": "forecast_analysis",
}


CORE_SECTION_IDS = {
    "sales_trends",
    "product_and_category",
    "segment_and_region",
    "discount_and_profit",
    "modeling",
    "forecast",
    "metric_distributions",
    "country_market",
    "order_structure",
}


def module_map(report: AnalysisReport) -> dict[str, object]:
    return {module.module_id: module for module in report.modules}


def narrative_map(narrative: NotebookNarrative | None) -> dict[str, NotebookSectionNarrative]:
    if narrative is None:
        return {}
    return {section.section_id: section for section in narrative.sections}


def module_for_section(section_id: str, report: AnalysisReport):
    module_id = SECTION_TO_MODULE.get(section_id)
    if module_id is None:
        return None
    return next((item for item in report.modules if item.module_id == module_id), None)


def first_metric(report: AnalysisReport, *metric_names: str) -> Any:
    for module in report.modules:
        for metric_name in metric_names:
            value = module.summary_metrics.get(metric_name)
            if value is not None:
                return value
    return None
