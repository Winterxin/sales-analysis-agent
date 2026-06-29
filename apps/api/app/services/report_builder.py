from __future__ import annotations

import re

from app.schemas.report import AnalysisReport
from app.services.output_language import is_english_output


ENGLISH_MODULE_TITLES = {
    "data_quality_check": "Data Quality Check",
    "metric_distribution_analysis": "Metric Distribution Analysis",
    "sales_trend_analysis": "Sales Trend Analysis",
    "product_contribution_analysis": "Product and Category Analysis",
    "dimension_breakdown_analysis": "Segment and Region Analysis",
    "country_market_analysis": "Country Market Analysis",
    "order_structure_analysis": "Order Structure Analysis",
    "discount_profit_analysis": "Discount and Profit Analysis",
    "loss_risk_modeling": "Loss-risk Modeling",
    "forecast_analysis": "Forecast Analysis",
}


def _contains_cjk(text: object) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text)))


def build_html_report(report: AnalysisReport, output_language: str | None = None) -> str:
    if output_language is not None and is_english_output(output_language):
        summary_items = [item for item in report.summary if not _contains_cjk(item)]
        if not summary_items:
            summary_items = ["Summary is available in report.json."]
        sections = [
            "<html lang='en'><head><meta charset='utf-8'><title>Sales Analysis Report</title></head><body>",
            f"<h1>Sales Analysis Report</h1><p>Task: {report.task_id}</p>",
            "<h2>Summary</h2>",
            "<ul>",
        ]
        sections.extend(f"<li>{item}</li>" for item in summary_items)
        sections.extend(["</ul>", "<h2>Analysis Modules</h2>", "<ul>"])
        for module in report.modules:
            title = ENGLISH_MODULE_TITLES.get(module.module_id, "Analysis Module")
            sections.append(f"<li>{title}</li>")
        sections.append("</ul></body></html>")
        return "".join(sections)

    sections = [
        "<html><head><meta charset='utf-8'><title>Sales Analysis Report</title></head><body>",
        f"<h1>Sales Analysis Report</h1><p>Task: {report.task_id}</p>",
        "<h2>Summary</h2>",
        "<ul>",
    ]
    sections.extend(f"<li>{item}</li>" for item in report.summary)
    sections.append("</ul>")
    for module in report.modules:
        sections.append(f"<h3>{module.title}</h3>")
        sections.append("<ul>")
        sections.extend(f"<li>{finding}</li>" for finding in module.findings)
        sections.append("</ul>")
    sections.append("</body></html>")
    return "".join(sections)
