from __future__ import annotations

import pandas as pd

from app.analysis.runner import run_analysis
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_planner import build_analysis_plan
from app.services.chart_selection_planner import build_chart_selection_plan
from app.services.csv_ingestion import ingest_csv
from app.services.dataset_profile import build_dataset_profile, select_analysis_focuses
from app.services.notebook_planner import build_section_priority
from app.schemas.notebook_content import NotebookSectionContent
from app.services.notebook_content_planner import _apply_chart_selection_to_section
from app.services.schema_rules import map_schema_fields


def test_store_weekly_sales_without_quantity_reaches_chart_selection(tmp_path) -> None:
    csv_path = tmp_path / "weekly_sales.csv"
    pd.DataFrame(
        {
            "Store": [1, 1, 2, 2, 3],
            "Dept": [10, 20, 10, 20, 30],
            "Date": ["2024-01-05", "2024-01-12", "2024-02-02", "2024-02-09", "2024-03-01"],
            "Weekly_Sales": [1200.0, 1500.0, 900.0, 1100.0, 800.0],
            "IsHoliday": [False, False, True, False, False],
        }
    ).to_csv(csv_path, index=False)

    schema_mapping = map_schema_fields(["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"])
    ingestion = ingest_csv(csv_path)
    plan = build_analysis_plan(schema_mapping, ingestion)
    report = run_analysis(
        task_id="weekly-sales",
        csv_path=csv_path,
        schema_mapping=schema_mapping,
        plan=plan,
    )
    profile = build_dataset_profile(pd.read_csv(csv_path), schema_mapping)
    focus = select_analysis_focuses(profile)
    section_priority = build_section_priority(
        schema_mapping=schema_mapping,
        analysis_plan=plan,
        dataset_profile=profile,
        analysis_focus=focus,
    )
    chart_plan = build_chart_selection_plan(
        schema_mapping=schema_mapping,
        dataset_profile=profile,
        analysis_focus=focus,
        section_priority=section_priority,
    )

    trend_module = next(module for module in report.modules if module.module_id == "sales_trend_analysis")
    assert trend_module.summary_metrics["skipped_quantity_dependent_analysis"] is True
    assert any("missing quantity" in warning for warning in trend_module.warnings)
    assert chart_plan["selected_charts"]
    selected_ids = {chart["chart_id"] for chart in chart_plan["selected_charts"]}
    assert "sales_trends_monthly_line" in selected_ids or "category_concentration_bar" in selected_ids


def test_sales_sample_and_online_retail_alignment_regression_smoke() -> None:
    sales_section = NotebookSectionContent(
        section_id="sales_trends",
        markdown_blocks=[],
        code_cells=[
            "# chart_id: sales_trends_monthly_line\nfig.show()",
            "# chart_id: profit_distribution_if_available\nfig.show()",
        ],
    )
    retail_section = NotebookSectionContent(
        section_id="order_structure",
        markdown_blocks=[],
        code_cells=[
            "# chart_id: invoice_value_distribution\nfig.show()",
            "# chart_id: quantity_distribution\nfig.show()",
        ],
    )

    sales_filtered = _apply_chart_selection_to_section(
        sales_section,
        {
            "selected_charts": [
                {"section_id": "sales_trends", "chart_id": "sales_trends_monthly_line"},
            ],
            "intent_guided_selection_trace": {
                "remove_only_chart_ids": ["profit_distribution_if_available"],
            },
        },
    )
    retail_filtered = _apply_chart_selection_to_section(
        retail_section,
        {
            "selected_charts": [
                {"section_id": "order_structure", "chart_id": "invoice_value_distribution"},
            ],
            "llm_selection_trace": {"remove_only_chart_ids": ["quantity_distribution"]},
        },
    )

    sales_code = "\n".join(sales_filtered.code_cells)
    retail_code = "\n".join(retail_filtered.code_cells)
    assert "sales_trends_monthly_line" in sales_code
    assert "profit_distribution_if_available" not in sales_code
    assert "invoice_value_distribution" in retail_code
    assert "quantity_distribution" not in retail_code
