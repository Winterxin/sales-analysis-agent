from __future__ import annotations

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_content_planner import build_notebook_content


def test_weekly_sales_without_quantity_does_not_generate_quantity_column_access() -> None:
    outline = NotebookOutline(
        title="Weekly Sales Notebook",
        sections=[
            NotebookSection(section_id="data_cleaning", title="Data Cleaning", purpose="Prepare data."),
            NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze sales trend."),
            NotebookSection(
                section_id="metric_distributions",
                title="Metric Distributions",
                purpose="Profile numeric metrics.",
            ),
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="retail_store_weekly",
        field_mapping={
            "Date": "order_datetime",
            "Weekly_Sales": "sales_amount",
            "Store": "store",
            "Dept": "category",
        },
        confidence=0.9,
    )
    chart_selection_plan = {
        "selected_charts": [
            {"chart_id": "sales_trends_monthly_line", "section_id": "sales_trends", "chart_kind": "line"},
            {"chart_id": "metric_sales_distribution", "section_id": "metric_distributions", "chart_kind": "histogram"},
        ],
        "rejected_charts": [
            {"chart_id": "quantity_distribution", "section_id": "metric_distributions", "chart_kind": "histogram"}
        ],
    }

    content = build_notebook_content(
        outline=outline,
        narrative=NotebookNarrative(sections=[], suggested_followups=[]),
        report=AnalysisReport(
            task_id="task-1",
            dataset_type="retail_store_weekly",
            module_count=1,
            summary=[],
            modules=[ModuleReport(module_id="sales_trend_analysis", title="Sales Trend", chart_type="line")],
        ),
        schema_mapping=schema_mapping,
        analysis_plan=AnalysisPlan(analysis_plan=["sales_trend_analysis"], chart_preferences={}, reasoning_summary=[]),
        chart_selection_plan=chart_selection_plan,
    )

    combined_code = "\n\n".join(cell for section in content.sections for cell in section.code_cells)

    assert "clean_df['Quantity']" not in combined_code
    assert "\"Quantity\"" not in combined_code
    assert "'Quantity'" not in combined_code
    assert "quantity_distribution_df" not in combined_code
    assert "selected_chart_ids = ['sales_trends_monthly_line', 'metric_sales_distribution']" in combined_code
