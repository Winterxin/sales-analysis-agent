from __future__ import annotations

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_content_planner import build_notebook_content


def test_forecast_chart_analysis_avoids_blocked_template_opener() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="forecast", title="Forecast", purpose="Project trend.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="forecast_analysis",
                title="Forecast",
                chart_type="line",
                findings=[],
                summary_metrics={"baseline_sales_amount": 2172.98, "horizon_days": 28},
                tables={},
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["forecast_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )
    combined_markdown = "\n".join(
        block for section in content.sections for block in section.markdown_blocks
    )

    assert "当前展示的是" not in combined_markdown


def test_forecast_chart_analysis_shows_baseline_metrics_and_high_mape_caution() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="forecast", title="Forecast", purpose="Project trend.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="forecast_analysis",
                title="Forecast",
                chart_type="line",
                findings=["已使用时间顺序切分回测销售额预测 baseline。"],
                summary_metrics={
                    "forecast_status": "baseline_evaluated",
                    "baseline_sales_amount": 2172.98,
                    "horizon_days": 28,
                    "best_baseline": "moving_average_7",
                    "best_baseline_mae": 2542.4,
                    "best_baseline_rmse": 2853.49,
                    "best_baseline_mape": 1690.51,
                    "time_split": "chronological",
                },
                tables={
                    "baseline_metrics": [
                        {"baseline": "naive_last_value", "mae": 3000.0, "rmse": 3200.0, "mape": 1900.0},
                        {"baseline": "moving_average_7", "mae": 2542.4, "rmse": 2853.49, "mape": 1690.51},
                    ]
                },
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["forecast_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )
    combined_markdown = "\n".join(
        block for section in content.sections for block in section.markdown_blocks
    )

    assert "baseline，不是生产级预测" in combined_markdown
    assert "MAE" in combined_markdown
    assert "RMSE" in combined_markdown
    assert "MAPE" in combined_markdown
    assert "1690.51%" in combined_markdown
    assert "波动较大" in combined_markdown
    assert "预测难度高" in combined_markdown
