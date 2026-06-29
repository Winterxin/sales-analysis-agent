from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.analysis.runner import run_analysis
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_builder import build_notebook
from app.services.notebook_planner import build_notebook_outline


def _schema_mapping() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Quantity": "quantity",
            "Category": "category",
            "Region": "region",
        },
        confidence=0.95,
    )


def _modeling_csv(path: Path, rows: int = 120) -> Path:
    data: list[dict[str, object]] = []
    for index in range(rows):
        is_loss = index % 4 == 0
        data.append(
            {
                "Order Date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index),
                "Sales": 100 + index,
                "Profit": -15 if is_loss else 25,
                "Discount": 0.35 if is_loss else 0.05,
                "Quantity": 1 + (index % 5),
                "Category": "Furniture" if is_loss else "Technology",
                "Region": "West" if index % 2 == 0 else "East",
            }
        )
    csv_path = path / "modeling.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def _invalid_profit_csv(path: Path) -> Path:
    data: list[dict[str, object]] = []
    profits: list[object] = [-15] * 20 + [25] * 25 + ["not available"] * 20
    for index, profit in enumerate(profits):
        data.append(
            {
                "Order Date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index),
                "Sales": 100 + index,
                "Profit": profit,
                "Discount": 0.35 if isinstance(profit, int) and profit < 0 else 0.05,
                "Quantity": 1 + (index % 5),
                "Category": "Furniture" if index % 2 == 0 else "Technology",
                "Region": "West" if index % 2 == 0 else "East",
            }
        )
    csv_path = path / "invalid_profit.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def _modeling_plan() -> AnalysisPlan:
    return AnalysisPlan(
        analysis_plan=["data_quality_check", "loss_risk_modeling"],
        chart_preferences={"loss_risk_modeling": "modeling"},
        reasoning_summary=[],
    )


def _markdown_text(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )


def _table_data_row_count_after(markdown: str, heading: str) -> int:
    start = markdown.index(heading)
    tail = markdown[start:]
    next_h3 = tail.find("\n### ", 1)
    next_h4 = tail.find("\n#### ", 1)
    next_candidates = [value for value in (next_h3, next_h4) if value != -1]
    next_heading = min(next_candidates) if next_candidates else -1
    section = tail if next_heading == -1 else tail[:next_heading]
    return sum(
        1
        for line in section.splitlines()
        if line.startswith("|") and "---" not in line and not line.startswith("| row_index")
    )


def test_runner_executes_loss_risk_modeling_module(tmp_path: Path) -> None:
    report = run_analysis(
        task_id="task-modeling",
        csv_path=_modeling_csv(tmp_path),
        schema_mapping=_schema_mapping(),
        plan=_modeling_plan(),
    )

    module_ids = [module.module_id for module in report.modules]
    assert "loss_risk_modeling" in module_ids
    modeling_module = next(module for module in report.modules if module.module_id == "loss_risk_modeling")
    assert modeling_module.chart_type == "modeling"


def test_runner_keeps_invalid_profit_out_of_modeling_target(tmp_path: Path) -> None:
    report = run_analysis(
        task_id="tatest-api-key",
        csv_path=_invalid_profit_csv(tmp_path),
        schema_mapping=_schema_mapping(),
        plan=AnalysisPlan(
            analysis_plan=["loss_risk_modeling"],
            chart_preferences={"loss_risk_modeling": "modeling"},
            reasoning_summary=[],
        ),
    )

    modeling_module = next(module for module in report.modules if module.module_id == "loss_risk_modeling")
    assert not modeling_module.tables.get("model_comparison")
    assert any(
        "insufficient_sample_size" in warning or "no_valid_profit_values" in warning
        for warning in modeling_module.warnings
    )


def test_notebook_includes_modeling_section_for_successful_modeling(tmp_path: Path) -> None:
    report = run_analysis(
        task_id="task-modeling",
        csv_path=_modeling_csv(tmp_path),
        schema_mapping=_schema_mapping(),
        plan=_modeling_plan(),
    )
    outline = build_notebook_outline(_schema_mapping(), _modeling_plan())

    notebook_path = build_notebook(
        task_id="task-modeling",
        output_dir=tmp_path / "notebook",
        report=report,
        schema_mapping=_schema_mapping(),
        plan=_modeling_plan(),
        outline=outline,
    )

    markdown = _markdown_text(notebook_path)
    assert "Modeling Analysis: Loss Risk Review" in markdown
    assert "is_loss = profit < 0" in markdown
    for heading in [
        "Modeling Analysis",
        "threshold",
        "阈值取舍与混淆矩阵",
        "风险特征解释",
        "Usage Limits",
    ]:
        assert heading in markdown
    for old_heading in [
        "### Modeling Goal",
        "### Target Definition",
        "### Data and Validation Design",
        "### Model Selection Explanation",
        "### Feature Importance Business Grouping",
        "### Error Sample Analysis",
        "### High-risk Order Examples",
    ]:
        assert old_heading not in markdown
    assert "False Positive Examples" not in markdown
    assert "feature_importance" not in markdown
    assert _table_data_row_count_after(markdown, "High-risk Examples") <= 3
    assert _table_data_row_count_after(markdown, "False Negative") <= 3
    assert "Usage Limits" in markdown


def test_notebook_includes_modeling_skip_reason_when_modeling_is_skipped(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="loss_risk_modeling",
                title="亏损风险建模",
                chart_type="modeling",
                warnings=["modeling_plan_skipped: single_target_class"],
            )
        ],
    )
    outline = build_notebook_outline(_schema_mapping(), _modeling_plan())

    notebook_path = build_notebook(
        task_id="tatest-api-key",
        output_dir=tmp_path / "skipped",
        report=report,
        schema_mapping=_schema_mapping(),
        plan=_modeling_plan(),
        outline=outline,
    )

    markdown = _markdown_text(notebook_path)
    assert "Modeling Analysis: Loss Risk Review" in markdown
    assert "The current data is not suitable for loss-risk modeling" in markdown
    assert "single_target_class" in markdown
    assert "causal claims" in markdown
    assert "Threshold Analysis" not in markdown
    assert "Error Sample Analysis" not in markdown
