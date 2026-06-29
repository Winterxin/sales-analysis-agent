from __future__ import annotations

from app.schemas.notebook_narrative import NotebookNarrative, NotebookSectionNarrative
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook.evidence_formatter import collect_evidence_items, collect_risk_items
from app.services.notebook.field_utils import field_list, mapped_fields
from app.services.notebook.formatters import format_percent, format_scalar
from app.services.notebook.report_utils import first_metric, module_map, narrative_map


def test_notebook_public_formatters_and_field_utils_keep_existing_text() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
    )

    assert format_scalar(1234.5) == "1,234.50"
    assert format_percent(0.1872) == "18.72%"
    assert mapped_fields(schema_mapping) == {"order_datetime", "sales_amount"}
    assert field_list({"sales_amount", "order_datetime"}) == "订单日期, 销售额"


def test_notebook_public_report_and_evidence_utils_read_existing_report_shapes() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["总体利润率存在风险。"],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣利润分析",
                chart_type="scatter",
                summary_metrics={"total_sales_amount": 1234.5},
                findings=["高折扣订单存在利润侵蚀风险。"],
                warnings=["负利润记录需要复盘。"],
                tables={"top": [{"sku": "A", "sales": 10}]},
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="discount_and_profit",
                intro="折扣利润说明。",
            )
        ],
    )

    assert module_map(report)["discount_profit_analysis"].title == "折扣利润分析"
    assert narrative_map(narrative)["discount_and_profit"].intro == "折扣利润说明。"
    assert first_metric(report, "total_sales_amount") == 1234.5
    assert "高折扣订单存在利润侵蚀风险。" in collect_evidence_items(report)
    assert "负利润记录需要复盘。" in collect_risk_items(report)
