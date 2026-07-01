from __future__ import annotations

import re

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_content_planner import build_notebook_content_with_trace


CORE_SECTIONS = [
    "dataset_and_schema",
    "data_cleaning",
    "metric_distributions",
    "sales_trends",
    "product_and_category",
    "segment_and_region",
    "discount_and_profit",
    "forecast",
]


def _outline(section_ids: list[str]) -> NotebookOutline:
    return NotebookOutline(
        title="Regression Notebook",
        sections=[
            NotebookSection(
                section_id=section_id,
                title=section_id.replace("_", " ").title(),
                purpose=f"Exercise fallback content for {section_id}.",
            )
            for section_id in section_ids
        ],
    )


def _analysis_plan(section_ids: list[str]) -> AnalysisPlan:
    module_by_section = {
        "data_cleaning": "data_quality_check",
        "metric_distributions": "metric_distribution_analysis",
        "sales_trends": "sales_trend_analysis",
        "product_and_category": "product_contribution_analysis",
        "segment_and_region": "dimension_breakdown_analysis",
        "discount_and_profit": "discount_profit_analysis",
        "forecast": "forecast_analysis",
        "country_market": "country_market_analysis",
        "order_structure": "order_structure_analysis",
    }
    return AnalysisPlan(
        analysis_plan=[
            module_by_section[section_id]
            for section_id in section_ids
            if section_id in module_by_section
        ],
        chart_preferences={},
        reasoning_summary=[],
    )


def _chart_selection(selected: dict[str, list[str]]) -> dict[str, object]:
    return {
        "selected_charts": [
            {
                "section_id": section_id,
                "chart_id": chart_id,
                "title": chart_id.replace("_", " ").title(),
            }
            for section_id, chart_ids in selected.items()
            for chart_id in chart_ids
        ]
    }


def _build_content(
    *,
    section_ids: list[str],
    field_mapping: dict[str, str],
    selected_charts: dict[str, list[str]] | None = None,
    output_language: str | None = None,
):
    content, trace = build_notebook_content_with_trace(
        outline=_outline(section_ids),
        narrative=NotebookNarrative(sections=[], suggested_followups=[]),
        report=AnalysisReport(
            task_id="notebook-fallback-regression",
            dataset_type="sales_transaction",
            module_count=0,
            summary=[],
            modules=[],
        ),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping=field_mapping,
            confidence=0.9,
            missing_required_fields=[],
            uncertain_fields=[],
        ),
        analysis_plan=_analysis_plan(section_ids),
        chart_selection_plan=_chart_selection(selected_charts or {}),
        output_language=output_language,
    )
    return content, trace


def _sections_by_id(content) -> dict[str, object]:
    return {section.section_id: section for section in content.sections}


def _combined_code(section) -> str:
    return "\n\n".join(section.code_cells)


def _combined_markdown(section) -> str:
    return "\n\n".join(section.markdown_blocks)


def test_superstore_fallback_plan_keeps_core_sections_charts_and_english_output() -> None:
    section_ids = CORE_SECTIONS + ["country_market", "order_structure"]
    content, trace = _build_content(
        section_ids=section_ids,
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Profit": "profit",
            "Discount": "discount",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Product Name": "product_name",
            "Segment": "segment",
            "Region": "region",
            "Country": "country",
            "Customer ID": "customer_id",
            "Order ID": "order_id",
            "Unit Price": "unit_price",
            "DEALSIZE": "deal_size",
            "STATUS": "order_status",
        },
        selected_charts={
            "metric_distributions": ["metric_sales_distribution", "quantity_distribution"],
            "sales_trends": ["sales_trends_monthly_line"],
            "product_and_category": ["category_profit_margin_bar", "top_product_profit_gap_bar"],
            "segment_and_region": ["segment_region_low_margin_table_or_bar"],
            "discount_and_profit": ["discount_vs_profit_scatter", "discount_profit_quality_bar"],
            "country_market": ["country_sales_bar"],
            "order_structure": ["status_order_count_bar"],
        },
        output_language="en",
    )

    sections = _sections_by_id(content)
    assert [section.section_id for section in content.sections] == section_ids
    assert trace.status == "disabled"

    assert "read_csv_with_encoding_fallback" in _combined_code(sections["dataset_and_schema"])
    assert "# chart_id: sales_trends_monthly_line" in _combined_code(sections["sales_trends"])
    assert "monthly_sales =" in _combined_code(sections["sales_trends"])
    assert "# chart_id: category_profit_margin_bar" in _combined_code(sections["product_and_category"])
    assert "# chart_id: top_product_profit_gap_bar" in _combined_code(sections["product_and_category"])
    assert "# chart_id: segment_region_low_margin_table_or_bar" in _combined_code(sections["segment_and_region"])
    assert "# chart_id: discount_profit_quality_bar" in _combined_code(sections["discount_and_profit"])
    assert "# chart_id: country_sales_bar" in _combined_code(sections["country_market"])
    assert "# chart_id: status_order_count_bar" in _combined_code(sections["order_structure"])

    markdown = "\n\n".join(_combined_markdown(section) for section in content.sections)
    assert "### Section Takeaway" in markdown
    assert not re.search(r"[\u4e00-\u9fff]", markdown)


def test_pakistan_ecommerce_fallback_plan_keeps_thin_schema_product_and_order_structure() -> None:
    section_ids = [
        "dataset_and_schema",
        "data_cleaning",
        "metric_distributions",
        "sales_trends",
        "product_and_category",
        "order_structure",
    ]
    content, _trace = _build_content(
        section_ids=section_ids,
        field_mapping={
            "created_at": "order_datetime",
            "grand_total": "sales_amount",
            "qty_ordered": "quantity",
            "category_name_1": "category",
            "item_id": "sku",
            "increment_id": "order_id",
            "payment_method": "payment_method",
            "BI Status": "order_status",
        },
        selected_charts={
            "metric_distributions": ["metric_sales_distribution", "quantity_distribution"],
            "sales_trends": ["sales_trends_monthly_line"],
            "order_structure": ["status_order_count_bar"],
        },
    )

    sections = _sections_by_id(content)
    assert [section.section_id for section in content.sections] == section_ids
    assert "# chart_id: quantity_distribution" in _combined_code(sections["metric_distributions"])
    assert "# chart_id: sales_trends_monthly_line" in _combined_code(sections["sales_trends"])
    assert "category_sales" in _combined_code(sections["product_and_category"])
    assert "category_name_1" in _combined_code(sections["product_and_category"])
    assert "# chart_id: status_order_count_bar" in _combined_code(sections["order_structure"])
    assert "BI Status" in _combined_code(sections["order_structure"])


def test_no_discount_dataset_degrades_discount_section_to_profit_overview() -> None:
    content, _trace = _build_content(
        section_ids=["discount_and_profit"],
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Category": "category",
            "Product Name": "product_name",
        },
    )

    section = content.sections[0]
    code = _combined_code(section)
    markdown = _combined_markdown(section)

    assert section.section_id == "discount_and_profit"
    assert "利润概览" in markdown
    assert "profit_overview" in code
    assert "Discount" not in code
    assert "Product Name" in code
    assert "Category" in code


def test_sparse_field_fallback_plan_preserves_sections_without_optional_dimension_leakage() -> None:
    content, _trace = _build_content(
        section_ids=[
            "metric_distributions",
            "sales_trends",
            "product_and_category",
            "segment_and_region",
            "discount_and_profit",
        ],
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
        },
        selected_charts={
            "metric_distributions": ["metric_sales_distribution"],
            "sales_trends": ["sales_trends_monthly_line"],
        },
    )

    sections = _sections_by_id(content)
    assert set(sections) == {
        "metric_distributions",
        "sales_trends",
        "product_and_category",
        "segment_and_region",
        "discount_and_profit",
    }
    metric_code = _combined_code(sections["metric_distributions"])
    assert "metric_sales_distribution" in metric_code
    assert "missing_selected_chart_render_reason" in metric_code
    assert "# chart_id: sales_trends_monthly_line" in _combined_code(sections["sales_trends"])
    assert "Product Name" not in _combined_code(sections["product_and_category"])
    assert "Category" not in _combined_code(sections["product_and_category"])
    assert "Segment" not in _combined_code(sections["segment_and_region"])
    assert "Region" not in _combined_code(sections["segment_and_region"])
    assert "profit_overview" in _combined_code(sections["discount_and_profit"])
