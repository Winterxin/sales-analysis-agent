from __future__ import annotations

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.ingestion import IngestionSummary
from app.schemas.notebook_outline import NotebookOutline
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_planner import build_analysis_plan
import app.services.notebook_planner as notebook_planner
from app.services.notebook_planner import (
    build_notebook_outline,
    build_notebook_outline_with_trace,
)


def test_notebook_planner_builds_rich_sales_notebook_outline() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Segment": "segment",
            "Region": "region",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Product Name": "product_name",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Discount": "discount",
            "Profit": "profit",
        },
        confidence=0.92,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "metric_distribution_analysis",
            "sales_trend_analysis",
            "product_contribution_analysis",
            "dimension_breakdown_analysis",
            "forecast_analysis",
        ],
        chart_preferences={
            "metric_distribution_analysis": "histogram",
            "sales_trend_analysis": "line",
            "product_contribution_analysis": "bar",
            "dimension_breakdown_analysis": "stacked_bar",
            "forecast_analysis": "line",
        },
        reasoning_summary=[],
    )

    outline = build_notebook_outline(schema_mapping, analysis_plan)

    assert isinstance(outline, NotebookOutline)
    assert outline.title == "Sales Data Analysis Notebook"
    assert [section.section_id for section in outline.sections] == [
        "title_and_goal",
        "dataset_and_schema",
        "data_cleaning",
        "metric_distributions",
        "sales_trends",
        "product_and_category",
        "segment_and_region",
        "discount_and_profit",
        "conclusions",
    ]
    assert all(section.title != "Forecast and Outlook" for section in outline.sections)


def test_analysis_plan_enables_metric_distribution_when_core_numeric_fields_exist() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Discount": "discount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    ingestion = IngestionSummary(
        columns=["Order ID", "Order Date", "Sales", "Quantity", "Discount", "Profit"],
        preview_rows=[],
        datetime_candidates=["Order Date"],
        numeric_candidates=["Sales", "Quantity", "Discount", "Profit"],
        row_count=100,
        column_count=6,
        delimiter=",",
        encoding="utf-8",
    )

    plan = build_analysis_plan(schema_mapping, ingestion)

    assert "metric_distribution_analysis" in plan.analysis_plan
    assert plan.analysis_plan.index("metric_distribution_analysis") == 1
    assert plan.chart_preferences["metric_distribution_analysis"] == "histogram"


def test_analysis_plan_adds_loss_risk_modeling_when_profit_exists() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
        },
        confidence=0.9,
    )
    ingestion = IngestionSummary(
        columns=["Order Date", "Sales", "Profit", "Discount"],
        preview_rows=[],
        datetime_candidates=["Order Date"],
        numeric_candidates=["Sales", "Profit", "Discount"],
        row_count=100,
        column_count=4,
        delimiter=",",
        encoding="utf-8",
    )

    plan = build_analysis_plan(schema_mapping, ingestion)

    assert "loss_risk_modeling" in plan.analysis_plan
    assert plan.chart_preferences["loss_risk_modeling"] == "modeling"
    assert any("亏损风险建模" in item or "loss-risk modeling" in item.lower() for item in plan.reasoning_summary)


class FakeNotebookLLMClient:
    enabled = True

    def suggest_notebook_outline(
        self, schema_mapping, analysis_plan, fallback_outline, language_instruction: str | None = None
    ):
        return {
            "title": "Sales Notebook",
            "sections": [
                {
                    "section_id": "title_and_goal",
                    "title": "Analysis Goal",
                    "purpose": "Explain notebook scope.",
                },
                {
                    "section_id": "dataset_and_schema",
                    "title": "Dataset Overview",
                    "purpose": "Summarize fields.",
                },
                {
                    "section_id": "sales_trends",
                    "title": "Trend Story",
                    "purpose": "Explain time movement.",
                },
                {
                    "section_id": "product_and_category",
                    "title": "Product Story",
                    "purpose": "Explain product concentration.",
                },
                {
                    "section_id": "conclusions",
                    "title": "Wrap Up",
                    "purpose": "State conclusions.",
                },
                {
                    "section_id": "invented_section",
                    "title": "Invalid",
                    "purpose": "Should be removed.",
                },
            ],
        }


class FakeReorderedNotebookLLMClient:
    enabled = True

    def suggest_notebook_outline(
        self, schema_mapping, analysis_plan, fallback_outline, language_instruction: str | None = None
    ):
        return {
            "title": "Sales Notebook",
            "sections": [
                {"section_id": "title_and_goal", "purpose": "Goal."},
                {"section_id": "dataset_and_schema", "purpose": "Schema."},
                {"section_id": "data_cleaning", "purpose": "Clean."},
                {"section_id": "discount_and_profit", "purpose": "Discount first."},
                {"section_id": "product_and_category", "purpose": "Product later."},
                {"section_id": "sales_trends", "purpose": "Trend later."},
                {"section_id": "modeling", "purpose": "Modeling."},
                {"section_id": "conclusions", "purpose": "Wrap."},
                {"section_id": "invented_section", "purpose": "Should be removed."},
            ],
        }


class FakeMissingModelingNotebookLLMClient:
    enabled = True

    def suggest_notebook_outline(
        self, schema_mapping, analysis_plan, fallback_outline, language_instruction: str | None = None
    ):
        return {
            "title": "Sales Notebook",
            "sections": [
                {"section_id": "title_and_goal", "purpose": "Goal."},
                {"section_id": "dataset_and_schema", "purpose": "Schema."},
                {"section_id": "data_cleaning", "purpose": "Clean."},
                {"section_id": "sales_trends", "purpose": "Trend."},
                {"section_id": "discount_and_profit", "purpose": "Discount."},
                {"section_id": "conclusions", "purpose": "Wrap."},
            ],
        }


def test_notebook_planner_accepts_llm_outline_but_sanitizes_unknown_sections() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Product Name": "product_name",
            "Sales": "sales_amount",
            "Quantity": "quantity",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "product_contribution_analysis",
        ],
        chart_preferences={},
        reasoning_summary=[],
    )

    outline = build_notebook_outline(
        schema_mapping,
        analysis_plan,
        llm_client=FakeNotebookLLMClient(),
    )

    assert outline.title == "Sales Data Analysis Notebook"
    assert [section.section_id for section in outline.sections] == [
        "title_and_goal",
        "dataset_and_schema",
        "data_cleaning",
        "sales_trends",
        "product_and_category",
        "conclusions",
    ]
    assert [section.title for section in outline.sections] == [
        "Analysis Goal",
        "Dataset and Field Mapping",
        "Data Cleaning and Preparation",
        "Sales Trend Analysis",
        "Product and Category Analysis",
        "Conclusions and Recommended Actions",
    ]


def test_notebook_planner_keeps_discount_section_as_profit_overview_when_profit_exists() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Product Name": "product_name",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "product_contribution_analysis",
            "discount_profit_analysis",
        ],
        chart_preferences={},
        reasoning_summary=[],
    )

    outline = build_notebook_outline(schema_mapping, analysis_plan)

    assert "discount_and_profit" in [section.section_id for section in outline.sections]


def test_notebook_planner_adds_modeling_section_when_modeling_is_planned() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount", "Profit": "profit"},
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "discount_profit_analysis",
            "loss_risk_modeling",
        ],
        chart_preferences={"loss_risk_modeling": "modeling"},
        reasoning_summary=[],
    )

    outline = build_notebook_outline(schema_mapping, analysis_plan)

    assert "modeling" in [section.section_id for section in outline.sections]


def test_analysis_plan_does_not_enable_product_module_without_product_fields() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Region": "region",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    ingestion = IngestionSummary(
        columns=["Order ID", "Order Date", "Sales", "Quantity", "Region"],
        preview_rows=[],
        datetime_candidates=["Order Date"],
        numeric_candidates=["Sales", "Quantity"],
        row_count=100,
        column_count=5,
        delimiter=",",
        encoding="utf-8",
    )

    plan = build_analysis_plan(schema_mapping, ingestion)

    assert "product_contribution_analysis" not in plan.analysis_plan


def test_notebook_planner_skips_segment_region_when_only_channel_and_store_exist() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Product Name": "product_name",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Channel": "channel",
            "Store": "store",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "product_contribution_analysis",
            "dimension_breakdown_analysis",
        ],
        chart_preferences={},
        reasoning_summary=[],
    )

    outline = build_notebook_outline(schema_mapping, analysis_plan)

    assert "segment_and_region" not in [section.section_id for section in outline.sections]


def test_analysis_plan_skips_dimension_breakdown_when_only_channel_and_store_exist() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Product Name": "product_name",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Channel": "channel",
            "Store": "store",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    ingestion = IngestionSummary(
        columns=["Order ID", "Order Date", "Product Name", "Sales", "Quantity", "Channel", "Store"],
        preview_rows=[],
        datetime_candidates=["Order Date"],
        numeric_candidates=["Sales", "Quantity"],
        row_count=100,
        column_count=7,
        delimiter=",",
        encoding="utf-8",
    )

    plan = build_analysis_plan(schema_mapping, ingestion)

    assert "dimension_breakdown_analysis" not in plan.analysis_plan


def test_notebook_planner_reports_disabled_trace_without_llm() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Product Name": "product_name",
            "Sales": "sales_amount",
            "Quantity": "quantity",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "product_contribution_analysis",
        ],
        chart_preferences={},
        reasoning_summary=[],
    )

    outline, trace = build_notebook_outline_with_trace(schema_mapping, analysis_plan)

    assert outline.sections
    assert trace.stage == "notebook_outline"
    assert trace.status == "disabled"
    assert trace.llm_enabled is False
    assert trace.attempted is False
    assert trace.applied is False


def test_focus_priority_keeps_stable_default_section_order(monkeypatch) -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Segment": "segment",
            "Region": "region",
            "Category": "category",
            "Order ID": "order_id",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["loss_risk_modeling"],
        chart_preferences={"loss_risk_modeling": "modeling"},
        reasoning_summary=[],
    )

    def fake_priority(**kwargs):
        return {
            "core_sections": [
                "discount_and_profit",
                "segment_and_region",
                "product_and_category",
            ],
            "support_sections": [
                "metric_distributions",
                "sales_trends",
                "order_structure",
                "modeling",
            ],
            "skipped_sections": [],
            "section_reasons": {},
        }

    monkeypatch.setattr(notebook_planner, "build_section_priority", fake_priority)

    outline = build_notebook_outline(
        schema_mapping,
        analysis_plan,
        analysis_focus={"selected_focuses": ["discount_erosion_focus"]},
    )

    assert [section.section_id for section in outline.sections] == [
        "title_and_goal",
        "dataset_and_schema",
        "data_cleaning",
        "metric_distributions",
        "sales_trends",
        "product_and_category",
        "segment_and_region",
        "order_structure",
        "discount_and_profit",
        "modeling",
        "conclusions",
    ]


def test_focus_priority_can_use_dynamic_section_reordering_when_enabled(monkeypatch) -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Segment": "segment",
            "Region": "region",
            "Category": "category",
            "Order ID": "order_id",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["loss_risk_modeling"],
        chart_preferences={"loss_risk_modeling": "modeling"},
        reasoning_summary=[],
    )

    def fake_priority(**kwargs):
        return {
            "core_sections": [
                "discount_and_profit",
                "segment_and_region",
                "product_and_category",
            ],
            "support_sections": [
                "metric_distributions",
                "sales_trends",
                "order_structure",
                "modeling",
            ],
            "skipped_sections": [],
            "section_reasons": {},
        }

    monkeypatch.setattr(notebook_planner, "build_section_priority", fake_priority)

    outline = build_notebook_outline(
        schema_mapping,
        analysis_plan,
        analysis_focus={"selected_focuses": ["discount_erosion_focus"]},
        dynamic_section_reordering=True,
    )
    section_ids = [section.section_id for section in outline.sections]

    assert section_ids.index("discount_and_profit") < section_ids.index("product_and_category")


def test_discount_focus_default_outline_keeps_product_before_discount() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Product Name": "product_name",
            "Category": "category",
            "Segment": "segment",
            "Region": "region",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "product_contribution_analysis",
            "dimension_breakdown_analysis",
            "order_structure_analysis",
            "discount_profit_analysis",
            "loss_risk_modeling",
        ],
        chart_preferences={"loss_risk_modeling": "modeling"},
        reasoning_summary=[],
    )
    analysis_focus = {
        "selected_focuses": [
            "discount_erosion_focus",
            "profit_quality_focus",
            "segment_region_focus",
            "product_concentration_focus",
        ],
        "support_focuses": [
            "customer_order_structure_focus",
            "trend_volatility_focus",
        ],
    }

    outline = build_notebook_outline(
        schema_mapping,
        analysis_plan,
        dataset_profile={"has_discount": True, "has_profit": True, "country_count": 1},
        analysis_focus=analysis_focus,
    )
    section_ids = [section.section_id for section in outline.sections]

    assert section_ids == [
        "title_and_goal",
        "dataset_and_schema",
        "data_cleaning",
        "sales_trends",
        "product_and_category",
        "segment_and_region",
        "order_structure",
        "discount_and_profit",
        "modeling",
        "conclusions",
    ]
    assert section_ids.index("product_and_category") < section_ids.index("discount_and_profit")
    assert "discount_and_profit" in section_ids


def test_llm_outline_cannot_reorder_default_sections() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Product Name": "product_name",
            "Category": "category",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "product_contribution_analysis",
            "discount_profit_analysis",
            "loss_risk_modeling",
        ],
        chart_preferences={"loss_risk_modeling": "modeling"},
        reasoning_summary=[],
    )

    outline, trace = build_notebook_outline_with_trace(
        schema_mapping,
        analysis_plan,
        llm_client=FakeReorderedNotebookLLMClient(),
    )
    section_ids = [section.section_id for section in outline.sections]

    assert trace.status in {"llm_applied", "llm_no_change"}
    assert section_ids == [
        "title_and_goal",
        "dataset_and_schema",
        "data_cleaning",
        "sales_trends",
        "product_and_category",
        "discount_and_profit",
        "modeling",
        "conclusions",
    ]
    assert section_ids.index("product_and_category") < section_ids.index("discount_and_profit")
    assert "invented_section" not in section_ids


def test_llm_outline_cannot_delete_mandatory_modeling_when_planned() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "discount_profit_analysis",
            "loss_risk_modeling",
        ],
        chart_preferences={"loss_risk_modeling": "modeling"},
        reasoning_summary=[],
    )

    outline, trace = build_notebook_outline_with_trace(
        schema_mapping,
        analysis_plan,
        llm_client=FakeMissingModelingNotebookLLMClient(),
    )
    section_ids = [section.section_id for section in outline.sections]

    assert trace.status in {"llm_applied", "llm_no_change"}
    assert "modeling" in section_ids
    assert section_ids.index("modeling") < section_ids.index("conclusions")


def test_llm_outline_does_not_force_modeling_when_not_planned() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["data_quality_check", "sales_trend_analysis", "discount_profit_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    outline = build_notebook_outline(
        schema_mapping,
        analysis_plan,
        llm_client=FakeMissingModelingNotebookLLMClient(),
    )

    assert "modeling" not in [section.section_id for section in outline.sections]


def test_outline_includes_modeling_when_modeling_opportunity_exists_without_loss_module() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Customer ID": "customer_id",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["data_quality_check", "sales_trend_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    outline = build_notebook_outline(
        schema_mapping,
        analysis_plan,
        modeling_opportunity_plan={
            "recommended_modeling_task": "sales_amount_forecast_or_regression",
            "decision_status": "opportunity_only",
        },
    )

    section_ids = [section.section_id for section in outline.sections]
    assert "modeling" in section_ids
    assert section_ids.index("modeling") < section_ids.index("conclusions")


def test_focus_priority_excludes_skipped_sections_from_narrative_order(monkeypatch) -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Country": "country",
        },
        confidence=0.9,
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["country_market_analysis", "discount_profit_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    def fake_priority(**kwargs):
        return {
            "core_sections": ["discount_and_profit", "country_market"],
            "support_sections": ["metric_distributions", "sales_trends"],
            "skipped_sections": ["discount_and_profit", "country_market"],
            "section_reasons": {},
        }

    monkeypatch.setattr(notebook_planner, "build_section_priority", fake_priority)

    outline = build_notebook_outline(
        schema_mapping,
        analysis_plan,
        analysis_focus={"selected_focuses": ["country_market_focus"]},
    )
    section_ids = [section.section_id for section in outline.sections]

    assert section_ids == [
        "title_and_goal",
        "dataset_and_schema",
        "data_cleaning",
        "metric_distributions",
        "sales_trends",
        "conclusions",
    ]
    assert "country_market" not in section_ids
    assert "discount_and_profit" not in section_ids
