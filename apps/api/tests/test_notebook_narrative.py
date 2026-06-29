from __future__ import annotations

from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_narrative import build_notebook_narrative


def test_notebook_narrative_builds_richer_sales_story_from_report() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="Sales Trends",
                purpose="Explain time movement.",
            ),
            NotebookSection(
                section_id="discount_and_profit",
                title="Discount and Profit Analysis",
                purpose="Explain margin dynamics.",
            ),
            NotebookSection(
                section_id="conclusions",
                title="Conclusions",
                purpose="Wrap up.",
            ),
        ],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["Sales grew unevenly.", "Discount risk needs review."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={
                    "total_sales_amount": 2297200.86,
                    "total_quantity": 37873,
                    "peak_period": "2014-03-18",
                    "trough_period": "2015-07-19",
                },
                findings=["覆盖 1237 个时间粒度。"],
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                summary_metrics={"top_product": "Canon imageCLASS 2200 Advanced Copier"},
                findings=["Top 商品集中度较高。"],
            ),
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
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

    narrative = build_notebook_narrative(
        outline=outline,
        report=report,
        schema_mapping=schema_mapping,
    )

    assert narrative.sections[0].section_id == "sales_trends"
    assert "This section reviews sales trends" in narrative.sections[0].intro
    assert "Section purpose: Explain time movement." in narrative.sections[0].key_observations
    assert "Mapped field count: 5" in narrative.sections[0].key_observations
    assert narrative.sections[1].section_id == "discount_and_profit"
    assert narrative.sections[1].followup_question == "Which evidence should the business team validate first?"
    assert narrative.suggested_followups == ["validate_priority_slices"]


def test_notebook_narrative_fallback_includes_modeling_section() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="modeling",
                title="Modeling Analysis: Loss Risk Review",
                purpose="Explain loss risk modeling.",
            )
        ],
    )
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="loss_risk_modeling",
                title="Loss Risk Modeling",
                chart_type="modeling",
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Profit": "profit"},
        confidence=0.9,
    )

    narrative = build_notebook_narrative(
        outline=outline,
        report=report,
        schema_mapping=schema_mapping,
    )

    assert [section.section_id for section in narrative.sections] == ["modeling"]
    modeling_section = narrative.sections[0]
    assert "modeling analysis: loss risk review" in modeling_section.intro
    assert modeling_section.business_takeaway == (
        "Treat the result as decision support and validate it with business context before acting."
    )
    assert modeling_section.followup_question == "Which evidence should the business team validate first?"


class FakeNotebookNarrativeLLMClient:
    enabled = True

    def suggest_notebook_narrative(
        self, outline, report, schema_mapping, fallback_narrative, language_instruction: str | None = None
    ):
        return {
            "sections": [
                {
                    "section_id": "sales_trends",
                    "intro": "Custom trend intro.",
                    "key_observations": ["Observation A"],
                    "business_takeaway": "Trend takeaway.",
                    "followup_question": "What drove the peak?",
                },
                {
                    "section_id": "invented_section",
                    "intro": "Should be dropped.",
                    "key_observations": ["Invalid"],
                    "business_takeaway": "Invalid.",
                    "followup_question": "Invalid?",
                },
            ],
            "suggested_followups": ["discount_and_profit"],
        }


def test_notebook_narrative_allows_llm_enrichment_but_sanitizes_invalid_sections() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="Sales Trends",
                purpose="Explain time movement.",
            ),
            NotebookSection(
                section_id="conclusions",
                title="Conclusions",
                purpose="Wrap up.",
            ),
        ],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Summary."],
        modules=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )

    narrative = build_notebook_narrative(
        outline=outline,
        report=report,
        schema_mapping=schema_mapping,
        llm_client=FakeNotebookNarrativeLLMClient(),
    )

    assert [section.section_id for section in narrative.sections] == ["sales_trends"]
    assert narrative.sections[0].intro == "Custom trend intro."
    assert narrative.suggested_followups == ["discount_and_profit"]


def test_notebook_narrative_degrades_segment_region_story_when_only_one_dimension_exists() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Explain dimensional performance.",
            )
        ],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Segment view only."],
        modules=[
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="维度拆解分析",
                chart_type="bar",
                summary_metrics={
                    "primary_dimension": "segment",
                    "secondary_dimension": None,
                    "group_count": 3,
                },
                findings=["按 segment 完成单维拆解。"],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Segment": "segment",
            "Sales": "sales_amount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )

    narrative = build_notebook_narrative(
        outline=outline,
        report=report,
        schema_mapping=schema_mapping,
    )

    assert narrative.sections[0].section_id == "segment_and_region"
    assert "segment and region" in narrative.sections[0].intro.lower()
    assert "Section purpose: Explain dimensional performance." in narrative.sections[0].key_observations


def test_notebook_narrative_uses_concrete_metrics_for_dimension_and_discount_sections() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Explain dimensional performance.",
            ),
            NotebookSection(
                section_id="discount_and_profit",
                title="Discount and Profit",
                purpose="Explain margin dynamics.",
            ),
        ],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        summary=[],
        modules=[
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="维度拆解分析",
                chart_type="bar",
                summary_metrics={
                    "primary_dimension": "segment",
                    "secondary_dimension": "region",
                },
                tables={
                    "dimension_totals": [
                        {
                            "Segment": "Consumer",
                            "Region": "Central",
                            "Sales": 1161401.345,
                            "Profit": 134119.2092,
                        },
                        {
                            "Segment": "Home Office",
                            "Region": "West",
                            "Sales": 429653.1485,
                            "Profit": 60298.6785,
                        },
                    ]
                },
                findings=["segment 维度下 Home Office 的利润表现最弱，值得重点复盘。"],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                summary_metrics={
                    "high_discount_order_count": 1393,
                    "negative_profit_order_count": 1871,
                    "negative_profit_rate": 0.1872,
                },
                tables={
                    "discount_buckets": [
                        {"discount_bucket": "20-30%", "avg_profit": -45.67963612334801},
                        {"discount_bucket": "30%+", "avg_profit": -107.20993018867925},
                    ]
                },
                findings=["30%+ 折扣区间平均利润最低。"],
            ),
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Segment": "segment",
            "Region": "region",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )

    narrative = build_notebook_narrative(
        outline=outline,
        report=report,
        schema_mapping=schema_mapping,
    )

    segment_section = narrative.sections[0]
    assert "segment and region" in segment_section.intro.lower()
    assert "Mapped field count: 5" in segment_section.key_observations

    discount_section = narrative.sections[1]
    assert "discount and profit" in discount_section.intro.lower()
    assert "Mapped field count: 5" in discount_section.key_observations


def test_notebook_narrative_prefers_segment_region_matrix_for_combined_labels() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Explain dimensional performance.",
            )
        ],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="维度拆解分析",
                chart_type="bar",
                summary_metrics={
                    "primary_dimension": "segment",
                    "secondary_dimension": "region",
                },
                tables={
                    "dimension_totals": [
                        {"Segment": "Consumer", "Sales": 1200.0, "Profit": 100.0},
                        {"Segment": "Home Office", "Sales": 800.0, "Profit": 50.0},
                    ],
                    "segment_region_matrix": [
                        {
                            "Segment": "Consumer",
                            "Region": "Central",
                            "Sales": 700.0,
                            "Profit": -20.0,
                        },
                        {
                            "Segment": "Consumer",
                            "Region": "West",
                            "Sales": 900.0,
                            "Profit": 80.0,
                        },
                    ],
                },
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Segment": "segment",
            "Region": "region",
            "Sales": "sales_amount",
            "Profit": "profit",
        },
        confidence=0.9,
    )

    narrative = build_notebook_narrative(
        outline=outline,
        report=report,
        schema_mapping=schema_mapping,
    )

    combined = narrative.sections[0].intro + "\n" + "\n".join(narrative.sections[0].key_observations)
    assert "segment and region" in combined.lower()
    assert "Mapped field count: 4" in combined
