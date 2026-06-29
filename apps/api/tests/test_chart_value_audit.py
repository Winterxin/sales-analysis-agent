from __future__ import annotations

from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_value_audit import build_chart_value_audit


def _mapping() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Region": "region",
            "Segment": "segment",
        },
        confidence=1.0,
    )


def test_chart_value_audit_scores_candidates() -> None:
    audit = build_chart_value_audit(
        candidate_charts=[
            {
                "chart_id": "region_sales_profit_bar",
                "section_id": "segment_and_region",
                "chart_kind": "bar",
                "required_fields": ["region", "sales_amount", "profit"],
                "insight_type": "profit_quality",
                "business_question": "Which regions have strong sales and profit quality?",
                "availability_status": "selected",
            }
        ],
        schema_mapping=_mapping(),
        dataset_profile={},
        evidence_pack={"dataset_signature": {"dominant_story": "profit quality"}},
    )

    value = audit["chart_value_by_id"]["region_sales_profit_bar"]
    assert 0 <= value["overall_chart_value_score"] <= 1
    assert value["information_gain_score"] > 0.5


def test_same_section_multiple_bars_create_redundancy_risk() -> None:
    audit = build_chart_value_audit(
        candidate_charts=[
            {
                "chart_id": "segment_region_sales_bar",
                "section_id": "segment_and_region",
                "chart_kind": "bar",
                "required_fields": ["segment", "region", "sales_amount"],
                "insight_type": "segment_region_sales",
                "availability_status": "selected",
            },
            {
                "chart_id": "region_sales_profit_bar",
                "section_id": "segment_and_region",
                "chart_kind": "bar",
                "required_fields": ["region", "sales_amount", "profit"],
                "insight_type": "profit_quality",
                "availability_status": "selected",
            },
            {
                "chart_id": "segment_region_low_margin_table_or_bar",
                "section_id": "segment_and_region",
                "chart_kind": "bar",
                "required_fields": ["segment", "region", "sales_amount", "profit"],
                "insight_type": "profit_quality",
                "availability_status": "selected",
            },
        ],
        schema_mapping=_mapping(),
        dataset_profile={},
        evidence_pack={},
    )

    values = audit["chart_value_by_id"]
    assert values["segment_region_sales_bar"]["redundancy_risk_score"] > 0.5
    assert "same_section_bar_overlap" in values["segment_region_sales_bar"]["value_flags"]


def test_low_visual_gain_risk_flag_for_plain_overlapping_bar() -> None:
    audit = build_chart_value_audit(
        candidate_charts=[
            {
                "chart_id": "segment_region_sales_bar",
                "section_id": "segment_and_region",
                "chart_kind": "bar",
                "required_fields": ["segment", "region", "sales_amount"],
                "insight_type": "segment_region_sales",
                "business_question": "Which segment and region sells most?",
                "availability_status": "selected",
            },
            {
                "chart_id": "region_sales_bar",
                "section_id": "segment_and_region",
                "chart_kind": "bar",
                "required_fields": ["region", "sales_amount"],
                "insight_type": "region_sales",
                "business_question": "Which region sells most?",
                "availability_status": "selected",
            },
        ],
        schema_mapping=_mapping(),
        dataset_profile={},
        evidence_pack={},
    )

    value = audit["chart_value_by_id"]["region_sales_bar"]
    assert "low_visual_gain_risk" in value["value_flags"]
    assert "region_sales_bar" in audit["low_visual_gain_risk_chart_ids"]