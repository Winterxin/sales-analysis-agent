from __future__ import annotations

from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_selection_planner import build_chart_selection_plan
from app.services.intent_guided_chart_selection import apply_intent_guided_chart_selection


class FakeChartSelectionLLM:
    enabled = True
    source = "fake"
    configured_model = "fake-model"

    def __init__(self, payload: dict[str, object]):
        self.response_payload = payload
        self.payloads: list[dict[str, object]] = []

    def suggest_llm_chart_selection(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        return self.response_payload


def _mapping(fields: dict[str, str]) -> SchemaMapping:
    return SchemaMapping(dataset_type="sales_transaction", field_mapping=fields, confidence=1.0)


def _sample_kwargs() -> dict[str, object]:
    return {
        "schema_mapping": _mapping(
            {
                "ORDERDATE": "order_datetime",
                "ORDERNUMBER": "order_id",
                "PRODUCTLINE": "productline",
                "DEALSIZE": "deal_size",
                "STATUS": "order_status",
                "COUNTRY": "country",
                "QUANTITYORDERED": "quantity",
                "PRICEEACH": "unit_price",
                "SALES": "sales_amount",
            }
        ),
        "dataset_profile": {
            "row_count": 500,
            "country_count": 7,
            "productline_count": 6,
            "distinct_deal_size_count": 3,
            "distinct_order_status_count": 5,
            "top_country_sales_share": 0.34,
        },
        "analysis_focus": {
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
            ],
            "support_focuses": ["customer_order_structure_focus", "trend_volatility_focus"],
        },
        "section_priority": {
            "core_sections": ["country_market", "order_structure", "product_and_category"],
            "support_sections": ["metric_distributions", "sales_trends"],
            "skipped_sections": ["discount_and_profit"],
        },
        "evidence_pack": {
            "dataset_signature": {
                "shape_type": "productline_order_like",
                "dominant_story": "productline_mix",
                "available_fields": [
                    "country",
                    "deal_size",
                    "order_datetime",
                    "order_id",
                    "order_status",
                    "productline",
                    "quantity",
                    "sales_amount",
                    "unit_price",
                ],
            },
            "focus_evidence": {},
            "distinctive_facts": [],
            "limitations": [],
        },
    }


def _base_plan() -> dict[str, object]:
    return build_chart_selection_plan(**_sample_kwargs(), llm_client=None)


def _intent_plan() -> dict[str, object]:
    return {
        "mode": "shadow",
        "overall_chart_strategy": "Prefer productline, deal size, status, country, and quantity structure.",
        "validated_intents": [
            {
                "intent_id": "quantity_distribution",
                "priority": "high",
                "validation_status": "valid_supported",
                "candidate_matches": [
                    {
                        "chart_id": "quantity_distribution",
                        "match_quality": "exact_match",
                        "match_score": 0.96,
                        "match_reason": "Quantity distribution directly answers the intent.",
                    },
                    {
                        "chart_id": "sales_trends_monthly_line",
                        "match_quality": "weak_match",
                        "match_score": 0.4,
                        "match_reason": "Generic sales trend is only a fallback.",
                    },
                    {
                        "chart_id": "rolling_volatility_line",
                        "match_quality": "false_friend_match",
                        "match_score": 0.1,
                        "match_reason": "Volatility is not quantity distribution.",
                    },
                ],
            },
            {
                "intent_id": "productline_deal_size_cross_analysis",
                "priority": "medium",
                "validation_status": "valid_supported",
                "candidate_matches": [
                    {
                        "chart_id": "productline_deal_size_stacked_bar",
                        "match_quality": "strong_match",
                        "match_score": 0.86,
                        "match_reason": "Productline/deal-size structure.",
                    }
                ],
            },
        ],
        "unsupported_intents": [],
    }


def test_exact_and_strong_intents_add_support_but_weak_and_false_friend_do_not() -> None:
    result = apply_intent_guided_chart_selection(
        base_chart_selection_plan=_base_plan(),
        chart_intent_plan=_intent_plan(),
        **_sample_kwargs(),
        llm_client=None,
        llm_profile="full",
        enabled=True,
    )

    candidates = {item["chart_id"]: item for item in result["candidate_charts"]}
    assert candidates["quantity_distribution"]["intent_support_score"] > 0
    assert candidates["quantity_distribution"]["best_intent_match_quality"] == "exact_match"
    assert candidates["productline_deal_size_stacked_bar"]["intent_support_score"] > 0
    assert candidates["sales_trends_monthly_line"]["intent_support_score"] == 0
    assert candidates["rolling_volatility_line"]["intent_support_score"] == 0
    assert "sales_trends_monthly_line" in result["intent_guided_selection_trace"]["weak_match_ignored_candidate_ids"]
    assert "rolling_volatility_line" in result["intent_guided_selection_trace"]["false_friend_ignored_candidate_ids"]


def test_intent_guided_deterministic_can_update_selected_charts_and_records_trace() -> None:
    base = _base_plan()
    base_ids = [item["chart_id"] for item in base["selected_charts"]]

    result = apply_intent_guided_chart_selection(
        base_chart_selection_plan=base,
        chart_intent_plan=_intent_plan(),
        **_sample_kwargs(),
        llm_client=None,
        llm_profile="full",
        enabled=True,
    )

    final_ids = [item["chart_id"] for item in result["selected_charts"]]
    assert result["selection_mode"] == "intent_guided_deterministic"
    assert result["intent_guided_selection_trace"]["base_selected_chart_ids"] == base_ids
    assert result["intent_guided_selection_trace"]["final_selected_chart_ids"] == final_ids
    assert "quantity_distribution" in result["intent_guided_selection_trace"]["intent_supported_candidate_ids"]
    assert "quantity_distribution" in result["intent_guided_selection_trace"]["selected_due_to_intent"]
    assert "quantity_distribution" in final_ids


def test_intent_guided_selection_rejects_hard_rejected_candidates() -> None:
    kwargs = _sample_kwargs()
    missing_quantity_mapping = _mapping({"ORDERDATE": "order_datetime", "SALES": "sales_amount"})
    kwargs["schema_mapping"] = missing_quantity_mapping
    base = build_chart_selection_plan(**kwargs, llm_client=None)

    result = apply_intent_guided_chart_selection(
        base_chart_selection_plan=base,
        chart_intent_plan=_intent_plan(),
        **kwargs,
        llm_client=None,
        llm_profile="full",
        enabled=True,
    )

    assert "quantity_distribution" not in [item["chart_id"] for item in result["selected_charts"]]
    assert "quantity_distribution" in result["intent_guided_selection_trace"]["sanitizer_rejected_candidate_ids"]


def test_hard_rejected_intent_supported_candidates_are_separated_from_usable_trace() -> None:
    kwargs = _sample_kwargs()
    kwargs["schema_mapping"] = _mapping({"ORDERDATE": "order_datetime", "SALES": "sales_amount"})
    base = build_chart_selection_plan(**kwargs, llm_client=None)

    result = apply_intent_guided_chart_selection(
        base_chart_selection_plan=base,
        chart_intent_plan=_intent_plan(),
        **kwargs,
        llm_client=None,
        llm_profile="full",
        enabled=True,
    )

    trace = result["intent_guided_selection_trace"]
    assert "quantity_distribution" not in trace["intent_supported_candidate_ids"]
    assert "quantity_distribution" in trace["intent_supported_but_hard_rejected_candidate_ids"]
    assert trace["intent_supported_hard_rejection_reasons"]["quantity_distribution"].startswith(
        "missing required fields"
    )
    assert "quantity_distribution" not in trace["selected_due_to_intent"]


def test_intent_guided_llm_falls_back_for_invalid_output() -> None:
    base = _base_plan()
    llm = FakeChartSelectionLLM({"selected_chart_ids": ["not_a_chart"]})

    result = apply_intent_guided_chart_selection(
        base_chart_selection_plan=base,
        chart_intent_plan=_intent_plan(),
        **_sample_kwargs(),
        llm_client=llm,
        llm_profile="full",
        enabled=True,
    )

    assert result["selection_mode"] == "intent_guided_fallback"
    assert result["selected_charts"] == base["selected_charts"]
    assert result["intent_guided_selection_trace"]["fallback_reason"]
    payload = llm.payloads[0]
    assert "chart_intent_strategy" in payload
    assert "intent_supported_candidates" in payload
    assert "false_friend_matches" in payload
