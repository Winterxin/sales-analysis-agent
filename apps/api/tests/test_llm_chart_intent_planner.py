from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_selection_planner import build_chart_selection_plan
from app.services.llm_chart_intent_planner import build_llm_chart_intent_plan


class FakeIntentLLM:
    enabled = True
    source = "fake"
    configured_model = "fake-model"

    def __init__(self, payload: dict[str, object] | object):
        self.response_payload = payload
        self.payloads: list[dict[str, object]] = []
        self.last_completion_metrics = {
            "prompt_chars": 1234,
            "response_chars": 567,
        }

    def suggest_llm_chart_intent_plan(self, payload: dict[str, object]) -> object:
        self.payloads.append(payload)
        return self.response_payload


def _mapping() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Order ID": "order_id",
            "Product Name": "product_name",
            "Category": "category",
            "Segment": "segment",
            "Region": "region",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Quantity": "quantity",
        },
        confidence=1.0,
    )


def _kwargs() -> dict[str, object]:
    return {
        "schema_mapping": _mapping(),
        "dataset_profile": {
            "row_count": 1000,
            "product_count": 180,
            "category_count": 4,
            "segment_count": 3,
            "region_count": 4,
            "negative_profit_rate": 0.18,
        },
        "analysis_focus": {
            "selected_focuses": [
                "discount_erosion_focus",
                "profit_quality_focus",
                "segment_region_focus",
                "product_concentration_focus",
            ],
            "support_focuses": ["trend_volatility_focus"],
        },
        "section_priority": {
            "core_sections": ["discount_and_profit", "segment_and_region", "product_and_category"],
            "support_sections": ["sales_trends", "metric_distributions"],
            "skipped_sections": ["country_market"],
        },
        "evidence_pack": {
            "dataset_signature": {
                "shape_type": "superstore_like",
                "dominant_story": "discount_profit_quality",
                "available_fields": [
                    "category",
                    "discount",
                    "order_datetime",
                    "order_id",
                    "product_name",
                    "profit",
                    "quantity",
                    "region",
                    "sales_amount",
                    "segment",
                ],
            },
            "focus_evidence": {},
            "distinctive_facts": [],
            "limitations": [],
        },
    }


def _chart_selection_plan() -> dict[str, object]:
    return build_chart_selection_plan(**_kwargs())


def _sample_sales_kwargs() -> dict[str, object]:
    return {
        "schema_mapping": SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "ORDERDATE": "order_datetime",
                "ORDERNUMBER": "order_id",
                "PRODUCTLINE": "productline",
                "DEALSIZE": "deal_size",
                "STATUS": "order_status",
                "COUNTRY": "country",
                "QUANTITYORDERED": "quantity",
                "PRICEEACH": "unit_price",
                "SALES": "sales_amount",
            },
            confidence=1.0,
        ),
        "dataset_profile": {
            "row_count": 2800,
            "country_count": 7,
            "productline_count": 7,
            "distinct_deal_size_count": 3,
            "distinct_order_status_count": 6,
        },
        "analysis_focus": {
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
                "trend_volatility_focus",
            ],
            "support_focuses": [],
        },
        "section_priority": {
            "core_sections": ["product_and_category", "order_structure", "country_market"],
            "support_sections": ["sales_trends"],
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
            }
        },
    }


def _sample_sales_plan() -> dict[str, object]:
    return build_chart_selection_plan(**_sample_sales_kwargs())


def _build_intent_plan(intent: dict[str, object], chart_selection_plan: dict[str, object] | None = None) -> dict[str, object]:
    kwargs = _sample_sales_kwargs()
    return build_llm_chart_intent_plan(
        **kwargs,
        chart_selection_plan=chart_selection_plan or _sample_sales_plan(),
        llm_client=FakeIntentLLM(
            {
                "overall_chart_strategy": "观察产品线、国家和订单结构。",
                "chart_intents": [intent],
                "generic_fallback_reason": "",
            }
        ),
        llm_profile="full",
        llm_chart_intent_planning_enabled=True,
    )


def _match_by_id(intent: dict[str, object]) -> dict[str, dict[str, object]]:
    return {item["chart_id"]: item for item in intent["candidate_matches"]}  # type: ignore[index]


def _llm_payload() -> dict[str, object]:
    return {
        "overall_chart_strategy": "优先观察折扣、利润和客群区域之间的结构关系。",
        "chart_intents": [
            {
                "intent_id": "discount_profit_boxplot",
                "business_question": "不同折扣区间的利润率分布是否明显不同？",
                "suggested_chart_kind": "boxplot",
                "required_fields": ["discount", "profit", "sales_amount"],
                "optional_fields": ["category", "segment"],
                "recommended_dimensions": {
                    "x": "discount_bucket",
                    "y": "profit_margin",
                    "color_or_group": "category",
                },
                "why_this_chart": "箱线图能展示折扣区间下利润率分布和异常亏损。",
                "why_not_generic_chart": "普通柱形图只能展示均值。",
                "priority": "high",
                "fallback_if_unsupported": "discount_profit_quality_bar",
            },
            {
                "intent_id": "category_profit_treemap",
                "business_question": "类目利润贡献是否集中？",
                "suggested_chart_kind": "treemap",
                "required_fields": ["category", "profit", "sales_amount"],
                "optional_fields": [],
                "recommended_dimensions": {"label": "category", "size": "sales_amount", "color": "profit"},
                "why_this_chart": "能同时表达类目规模和利润质量。",
                "why_not_generic_chart": "普通柱形图难以同时表达规模和利润。",
                "priority": "medium",
                "fallback_if_unsupported": "category_profit_margin_bar",
            },
            {
                "intent_id": "country_profit_map",
                "business_question": "国家市场利润是否分化？",
                "suggested_chart_kind": "map",
                "required_fields": ["country", "profit"],
                "optional_fields": [],
                "recommended_dimensions": {"location": "country", "color": "profit"},
                "why_this_chart": "国家地图适合观察市场空间差异。",
                "why_not_generic_chart": "普通柱形图不表达地理位置。",
                "priority": "low",
                "fallback_if_unsupported": "country_sales_bar",
            },
        ],
        "generic_fallback_reason": "字段不足时退化到趋势和头部商品图。",
    }


def test_chart_intent_plan_validates_intents_and_keeps_selected_charts_shadow_only() -> None:
    chart_selection_plan = _chart_selection_plan()
    original_selected = deepcopy(chart_selection_plan["selected_charts"])
    llm = FakeIntentLLM(_llm_payload())

    plan = build_llm_chart_intent_plan(
        **_kwargs(),
        chart_selection_plan=chart_selection_plan,
        llm_client=llm,
        llm_profile="full",
        llm_chart_intent_planning_enabled=True,
    )

    assert chart_selection_plan["selected_charts"] == original_selected
    assert plan["mode"] == "shadow"
    assert plan["applied_to_selected_charts"] is False
    assert plan["trace"]["status"] == "success"

    by_id = {item["intent_id"]: item for item in plan["validated_intents"]}
    assert by_id["discount_profit_boxplot"]["validation_status"] == "valid_supported"
    assert "profit_margin_by_discount_box" in by_id["discount_profit_boxplot"]["matched_candidate_chart_ids"]
    assert by_id["category_profit_treemap"]["validation_status"] == "valid_unsupported"
    assert by_id["category_profit_treemap"] in plan["unsupported_intents"]
    assert by_id["country_profit_map"]["validation_status"] == "invalid_missing_fields"
    assert by_id["country_profit_map"] in plan["invalid_intents"]
    assert plan["new_template_opportunities"][0]["intent_id"] == "category_profit_treemap"


def test_chart_intent_planner_falls_back_on_invalid_llm_payload() -> None:
    plan = build_llm_chart_intent_plan(
        **_kwargs(),
        chart_selection_plan=_chart_selection_plan(),
        llm_client=FakeIntentLLM({"chart_intents": "not-a-list"}),
        llm_profile="full",
        llm_chart_intent_planning_enabled=True,
    )

    assert plan["mode"] == "shadow"
    assert plan["trace"]["status"] == "fallback"
    assert plan["trace"]["fallback_reason"] == "invalid_payload"
    assert plan["chart_intents"] == []
    assert plan["validated_intents"] == []


def test_top_n_product_intent_allows_high_cardinality_product_field() -> None:
    llm = FakeIntentLLM(
        {
            "overall_chart_strategy": "聚焦头部商品利润缺口。",
            "chart_intents": [
                {
                    "intent_id": "top_products_profit_gap",
                    "business_question": "头部商品是否存在销售高但利润不足的排名问题？",
                    "suggested_chart_kind": "bar",
                    "required_fields": ["product_name", "sales_amount", "profit"],
                    "optional_fields": [],
                    "recommended_dimensions": {"x": "top products", "y": "profit_gap"},
                    "why_this_chart": "Top 10 ranking avoids showing all products and highlights head items.",
                    "why_not_generic_chart": "普通销售排行无法解释利润缺口。",
                    "priority": "high",
                    "fallback_if_unsupported": "top_product_profit_gap_bar",
                }
            ],
            "generic_fallback_reason": "",
        }
    )

    plan = build_llm_chart_intent_plan(
        **_kwargs(),
        chart_selection_plan=_chart_selection_plan(),
        llm_client=llm,
        llm_profile="full",
        llm_chart_intent_planning_enabled=True,
    )

    intent = plan["validated_intents"][0]
    assert intent["validation_status"] == "valid_supported"
    assert "top_product_profit_gap_bar" in intent["matched_candidate_chart_ids"]
    assert intent["category_handling"] == "top_n"
    assert "Top-N/ranking truncation" in intent["category_handling_reason"]
    assert plan["invalid_intents"] == []


def test_full_product_name_chart_still_rejects_high_cardinality_without_top_n() -> None:
    llm = FakeIntentLLM(
        {
            "overall_chart_strategy": "展示所有商品的利润热力图。",
            "chart_intents": [
                {
                    "intent_id": "all_products_profit_heatmap",
                    "business_question": "所有商品利润是否分化？",
                    "suggested_chart_kind": "heatmap",
                    "required_fields": ["product_name", "sales_amount", "profit"],
                    "optional_fields": [],
                    "recommended_dimensions": {"x": "product_name", "y": "profit"},
                    "why_this_chart": "全量商品热力图显示每个商品利润。",
                    "why_not_generic_chart": "柱形图不够密集。",
                    "priority": "medium",
                    "fallback_if_unsupported": "",
                }
            ],
            "generic_fallback_reason": "",
        }
    )

    plan = build_llm_chart_intent_plan(
        **_kwargs(),
        chart_selection_plan=_chart_selection_plan(),
        llm_client=llm,
        llm_profile="full",
        llm_chart_intent_planning_enabled=True,
    )

    intent = plan["validated_intents"][0]
    assert intent["validation_status"] == "invalid_too_many_categories"
    assert intent["category_handling"] == "full_category"
    assert intent in plan["invalid_intents"]


def test_productline_monthly_trend_has_precise_match_quality() -> None:
    plan = _build_intent_plan(
        {
            "intent_id": "productline_monthly_trend",
            "business_question": "不同产品线的月度销售趋势是否同步？",
            "suggested_chart_kind": "line",
            "required_fields": ["productline", "order_datetime", "sales_amount"],
            "optional_fields": [],
            "recommended_dimensions": {"x": "month", "y": "sales_amount", "color_or_group": "productline"},
            "why_this_chart": "产品线月趋势能显示不同产品线的时间变化。",
            "why_not_generic_chart": "普通趋势图不能区分产品线。",
            "priority": "high",
            "fallback_if_unsupported": "productline_monthly_trend",
        }
    )

    intent = plan["validated_intents"][0]
    matches = _match_by_id(intent)
    assert matches["productline_monthly_trend"]["match_quality"] in {"exact_match", "strong_match"}
    assert matches["sales_trends_monthly_line"]["match_quality"] == "weak_match"
    assert matches["rolling_volatility_line"]["match_quality"] == "false_friend_match"
    assert "productline_monthly_trend" in intent["matched_candidate_chart_ids"]
    assert "sales_trends_monthly_line" not in intent["matched_candidate_chart_ids"]
    assert "rolling_volatility_line" not in intent["matched_candidate_chart_ids"]


def test_country_market_sales_trend_matches_country_monthly_trend_strongly() -> None:
    plan = _build_intent_plan(
        {
            "intent_id": "country_market_sales_trend",
            "business_question": "头部国家市场的月度销售趋势是否同步？",
            "suggested_chart_kind": "line",
            "required_fields": ["country", "order_datetime", "sales_amount"],
            "optional_fields": [],
            "recommended_dimensions": {"x": "month", "y": "sales_amount", "color_or_group": "country"},
            "why_this_chart": "国家月趋势能展示市场变化。",
            "why_not_generic_chart": "普通趋势图无法拆国家。",
            "priority": "high",
            "fallback_if_unsupported": "country_monthly_trend",
        }
    )

    intent = plan["validated_intents"][0]
    matches = _match_by_id(intent)
    assert matches["country_monthly_trend"]["match_quality"] in {"exact_match", "strong_match"}
    assert matches["rolling_volatility_line"]["match_quality"] == "false_friend_match"
    assert "country_monthly_trend" in intent["primary_matched_candidate_chart_ids"]


def test_only_weak_intent_is_valid_unsupported_and_not_compat_matched() -> None:
    chart_selection_plan = _sample_sales_plan()
    chart_selection_plan["candidate_charts"] = [
        item
        for item in chart_selection_plan["candidate_charts"]
        if item["chart_id"] in {"sales_trends_monthly_line", "rolling_volatility_line"}
    ]

    plan = _build_intent_plan(
        {
            "intent_id": "productline_monthly_trend",
            "business_question": "不同产品线的月度销售趋势是否同步？",
            "suggested_chart_kind": "line",
            "required_fields": ["productline", "order_datetime", "sales_amount"],
            "optional_fields": [],
            "recommended_dimensions": {"x": "month", "y": "sales_amount", "color_or_group": "productline"},
            "why_this_chart": "产品线月趋势能显示不同产品线的时间变化。",
            "why_not_generic_chart": "普通趋势图不能区分产品线。",
            "priority": "high",
            "fallback_if_unsupported": "sales_trends_monthly_line",
        },
        chart_selection_plan,
    )

    intent = plan["validated_intents"][0]
    assert intent["validation_status"] == "valid_unsupported"
    assert intent["matched_candidate_chart_ids"] == []
    assert intent["primary_matched_candidate_chart_ids"] == []
    assert intent["fallback_matched_candidate_chart_ids"] == ["sales_trends_monthly_line"]
    assert intent["false_friend_candidate_chart_ids"] == ["rolling_volatility_line"]
    assert "Only weak fallback candidates" in intent["unsupported_reason"]


def test_quantity_distribution_overview_matches_quantity_distribution_candidate() -> None:
    plan = _build_intent_plan(
        {
            "intent_id": "quantity_distribution_overview",
            "business_question": "What is the distribution of quantities per order line?",
            "suggested_chart_kind": "histogram",
            "required_fields": ["quantity"],
            "optional_fields": [],
            "recommended_dimensions": {"x": "quantity", "y": "record_count"},
            "why_this_chart": "A histogram shows the order-line quantity distribution directly.",
            "why_not_generic_chart": "A generic sales bar cannot show per-line quantity concentration.",
            "priority": "medium",
            "fallback_if_unsupported": "quantity_distribution",
        }
    )

    intent = plan["validated_intents"][0]
    matches = _match_by_id(intent)
    assert intent["validation_status"] == "valid_supported"
    assert matches["quantity_distribution"]["match_quality"] in {"exact_match", "strong_match"}
    assert "quantity_distribution" in intent["matched_candidate_chart_ids"]
    assert intent not in plan["unsupported_intents"]
    assert plan["new_template_opportunities"] == []


def test_chart_intent_prompt_uses_safe_compact_inputs() -> None:
    llm = FakeIntentLLM(_llm_payload())

    build_llm_chart_intent_plan(
        **_kwargs(),
        chart_selection_plan=_chart_selection_plan(),
        llm_client=llm,
        llm_profile="full",
        llm_chart_intent_planning_enabled=True,
    )

    payload = llm.payloads[0]
    payload_text = json.dumps(payload, ensure_ascii=False).lower()
    assert "dataset_signature" in payload
    assert "candidate_chart_summary" in payload
    assert "report" not in payload
    assert "notebook" not in payload_text
    assert "chart_payload" not in payload_text
    assert "fig.show" not in payload_text
    assert "clean_df" not in payload_text
    assert "table_preview" not in payload_text
    assert "modules" not in payload


