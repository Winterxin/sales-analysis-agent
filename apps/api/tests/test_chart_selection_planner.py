from __future__ import annotations

import json
from pathlib import Path

from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_candidate_registry import iter_chart_candidates
from app.services.chart_selection_planner import CHART_BUDGET, build_chart_selection_plan
from app.services.llm_chart_selector import _apply_bar_dedup_guard, _apply_chart_count_guard


class FakeChartSelectionLLM:
    enabled = True

    def __init__(self, payload: dict[str, object]):
        self.response_payload = payload
        self.payloads: list[dict[str, object]] = []

    def suggest_llm_chart_selection(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        return self.response_payload


class FailingChartSelectionLLM(FakeChartSelectionLLM):
    def __init__(self) -> None:
        super().__init__({})

    def suggest_llm_chart_selection(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        raise RuntimeError("chart selection unavailable")


def _mapping(fields: dict[str, str]) -> SchemaMapping:
    return SchemaMapping(dataset_type="sales_transaction", field_mapping=fields, confidence=1.0)


def _sample_sales_kwargs() -> dict[str, object]:
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
            "support_focuses": ["customer_order_structure_focus"],
        },
        "section_priority": {
            "core_sections": ["country_market", "order_structure", "product_and_category"],
            "support_sections": [],
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


def test_chart_selection_plan_respects_core_support_skipped_budgets() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "Order Date": "order_datetime",
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Product Name": "product_name",
                "Segment": "segment",
                "Region": "region",
                "Order ID": "order_id",
                "Quantity": "quantity",
            }
        ),
        dataset_profile={"country_count": 1},
        analysis_focus={
            "selected_focuses": [
                "discount_erosion_focus",
                "profit_quality_focus",
                "segment_region_focus",
                "product_concentration_focus",
            ],
            "support_focuses": ["customer_order_structure_focus", "trend_volatility_focus"],
        },
        section_priority={
            "core_sections": ["discount_and_profit", "segment_and_region", "product_and_category"],
            "support_sections": ["order_structure", "sales_trends", "metric_distributions"],
            "skipped_sections": ["country_market"],
        },
    )

    counts: dict[str, int] = {}
    for chart in plan["selected_charts"]:
        counts[chart["section_id"]] = counts.get(chart["section_id"], 0) + 1
    assert counts["order_structure"] <= 1
    assert counts["sales_trends"] <= 1
    assert counts["metric_distributions"] <= 1
    assert "country_market" not in counts


def test_required_fields_and_country_count_reject_charts() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping({"Country": "country", "Sales": "sales_amount"}),
        dataset_profile={"country_count": 1, "top_country_sales_share": 1.0},
        analysis_focus={"selected_focuses": ["country_market_focus"]},
        section_priority={
            "core_sections": ["country_market"],
            "support_sections": [],
            "skipped_sections": [],
        },
    )

    assert not plan["selected_charts"]
    reasons = {item["chart_id"]: item["reason"] for item in plan["rejected_charts"]}
    assert reasons["country_sales_bar"] == "country field has no useful market slice"
    assert "missing required fields" in reasons["country_avg_order_value_bar"]


def test_online_retail_does_not_select_discount_profit_charts() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "InvoiceNo": "order_id",
                "InvoiceDate": "order_datetime",
                "CustomerID": "customer_id",
                "Country": "country",
                "Description": "product_name",
                "Quantity": "quantity",
                "UnitPrice": "unit_price",
                "__sales_amount": "sales_amount",
            }
        ),
        dataset_profile={"country_count": 38, "top_country_sales_share": 0.84},
        analysis_focus={
            "selected_focuses": [
                "country_market_focus",
                "customer_order_structure_focus",
                "product_concentration_focus",
            ],
            "support_focuses": ["trend_volatility_focus"],
        },
        section_priority={
            "core_sections": ["country_market", "order_structure", "product_and_category"],
            "support_sections": ["sales_trends", "metric_distributions"],
            "skipped_sections": ["discount_and_profit"],
        },
    )

    selected_ids = {chart["chart_id"] for chart in plan["selected_charts"]}
    assert "discount_profit_quality_bar" not in selected_ids
    assert "country_sales_bar" in selected_ids
    assert "basket_size_distribution" in selected_ids


def test_sample_sales_selects_productline_deal_size_status_charts() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
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
        dataset_profile={"country_count": 19, "top_country_sales_share": 0.36},
        analysis_focus={
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
            ],
            "support_focuses": ["customer_order_structure_focus"],
        },
        section_priority={
            "core_sections": ["product_and_category", "order_structure", "country_market"],
            "support_sections": [],
            "skipped_sections": ["discount_and_profit"],
        },
    )

    selected_ids = {chart["chart_id"] for chart in plan["selected_charts"]}
    assert "productline_sales_bar" in selected_ids
    assert "deal_size_sales_bar" in selected_ids
    assert "order_status_breakdown" in selected_ids
    assert "country_sales_bar" in selected_ids


def test_chart_registry_has_productline_deal_size_status_country_candidates() -> None:
    candidates = {item["chart_id"]: item for item in iter_chart_candidates()}
    expected = {
        "productline_sales_donut",
        "productline_monthly_trend",
        "productline_quantity_sales_combo",
        "deal_size_sales_donut",
        "deal_size_order_count_bar",
        "deal_size_avg_order_value_bar",
        "status_sales_donut",
        "status_order_count_bar",
        "productline_deal_size_stacked_bar",
        "productline_deal_size_heatmap",
        "country_productline_heatmap",
        "status_deal_size_stacked_bar",
        "country_sales_donut",
    }

    assert expected <= set(candidates)
    for chart_id in expected:
        candidate = candidates[chart_id]
        assert candidate["required_fields"]
        assert candidate["supported_focuses"]
        assert candidate["business_question"]
        assert candidate["insight_type"]
        assert isinstance(candidate["priority"], float)


def test_quantity_distribution_candidate_is_available_when_quantity_exists() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "Order Date": "order_datetime",
                "Sales": "sales_amount",
                "Quantity": "quantity",
            }
        ),
        dataset_profile={"row_count": 120, "quantity_distinct_count": 8},
        analysis_focus={"selected_focuses": [], "support_focuses": ["customer_order_structure_focus"]},
        section_priority={
            "core_sections": [],
            "support_sections": ["metric_distributions"],
            "skipped_sections": [],
        },
    )

    candidate = {item["chart_id"]: item for item in plan["candidate_charts"]}["quantity_distribution"]
    assert candidate["section_id"] == "metric_distributions"
    assert candidate["chart_kind"] == "histogram"
    assert candidate["required_fields"] == ["quantity"]
    assert candidate["business_question"] == "What is the distribution of quantities per order line?"
    assert candidate["insight_type"] in {"quantity_distribution", "order_quantity_distribution"}
    assert not str(candidate.get("rejection_reason") or "").startswith("missing required fields")
    assert candidate.get("rejection_reason") != "skipped section"


def test_quantity_distribution_rejected_when_quantity_missing() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping({"Order Date": "order_datetime", "Sales": "sales_amount"}),
        dataset_profile={"row_count": 120},
        analysis_focus={"selected_focuses": [], "support_focuses": ["customer_order_structure_focus"]},
        section_priority={
            "core_sections": [],
            "support_sections": ["metric_distributions"],
            "skipped_sections": [],
        },
    )

    candidate = {item["chart_id"]: item for item in plan["candidate_charts"]}["quantity_distribution"]
    assert candidate["availability_status"] == "rejected"
    assert str(candidate["rejection_reason"]).startswith("missing required fields")
    assert all(chart["chart_id"] != "quantity_distribution" for chart in plan["selected_charts"])


def test_chart_selection_respects_budget_after_new_candidates() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "ORDERDATE": "order_datetime",
                "ORDERNUMBER": "order_id",
                "PRODUCTLINE": "productline",
                "DEALSIZE": "deal_size",
                "STATUS": "order_status",
                "COUNTRY": "country",
                "QUANTITYORDERED": "quantity",
                "SALES": "sales_amount",
            }
        ),
        dataset_profile={
            "country_count": 7,
            "productline_count": 6,
            "distinct_deal_size_count": 3,
            "distinct_order_status_count": 5,
            "top_country_sales_share": 0.34,
        },
        analysis_focus={
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
            ],
            "support_focuses": ["trend_volatility_focus"],
        },
        section_priority={
            "core_sections": ["product_and_category", "order_structure", "country_market"],
            "support_sections": ["sales_trends"],
            "skipped_sections": [],
        },
    )

    counts: dict[str, int] = {}
    for chart in plan["selected_charts"]:
        counts[chart["section_id"]] = counts.get(chart["section_id"], 0) + 1
    for section_id in ("product_and_category", "order_structure", "country_market"):
        assert counts.get(section_id, 0) <= CHART_BUDGET["core_max_per_section"]
    assert counts.get("sales_trends", 0) <= CHART_BUDGET["support_max_per_section"]


def test_pie_donut_not_selected_for_too_many_categories() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping({"Country": "country", "Sales": "sales_amount"}),
        dataset_profile={"country_count": 19, "top_country_sales_share": 0.3},
        analysis_focus={"selected_focuses": ["country_market_focus"]},
        section_priority={"core_sections": ["country_market"], "support_sections": []},
    )

    selected_ids = {chart["chart_id"] for chart in plan["selected_charts"]}
    rejected_reasons = {chart["chart_id"]: chart["reason"] for chart in plan["rejected_charts"]}
    assert "country_sales_bar" in selected_ids
    assert "country_sales_donut" not in selected_ids
    assert rejected_reasons["country_sales_donut"] == "too many categories for pie/donut"


def test_no_hardcoded_dataset_specific_chart_logic() -> None:
    service_root = Path(__file__).resolve().parents[1] / "app" / "services"
    service_text = "\n".join(path.read_text(encoding="utf-8") for path in service_root.rglob("*.py"))

    assert "sample_superstore" not in service_text
    assert "sales_data_sample" not in service_text


def test_selected_charts_include_evidence_metadata() -> None:
    evidence_pack = {
        "dataset_signature": {"dominant_story": "discount_loss", "shape_type": "superstore_like"},
        "focus_evidence": {
            "discount_erosion_focus": [
                {
                    "evidence_id": "negative_profit_rate",
                    "formatted": "18.72%",
                    "business_meaning": "负利润记录占比较高。",
                    "strength": 0.8,
                }
            ]
        },
        "distinctive_facts": [],
        "limitations": [],
    }
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Category": "category",
            }
        ),
        dataset_profile={"negative_profit_rate": 0.1872},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        section_priority={"core_sections": ["discount_and_profit"], "support_sections": []},
        evidence_pack=evidence_pack,
    )

    selected = plan["selected_charts"][0]
    assert selected["selection_source"] == "evidence_ranker"
    assert selected["evidence_ids"]
    assert "18.72%" in selected["evidence_summary"]
    assert isinstance(selected["score"], float)
    assert selected["ranking_factors"]


def test_evidence_ranking_changes_selected_chart_ids_for_different_superstore_profiles() -> None:
    base_kwargs = {
        "schema_mapping": _mapping(
            {
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Category": "category",
                "Segment": "segment",
                "Region": "region",
                "Product Name": "product_name",
            }
        ),
        "analysis_focus": {
            "selected_focuses": [
                "discount_erosion_focus",
                "profit_quality_focus",
                "segment_region_focus",
                "product_concentration_focus",
            ]
        },
        "section_priority": {
            "core_sections": ["discount_and_profit", "product_and_category", "segment_and_region"],
            "support_sections": [],
        },
    }
    loss_pack = {
        "dataset_signature": {"dominant_story": "discount_loss", "shape_type": "superstore_like"},
        "focus_evidence": {
            "discount_erosion_focus": [
                {"evidence_id": "negative_profit_rate", "formatted": "22.00%", "strength": 0.9}
            ]
        },
        "distinctive_facts": [],
        "limitations": [],
    }
    concentration_pack = {
        "dataset_signature": {"dominant_story": "product_concentration", "shape_type": "superstore_like"},
        "focus_evidence": {
            "product_concentration_focus": [
                {"evidence_id": "top_product_sales_share", "formatted": "35.00%", "strength": 0.9}
            ]
        },
        "distinctive_facts": [],
        "limitations": [],
    }

    loss_ids = {
        chart["chart_id"]
        for chart in build_chart_selection_plan(
            dataset_profile={"negative_profit_rate": 0.22}, evidence_pack=loss_pack, **base_kwargs
        )["selected_charts"]
    }
    concentration_ids = {
        chart["chart_id"]
        for chart in build_chart_selection_plan(
            dataset_profile={"top_product_sales_share": 0.35},
            evidence_pack=concentration_pack,
            **base_kwargs,
        )["selected_charts"]
    }

    assert loss_ids != concentration_ids


def test_dataset_specificity_scores_rank_dataset_feature_charts_above_generic_charts() -> None:
    evidence_pack = {
        "dataset_signature": {
            "shape_type": "productline_order_like",
            "dominant_story": "productline_mix",
            "available_fields": [
                "country",
                "deal_size",
                "order_datetime",
                "order_id",
                "order_status",
                "product_name",
                "productline",
                "sales_amount",
            ],
        },
        "focus_evidence": {},
        "distinctive_facts": [],
        "limitations": [],
    }
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "ORDERDATE": "order_datetime",
                "ORDERNUMBER": "order_id",
                "PRODUCTNAME": "product_name",
                "PRODUCTLINE": "productline",
                "DEALSIZE": "deal_size",
                "STATUS": "order_status",
                "COUNTRY": "country",
                "SALES": "sales_amount",
            }
        ),
        dataset_profile={"country_count": 19, "top_country_sales_share": 0.36},
        analysis_focus={
            "selected_focuses": [
                "trend_volatility_focus",
                "product_concentration_focus",
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
            ],
            "support_focuses": [],
        },
        section_priority={
            "core_sections": ["sales_trends", "product_and_category", "order_structure", "country_market"],
            "support_sections": [],
            "skipped_sections": [],
        },
        evidence_pack=evidence_pack,
    )

    candidate_scores = {
        chart["chart_id"]: chart["dataset_specificity_score"]
        for chart in plan["candidate_charts"]
    }
    for chart in plan["candidate_charts"] + plan["selected_charts"]:
        assert 0.0 <= chart["dataset_specificity_score"] <= 1.0

    assert candidate_scores["sales_trends_monthly_line"] < candidate_scores["productline_sales_bar"]
    assert candidate_scores["product_category_bar"] < candidate_scores["deal_size_sales_bar"]
    assert candidate_scores["product_category_bar"] < candidate_scores["order_status_breakdown"]
    assert candidate_scores["product_category_bar"] < candidate_scores["country_sales_bar"]


def test_chart_strategy_diagnostics_preserve_selected_charts() -> None:
    kwargs = {
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
        "dataset_profile": {"country_count": 19, "top_country_sales_share": 0.36},
        "analysis_focus": {
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
            ],
            "support_focuses": ["customer_order_structure_focus"],
        },
        "section_priority": {
            "core_sections": ["product_and_category", "order_structure", "country_market"],
            "support_sections": [],
            "skipped_sections": ["discount_and_profit"],
        },
    }

    plan = build_chart_selection_plan(**kwargs)
    assert [chart["chart_id"] for chart in plan["selected_charts"]] == [
        "country_sales_bar",
        "country_concentration_pareto",
        "country_avg_order_value_bar",
        "deal_size_sales_bar",
        "order_status_breakdown",
        "deal_size_avg_order_value_bar",
        "productline_sales_bar",
        "productline_monthly_trend",
    ]
    assert plan["diagnostics"]["selected_chart_ids"] == [
        chart["chart_id"] for chart in plan["selected_charts"]
    ]
    assert plan["diagnostics"]["generic_chart_count"] + plan["diagnostics"]["dataset_specific_chart_count"] == len(
        plan["selected_charts"]
    )


def test_root_cause_diagnostics_explain_rejections_budget_and_specificity_gap() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
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
        dataset_profile={
            "country_count": 19,
            "productline_count": 7,
            "distinct_deal_size_count": 3,
            "distinct_order_status_count": 6,
            "top_country_sales_share": 0.36,
        },
        analysis_focus={
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
            ],
            "support_focuses": ["customer_order_structure_focus"],
        },
        section_priority={
            "core_sections": ["product_and_category", "order_structure", "country_market"],
            "support_sections": ["sales_trends"],
            "skipped_sections": ["discount_and_profit"],
        },
        evidence_pack={
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
    )

    diagnostics = plan["diagnostics"]
    assert {
        "selected_reason_breakdown",
        "rejected_reason_distribution",
        "hard_rejection_distribution",
        "soft_rejection_distribution",
        "section_budget_pressure",
        "candidate_specificity_distribution",
        "selected_vs_available_specificity_gap",
        "top_unselected_dataset_specific_charts",
    } <= set(diagnostics)
    assert [chart["chart_id"] for chart in plan["selected_charts"]] == diagnostics["selected_chart_ids"]

    reason_distribution = diagnostics["rejected_reason_distribution"]
    assert reason_distribution["skipped section"] >= 1
    assert reason_distribution["missing required fields"] >= 1
    assert reason_distribution["too many categories"] >= 1
    assert reason_distribution["lower evidence score"] >= 1
    assert reason_distribution["redundant insight type"] >= 1
    assert diagnostics["hard_rejection_distribution"]["skipped section"] >= 1
    assert diagnostics["hard_rejection_distribution"]["missing required fields"] >= 1
    assert diagnostics["hard_rejection_distribution"]["too many categories"] >= 1
    assert diagnostics["soft_rejection_distribution"]["lower evidence score"] >= 1
    assert diagnostics["soft_rejection_distribution"]["redundant insight type"] >= 1

    pressure = diagnostics["section_budget_pressure"]
    assert pressure["order_structure"]["available_candidate_count"] >= pressure["order_structure"]["selected_count"]
    assert pressure["product_and_category"]["rejected_by_redundant_insight_count"] >= 1
    assert pressure["order_structure"]["budget_limit"] == CHART_BUDGET["core_max_per_section"]

    specificity_distribution = diagnostics["candidate_specificity_distribution"]
    assert specificity_distribution["high"] >= 1
    assert specificity_distribution["low"] >= 1

    gap = diagnostics["selected_vs_available_specificity_gap"]
    assert gap["max_available_specificity"] >= gap["max_selected_specificity"]
    assert gap["avg_available_specificity"] >= 0.0
    assert isinstance(gap["gap"], float)

    selected_ids = {chart["chart_id"] for chart in plan["selected_charts"]}
    top_unselected = diagnostics["top_unselected_dataset_specific_charts"]
    assert top_unselected
    assert all(chart["chart_id"] not in selected_ids for chart in top_unselected)
    assert all(chart["dataset_specificity_score"] >= 0.55 for chart in top_unselected)
    assert all(
        chart["rejection_reason"] in {"lower evidence score", "redundant insight type", "core budget exceeded", "support budget exceeded", "budget exceeded"}
        for chart in top_unselected
    )
    assert all(chart["why_it_matters"] for chart in top_unselected)
    assert any(chart.get("score") is not None and chart.get("ranking_factors") for chart in top_unselected)


def test_core_budget_exceeded_is_reported_as_soft_rejection_without_changing_selection() -> None:
    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
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
        dataset_profile={
            "country_count": 7,
            "productline_count": 6,
            "distinct_deal_size_count": 3,
            "distinct_order_status_count": 5,
            "top_country_sales_share": 0.34,
        },
        analysis_focus={
            "selected_focuses": [
                "productline_performance_focus",
                "deal_size_focus",
                "order_status_focus",
                "country_market_focus",
            ],
            "support_focuses": ["customer_order_structure_focus"],
        },
        section_priority={
            "core_sections": ["country_market", "order_structure", "product_and_category"],
            "support_sections": [],
            "skipped_sections": [],
        },
    )

    selected_ids = [chart["chart_id"] for chart in plan["selected_charts"]]
    rejected_reasons = {chart["chart_id"]: chart["reason"] for chart in plan["rejected_charts"]}
    assert selected_ids == plan["diagnostics"]["selected_chart_ids"]
    assert "country_order_count_bar" not in selected_ids
    assert rejected_reasons["country_order_count_bar"] == "core budget exceeded"
    assert plan["diagnostics"]["soft_rejection_distribution"]["core budget exceeded"] >= 1
    assert "core budget exceeded" not in plan["diagnostics"]["hard_rejection_distribution"]


def test_llm_chart_selection_stays_deterministic_for_quick_profile() -> None:
    llm = FakeChartSelectionLLM({"selected_chart_ids": ["productline_deal_size_stacked_bar"]})

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="quick",
    )

    assert plan["selection_mode"] == "deterministic"
    assert plan["llm_selection_trace"]["attempted"] is False
    assert "quick" in plan["llm_selection_trace"]["reason"]
    assert llm.payloads == []
    assert plan["deterministic_selected_charts"] == plan["selected_charts"]


def test_quick_profile_can_use_lightweight_llm_chart_selection_when_enabled() -> None:
    llm = FakeChartSelectionLLM(
        {
            "selected_chart_ids": [
                "productline_deal_size_stacked_bar",
                "deal_size_sales_bar",
                "country_sales_bar",
            ],
            "selection_rationale": {
                "productline_deal_size_stacked_bar": "保留产品线与交易规模结构。",
                "deal_size_sales_bar": "保留交易规模结构。",
                "country_sales_bar": "保留国家市场结构。",
            },
            "overall_strategy": "Quick 保留轻量单轮图表选择，不进入意图规划。",
        }
    )

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="quick",
        llm_chart_selection_enabled=True,
    )

    assert plan["selection_mode"] == "llm_sanitized"
    assert plan["llm_selection_trace"]["attempted"] is True
    assert plan["llm_selection_trace"]["applied"] is True
    assert llm.payloads


def test_quick_lightweight_llm_chart_selection_falls_back_to_deterministic_on_error() -> None:
    llm = FailingChartSelectionLLM()

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="quick",
        llm_chart_selection_enabled=True,
    )

    assert plan["selection_mode"] == "llm_fallback"
    assert plan["llm_selection_trace"]["attempted"] is True
    assert plan["llm_selection_trace"]["applied"] is False
    assert "llm_error" in plan["llm_selection_trace"]["fallback_reason"]
    assert plan["selected_charts"] == plan["deterministic_selected_charts"]
    assert llm.payloads


def test_llm_chart_selection_uses_sanitized_full_profile_decision() -> None:
    llm = FakeChartSelectionLLM(
        {
            "selected_chart_ids": [
                "productline_deal_size_stacked_bar",
                "deal_size_sales_bar",
                "country_sales_bar",
            ],
            "selection_rationale": {
                "productline_deal_size_stacked_bar": "产品线和交易规模一起解释当前数据主线。",
                "deal_size_sales_bar": "交易规模结构是当前订单结构重点。",
                "country_sales_bar": "国家市场贡献需要保留。",
            },
            "overall_strategy": "优先使用 PRODUCTLINE、DEALSIZE 和 COUNTRY 字段解释业务结构。",
            "rejected_high_specific_notes": {
                "status_deal_size_stacked_bar": "与交易规模结构有重叠，暂不优先。"
            },
        }
    )

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="full",
    )

    assert plan["selection_mode"] == "llm_sanitized"
    assert plan["llm_selected_chart_ids"] == [
        "productline_deal_size_stacked_bar",
        "deal_size_sales_bar",
        "country_sales_bar",
    ]
    assert plan["llm_sanitized_selected_chart_ids"][:3] == plan["llm_selected_chart_ids"]
    assert [chart["chart_id"] for chart in plan["selected_charts"]] == plan["llm_sanitized_selected_chart_ids"]
    assert plan["llm_selection_trace"]["chart_count_guard_applied"] is False
    assert plan["llm_selection_trace"]["restored_base_chart_ids"] == []
    assert plan["llm_selection_trace"]["chart_count_guard_reason"] == "final_already_has_bar_chart"
    assert all(chart["selection_source"] == "llm_sanitized" for chart in plan["selected_charts"])
    assert plan["llm_selection_rationale"]["overall_strategy"].startswith("优先使用")
    assert plan["llm_selection_trace"]["applied"] is True
    assert llm.payloads


def test_llm_chart_count_guard_restores_one_safe_base_bar_when_bars_are_removed() -> None:
    llm = FakeChartSelectionLLM(
        {
            "selected_chart_ids": ["sales_trends_monthly_line", "invoice_value_distribution"],
            "selection_rationale": {
                "sales_trends_monthly_line": "保留销售趋势。",
                "invoice_value_distribution": "保留订单金额分布。",
            },
            "overall_strategy": "去除冗余柱形图，但不能让分类对比完全消失。",
        }
    )

    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "Date": "order_datetime",
                "Sales": "sales_amount",
                "Category": "category",
                "Segment": "segment",
                "Region": "region",
                "Quantity": "quantity",
                "Order ID": "order_id",
            }
        ),
        dataset_profile={
            "category_count": 4,
            "segment_count": 3,
            "region_count": 4,
            "country_count": 1,
        },
        analysis_focus={
            "selected_focuses": [
                "product_concentration_focus",
                "segment_region_focus",
                "sales_trend_focus",
                "metric_distribution_focus",
            ],
            "support_focuses": ["customer_order_structure_focus"],
        },
        section_priority={
            "core_sections": ["product_and_category", "segment_and_region", "sales_trends"],
            "support_sections": ["metric_distributions", "order_structure"],
            "skipped_sections": [],
        },
        llm_client=llm,
        llm_profile="quick",
        llm_chart_selection_enabled=True,
    )

    selected_ids = [chart["chart_id"] for chart in plan["selected_charts"]]
    selected_kinds = [chart["chart_kind"] for chart in plan["selected_charts"]]

    assert plan["selection_mode"] == "llm_sanitized"
    assert len(selected_ids) == 3
    assert selected_kinds.count("bar") == 1
    assert set(plan["llm_selection_trace"]["restored_base_chart_ids"]).issubset(
        {chart["chart_id"] for chart in plan["deterministic_selected_charts"]}
    )
    assert plan["llm_selection_trace"]["chart_count_guard_applied"] is True
    assert plan["llm_selection_trace"]["chart_count_before_guard"] == 2
    assert plan["llm_selection_trace"]["chart_count_after_guard"] == 3
    assert plan["llm_selection_trace"]["final_bar_chart_count"] == 1
    assert plan["llm_selection_trace"]["chart_count_guard_reason"] in {
        "restored_one_safe_base_bar_chart",
        "restored_one_safe_base_chart_for_minimum_count_and_bar_coverage",
    }


def test_llm_chart_count_guard_does_not_invent_bar_when_base_has_no_bar() -> None:
    selected, trace = _apply_chart_count_guard(
        selected=[
            {"chart_id": "trend", "chart_kind": "line", "insight_type": "trend"},
            {"chart_id": "distribution", "chart_kind": "histogram", "insight_type": "distribution"},
        ],
        raw_ids=["trend", "distribution"],
        result={},
        candidate_charts=[
            {"chart_id": "trend", "chart_kind": "line", "required_fields": ["sales_amount"], "section_role": "core"},
            {"chart_id": "distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
            {"chart_id": "scatter", "chart_kind": "scatter", "required_fields": ["sales_amount"], "section_role": "core"},
            {"chart_id": "table", "chart_kind": "table", "required_fields": ["sales_amount"], "section_role": "core"},
        ],
        schema_mapping=_mapping({"Sales": "sales_amount"}),
        chart_budget={},
        deterministic_chart_selection_plan={
            "selected_charts": [
                {"chart_id": "trend", "chart_kind": "line", "required_fields": ["sales_amount"], "section_role": "core"},
                {"chart_id": "distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
                {"chart_id": "scatter", "chart_kind": "scatter", "required_fields": ["sales_amount"], "section_role": "core"},
                {"chart_id": "table", "chart_kind": "table", "required_fields": ["sales_amount"], "section_role": "core"},
            ],
            "rejected_charts": [],
        },
        base_selected_by_id={},
    )

    assert len(selected) == 3
    assert all(chart["chart_kind"] != "bar" for chart in selected)
    assert trace["restored_base_chart_ids"] in [["scatter"], ["table"]]
    assert trace["chart_count_guard_applied"] is True
    assert trace["chart_count_guard_reason"] == "restored_one_safe_base_chart_for_minimum_count"


def test_llm_chart_count_guard_does_not_restore_when_final_already_has_bar() -> None:
    llm = FakeChartSelectionLLM(
        {
            "selected_chart_ids": [
                "sales_trends_monthly_line",
                "invoice_value_distribution",
                "category_concentration_bar",
            ],
            "selection_rationale": {
                "sales_trends_monthly_line": "保留销售趋势。",
                "invoice_value_distribution": "保留订单金额分布。",
                "category_concentration_bar": "保留分类结构对比。",
            },
        }
    )

    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "Date": "order_datetime",
                "Sales": "sales_amount",
                "Category": "category",
                "Segment": "segment",
                "Region": "region",
                "Quantity": "quantity",
                "Order ID": "order_id",
            }
        ),
        dataset_profile={"category_count": 4, "segment_count": 3, "region_count": 4, "country_count": 1},
        analysis_focus={
            "selected_focuses": [
                "product_concentration_focus",
                "segment_region_focus",
                "sales_trend_focus",
                "metric_distribution_focus",
            ],
            "support_focuses": ["customer_order_structure_focus"],
        },
        section_priority={
            "core_sections": ["product_and_category", "segment_and_region", "sales_trends"],
            "support_sections": ["metric_distributions", "order_structure"],
            "skipped_sections": [],
        },
        llm_client=llm,
        llm_profile="quick",
        llm_chart_selection_enabled=True,
    )

    assert [chart["chart_kind"] for chart in plan["selected_charts"]].count("bar") == 1
    assert plan["llm_selection_trace"]["restored_base_chart_ids"] == []
    assert plan["llm_selection_trace"]["chart_count_guard_applied"] is False
    assert plan["llm_selection_trace"]["chart_count_guard_reason"] == "final_already_has_bar_chart"


def test_llm_chart_count_guard_does_not_restore_when_base_chart_count_is_small() -> None:
    selected, trace = _apply_chart_count_guard(
        selected=[
            {"chart_id": "trend", "chart_kind": "line", "insight_type": "trend"},
            {"chart_id": "distribution", "chart_kind": "histogram", "insight_type": "distribution"},
        ],
        raw_ids=["trend", "distribution"],
        result={},
        candidate_charts=[
            {"chart_id": "trend", "chart_kind": "line", "required_fields": ["sales_amount"], "section_role": "core"},
            {"chart_id": "distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
            {"chart_id": "category_bar", "chart_kind": "bar", "required_fields": ["category", "sales_amount"], "section_role": "core"},
        ],
        schema_mapping=_mapping({"Sales": "sales_amount", "Category": "category"}),
        chart_budget={},
        deterministic_chart_selection_plan={
            "selected_charts": [
                {"chart_id": "trend", "chart_kind": "line", "required_fields": ["sales_amount"], "section_role": "core"},
                {"chart_id": "distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
                {"chart_id": "category_bar", "chart_kind": "bar", "required_fields": ["category", "sales_amount"], "section_role": "core"},
            ],
            "rejected_charts": [],
        },
        base_selected_by_id={},
    )

    assert len(selected) == 2
    assert all(chart["chart_kind"] != "bar" for chart in selected)
    assert trace["restored_base_chart_ids"] == []
    assert trace["chart_count_guard_applied"] is False
    assert trace["chart_count_guard_reason"] == "base_chart_count_below_guard_threshold"


def test_llm_chart_count_guard_restores_only_from_safe_base_selected_charts() -> None:
    llm = FakeChartSelectionLLM(
        {
            "selected_chart_ids": ["sales_trends_monthly_line", "invoice_value_distribution"],
            "selection_rationale": {
                "sales_trends_monthly_line": "保留销售趋势。",
                "invoice_value_distribution": "保留订单金额分布。",
            },
        }
    )

    plan = build_chart_selection_plan(
        schema_mapping=_mapping(
            {
                "Date": "order_datetime",
                "Sales": "sales_amount",
                "Category": "category",
                "Segment": "segment",
                "Region": "region",
                "Quantity": "quantity",
                "Order ID": "order_id",
            }
        ),
        dataset_profile={"category_count": 4, "segment_count": 3, "region_count": 4, "country_count": 1},
        analysis_focus={
            "selected_focuses": [
                "product_concentration_focus",
                "segment_region_focus",
                "sales_trend_focus",
                "metric_distribution_focus",
            ],
            "support_focuses": ["customer_order_structure_focus"],
        },
        section_priority={
            "core_sections": ["product_and_category", "segment_and_region", "sales_trends"],
            "support_sections": ["metric_distributions", "order_structure"],
            "skipped_sections": ["country_market"],
        },
        llm_client=llm,
        llm_profile="quick",
        llm_chart_selection_enabled=True,
    )

    restored_ids = plan["llm_selection_trace"]["restored_base_chart_ids"]
    deterministic_ids = {chart["chart_id"] for chart in plan["deterministic_selected_charts"]}
    rejected_ids = {chart["chart_id"] for chart in plan["rejected_charts"]}

    assert len(restored_ids) == 1
    assert set(restored_ids) <= deterministic_ids
    assert not set(restored_ids) & rejected_ids
    assert restored_ids[0] in [chart["chart_id"] for chart in plan["selected_charts"]]


def _bar_dedup_candidate(
    chart_id: str,
    *,
    title: str,
    required_fields: list[str],
    optional_fields: list[str] | None = None,
    business_question: str,
    insight_type: str = "mix",
    evidence_ids: list[str] | None = None,
    score: float = 0.7,
    dataset_specificity_score: float = 0.7,
) -> dict[str, object]:
    return {
        "chart_id": chart_id,
        "section_id": "test_section",
        "chart_kind": "bar",
        "title": title,
        "required_fields": required_fields,
        "optional_fields": optional_fields or [],
        "business_question": business_question,
        "insight_type": insight_type,
        "evidence_ids": evidence_ids or [],
        "score": score,
        "dataset_specificity_score": dataset_specificity_score,
    }


def _bar_dedup_audit() -> dict[str, object]:
    return {
        "chart_value_by_id": {
            "category_concentration_bar": {"overall_chart_value_score": 0.72},
            "segment_region_sales_bar": {"overall_chart_value_score": 0.42},
            "order_status_breakdown": {"overall_chart_value_score": 0.68},
            "discount_profit_quality_bar": {"overall_chart_value_score": 0.74},
        },
        "low_visual_gain_risk_chart_ids": ["segment_region_sales_bar"],
        "low_information_gain_chart_ids": ["segment_region_sales_bar"],
    }


def test_bar_dedup_guard_removes_semantically_duplicate_fallback_bar() -> None:
    category_bar = _bar_dedup_candidate(
        "category_concentration_bar",
        title="类目集中度对比",
        required_fields=["category", "sales_amount"],
        business_question="销售额是否集中在少数类目？",
        insight_type="concentration",
        evidence_ids=["top_category_sales_share"],
        score=0.77,
        dataset_specificity_score=0.82,
    )
    fallback_bar = _bar_dedup_candidate(
        "segment_region_sales_bar",
        title="客群/区域销售额对比",
        required_fields=["sales_amount"],
        optional_fields=["segment", "region", "category", "profit"],
        business_question="哪些客群、区域或类目贡献主要销售额？",
        evidence_ids=["region_count", "segment_count", "category_count"],
        score=0.9,
        dataset_specificity_score=0.5,
    )

    selected, trace = _apply_bar_dedup_guard(
        selected=[
            {"chart_id": "invoice_value_distribution", "chart_kind": "histogram"},
            dict(category_bar),
            {"chart_id": "sales_trends_monthly_line", "chart_kind": "line"},
            dict(fallback_bar),
        ],
        candidate_charts=[category_bar, fallback_bar],
        schema_mapping=_mapping({"Category": "category", "Sales": "sales_amount"}),
        chart_value_audit=_bar_dedup_audit(),
    )

    assert [chart["chart_id"] for chart in selected] == [
        "invoice_value_distribution",
        "category_concentration_bar",
        "sales_trends_monthly_line",
    ]
    assert trace["bar_dedup_guard_applied"] is True
    assert trace["removed_duplicate_bar_chart_ids"] == ["segment_region_sales_bar"]
    assert trace["retained_bar_chart_ids"] == ["category_concentration_bar"]
    assert trace["final_bar_chart_count"] == 1
    assert trace["bar_dedup_before_count"] == 4
    assert trace["bar_dedup_after_count"] == 3


def test_bar_dedup_guard_keeps_different_category_structures() -> None:
    category_bar = _bar_dedup_candidate(
        "category_concentration_bar",
        title="类目集中度对比",
        required_fields=["category", "sales_amount"],
        business_question="销售额是否集中在少数类目？",
        insight_type="concentration",
    )
    status_bar = _bar_dedup_candidate(
        "order_status_breakdown",
        title="STATUS 销售额结构",
        required_fields=["order_status", "sales_amount"],
        business_question="订单状态是否影响可确认销售额和履约质量？",
        insight_type="structure",
        evidence_ids=["distinct_order_status_count"],
    )

    selected, trace = _apply_bar_dedup_guard(
        selected=[dict(category_bar), dict(status_bar)],
        candidate_charts=[category_bar, status_bar],
        schema_mapping=_mapping({"Category": "category", "Status": "order_status", "Sales": "sales_amount"}),
        chart_value_audit=_bar_dedup_audit(),
    )

    assert [chart["chart_id"] for chart in selected] == ["category_concentration_bar", "order_status_breakdown"]
    assert trace["bar_dedup_guard_applied"] is False
    assert trace["bar_dedup_reason"] == "bars_not_semantically_duplicate"


def test_bar_dedup_guard_skips_single_bar() -> None:
    category_bar = _bar_dedup_candidate(
        "category_concentration_bar",
        title="类目集中度对比",
        required_fields=["category", "sales_amount"],
        business_question="销售额是否集中在少数类目？",
    )

    selected, trace = _apply_bar_dedup_guard(
        selected=[dict(category_bar), {"chart_id": "sales_trends_monthly_line", "chart_kind": "line"}],
        candidate_charts=[category_bar],
        schema_mapping=_mapping({"Category": "category", "Sales": "sales_amount"}),
        chart_value_audit=_bar_dedup_audit(),
    )

    assert [chart["chart_id"] for chart in selected] == ["category_concentration_bar", "sales_trends_monthly_line"]
    assert trace["bar_dedup_guard_applied"] is False
    assert trace["bar_dedup_reason"] == "only_one_bar"


def test_bar_dedup_guard_keeps_single_bar_restored_by_count_guard() -> None:
    category_bar = _bar_dedup_candidate(
        "category_concentration_bar",
        title="类目集中度对比",
        required_fields=["category", "sales_amount"],
        business_question="销售额是否集中在少数类目？",
        evidence_ids=["top_category_sales_share"],
    )
    restored, count_trace = _apply_chart_count_guard(
        selected=[
            {"chart_id": "invoice_value_distribution", "chart_kind": "histogram", "insight_type": "structure"},
            {"chart_id": "sales_trends_monthly_line", "chart_kind": "line", "insight_type": "trend"},
        ],
        raw_ids=["invoice_value_distribution", "sales_trends_monthly_line"],
        result={},
        candidate_charts=[
            {"chart_id": "invoice_value_distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
            {"chart_id": "sales_trends_monthly_line", "chart_kind": "line", "required_fields": ["order_datetime", "sales_amount"], "section_role": "core"},
            category_bar,
            {"chart_id": "metric_sales_distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
        ],
        schema_mapping=_mapping({"Date": "order_datetime", "Category": "category", "Sales": "sales_amount"}),
        chart_budget={},
        deterministic_chart_selection_plan={
            "selected_charts": [
                {"chart_id": "invoice_value_distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
                {"chart_id": "sales_trends_monthly_line", "chart_kind": "line", "required_fields": ["order_datetime", "sales_amount"], "section_role": "core"},
                category_bar,
                {"chart_id": "metric_sales_distribution", "chart_kind": "histogram", "required_fields": ["sales_amount"], "section_role": "core"},
            ],
            "rejected_charts": [],
        },
        base_selected_by_id={},
    )

    selected, dedup_trace = _apply_bar_dedup_guard(
        selected=restored,
        candidate_charts=[category_bar],
        schema_mapping=_mapping({"Date": "order_datetime", "Category": "category", "Sales": "sales_amount"}),
        chart_value_audit=_bar_dedup_audit(),
    )

    assert count_trace["chart_count_guard_applied"] is True
    assert [chart["chart_kind"] for chart in selected].count("bar") == 1
    assert dedup_trace["bar_dedup_guard_applied"] is False
    assert dedup_trace["bar_dedup_reason"] == "only_one_bar"


def test_bar_dedup_guard_keeps_bars_with_different_metric_or_question() -> None:
    category_bar = _bar_dedup_candidate(
        "category_concentration_bar",
        title="类目集中度对比",
        required_fields=["category", "sales_amount"],
        business_question="销售额是否集中在少数类目？",
        insight_type="concentration",
    )
    risk_bar = _bar_dedup_candidate(
        "discount_profit_quality_bar",
        title="各折扣区间利润质量：平均利润与亏损率",
        required_fields=["discount", "profit", "sales_amount"],
        business_question="高折扣是否显著侵蚀利润质量？",
        insight_type="risk",
        evidence_ids=["negative_profit_rate", "profit_margin_spread"],
    )

    selected, trace = _apply_bar_dedup_guard(
        selected=[dict(category_bar), dict(risk_bar)],
        candidate_charts=[category_bar, risk_bar],
        schema_mapping=_mapping({"Category": "category", "Discount": "discount", "Profit": "profit", "Sales": "sales_amount"}),
        chart_value_audit=_bar_dedup_audit(),
    )

    assert [chart["chart_id"] for chart in selected] == ["category_concentration_bar", "discount_profit_quality_bar"]
    assert trace["bar_dedup_guard_applied"] is False
    assert trace["bar_dedup_reason"] == "bars_not_semantically_duplicate"


def test_bar_dedup_guard_trace_contains_counts_and_groups() -> None:
    category_bar = _bar_dedup_candidate(
        "category_concentration_bar",
        title="类目集中度对比",
        required_fields=["category", "sales_amount"],
        business_question="销售额是否集中在少数类目？",
        evidence_ids=["top_category_sales_share"],
    )

    selected, trace = _apply_bar_dedup_guard(
        selected=[dict(category_bar)],
        candidate_charts=[category_bar],
        schema_mapping=_mapping({"Category": "category", "Sales": "sales_amount"}),
        chart_value_audit=_bar_dedup_audit(),
    )

    assert selected == [category_bar]
    assert trace["bar_dedup_before_count"] == 1
    assert trace["bar_dedup_after_count"] == 1
    assert trace["removed_duplicate_bar_chart_ids"] == []
    assert trace["retained_bar_chart_ids"] == ["category_concentration_bar"]
    assert trace["bar_dedup_reason"] == "only_one_bar"
    assert trace["bar_dedup_groups"] == []


def test_llm_chart_selection_falls_back_for_unknown_chart_id() -> None:
    llm = FakeChartSelectionLLM({"selected_chart_ids": ["not_a_chart"]})

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="full",
    )

    assert plan["selection_mode"] == "llm_fallback"
    assert plan["selected_charts"] == plan["deterministic_selected_charts"]
    assert plan["llm_selection_trace"]["invalid_chart_ids"] == ["not_a_chart"]
    assert "invalid_chart_ids" in plan["llm_selection_trace"]["fallback_reason"]


def test_llm_chart_selection_rejects_missing_required_field_chart() -> None:
    llm = FakeChartSelectionLLM({"selected_chart_ids": ["discount_profit_quality_bar"]})

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="full",
    )

    assert plan["selection_mode"] == "llm_fallback"
    assert "discount_profit_quality_bar" in plan["llm_selection_trace"]["hard_rejected_chart_ids"]
    assert plan["selected_charts"] == plan["deterministic_selected_charts"]


def test_llm_chart_selection_sanitizes_section_budget_overflow() -> None:
    llm = FakeChartSelectionLLM(
        {
            "selected_chart_ids": [
                "country_sales_bar",
                "country_concentration_pareto",
                "country_avg_order_value_bar",
                "country_order_count_bar",
            ],
            "selection_rationale": {
                "country_sales_bar": "国家销售贡献。",
                "country_concentration_pareto": "国家集中度。",
                "country_avg_order_value_bar": "国家客单价。",
                "country_order_count_bar": "国家订单数。",
            },
        }
    )

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="full",
    )

    assert plan["selection_mode"] == "llm_sanitized"
    assert "country_order_count_bar" in plan["llm_selection_trace"]["budget_rejected_chart_ids"]
    assert [chart["section_id"] for chart in plan["selected_charts"]].count("country_market") == CHART_BUDGET["core_max_per_section"]


def test_llm_chart_selection_prompt_is_safe_and_compact() -> None:
    llm = FakeChartSelectionLLM({"selected_chart_ids": ["country_sales_bar"]})

    build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="full",
    )

    payload = llm.payloads[0]
    payload_text = json.dumps(payload, ensure_ascii=False)
    assert "candidate_charts" in payload
    assert "dataset_signature" in payload
    assert "diagnostics" in payload
    assert "report" not in payload
    assert "notebook" not in payload_text.lower()
    assert "fig.show" not in payload_text
    assert "clean_df" not in payload_text
    assert "table_preview" not in payload_text
    assert "modules" not in payload


def test_llm_sanitized_selected_charts_are_legal_candidates() -> None:
    llm = FakeChartSelectionLLM(
        {
            "selected_chart_ids": [
                "productline_deal_size_stacked_bar",
                "country_sales_bar",
                "order_status_breakdown",
            ]
        }
    )

    plan = build_chart_selection_plan(
        **_sample_sales_kwargs(),
        llm_client=llm,
        llm_profile="full",
    )

    candidate_by_id = {chart["chart_id"]: chart for chart in plan["candidate_charts"]}
    hard_reasons = {
        "missing required fields",
        "skipped section",
        "country field has no useful market slice",
        "too many categories",
        "limited by data availability",
    }
    for chart in plan["selected_charts"]:
        candidate = candidate_by_id[chart["chart_id"]]
        assert chart["chart_id"] in candidate_by_id
        assert not any(str(candidate.get("rejection_reason", "")).startswith(reason) for reason in hard_reasons)
        assert set(candidate["required_fields"]) <= set(_sample_sales_kwargs()["schema_mapping"].field_mapping.values())  # type: ignore[index,union-attr]


