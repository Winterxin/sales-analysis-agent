from __future__ import annotations

from app.services.chart_portfolio_diversity import apply_portfolio_diversity_gate


def _candidate(
    chart_id: str,
    section_id: str,
    chart_kind: str,
    required_fields: list[str],
    score: float,
    *,
    rejection_reason: str = "",
) -> dict[str, object]:
    return {
        "chart_id": chart_id,
        "section_id": section_id,
        "chart_kind": chart_kind,
        "title": chart_id,
        "required_fields": required_fields,
        "score": score,
        "section_role": "core",
        "insight_type": chart_id,
        "rejection_reason": rejection_reason,
    }


def test_diversity_gate_replaces_bar_with_safe_non_bar_and_updates_trace() -> None:
    plan = {
        "selected_charts": [
            _candidate("category_profit_margin_bar", "product_and_category", "bar", ["category", "sales_amount"], 0.86),
            _candidate("top_product_profit_gap_bar", "product_and_category", "bar", ["product_name", "sales_amount"], 0.72),
            _candidate("region_sales_profit_bar", "segment_and_region", "bar", ["region", "sales_amount"], 0.75),
            _candidate("sales_trends_monthly_line", "sales_trends", "line", ["order_datetime", "sales_amount"], 0.9),
        ],
        "candidate_charts": [
            _candidate("category_profit_margin_bar", "product_and_category", "bar", ["category", "sales_amount"], 0.86),
            _candidate("top_product_profit_gap_bar", "product_and_category", "bar", ["product_name", "sales_amount"], 0.72),
            _candidate("region_sales_profit_bar", "segment_and_region", "bar", ["region", "sales_amount"], 0.75),
            _candidate("sales_trends_monthly_line", "sales_trends", "line", ["order_datetime", "sales_amount"], 0.9),
            _candidate("profit_distribution_if_available", "metric_distributions", "histogram", ["profit"], 0.78),
            _candidate("unsafe_discount_scatter", "discount_and_profit", "scatter", ["discount", "profit"], 0.99),
        ],
        "intent_guided_selection_trace": {
            "remove_only_chart_ids": ["profit_distribution_if_available"],
            "base_removed_chart_ids": ["profit_distribution_if_available"],
            "final_selected_chart_ids": [
                "category_profit_margin_bar",
                "top_product_profit_gap_bar",
                "region_sales_profit_bar",
                "sales_trends_monthly_line",
            ],
        },
    }

    updated = apply_portfolio_diversity_gate(
        plan,
        mapped_fields={"category", "product_name", "region", "order_datetime", "sales_amount", "profit"},
        renderable_chart_ids={"profit_distribution_if_available", "sales_trends_monthly_line"},
    )

    selected_ids = [item["chart_id"] for item in updated["selected_charts"]]
    trace = updated["intent_guided_selection_trace"]

    assert "profit_distribution_if_available" in selected_ids
    assert "top_product_profit_gap_bar" not in selected_ids
    assert "profit_distribution_if_available" not in trace["remove_only_chart_ids"]
    assert "profit_distribution_if_available" not in trace["base_removed_chart_ids"]
    assert trace["final_selected_chart_ids"] == selected_ids
    assert trace["portfolio_diversity_status"] == "improved"
    assert trace["bar_family_ratio_after"] < trace["bar_family_ratio_before"]


def test_diversity_gate_does_not_change_low_risk_portfolio() -> None:
    plan = {
        "selected_charts": [
            _candidate("country_sales_bar", "country_market", "bar", ["country", "sales_amount"], 0.9),
            _candidate("sales_trends_monthly_line", "sales_trends", "line", ["order_datetime", "sales_amount"], 0.85),
            _candidate("invoice_value_distribution", "order_structure", "histogram", ["order_id", "sales_amount"], 0.8),
            _candidate("profit_distribution_if_available", "metric_distributions", "histogram", ["profit"], 0.7),
        ],
        "candidate_charts": [],
        "intent_guided_selection_trace": {"final_selected_chart_ids": []},
    }

    updated = apply_portfolio_diversity_gate(
        plan,
        mapped_fields={"country", "sales_amount", "order_datetime", "order_id", "profit"},
        renderable_chart_ids={"country_sales_bar", "sales_trends_monthly_line", "invoice_value_distribution"},
    )

    assert updated["selected_charts"] == plan["selected_charts"]
    assert updated["intent_guided_selection_trace"]["portfolio_diversity_status"] == "not_needed"
