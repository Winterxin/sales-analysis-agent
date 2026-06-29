from __future__ import annotations

from app.schemas.report import ModuleReport


def _discount_module() -> ModuleReport:
    return ModuleReport(
        module_id="discount_profit_analysis",
        title="Discount Profit",
        chart_type="scatter",
        summary_metrics={
            "negative_profit_order_count": 4,
            "high_discount_order_count": 4,
            "negative_profit_rate": 0.57,
            "worst_discount_bucket": "30%+",
            "high_sales_low_profit_count": 2,
        },
        tables={
            "discount_profit_risk_buckets": [
                {
                    "discount_bucket": "0%",
                    "avg_profit": 80.0,
                    "negative_profit_rate": 0.0,
                    "profit_margin": 0.1,
                    "order_count": 2,
                },
                {
                    "discount_bucket": "30%+",
                    "avg_profit": -46.25,
                    "negative_profit_rate": 0.75,
                    "profit_margin": -0.05,
                    "order_count": 4,
                },
            ],
            "high_sales_low_profit_items": [
                {
                    "Product Name": "F",
                    "Sales": 1000.0,
                    "Profit": -90.0,
                    "profit_margin": -0.09,
                    "avg_discount": 0.45,
                }
            ],
            "category_discount_risk": [
                {
                    "Category": "Furniture",
                    "discount_bucket": "30%+",
                    "negative_profit_rate": 0.67,
                    "profit_margin": -0.06,
                    "Sales": 3100.0,
                },
                {
                    "Category": "Tech",
                    "discount_bucket": "0-10%",
                    "negative_profit_rate": 0.0,
                    "profit_margin": 0.1,
                    "Sales": 1600.0,
                },
            ],
        },
        findings=[],
    )


def test_fallback_discount_profit_strategy_recommends_risk_views() -> None:
    from app.services.discount_profit_strategy import build_discount_profit_strategy

    strategy = build_discount_profit_strategy(_discount_module(), llm_client=None)

    charts = [view["chart"] for view in strategy["views"]]

    assert "discount_profit_scatter" in charts
    assert "discount_bucket_boxplot" in charts
    assert "bucket_profit_quality_bar" in charts
    assert "high_sales_low_profit_bar" in charts
    assert "category_discount_risk_heatmap" in charts


class UnsafeDiscountProfitStrategyLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.last_payload = None

    def suggest_discount_profit_strategy(self, payload):
        self.last_payload = payload
        return {
            "views": [
                {
                    "chart": "bucket_profit_quality_bar",
                    "reason": "The 30%+ bucket has visible profit erosion.",
                },
                {"chart": "delete_files", "reason": "unsafe"},
                {
                    "chart": "category_discount_risk_heatmap",
                    "reason": "Category by discount bucket shows concentration.",
                },
            ]
        }


def test_llm_discount_profit_strategy_is_sanitized_to_safe_views() -> None:
    from app.services.discount_profit_strategy import build_discount_profit_strategy

    llm = UnsafeDiscountProfitStrategyLLM()

    strategy = build_discount_profit_strategy(_discount_module(), llm_client=llm)

    charts = [view["chart"] for view in strategy["views"]]
    assert "delete_files" not in charts
    assert "bucket_profit_quality_bar" in charts
    assert "category_discount_risk_heatmap" in charts
    assert llm.last_payload["allowed_charts"] == [
        "discount_profit_scatter",
        "discount_bucket_boxplot",
        "bucket_profit_quality_bar",
        "high_sales_low_profit_bar",
        "category_discount_risk_heatmap",
    ]
