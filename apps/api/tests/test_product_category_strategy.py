from __future__ import annotations

from app.schemas.report import ModuleReport


def _product_module() -> ModuleReport:
    return ModuleReport(
        module_id="product_contribution_analysis",
        title="Product Contribution",
        chart_type="bar",
        summary_metrics={
            "distinct_products": 120,
            "top_1_sales_share": 0.18,
            "top_10_sales_share": 0.64,
            "products_to_80pct_sales": 22,
        },
        tables={
            "top_products": [
                {
                    "Product Name": "A",
                    "Sales": 1000.0,
                    "Profit": 50.0,
                    "sales_share": 0.18,
                    "cumulative_sales_share": 0.18,
                    "profit_margin": 0.05,
                }
            ],
            "category_sales": [
                {
                    "Category": "Tech",
                    "Sub-Category": "Phones",
                    "Sales": 3200.0,
                    "Profit": 500.0,
                    "profit_margin": 0.1562,
                }
            ],
            "category_profit_quality": [
                {
                    "Category": "Furniture",
                    "Sales": 1600.0,
                    "Profit": -80.0,
                    "profit_margin": -0.05,
                }
            ],
            "high_sales_low_profit_products": [
                {
                    "Product Name": "A",
                    "Sales": 1000.0,
                    "Profit": 50.0,
                    "profit_margin": 0.05,
                }
            ],
        },
    )


def test_fallback_product_category_strategy_recommends_profit_and_concentration_views() -> None:
    from app.services.product_category_strategy import build_product_category_strategy

    strategy = build_product_category_strategy(_product_module(), llm_client=None)

    charts = [view["chart"] for view in strategy["views"]]

    assert len(charts) <= 4
    assert "category_treemap" in charts
    assert "product_pareto" in charts
    assert "category_profit_margin_bar" in charts
    assert "product_profit_bridge_scatter" in charts


class UnsafeProductStrategyLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.last_payload = None

    def suggest_product_category_strategy(self, payload):
        self.last_payload = payload
        return {
            "views": [
                {"chart": "product_profit_bridge_scatter", "reason": "Compare scale and margin."},
                {"chart": "delete_files", "reason": "unsafe"},
                {"chart": "category_profit_margin_bar", "reason": "Find weak categories."},
            ]
        }


def test_llm_product_category_strategy_is_sanitized_to_safe_views() -> None:
    from app.services.product_category_strategy import build_product_category_strategy

    llm = UnsafeProductStrategyLLM()

    strategy = build_product_category_strategy(_product_module(), llm_client=llm)

    charts = [view["chart"] for view in strategy["views"]]
    assert "delete_files" not in charts
    assert "product_profit_bridge_scatter" in charts
    assert "category_profit_margin_bar" in charts
    assert llm.last_payload["allowed_charts"] == [
        "category_treemap",
        "top_product_bar",
        "product_pareto",
        "category_profit_margin_bar",
        "product_profit_bridge_scatter",
        "high_sales_low_profit_bar",
    ]


class FocusedProductStrategyLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_product_category_strategy(self, payload):
        return {
            "views": [
                {
                    "chart": "product_pareto",
                    "reason": "Use Pareto because product contribution is long-tail.",
                },
                {
                    "chart": "product_profit_bridge_scatter",
                    "reason": "Bridge sales and profit instead of another simple top bar.",
                },
            ]
        }


def test_llm_product_category_strategy_does_not_append_unselected_fallback_views() -> None:
    from app.services.product_category_strategy import build_product_category_strategy

    strategy = build_product_category_strategy(
        _product_module(),
        llm_client=FocusedProductStrategyLLM(),
    )

    assert strategy["source"] == "llm_sanitized"
    assert [view["chart"] for view in strategy["views"]] == [
        "product_pareto",
        "product_profit_bridge_scatter",
    ]
