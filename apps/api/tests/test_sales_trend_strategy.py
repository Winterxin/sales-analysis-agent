from __future__ import annotations

from app.schemas.report import ModuleReport
from app.services.sales_trend_strategy import build_sales_trend_strategy


def _trend_module() -> ModuleReport:
    return ModuleReport(
        module_id="sales_trend_analysis",
        title="Sales Trend",
        chart_type="line",
        summary_metrics={
            "daily_grain_count": 420,
            "monthly_grain_count": 14,
            "year_count": 2,
            "has_multi_year": True,
            "peak_period": "2024-12-18",
            "trough_period": "2024-01-03",
        },
        tables={
            "daily_totals": [
                {"period": "2024-01-01", "Sales": 120.0, "Quantity": 3},
                {"period": "2024-01-02", "Sales": 118.0, "Quantity": 2},
            ],
            "monthly_totals": [
                {"order_month": "2024-01", "Sales": 3800.0, "Quantity": 95},
                {"order_month": "2024-02", "Sales": 4200.0, "Quantity": 102},
            ],
            "weekday_profile": [
                {"weekday": "Monday", "weekday_cn": "周一", "avg_sales": 158.0, "avg_quantity": 2.8},
                {"weekday": "Sunday", "weekday_cn": "周日", "avg_sales": 131.0, "avg_quantity": 2.3},
            ],
            "year_month_totals": [
                {"year": 2024, "month": 1, "month_name": "Jan", "Sales": 3800.0},
                {"year": 2025, "month": 1, "month_name": "Jan", "Sales": 4500.0},
            ],
        },
        findings=[],
    )


def test_fallback_sales_trend_strategy_recommends_richer_eda_views() -> None:
    strategy = build_sales_trend_strategy(_trend_module(), llm_client=None)

    charts = [view["chart"] for view in strategy["views"]]

    assert charts[0] == "daily_rolling_line"
    assert "monthly_line" in charts
    assert "weekday_bar" in charts
    assert "year_month_heatmap" in charts


class UnsafeTrendStrategyLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.last_payload = None

    def suggest_sales_trend_strategy(self, payload):
        self.last_payload = payload
        return {
            "views": [
                {"chart": "daily_rolling_line", "reason": "Use daily and rolling views."},
                {"chart": "shell_exec", "reason": "unsafe"},
                {"chart": "year_month_heatmap", "reason": "Show seasonality."},
            ]
        }


def test_llm_sales_trend_strategy_is_sanitized_to_safe_views() -> None:
    llm = UnsafeTrendStrategyLLM()

    strategy = build_sales_trend_strategy(_trend_module(), llm_client=llm)

    charts = [view["chart"] for view in strategy["views"]]
    assert "shell_exec" not in charts
    assert "daily_rolling_line" in charts
    assert "year_month_heatmap" in charts
    assert llm.last_payload["allowed_charts"] == [
        "daily_rolling_line",
        "monthly_line",
        "weekday_bar",
        "year_month_heatmap",
    ]
