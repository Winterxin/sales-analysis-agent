from __future__ import annotations

import pandas as pd

from app.analysis.modules import sales_trend


def test_sales_trend_profiles_daily_monthly_weekday_and_seasonality_views() -> None:
    dates = pd.date_range("2024-01-01", periods=420, freq="D")
    frame = pd.DataFrame(
        {
            "Order Date": dates,
            "Sales": [
                100
                + (date.month * 5)
                + (20 if date.weekday() in {0, 1} else 0)
                + (60 if date.month in {11, 12} else 0)
                for date in dates
            ],
            "Quantity": [2 + (1 if date.weekday() in {4, 5} else 0) for date in dates],
        }
    )

    result = sales_trend.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
            "quantity": "Quantity",
        },
    )

    assert result.module_id == "sales_trend_analysis"
    assert result.summary_metrics["daily_grain_count"] == 420
    assert result.summary_metrics["monthly_grain_count"] >= 13
    assert result.summary_metrics["year_count"] >= 2
    assert result.summary_metrics["has_multi_year"] is True
    assert "monthly_totals" in result.tables
    assert "weekday_profile" in result.tables
    assert "year_month_totals" in result.tables
    assert any(row["weekday_cn"] == "周一" for row in result.tables["weekday_profile"])
    assert any(
        row["year"] == 2024 and row["month"] == 12 for row in result.tables["year_month_totals"]
    )


def test_sales_trend_degrades_when_quantity_is_missing_for_weekly_store_sales() -> None:
    frame = pd.DataFrame(
        {
            "Store": [1, 1, 2, 2],
            "Dept": [10, 20, 10, 20],
            "Date": ["2024-01-05", "2024-01-12", "2024-02-02", "2024-02-09"],
            "Weekly_Sales": [1200.0, 1500.0, 900.0, 1100.0],
        }
    )

    result = sales_trend.run(
        frame,
        {
            "order_datetime": "Date",
            "sales_amount": "Weekly_Sales",
            "store": "Store",
            "category": "Dept",
        },
    )

    assert result.module_id == "sales_trend_analysis"
    assert result.summary_metrics["total_sales_amount"] == 4700.0
    assert "total_quantity" not in result.summary_metrics
    assert result.summary_metrics["skipped_quantity_dependent_analysis"] is True
    assert "quantity" not in result.tables["daily_totals"][0]
    assert any("missing quantity" in warning for warning in result.warnings)
