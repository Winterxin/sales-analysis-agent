from __future__ import annotations

import pandas as pd

from app.analysis.modules import metric_distribution


def test_metric_distribution_profiles_core_numeric_metrics() -> None:
    frame = pd.DataFrame(
        {
            "Order Date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "Sales": [120.0, 150.0, 180.0, 220.0, 900.0, 1500.0],
            "Quantity": [1, 2, 2, 3, 4, 8],
            "Discount": [0.0, 0.1, 0.0, 0.2, 0.3, 0.4],
            "Profit": [10.0, 15.0, 22.0, 18.0, -30.0, -120.0],
        }
    )

    result = metric_distribution.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
            "quantity": "Quantity",
            "discount": "Discount",
            "profit": "Profit",
        },
    )

    assert result.module_id == "metric_distribution_analysis"
    assert result.summary_metrics["metric_count"] == 4
    assert len(result.tables["metric_quantiles"]) == 4
    assert len(result.tables["metric_outliers"]) == 4
    assert "metric_correlations" in result.tables
    assert "relationship_candidates" in result.tables
    assert any(
        row["x_metric"] == "discount" and row["y_metric"] == "profit"
        for row in result.tables["metric_correlations"]
    )
    assert any(
        row["x_metric"] == "discount" and row["y_metric"] == "profit"
        for row in result.tables["relationship_candidates"]
    )
    assert any(row["metric"] == "sales_amount" for row in result.tables["metric_quantiles"])
    assert any(row["metric"] == "profit" for row in result.tables["metric_outliers"])
    sales_quantiles = next(
        row for row in result.tables["metric_quantiles"] if row["metric"] == "sales_amount"
    )
    profit_quantiles = next(
        row for row in result.tables["metric_quantiles"] if row["metric"] == "profit"
    )
    assert sales_quantiles["p95"] is not None
    assert sales_quantiles["skew"] is not None
    assert sales_quantiles["mean_median_ratio"] is not None
    assert sales_quantiles["p99_median_ratio"] is not None
    assert sales_quantiles["non_negative"] is True
    assert profit_quantiles["non_negative"] is False
