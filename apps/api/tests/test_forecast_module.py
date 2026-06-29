from __future__ import annotations

import pandas as pd

from app.analysis.modules import forecast


def test_forecast_returns_28_day_projection() -> None:
    frame = pd.DataFrame(
        {
            "Order Date": pd.date_range("2026-01-01", periods=40, freq="D"),
            "Sales": [100 + day for day in range(40)],
            "Quantity": [2 + (day % 3) for day in range(40)],
        }
    )

    result = forecast.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
            "quantity": "Quantity",
        },
    )

    assert result.module_id == "forecast_analysis"
    assert result.summary_metrics["horizon_days"] == 28
    assert len(result.tables["forecast"]) == 28


def test_forecast_returns_time_split_baseline_metrics() -> None:
    frame = pd.DataFrame(
        {
            "Order Date": pd.date_range("2026-01-01", periods=80, freq="D"),
            "Sales": [100 + day * 2 for day in range(80)],
            "Quantity": [1 + (day % 4) for day in range(80)],
        }
    )

    result = forecast.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
            "quantity": "Quantity",
        },
    )

    assert result.summary_metrics["forecast_status"] == "regression_evaluated"
    assert result.summary_metrics["time_split"] == "chronological"
    assert result.summary_metrics["regression_time_split"] == "chronological"
    assert result.summary_metrics["day_span"] == 79
    assert result.summary_metrics["series_granularity"] == "daily"
    assert result.summary_metrics["date_gap_ratio"] == 0
    assert result.summary_metrics["zero_or_missing_period_ratio"] == 0
    assert result.summary_metrics["backtest_period_count"] > 0
    assert result.summary_metrics["best_baseline"] in {"naive_last_value", "moving_average_7"}
    assert {row["baseline"] for row in result.tables["baseline_metrics"]} == {
        "naive_last_value",
        "moving_average_7",
    }
    assert all(row["mae"] >= 0 for row in result.tables["baseline_metrics"])
    assert len(result.tables["baseline_backtest"]) == result.summary_metrics["backtest_period_count"]


def test_forecast_runs_lightweight_regression_without_random_split_or_leakage() -> None:
    frame = pd.DataFrame(
        {
            "Order Date": pd.date_range("2026-01-02", periods=120, freq="W-FRI"),
            "Weekly_Sales": [
                1000
                + week * 18
                + (week % 4) * 35
                for week in range(120)
            ],
        }
    )

    result = forecast.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Weekly_Sales",
        },
    )

    assert result.summary_metrics["forecast_status"] == "regression_evaluated"
    assert result.summary_metrics["regression_time_split"] == "chronological"
    assert result.summary_metrics["regression_cv_strategy"] == "TimeSeriesSplit_3"
    assert result.summary_metrics["regression_feature_leakage_guard"] == "shifted_lag_rolling_only"
    assert result.summary_metrics["regression_effective_sample_count"] >= 30
    assert result.summary_metrics["best_model"] in {
        "naive_last_value",
        "moving_average_7",
        "Ridge Regression",
        "RandomForestRegressor-small",
    }
    model_names = {row["model"] for row in result.tables["model_comparison"]}
    assert {"naive_last_value", "moving_average_7", "Ridge Regression", "RandomForestRegressor-small"} <= model_names
    cv_fold_rows = result.tables["regression_cv_folds"]
    assert cv_fold_rows
    assert {row["model"] for row in cv_fold_rows} == model_names
    assert {row["fold"] for row in cv_fold_rows} == {1, 2, 3}
    assert all(row["train_start"] <= row["train_end"] < row["valid_start"] <= row["valid_end"] for row in cv_fold_rows)
    assert all(key in cv_fold_rows[0] for key in ["mae", "rmse", "mape", "r2"])
    cv_summary_rows = result.tables["regression_cv_summary"]
    assert cv_summary_rows
    assert {row["model"] for row in cv_summary_rows} == model_names
    assert all(
        key in cv_summary_rows[0]
        for key in ["cv_mae_mean", "cv_mae_std", "cv_mape_mean", "cv_mape_std", "cv_r2_mean", "cv_r2_std"]
    )
    feature_names = {row["feature"] for row in result.tables["feature_importance"]}
    assert "lag_1" in feature_names
    assert any(name.startswith("rolling_mean_") for name in feature_names)
    assert len(result.tables["worst_error_periods"]) <= 5
    assert all("predicted" in row for row in result.tables["regression_backtest"])
    assert all(row["error_direction"] in {"over", "under", "exact"} for row in result.tables["regression_backtest"])
    assert all("error_direction" in row for row in result.tables["worst_error_periods"])
    assert result.summary_metrics["over_prediction_count"] + result.summary_metrics["under_prediction_count"] <= len(
        result.tables["regression_backtest"]
    )
    selection_basis = result.summary_metrics["selection_basis"]
    assert selection_basis["selected_model"] == result.summary_metrics["best_model"]
    assert selection_basis["holdout_mae_rank"] == 1
    assert "cv_mae_rank" in selection_basis
    assert selection_basis["cv_stability_note"] in {
        "holdout_and_cv_aligned",
        "cv_mean_best_but_variance_noticeable",
        "holdout_best_cv_near_top",
        "holdout_best_cv_not_stable",
        "cv_unavailable",
    }


def test_forecast_keeps_baseline_only_when_lag_rolling_samples_are_insufficient() -> None:
    frame = pd.DataFrame(
        {
            "Order Date": pd.date_range("2026-01-01", periods=36, freq="D"),
            "Sales": [100 + day for day in range(36)],
        }
    )

    result = forecast.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
        },
    )

    assert result.summary_metrics["forecast_status"] == "baseline_evaluated"
    assert result.summary_metrics["regression_status"] == "skipped"
    assert result.summary_metrics["regression_skip_reason"] == "insufficient_regression_periods"
    assert "model_comparison" not in result.tables


def test_forecast_skips_when_time_series_has_too_few_periods() -> None:
    frame = pd.DataFrame(
        {
            "Order Date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "Sales": [100 + day for day in range(10)],
        }
    )

    result = forecast.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
        },
    )

    assert result.summary_metrics["forecast_status"] == "skipped"
    assert "insufficient_forecast_periods" in result.warnings
    assert "forecast" not in result.tables


def test_forecast_skips_when_time_span_is_too_short_for_backtest() -> None:
    frame = pd.DataFrame(
        {
            "Order Date": pd.date_range("2026-01-01", periods=28, freq="D"),
            "Sales": [100 + day for day in range(28)],
        }
    )

    result = forecast.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
        },
    )

    assert result.summary_metrics["forecast_status"] == "skipped"
    assert result.summary_metrics["day_span"] == 27
    assert "insufficient_time_span" in result.warnings


def test_forecast_skips_sparse_daily_series_with_large_gap_ratio() -> None:
    dates = list(pd.date_range("2026-01-01", periods=20, freq="D")) + list(
        pd.date_range("2026-12-01", periods=20, freq="D")
    )
    frame = pd.DataFrame(
        {
            "Order Date": dates,
            "Sales": [100 + day for day in range(len(dates))],
        }
    )

    result = forecast.run(
        frame,
        {
            "order_datetime": "Order Date",
            "sales_amount": "Sales",
        },
    )

    assert result.summary_metrics["forecast_status"] == "skipped"
    assert result.summary_metrics["series_granularity"] == "daily"
    assert result.summary_metrics["date_gap_ratio"] > 0.5
    assert "sparse_time_series" in result.warnings
