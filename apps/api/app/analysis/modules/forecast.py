from __future__ import annotations

from datetime import timedelta
import math

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.analysis.contracts import ModuleResult


MIN_FORECAST_PERIODS = 28
MIN_FORECAST_DAY_SPAN = 30
MIN_TRAIN_PERIODS = 14
MIN_BACKTEST_PERIODS = 7
FORECAST_HORIZON_DAYS = 28
MOVING_AVERAGE_WINDOW = 7
MAX_DATE_GAP_RATIO = 0.4
MAX_ZERO_OR_MISSING_PERIOD_RATIO = 0.5
MIN_REGRESSION_PERIODS = 40
MIN_REGRESSION_EFFECTIVE_SAMPLES = 30
MIN_REGRESSION_TRAIN_PERIODS = 20
HIGH_REGRESSION_MAPE_THRESHOLD = 100.0


def _skipped_result(reason: str, summary_metrics: dict[str, object] | None = None) -> ModuleResult:
    metrics = {"forecast_status": "skipped", "skip_reason": reason}
    if summary_metrics:
        metrics.update(summary_metrics)
    return ModuleResult(
        module_id="forecast_analysis",
        title="基础预测",
        chart_type="line",
        summary_metrics=metrics,
        warnings=[reason],
    )


def _metric_row(baseline: str, actual: pd.Series, predicted: pd.Series) -> dict[str, object]:
    errors = actual - predicted
    mae = float(errors.abs().mean())
    rmse = float(math.sqrt((errors**2).mean()))
    non_zero_actual = actual[actual.abs() > 1e-9]
    if non_zero_actual.empty:
        mape: float | None = None
    else:
        aligned_predicted = predicted.loc[non_zero_actual.index]
        mape = float(((non_zero_actual - aligned_predicted).abs() / non_zero_actual.abs()).mean() * 100)
    return {
        "baseline": baseline,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2) if mape is not None else None,
    }


def _mape(actual: pd.Series, predicted: pd.Series) -> float | None:
    non_zero_actual = actual[actual.abs() > 1e-9]
    if non_zero_actual.empty:
        return None
    aligned_predicted = predicted.loc[non_zero_actual.index]
    return float(((non_zero_actual - aligned_predicted).abs() / non_zero_actual.abs()).mean() * 100)


def _r2(actual: pd.Series, predicted: pd.Series) -> float | None:
    if len(actual) < 2 or float(actual.var()) <= 1e-12:
        return None
    score = float(r2_score(actual, predicted))
    return score if math.isfinite(score) else None


def _regression_metric_row(
    model: str,
    model_type: str,
    actual: pd.Series,
    predicted: pd.Series,
    *,
    baseline_mae: float,
) -> dict[str, object]:
    errors = actual - predicted
    mae = float(errors.abs().mean())
    rmse = float(math.sqrt((errors**2).mean()))
    improvement = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 1e-9 else 0.0
    mape = _mape(actual, predicted)
    r2 = _r2(actual, predicted)
    return {
        "model": model,
        "model_type": model_type,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2) if mape is not None else None,
        "r2": round(r2, 4) if r2 is not None else None,
        "improvement_vs_baseline": round(improvement, 2),
    }


def _parse_datetime(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def _series_quality_metrics(daily: pd.DataFrame, sales_col: str) -> dict[str, object]:
    periods = daily["period"].sort_values()
    day_span = int((periods.max() - periods.min()).days) if len(periods) else 0
    deltas = periods.diff().dt.days.dropna()
    median_gap_days = float(deltas.median()) if not deltas.empty else 0.0
    if median_gap_days <= 2:
        granularity = "daily"
        expected_step_days = 1
    elif median_gap_days <= 10:
        granularity = "weekly"
        expected_step_days = 7
    elif 25 <= median_gap_days <= 35:
        granularity = "monthly"
        expected_step_days = 30
    else:
        granularity = "irregular"
        expected_step_days = max(1, int(round(median_gap_days or 1)))
    expected_period_count = max(1, int(day_span // expected_step_days) + 1)
    missing_period_count = max(0, expected_period_count - len(daily))
    zero_period_count = int((daily[sales_col].astype(float).abs() <= 1e-9).sum())
    date_gap_ratio = missing_period_count / expected_period_count
    zero_or_missing_ratio = (missing_period_count + zero_period_count) / expected_period_count
    return {
        "day_span": day_span,
        "median_gap_days": round(median_gap_days, 2),
        "series_granularity": granularity,
        "expected_period_count": expected_period_count,
        "observed_period_count": int(len(daily)),
        "date_gap_ratio": round(date_gap_ratio, 4),
        "zero_or_missing_period_ratio": round(zero_or_missing_ratio, 4),
    }


def _regression_windows(granularity: str) -> tuple[list[int], int]:
    if granularity == "daily":
        return [1, 2, 7], 7
    return [1, 2, 4], 4


def _build_regression_frame(daily: pd.DataFrame, sales_col: str, granularity: str) -> pd.DataFrame:
    features = pd.DataFrame({"period": daily["period"], "target": daily[sales_col].astype(float)})
    periods = pd.to_datetime(features["period"], errors="coerce")
    features["year"] = periods.dt.year
    features["month"] = periods.dt.month
    features["quarter"] = periods.dt.quarter
    features["week"] = periods.dt.isocalendar().week.astype(int)
    features["weekday"] = periods.dt.weekday
    lags, rolling_window = _regression_windows(granularity)
    target = features["target"]
    for lag in lags:
        features[f"lag_{lag}"] = target.shift(lag)
    shifted = target.shift(1)
    features[f"rolling_mean_{rolling_window}"] = shifted.rolling(rolling_window).mean()
    features[f"rolling_std_{rolling_window}"] = shifted.rolling(rolling_window).std()
    return features.dropna().reset_index(drop=True)


def _regression_estimators() -> dict[str, object]:
    return {
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForestRegressor-small": RandomForestRegressor(
            n_estimators=60,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ),
    }


def _cv_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model") or ""), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for model, model_rows in by_model.items():
        summary: dict[str, object] = {"model": model}
        for metric in ("mae", "mape", "r2"):
            values = [
                float(row[metric])
                for row in model_rows
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                summary[f"cv_{metric}_mean"] = round(mean, 2) if metric != "r2" else round(mean, 4)
                summary[f"cv_{metric}_std"] = round(math.sqrt(variance), 2) if metric != "r2" else round(math.sqrt(variance), 4)
            else:
                summary[f"cv_{metric}_mean"] = None
                summary[f"cv_{metric}_std"] = None
        summary_rows.append(summary)
    return summary_rows


def _cv_fold_rows(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    estimators: dict[str, object],
) -> list[dict[str, object]]:
    if len(feature_df) < 80:
        return []
    rows: list[dict[str, object]] = []
    splitter = TimeSeriesSplit(n_splits=3)
    for fold, (train_index, valid_index) in enumerate(splitter.split(feature_df), start=1):
        train_fold = feature_df.iloc[train_index]
        valid_fold = feature_df.iloc[valid_index]
        actual = valid_fold["target"].astype(float)
        train_target = train_fold["target"].astype(float)
        fold_predictions: dict[str, pd.Series] = {
            "naive_last_value": pd.Series(float(train_target.iloc[-1]), index=actual.index),
            "moving_average_7": pd.Series(float(train_target.tail(7).mean()), index=actual.index),
        }
        for model_name, estimator in estimators.items():
            fitted = clone(estimator)
            fitted.fit(train_fold[feature_cols], train_fold["target"])
            fold_predictions[model_name] = pd.Series(fitted.predict(valid_fold[feature_cols]), index=actual.index)
        for model_name, predicted in fold_predictions.items():
            metric = _regression_metric_row(model_name, "cv", actual, predicted, baseline_mae=0)
            rows.append(
                {
                    "fold": fold,
                    "train_start": pd.Timestamp(train_fold["period"].iloc[0]).date().isoformat(),
                    "train_end": pd.Timestamp(train_fold["period"].iloc[-1]).date().isoformat(),
                    "valid_start": pd.Timestamp(valid_fold["period"].iloc[0]).date().isoformat(),
                    "valid_end": pd.Timestamp(valid_fold["period"].iloc[-1]).date().isoformat(),
                    "model": model_name,
                    "mae": metric["mae"],
                    "rmse": metric["rmse"],
                    "mape": metric["mape"],
                    "r2": metric["r2"],
                }
            )
    return rows


def _cv_metrics_from_summary(model_name: str, cv_summary: list[dict[str, object]]) -> dict[str, object]:
    row = next((item for item in cv_summary if item.get("model") == model_name), None)
    if not row:
        return {}
    return {
        key: row[key]
        for key in ("cv_mae_mean", "cv_mae_std", "cv_mape_mean", "cv_mape_std", "cv_r2_mean", "cv_r2_std")
        if row.get(key) is not None
    }


def _regression_selection_basis(
    comparison: list[dict[str, object]],
    cv_summary: list[dict[str, object]],
    *,
    best_model: str,
) -> dict[str, object]:
    holdout_ranks = {
        str(row.get("model")): rank
        for rank, row in enumerate(
            sorted(comparison, key=lambda row: float(row.get("mae") or float("inf"))),
            start=1,
        )
    }
    cv_rank_rows = [
        row
        for row in cv_summary
        if row.get("cv_mae_mean") is not None and math.isfinite(float(row.get("cv_mae_mean")))
    ]
    cv_ranks = {
        str(row.get("model")): rank
        for rank, row in enumerate(
            sorted(cv_rank_rows, key=lambda row: float(row.get("cv_mae_mean") or float("inf"))),
            start=1,
        )
    }
    cv_best_model = next(iter(cv_ranks), None)
    selected_cv = next((row for row in cv_summary if str(row.get("model")) == best_model), {})
    cv_mae_mean = selected_cv.get("cv_mae_mean")
    cv_mae_std = selected_cv.get("cv_mae_std")
    cv_rank = cv_ranks.get(best_model)
    stability_note = "cv_unavailable"
    if cv_rank is not None:
        cv_ratio = (
            float(cv_mae_std) / float(cv_mae_mean)
            if cv_mae_mean is not None and cv_mae_std is not None and float(cv_mae_mean) > 1e-9
            else None
        )
        if cv_rank == 1 and (cv_ratio is None or cv_ratio <= 0.2):
            stability_note = "holdout_and_cv_aligned"
        elif cv_rank == 1:
            stability_note = "cv_mean_best_but_variance_noticeable"
        elif cv_rank <= 2:
            stability_note = "holdout_best_cv_near_top"
        else:
            stability_note = "holdout_best_cv_not_stable"
    if stability_note in {"holdout_and_cv_aligned", "cv_mean_best_but_variance_noticeable"}:
        selected_reason = (
            f"{best_model} has the lowest holdout MAE and the best CV MAE mean; "
            "CV variance still controls how strongly this model should be trusted."
        )
    elif stability_note == "holdout_best_cv_near_top":
        selected_reason = (
            f"{best_model} has the lowest holdout MAE, while CV ranks it near the top; "
            "treat the selection as useful but not fully stable."
        )
    elif stability_note == "holdout_best_cv_not_stable":
        selected_reason = (
            f"{best_model} has the lowest holdout MAE, but CV does not confirm the same advantage; "
            "downgrade the conclusion and keep the model as a monitoring benchmark."
        )
    else:
        selected_reason = (
            f"{best_model} is selected by holdout MAE because TimeSeriesSplit CV is unavailable or insufficient."
        )
    return {
        "selected_model": best_model,
        "holdout_mae_rank": holdout_ranks.get(best_model),
        "cv_mae_rank": cv_rank,
        "cv_best_model": cv_best_model,
        "cv_stability_note": stability_note,
        "selected_reason": selected_reason,
    }


def _feature_importance_rows(model_name: str, estimator: object, feature_cols: list[str]) -> list[dict[str, object]]:
    if model_name == "Ridge Regression":
        ridge = getattr(estimator, "named_steps", {}).get("ridge")
        coefficients = getattr(ridge, "coef_", [])
        rows = [
            {"feature": feature, "importance": round(abs(float(coef)), 6), "importance_type": "absolute_coefficient"}
            for feature, coef in zip(feature_cols, coefficients)
        ]
    elif hasattr(estimator, "feature_importances_"):
        rows = [
            {"feature": feature, "importance": round(float(value), 6), "importance_type": "feature_importance"}
            for feature, value in zip(feature_cols, estimator.feature_importances_)
        ]
    else:
        rows = []
    return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)[:8]


def _worst_error_rows(backtest_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        backtest_rows,
        key=lambda row: float(row.get("absolute_error") or 0),
        reverse=True,
    )[:5]


def _regression_payload(
    daily: pd.DataFrame,
    sales_col: str,
    date_col: str,
    *,
    quality_metrics: dict[str, object],
    backtest_periods: int,
    best_baseline: str,
    baseline_metrics: list[dict[str, object]],
    naive_value: float,
    moving_average_value: float,
) -> tuple[dict[str, object], dict[str, list[dict[str, object]]], dict[str, object], list[str]]:
    warnings: list[str] = []
    observed_periods = int(quality_metrics.get("observed_period_count") or len(daily))
    if observed_periods < MIN_REGRESSION_PERIODS:
        return (
            {
                "regression_status": "skipped",
                "regression_skip_reason": "insufficient_regression_periods",
            },
            {},
            {},
            warnings,
        )

    granularity = str(quality_metrics.get("series_granularity") or "daily")
    lags, rolling_window = _regression_windows(granularity)
    feature_df = _build_regression_frame(daily, sales_col, granularity)
    if len(feature_df) < MIN_REGRESSION_EFFECTIVE_SAMPLES:
        return (
            {
                "regression_status": "skipped",
                "regression_skip_reason": "insufficient_lag_rolling_samples",
                "regression_effective_sample_count": int(len(feature_df)),
                "regression_feature_leakage_guard": "shifted_lag_rolling_only",
            },
            {},
            {},
            warnings,
        )

    regression_backtest_periods = min(
        backtest_periods,
        max(MIN_BACKTEST_PERIODS, int(round(len(feature_df) * 0.2))),
        len(feature_df) - MIN_REGRESSION_TRAIN_PERIODS,
    )
    if regression_backtest_periods < MIN_BACKTEST_PERIODS:
        return (
            {
                "regression_status": "skipped",
                "regression_skip_reason": "insufficient_regression_train_backtest_periods",
                "regression_effective_sample_count": int(len(feature_df)),
                "regression_feature_leakage_guard": "shifted_lag_rolling_only",
            },
            {},
            {},
            warnings,
        )

    train = feature_df.iloc[:-regression_backtest_periods].copy()
    test = feature_df.iloc[-regression_backtest_periods:].copy()
    actual = test["target"].astype(float)
    baseline_mae = float(min(float(row["mae"]) for row in baseline_metrics))
    naive_predicted = pd.Series(naive_value, index=actual.index)
    moving_average_predicted = pd.Series(moving_average_value, index=actual.index)
    comparison = [
        _regression_metric_row("naive_last_value", "baseline", actual, naive_predicted, baseline_mae=baseline_mae),
        _regression_metric_row("moving_average_7", "baseline", actual, moving_average_predicted, baseline_mae=baseline_mae),
    ]
    feature_cols = [
        column for column in feature_df.columns if column not in {"period", "target"}
    ]
    estimators = _regression_estimators()
    cv_fold_rows = _cv_fold_rows(feature_df, feature_cols, estimators)
    cv_summary_rows = _cv_summary(cv_fold_rows)
    for row in comparison:
        row.update(_cv_metrics_from_summary(str(row["model"]), cv_summary_rows))
    predictions: dict[str, pd.Series] = {
        "naive_last_value": naive_predicted,
        "moving_average_7": moving_average_predicted,
    }
    fitted_estimators: dict[str, object] = {}
    for model_name, estimator in estimators.items():
        fitted = clone(estimator)
        fitted.fit(train[feature_cols], train["target"])
        predicted = pd.Series(fitted.predict(test[feature_cols]), index=actual.index)
        row = _regression_metric_row(model_name, "regression", actual, predicted, baseline_mae=baseline_mae)
        row.update(_cv_metrics_from_summary(model_name, cv_summary_rows))
        comparison.append(row)
        predictions[model_name] = predicted
        fitted_estimators[model_name] = fitted

    best_row = min(comparison, key=lambda row: float(row["mae"]))
    best_model = str(best_row["model"])
    selection_basis = _regression_selection_basis(comparison, cv_summary_rows, best_model=best_model)
    best_predictions = predictions[best_model]
    best_regression_row = min(
        [row for row in comparison if row["model_type"] == "regression"],
        key=lambda row: float(row["mae"]),
    )
    feature_model = best_model if best_model in fitted_estimators else str(best_regression_row["model"])
    feature_importance = _feature_importance_rows(feature_model, fitted_estimators[feature_model], feature_cols)

    backtest_rows = []
    for index, row in test.iterrows():
        actual_value = float(row["target"])
        predicted_value = float(best_predictions.loc[index])
        absolute_error = abs(actual_value - predicted_value)
        percentage_error = (absolute_error / abs(actual_value) * 100) if abs(actual_value) > 1e-9 else None
        if predicted_value > actual_value:
            error_direction = "over"
        elif predicted_value < actual_value:
            error_direction = "under"
        else:
            error_direction = "exact"
        backtest_rows.append(
            {
                "period": pd.Timestamp(row["period"]).date().isoformat(),
                "actual": round(actual_value, 2),
                "predicted": round(predicted_value, 2),
                "model": best_model,
                "absolute_error": round(absolute_error, 2),
                "percentage_error": round(percentage_error, 2) if percentage_error is not None else None,
                "error_direction": error_direction,
            }
        )

    best_mape = best_row.get("mape")
    best_r2 = best_row.get("r2")
    improvement = float(best_row.get("improvement_vs_baseline") or 0)
    regression_status = "usable"
    weak_reasons: list[str] = []
    if best_model == best_baseline or best_row.get("model_type") == "baseline":
        weak_reasons.append("baseline_preferred")
    if best_mape is None or float(best_mape) > HIGH_REGRESSION_MAPE_THRESHOLD:
        weak_reasons.append("high_mape")
    if best_r2 is not None and float(best_r2) <= 0:
        weak_reasons.append("non_positive_r2")
    if improvement <= 0:
        weak_reasons.append("no_improvement_vs_baseline")
    if weak_reasons:
        regression_status = "weak"
    over_errors = [float(row["absolute_error"]) for row in backtest_rows if row.get("error_direction") == "over"]
    under_errors = [float(row["absolute_error"]) for row in backtest_rows if row.get("error_direction") == "under"]

    summary = {
        "forecast_status": "regression_evaluated",
        "regression_status": regression_status,
        "regression_weak_reasons": weak_reasons,
        "regression_time_split": "chronological",
        "regression_cv_strategy": "TimeSeriesSplit_3" if len(feature_df) >= 80 else "chronological_holdout",
        "regression_feature_leakage_guard": "shifted_lag_rolling_only",
        "regression_effective_sample_count": int(len(feature_df)),
        "regression_train_period_count": int(len(train)),
        "regression_backtest_period_count": int(len(test)),
        "regression_date_column": date_col,
        "regression_sales_amount_column": sales_col,
        "regression_lag_periods": lags,
        "regression_rolling_window": rolling_window,
        "regression_feature_columns": feature_cols,
        "regression_random_forest_params": {
            "n_estimators": 60,
            "max_depth": 8,
            "min_samples_leaf": 3,
            "random_state": 42,
        },
        "best_model": best_model,
        "best_model_mae": best_row["mae"],
        "best_model_rmse": best_row["rmse"],
        "best_model_mape": best_row["mape"],
        "best_model_r2": best_row["r2"],
        "baseline_model": best_baseline,
        "baseline_mae": round(baseline_mae, 2),
        "improvement_vs_baseline": best_row["improvement_vs_baseline"],
        "top_regression_feature_source": feature_model,
        "over_prediction_count": len(over_errors),
        "under_prediction_count": len(under_errors),
        "max_over_prediction": round(max(over_errors), 2) if over_errors else 0.0,
        "max_under_prediction": round(max(under_errors), 2) if under_errors else 0.0,
        "selection_basis": selection_basis,
    }
    summary.update(_cv_metrics_from_summary(best_model, cv_summary_rows))
    tables = {
        "regression_cv_folds": cv_fold_rows,
        "regression_cv_summary": cv_summary_rows,
        "model_comparison": comparison,
        "regression_backtest": backtest_rows,
        "feature_importance": feature_importance,
        "worst_error_periods": _worst_error_rows(backtest_rows),
    }
    chart_payload = {
        "regression_backtest": backtest_rows,
        "feature_importance": feature_importance,
    }
    return summary, tables, chart_payload, warnings


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    date_col = canonical_columns["order_datetime"]
    sales_col = canonical_columns["sales_amount"]

    working = frame[[date_col, sales_col]].copy()
    working["period"] = _parse_datetime(working[date_col]).dt.normalize()
    working[sales_col] = pd.to_numeric(working[sales_col], errors="coerce")
    working = working.dropna(subset=["period", sales_col])
    if working.empty:
        return _skipped_result("missing_forecast_target_values")

    daily = (
        working.groupby("period", as_index=False)
        .agg({sales_col: "sum"})
        .sort_values("period")
    )
    if len(daily) < MIN_FORECAST_PERIODS:
        return _skipped_result("insufficient_forecast_periods")

    quality_metrics = _series_quality_metrics(daily, sales_col)
    if int(quality_metrics["day_span"]) < MIN_FORECAST_DAY_SPAN:
        return _skipped_result("insufficient_time_span", quality_metrics)
    if float(quality_metrics["date_gap_ratio"]) > MAX_DATE_GAP_RATIO:
        return _skipped_result("sparse_time_series", quality_metrics)
    if float(quality_metrics["zero_or_missing_period_ratio"]) > MAX_ZERO_OR_MISSING_PERIOD_RATIO:
        return _skipped_result("too_many_zero_or_missing_periods", quality_metrics)

    backtest_periods = max(MIN_BACKTEST_PERIODS, int(round(len(daily) * 0.2)))
    backtest_periods = min(backtest_periods, FORECAST_HORIZON_DAYS, len(daily) - MIN_TRAIN_PERIODS)
    if backtest_periods < MIN_BACKTEST_PERIODS:
        return _skipped_result("insufficient_train_backtest_periods")

    train = daily.iloc[:-backtest_periods].copy()
    backtest = daily.iloc[-backtest_periods:].copy()
    actual = backtest[sales_col].astype(float)
    naive_value = float(train[sales_col].iloc[-1])
    moving_average_value = float(train[sales_col].tail(MOVING_AVERAGE_WINDOW).mean())
    naive_predicted = pd.Series(naive_value, index=actual.index)
    moving_average_predicted = pd.Series(moving_average_value, index=actual.index)
    baseline_metrics = [
        _metric_row("naive_last_value", actual, naive_predicted),
        _metric_row("moving_average_7", actual, moving_average_predicted),
    ]
    best_metric = min(baseline_metrics, key=lambda row: float(row["mae"]))
    best_baseline = str(best_metric["baseline"])
    baseline = naive_value if best_baseline == "naive_last_value" else moving_average_value
    last_date = pd.Timestamp(daily["period"].max())
    forecast_rows = []
    for offset in range(1, FORECAST_HORIZON_DAYS + 1):
        forecast_rows.append(
            {
                "period": (last_date + timedelta(days=offset)).date().isoformat(),
                "forecast_sales_amount": round(baseline, 2),
                "baseline": best_baseline,
            }
        )

    backtest_rows = []
    for index, row in backtest.iterrows():
        naive_forecast = round(naive_value, 2)
        moving_average_forecast = round(moving_average_value, 2)
        backtest_rows.append(
            {
                "period": pd.Timestamp(row["period"]).date().isoformat(),
                "actual_sales_amount": round(float(row[sales_col]), 2),
                "naive_last_value": naive_forecast,
                "moving_average_7": moving_average_forecast,
                "best_baseline_forecast": naive_forecast
                if best_baseline == "naive_last_value"
                else moving_average_forecast,
            }
        )

    warnings = []
    if any(row["mape"] is None for row in baseline_metrics):
        warnings.append("mape_unavailable_for_zero_actuals")
    regression_summary, regression_tables, regression_chart_payload, regression_warnings = _regression_payload(
        daily,
        sales_col,
        date_col,
        quality_metrics=quality_metrics,
        backtest_periods=backtest_periods,
        best_baseline=best_baseline,
        baseline_metrics=baseline_metrics,
        naive_value=naive_value,
        moving_average_value=moving_average_value,
    )
    warnings.extend(regression_warnings)
    forecast_status = str(regression_summary.get("forecast_status") or "baseline_evaluated")

    return ModuleResult(
        module_id="forecast_analysis",
        title="基础预测",
        chart_type="line",
        summary_metrics={
            "forecast_status": forecast_status,
            "time_split": "chronological",
            **quality_metrics,
            "train_period_count": int(len(train)),
            "backtest_period_count": int(len(backtest)),
            "horizon_days": FORECAST_HORIZON_DAYS,
            "baseline_sales_amount": round(baseline, 2),
            "best_baseline": best_baseline,
            "best_baseline_mae": best_metric["mae"],
            "best_baseline_rmse": best_metric["rmse"],
            "best_baseline_mape": best_metric["mape"],
            **regression_summary,
        },
        tables={
            "forecast": forecast_rows,
            "baseline_metrics": baseline_metrics,
            "baseline_backtest": backtest_rows,
            **regression_tables,
        },
        chart_payload={
            "x": [row["period"] for row in forecast_rows],
            "y": [row["forecast_sales_amount"] for row in forecast_rows],
            "series_name": "forecast_sales_amount",
            **regression_chart_payload,
        },
        findings=[
            "已使用时间顺序切分回测销售额预测 baseline。",
            f"当前最优 baseline 为 {best_baseline}，回测 MAE 为 {best_metric['mae']}。",
            (
                f"轻量销售额回归已完成，最佳模型为 {regression_summary.get('best_model')}。"
                if forecast_status == "regression_evaluated"
                else "轻量销售额回归未启用，保留 baseline 作为对照。"
            ),
        ],
        warnings=warnings,
    )
