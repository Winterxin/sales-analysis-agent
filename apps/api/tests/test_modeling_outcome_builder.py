from __future__ import annotations

from typing import Any

from app.services.modeling_outcome_builder import build_modeling_outcome


def _report(*, loss_metrics: dict[str, Any] | None = None, forecast_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    if loss_metrics is not None:
        modules.append(
            {
                "module_id": "loss_risk_modeling",
                "summary_metrics": loss_metrics,
                "tables": {
                    "model_comparison": [{"model": loss_metrics.get("best_model", "LogisticRegression")}],
                    "confusion_matrix": [{"actual": "loss", "predicted": "loss", "count": 12}],
                    "feature_importance_grouped": [{"feature_group": "discount", "total_importance": 4.2}],
                },
                "warnings": [],
            }
        )
    if forecast_metrics is not None:
        modules.append(
            {
                "module_id": "forecast_analysis",
                "summary_metrics": forecast_metrics,
                "tables": {
                    "baseline_metrics": [
                        {"baseline": "naive_last_value", "mae": forecast_metrics.get("best_baseline_mae")},
                    ],
                    "model_comparison": [
                        {"model": "naive_last_value", "mae": forecast_metrics.get("best_baseline_mae")},
                        {"model": "Ridge Regression", "mae": forecast_metrics.get("best_model_mae")},
                    ],
                    "regression_cv_summary": [
                        {
                            "model": "Ridge Regression",
                            "cv_mae_mean": forecast_metrics.get("cv_mae_mean"),
                            "cv_mae_std": forecast_metrics.get("cv_mae_std"),
                            "cv_mape_mean": forecast_metrics.get("cv_mape_mean"),
                            "cv_mape_std": forecast_metrics.get("cv_mape_std"),
                            "cv_r2_mean": forecast_metrics.get("cv_r2_mean"),
                            "cv_r2_std": forecast_metrics.get("cv_r2_std"),
                        }
                    ],
                    "feature_importance": [
                        {"feature": "lag_1", "importance": 0.52},
                        {"feature": "rolling_mean_4", "importance": 0.31},
                    ],
                    "worst_error_periods": [
                        {
                            "period": "2024-01-01",
                            "actual": 120,
                            "predicted": 118,
                            "absolute_error": 2,
                            "error_direction": "under",
                        },
                        {
                            "period": "2024-01-08",
                            "actual": 110,
                            "predicted": 130,
                            "absolute_error": 20,
                            "error_direction": "over",
                        },
                    ],
                    "regression_backtest": [
                        {"period": "2024-01-01", "actual": 120, "predicted": 118, "error_direction": "under"},
                        {"period": "2024-01-08", "actual": 110, "predicted": 130, "error_direction": "over"},
                        {"period": "2024-01-15", "actual": 100, "predicted": 115, "error_direction": "over"},
                    ],
                    "forecast": [{"period": "2024-01-01", "actual": 120, "forecast": 118}],
                },
                "warnings": [],
            }
        )
    return {"modules": modules}


def _opportunity_decision() -> dict[str, str]:
    return {
        "decision_summary": "当前推荐销售额预测，但本轮仅记录机会。",
        "notebook_message": "后续可评估销售额预测或回归，但本轮仅记录机会，不训练模型。",
        "risk_warning": "G6-2 只记录建模机会，不训练新模型。",
    }


def test_strong_loss_risk_is_primary_even_when_forecast_exists() -> None:
    outcome = build_modeling_outcome(
        _report(
            loss_metrics={
                "model_quality_status": "strong",
                "best_model": "LogisticRegression",
                "best_f1": 0.82,
                "best_recall": 0.96,
                "best_precision": 0.71,
                "false_positive_count": 183,
                "false_negative_count": 19,
            },
            forecast_metrics={
                "forecast_status": "baseline_evaluated",
                "best_baseline_mape": 1690.51,
                "best_baseline": "moving_average_7",
            },
        ),
        dataset="sample_superstore",
    )

    assert outcome["primary_modeling_task"] == "loss_risk_classification"
    assert outcome["modeling_status"] == "strong_model"
    assert outcome["modeling_value_level"] == "high"
    assert outcome["metrics_summary"]["best_f1"] == 0.82
    assert "loss_risk_modeling" in outcome["source_modules"]
    assert outcome["should_show_in_notebook"] is True


def test_weak_loss_risk_keeps_downgraded_review_language() -> None:
    outcome = build_modeling_outcome(
        _report(
            loss_metrics={
                "model_quality_status": "weak",
                "best_model": "LogisticRegression",
                "best_f1": 0.0976,
                "best_precision": 0.0526,
                "best_recall": 0.6667,
                "target_positive_count": 11,
                "test_positive_count": 3,
                "weak_reasons": ["too_few_test_positives", "low_precision"],
            }
        ),
        dataset="sales_sample_20000",
    )

    assert outcome["primary_modeling_task"] == "loss_risk_classification"
    assert outcome["modeling_status"] == "weak_model"
    assert outcome["modeling_value_level"] == "low"
    assert any("不建议直接用于复核排序" in item for item in outcome["limitations"])
    assert any("探索性" in item for item in outcome["main_findings"])


def test_forecast_baseline_becomes_primary_when_no_loss_model_exists() -> None:
    outcome = build_modeling_outcome(
        _report(
            forecast_metrics={
                "forecast_status": "baseline_evaluated",
                "best_baseline": "naive_last_value",
                "best_baseline_mae": 253982.68,
                "best_baseline_rmse": 301514.77,
                "best_baseline_mape": 11.66,
                "time_split": "chronological",
                "series_granularity": "weekly",
            }
        ),
        modeling_opportunity_decision=_opportunity_decision(),
        dataset="train_sample_20000",
    )

    assert outcome["primary_modeling_task"] == "sales_amount_forecast_baseline"
    assert outcome["modeling_status"] == "baseline_evaluated"
    assert outcome["modeling_value_level"] == "medium"
    assert outcome["metrics_summary"]["best_baseline_mape"] == 11.66
    assert outcome["conflict_resolution"]["forecast_baseline_supersedes_opportunity_only"] is True
    assert any("不是生产级预测" in item for item in outcome["limitations"])


def test_sales_regression_becomes_primary_when_no_loss_model_exists() -> None:
    outcome = build_modeling_outcome(
        _report(
            forecast_metrics={
                "forecast_status": "regression_evaluated",
                "regression_status": "usable",
                "best_model": "Ridge Regression",
                "best_model_mae": 900.0,
                "best_model_rmse": 1100.0,
                "best_model_mape": 18.0,
                "best_model_r2": 0.42,
                "baseline_model": "naive_last_value",
                "baseline_mae": 1200.0,
                "improvement_vs_baseline": 25.0,
                "time_split": "chronological",
                "series_granularity": "weekly",
                "cv_mae_mean": 185000.0,
                "cv_mae_std": 12000.0,
                "cv_mape_mean": 9.7,
                "cv_mape_std": 1.2,
                "cv_r2_mean": 0.31,
                "cv_r2_std": 0.05,
                "selection_basis": {
                    "selected_model": "Ridge Regression",
                    "holdout_mae_rank": 1,
                    "cv_mae_rank": 1,
                    "cv_stability_note": "holdout_and_cv_aligned",
                    "selected_reason": "Ridge Regression has the lowest holdout MAE and the best CV MAE mean.",
                },
            }
        ),
        dataset="train_sample_20000",
    )

    assert outcome["primary_modeling_task"] == "sales_amount_regression"
    assert outcome["modeling_status"] == "regression_usable"
    assert outcome["modeling_value_level"] == "medium"
    assert outcome["metrics_summary"]["best_model"] == "Ridge Regression"
    assert outcome["metrics_summary"]["improvement_vs_baseline"] == 25.0
    assert outcome["metrics_summary"]["cv_mae_mean"] == 185000.0
    assert outcome["metrics_summary"]["cv_mape_std"] == 1.2
    assert outcome["metrics_summary"]["selection_basis"]["holdout_mae_rank"] == 1
    assert outcome["metrics_summary"]["selection_basis"]["cv_mae_rank"] == 1
    assert outcome["metrics_summary"]["top_features"] == ["lag_1", "rolling_mean_4"]
    assert outcome["metrics_summary"]["worst_error_periods"][0]["period"] == "2024-01-01"
    assert outcome["metrics_summary"]["worst_error_periods"][0]["error_direction"] == "under"
    assert outcome["metrics_summary"]["over_prediction_count"] == 2
    assert outcome["metrics_summary"]["under_prediction_count"] == 1
    assert "forecast_analysis" in outcome["source_modules"]


def test_weak_loss_risk_still_supersedes_sales_regression() -> None:
    outcome = build_modeling_outcome(
        _report(
            loss_metrics={
                "model_quality_status": "weak",
                "best_model": "LogisticRegression",
                "best_f1": 0.09,
                "best_precision": 0.05,
                "best_recall": 0.66,
                "target_positive_count": 11,
                "test_positive_count": 3,
            },
            forecast_metrics={
                "forecast_status": "regression_evaluated",
                "regression_status": "usable",
                "best_model": "Ridge Regression",
                "best_model_mape": 18.0,
                "improvement_vs_baseline": 25.0,
            },
        ),
        dataset="sales_sample_20000",
    )

    assert outcome["primary_modeling_task"] == "loss_risk_classification"
    assert outcome["modeling_status"] == "weak_model"
    assert outcome["source_modules"] == ["loss_risk_modeling"]


def test_weak_sales_regression_is_downgraded_to_monitoring_reference() -> None:
    outcome = build_modeling_outcome(
        _report(
            forecast_metrics={
                "forecast_status": "regression_evaluated",
                "regression_status": "weak",
                "best_model": "naive_last_value",
                "best_model_mae": 1200.0,
                "best_model_rmse": 1600.0,
                "best_model_mape": 180.0,
                "best_model_r2": -0.2,
                "baseline_model": "naive_last_value",
                "baseline_mae": 1200.0,
                "improvement_vs_baseline": 0.0,
            }
        ),
        dataset="pakistan_largest_ecommerce_dataset_sample_20000",
    )

    assert outcome["primary_modeling_task"] == "sales_amount_regression"
    assert outcome["modeling_status"] == "regression_weak"
    assert outcome["modeling_value_level"] == "low"
    assert "监控参照" in outcome["recommended_action"]
    assert any("不建议直接用于销售计划" in item for item in outcome["limitations"])


def test_high_mape_forecast_is_downgraded_to_monitoring_reference() -> None:
    outcome = build_modeling_outcome(
        _report(
            forecast_metrics={
                "forecast_status": "baseline_evaluated",
                "best_baseline": "moving_average_7",
                "best_baseline_mae": 113235.52,
                "best_baseline_rmse": 192336.53,
                "best_baseline_mape": 1053.75,
                "time_split": "chronological",
                "series_granularity": "daily",
            }
        ),
        modeling_opportunity_decision=_opportunity_decision(),
        dataset="pakistan_largest_ecommerce_dataset_sample_20000",
    )

    assert outcome["primary_modeling_task"] == "sales_amount_forecast_baseline"
    assert outcome["modeling_status"] == "weak_baseline"
    assert outcome["modeling_value_level"] == "low"
    assert any("MAPE 偏高" in item for item in outcome["limitations"])
    assert "监控参照" in outcome["recommended_action"]


def test_no_modeling_evidence_keeps_opportunity_only_outcome() -> None:
    outcome = build_modeling_outcome(
        _report(),
        modeling_opportunity_plan={"recommended_modeling_task": "sales_amount_forecast_or_regression"},
        modeling_opportunity_decision=_opportunity_decision(),
        dataset="online_retail",
    )

    assert outcome["primary_modeling_task"] == "modeling_opportunity_only"
    assert outcome["modeling_status"] == "opportunity_only"
    assert outcome["modeling_value_level"] == "none"
    assert outcome["should_show_in_notebook"] is True
