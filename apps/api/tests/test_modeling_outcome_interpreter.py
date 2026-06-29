from __future__ import annotations

from typing import Any

from app.services.modeling_outcome_interpreter import (
    OUTCOME_INTERPRETATION_KEYS,
    build_modeling_outcome_interpretation_with_trace,
)


class DisabledLLMClient:
    enabled = False
    source = "disabled"
    configured_model = None
    last_completion_metrics: dict[str, Any] = {}


class SuccessfulLLMClient:
    enabled = True
    source = "test"
    configured_model = "test-model"
    last_completion_metrics = {"elapsed_ms": 1, "prompt_chars": 10, "response_chars": 20}

    def suggest_modeling_outcome_interpretation(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, str]:
        self.payload = payload
        self.language_instruction = language_instruction
        return {
            "outcome_summary": "LLM判断：当前已完成销售额预测 baseline 回测，MAPE 为 11.66%，可作为探索性监控基线。",
            "business_interpretation": "短期周销售序列的 naive baseline 有一定参照价值，但不是生产级预测。",
            "recommended_action": "建议把该结果作为后续预测模型的对照基线，并继续监控偏离。",
            "risk_warning": "这是 baseline，不是生产级预测，不用于自动决策。",
        }


class UnsafeLLMClient(SuccessfulLLMClient):
    def suggest_modeling_outcome_interpretation(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, str]:
        return {
            "outcome_summary": "模型证明折扣导致亏损，可自动拒单。",
            "business_interpretation": "预测能力强，可以生产级预测。",
            "recommended_action": "直接自动审批。",
            "risk_warning": "自动决策。",
        }


class WeakRegressionLLMClient(SuccessfulLLMClient):
    def suggest_modeling_outcome_interpretation(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, str]:
        return {
            "outcome_summary": "当前销售额回归没有形成稳定预测能力，仅适合作为监控参照。",
            "business_interpretation": "误差偏高说明预测难度高，业务上应把它视为波动提示。",
            "recommended_action": "建议补充促销、节假日等解释变量后再评估。",
            "risk_warning": "不是生产级预测，不用于自动决策。",
        }


class SalesRegressionInternalTokenLLMClient(SuccessfulLLMClient):
    def suggest_modeling_outcome_interpretation(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, str]:
        self.payload = payload
        self.language_instruction = language_instruction
        return {
            "outcome_summary": "当前销售额回归可作为轻量监控参照，但不是生产级预测。",
            "business_interpretation": "模型结果说明历史销售信号有一定参考价值，但仍需要结合业务事件解读。",
            "recommended_action": "建议把该模型作为后续模型增强的对照，并补充促销和节假日变量。",
            "risk_warning": "不是生产级预测，不用于自动决策。",
            "model_selection_takeaway": (
                "选择 Ridge Regression 是因为 holdout_and_cv_aligned，"
                "说明最后回测窗口和交叉验证方向一致。该结果仍需要结合 R2 和业务场景判断。"
            ),
            "cv_stability_takeaway": (
                "CV 结果显示 holdout_best_cv_not_stable，需要谨慎看待。"
                "这更适合作为监控参照，而不是经营计划依据。"
            ),
            "prediction_fit_takeaway": (
                "Actual vs Best Model 显示模型能跟随部分趋势。"
                "但峰值捕捉仍有限，需要结合业务事件解释。"
            ),
            "feature_importance_takeaway": (
                "Top features 主要来自 lag 和 rolling。"
                "这说明模型依赖历史惯性，而不是促销或门店原因。"
            ),
            "error_analysis_takeaway": (
                "误差 Top 5 主要用于定位异常周期。"
                "若这些周期是高估或低估集中出现，应优先补充促销和节假日变量。"
            ),
            "final_regression_synthesis": (
                "综合来看，Ridge Regression 有一定监控参考价值。"
                "但 CV 未充分确认 holdout 优势，不能作为生产级预测。"
                "后续应补充促销、节假日和门店变量后再判断是否能改善。"
            ),
        }


def _forecast_outcome() -> dict[str, Any]:
    return {
        "dataset": "train_sample_20000",
        "primary_modeling_task": "sales_amount_forecast_baseline",
        "modeling_status": "baseline_evaluated",
        "modeling_value_level": "medium",
        "metrics_summary": {
            "best_baseline": "naive_last_value",
            "best_baseline_mape": 11.66,
        },
        "main_findings": ["销售额预测 baseline 已完成时间顺序回测。"],
        "limitations": ["这是 baseline，不是生产级预测，不用于自动决策。"],
        "recommended_action": "作为监控参照和后续模型对照基线。",
    }


def _sales_regression_outcome() -> dict[str, Any]:
    return {
        "dataset": "train_sample_20000",
        "primary_modeling_task": "sales_amount_regression",
        "modeling_status": "regression_usable",
        "modeling_value_level": "medium",
        "metrics_summary": {
            "best_model": "Ridge Regression",
            "best_model_mape": 8.9,
            "best_model_r2": 0.42,
            "improvement_vs_baseline": 29.13,
            "cv_mae_mean": 185000.0,
            "cv_mae_std": 12000.0,
            "cv_mape_mean": 9.7,
            "cv_mape_std": 1.2,
            "cv_r2_mean": 0.31,
            "cv_r2_std": 0.05,
            "top_features": ["lag_1", "rolling_mean_4"],
            "worst_error_periods": [
                {
                    "period": "2012-01-13",
                    "actual": 1500000.0,
                    "predicted": 1520000.0,
                    "absolute_error": 20000.0,
                    "percentage_error": 1.33,
                    "error_direction": "over",
                }
            ],
            "over_prediction_count": 1,
            "under_prediction_count": 2,
            "selection_basis": {
                "holdout_mae_rank": 1,
                "cv_mae_rank": 1,
                "cv_stability_note": "holdout_and_cv_aligned",
                "selected_reason": "holdout_and_cv_aligned",
            },
        },
        "main_findings": ["销售额回归已完成时间顺序回测，并与简单 baseline 做了对比。"],
        "limitations": ["这是轻量回归模型，不是生产级预测，不用于自动决策。"],
        "recommended_action": "作为销售监控参照和后续模型对照基线。",
    }


def test_disabled_llm_returns_deterministic_outcome_interpretation() -> None:
    interpretation, trace = build_modeling_outcome_interpretation_with_trace(
        _forecast_outcome(),
        llm_client=DisabledLLMClient(),
    )

    assert set(interpretation) == OUTCOME_INTERPRETATION_KEYS
    assert trace.status == "disabled"
    assert trace.applied is False
    assert "baseline" in interpretation["risk_warning"].lower()
    assert "不是生产级预测" in interpretation["risk_warning"]


def test_sales_regression_deterministic_interpretation_uses_forecast_safety_warning() -> None:
    interpretation, trace = build_modeling_outcome_interpretation_with_trace(
        _sales_regression_outcome(),
        llm_client=DisabledLLMClient(),
    )

    assert trace.status == "disabled"
    assert "Ridge Regression" in interpretation["outcome_summary"]
    assert "模型选择解释：" not in interpretation["model_selection_takeaway"]
    assert "Ridge Regression" in interpretation["model_selection_takeaway"]
    assert "CV" in interpretation["cv_stability_takeaway"] or "TimeSeriesSplit" in interpretation["cv_stability_takeaway"]
    assert "185000" in interpretation["cv_stability_takeaway"] or "12000" in interpretation["cv_stability_takeaway"]
    assert "Actual vs Best Model" in interpretation["prediction_fit_takeaway"]
    assert "lag / rolling" in interpretation["feature_importance_takeaway"]
    assert "误差 Top 5" in interpretation["error_analysis_takeaway"]
    assert "应解释" not in interpretation["error_analysis_takeaway"]
    assert "综合来看" in interpretation["final_regression_synthesis"]
    assert interpretation["final_regression_synthesis"].count("。") >= 3
    assert "不是生产级预测" in interpretation["risk_warning"]
    assert "不用于自动决策" in interpretation["risk_warning"]


def test_successful_llm_interpretation_is_applied_and_sanitized() -> None:
    llm_client = SuccessfulLLMClient()

    interpretation, trace = build_modeling_outcome_interpretation_with_trace(
        _forecast_outcome(),
        llm_client=llm_client,
    )

    assert trace.status == "llm_applied"
    assert trace.applied is True
    assert interpretation["outcome_summary"].startswith("LLM判断")
    assert llm_client.payload["primary_modeling_task"] == "sales_amount_forecast_baseline"


def test_sales_regression_prompt_requests_deeper_takeaways() -> None:
    llm_client = SalesRegressionInternalTokenLLMClient()

    interpretation, trace = build_modeling_outcome_interpretation_with_trace(
        _sales_regression_outcome(),
        llm_client=llm_client,
    )

    assert trace.status == "llm_applied"
    constraints = " ".join(llm_client.payload["constraints"])
    assert "at least two Chinese sentences" in constraints
    assert "metric judgment plus business meaning" in constraints
    assert "Do not expose internal enum values" in constraints
    joined = " ".join(interpretation.values())
    assert "holdout_and_cv_aligned" not in joined
    assert "holdout_best_cv_not_stable" not in joined
    assert "holdout 与 CV 方向一致" in joined
    assert "CV 未充分确认 holdout 优势" in joined


def test_unsafe_llm_interpretation_falls_back() -> None:
    interpretation, trace = build_modeling_outcome_interpretation_with_trace(
        _forecast_outcome(),
        llm_client=UnsafeLLMClient(),
    )

    assert trace.status == "fallback_invalid_payload"
    joined = " ".join(interpretation.values())
    assert "自动拒单" not in joined
    assert "自动审批" not in joined
    assert "导致" not in joined
    assert "不是生产级预测" in joined


def test_negated_stable_forecast_ability_is_allowed_for_weak_regression() -> None:
    interpretation, trace = build_modeling_outcome_interpretation_with_trace(
        {
            **_sales_regression_outcome(),
            "modeling_status": "regression_weak",
            "modeling_value_level": "low",
        },
        llm_client=WeakRegressionLLMClient(),
    )

    assert trace.status == "llm_applied"
    assert trace.applied is True
    assert "没有形成稳定预测能力" in interpretation["outcome_summary"]
    assert "没有形成稳定预测能力" in interpretation["final_regression_synthesis"]
    assert "不是生产级预测" in interpretation["risk_warning"]
