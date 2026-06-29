from __future__ import annotations

from typing import Any

from app.schemas.llm_trace import LLMStageTrace
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.output_language import user_facing_language_instruction


STAGE = "modeling_outcome_interpretation"
OUTCOME_INTERPRETATION_KEYS = {
    "outcome_summary",
    "business_interpretation",
    "recommended_action",
    "risk_warning",
    "model_selection_takeaway",
    "cv_stability_takeaway",
    "prediction_fit_takeaway",
    "feature_importance_takeaway",
    "error_analysis_takeaway",
    "final_regression_synthesis",
}
BANNED_PHRASES = (
    "导致",
    "证明",
    "因果关系成立",
    "因果成立",
    "自动拦截",
    "自动审批",
    "自动拒单",
    "自动处置",
    "自动处理",
    "稳定预测能力",
    "生产级预测能力",
    "预测能力强",
)
NEGATED_STABLE_FORECAST_PHRASES = (
    "没有形成稳定预测能力",
    "未形成稳定预测能力",
    "尚未形成稳定预测能力",
    "没有稳定预测能力",
    "不具备稳定预测能力",
)
SAFE_AUTO_DECISION_MARKERS = (
    "不用于自动决策",
    "不能用于自动决策",
    "不可用于自动决策",
    "不是自动决策",
)


def _trace(
    *,
    llm_client,
    status: str,
    reason: str,
    attempted: bool = False,
    applied: bool = False,
) -> LLMStageTrace:
    return build_llm_stage_trace(
        stage=STAGE,
        llm_client=llm_client,
        status=status,
        reason=reason,
        attempted=attempted,
        applied=applied,
    )


def _clip_text(value: Any, fallback: str, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip() or fallback
    return text[:limit]


def _humanize_internal_tokens(text: str) -> str:
    replacements = {
        "holdout_and_cv_aligned": "holdout 与 CV 方向一致",
        "cv_mean_best_but_variance_noticeable": "CV 均值最优但波动仍需关注",
        "holdout_best_cv_near_top": "holdout 最优且 CV 接近前列",
        "holdout_best_cv_not_stable": "CV 未充分确认 holdout 优势",
        "cv_unavailable": "CV 证据不足",
    }
    result = text
    for raw, humanized in replacements.items():
        result = result.replace(raw, humanized)
    return result


def _joined(outcome: dict[str, Any], key: str) -> str:
    value = outcome.get(key)
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value or "")


def _deterministic_interpretation(outcome: dict[str, Any]) -> dict[str, str]:
    task = str(outcome.get("primary_modeling_task") or "modeling_opportunity_only")
    status = str(outcome.get("modeling_status") or "unknown")
    value_level = str(outcome.get("modeling_value_level") or "none")
    findings = _joined(outcome, "main_findings") or "当前没有足够建模证据。"
    action = str(outcome.get("recommended_action") or "保留描述性分析，并等待更稳定的建模条件。")
    limits = _joined(outcome, "limitations")
    if task == "loss_risk_classification":
        risk_warning = "不代表因果关系，不用于自动决策。"
        if status == "weak_model":
            risk_warning = "当前只适合作为探索性参考，不建议直接用于复核排序，不代表因果关系，不用于自动决策。"
    elif task in {"sales_amount_forecast_baseline", "sales_amount_regression"}:
        risk_warning = "这是 baseline，不是生产级预测，不用于自动决策。"
    else:
        risk_warning = "当前只记录建模机会，不训练新模型，不用于自动决策。"
    if limits and all(marker not in limits for marker in ("不代表因果关系", "不是生产级预测", "不训练新模型")):
        limits = f"{limits} {risk_warning}"
    model_prefix = ""
    model_selection_takeaway = ""
    cv_stability_takeaway = ""
    prediction_fit_takeaway = ""
    feature_importance_takeaway = ""
    error_analysis_takeaway = ""
    final_regression_synthesis = ""
    if task == "sales_amount_regression":
        metrics = outcome.get("metrics_summary") if isinstance(outcome.get("metrics_summary"), dict) else {}
        best_model = metrics.get("best_model") if isinstance(metrics, dict) else None
        improvement = metrics.get("improvement_vs_baseline") if isinstance(metrics, dict) else None
        best_mape = metrics.get("best_model_mape") if isinstance(metrics, dict) else None
        best_r2 = metrics.get("best_model_r2") if isinstance(metrics, dict) else None
        cv_mae_mean = metrics.get("cv_mae_mean") if isinstance(metrics, dict) else None
        cv_mae_std = metrics.get("cv_mae_std") if isinstance(metrics, dict) else None
        cv_mape_mean = metrics.get("cv_mape_mean") if isinstance(metrics, dict) else None
        cv_r2_mean = metrics.get("cv_r2_mean") if isinstance(metrics, dict) else None
        if best_model:
            model_prefix = f"最佳模型为 {best_model}"
            if improvement is not None:
                model_prefix += f"，相对 baseline 改善 {improvement}%"
            model_prefix += "。"
        risk_warning = "这是轻量回归模型，不是生产级预测，不用于自动决策。"
        weak = status == "regression_weak"
        if weak:
            model_selection_takeaway = (
                f"当前 best model 为 {best_model or 'baseline'}，MAPE={best_mape}，"
                "复杂模型没有形成稳定增益。"
            )
            final_regression_synthesis = (
                "综合来看，当前销售额回归没有形成稳定预测能力，更适合作为预测难度证据和监控参照。"
                "复杂模型没有稳定超过 baseline，说明仅靠历史销售惯性和周期特征不足以解释当前波动。"
                "该结果不建议用于销售计划、库存、补货或经营目标制定。"
                "下一步应补充促销、节假日、门店、类目、库存或异常订单等业务变量后再评估。"
            )
            cv_stability_takeaway = (
                f"CV 稳定性显示 cv_mae_mean={cv_mae_mean}、cv_mae_std={cv_mae_std}、"
                f"cv_mape_mean={cv_mape_mean}、cv_r2_mean={cv_r2_mean}；若各折波动较大，"
                "应把结果视为预测难度证据，而不是稳定可用模型。"
            )
        else:
            model_selection_takeaway = (
                f"当前按回测 MAE 选择 {best_model or 'best model'}；"
                f"相对 baseline 改善 {improvement}%，R2={best_r2}，提升可作为监控参考但不宜过度包装。"
            )
            final_regression_synthesis = (
                f"综合来看，{best_model or '当前模型'} 相对 baseline 有一定改善，可作为销售监控和后续模型增强的对照。"
                f"不过当前 R2={best_r2}，说明模型只能解释有限的销售波动。"
                "它适合用于观察历史走势是否偏离，不适合直接用于销售计划、库存、补货或经营目标制定。"
                "下一步应补充促销、节假日、商品结构、库存和渠道活动等业务变量，再判断预测能力是否真正提升。"
            )
            cv_stability_takeaway = (
                f"CV 稳定性显示 cv_mae_mean={cv_mae_mean}、cv_mae_std={cv_mae_std}、"
                f"cv_mape_mean={cv_mape_mean}、cv_r2_mean={cv_r2_mean}；"
                "如果 mean 有改善但 std 仍不低，说明模型有参考价值但跨窗口稳定性仍需谨慎看待。"
            )
        prediction_fit_takeaway = (
            "Actual vs Best Model 用于观察回测窗口内趋势、峰值和低谷是否被跟上。"
            "如果预测线过于平滑或漏掉峰谷，说明仅靠历史特征还不足以支撑经营计划。"
        )
        feature_importance_takeaway = (
            "当前特征信号主要来自 calendar、lag / rolling 等历史销售惯性和周期信息，"
            "说明模型更多在捕捉走势延续，而不是促销或门店等业务原因。"
        )
        error_analysis_takeaway = (
            "误差 Top 5 用于定位模型在哪些周期错得最明显，而不是再次评价整体指标。"
            "若误差集中在尖峰或低谷，通常提示缺少促销、节假日、门店、类目或异常订单变量。"
        )
    risk_warning_text = risk_warning
    if limits:
        cleaned_limits = limits.replace(risk_warning, "").strip()
        risk_warning_text = f"{risk_warning} {cleaned_limits}".strip()
    return {
        "outcome_summary": f"统一建模结论：primary={task}，status={status}，value={value_level}。{model_prefix}{findings}",
        "business_interpretation": str(outcome.get("business_interpretation") or findings),
        "recommended_action": action,
        "risk_warning": risk_warning_text,
        "model_selection_takeaway": model_selection_takeaway,
        "cv_stability_takeaway": cv_stability_takeaway,
        "prediction_fit_takeaway": prediction_fit_takeaway,
        "feature_importance_takeaway": feature_importance_takeaway,
        "error_analysis_takeaway": error_analysis_takeaway,
        "final_regression_synthesis": final_regression_synthesis,
    }


def _has_banned_content(text: str) -> bool:
    checked_text = text
    for phrase in NEGATED_STABLE_FORECAST_PHRASES:
        checked_text = checked_text.replace(phrase, "")
    if any(phrase in checked_text for phrase in BANNED_PHRASES):
        return True
    if "自动决策" in text and not any(marker in text for marker in SAFE_AUTO_DECISION_MARKERS):
        return True
    return False


def _sanitize(raw: dict[str, Any], fallback: dict[str, str], outcome: dict[str, Any]) -> dict[str, str] | None:
    sanitized = {
        "outcome_summary": _clip_text(raw.get("outcome_summary"), fallback["outcome_summary"], limit=260),
        "business_interpretation": _clip_text(raw.get("business_interpretation"), fallback["business_interpretation"]),
        "recommended_action": _clip_text(raw.get("recommended_action"), fallback["recommended_action"]),
        "risk_warning": _clip_text(raw.get("risk_warning"), fallback["risk_warning"]),
        "model_selection_takeaway": _clip_text(
            raw.get("model_selection_takeaway"),
            fallback["model_selection_takeaway"],
        ),
        "cv_stability_takeaway": _clip_text(
            raw.get("cv_stability_takeaway"),
            fallback["cv_stability_takeaway"],
        ),
        "prediction_fit_takeaway": _clip_text(
            raw.get("prediction_fit_takeaway"),
            fallback["prediction_fit_takeaway"],
        ),
        "feature_importance_takeaway": _clip_text(
            raw.get("feature_importance_takeaway"),
            fallback["feature_importance_takeaway"],
        ),
        "error_analysis_takeaway": _clip_text(
            raw.get("error_analysis_takeaway"),
            fallback["error_analysis_takeaway"],
        ),
        "final_regression_synthesis": _clip_text(
            raw.get("final_regression_synthesis"),
            fallback["final_regression_synthesis"],
            limit=360,
        ),
    }
    sanitized = {key: _humanize_internal_tokens(value) for key, value in sanitized.items()}
    if any(_has_banned_content(value) for value in sanitized.values()):
        return None
    task = str(outcome.get("primary_modeling_task") or "")
    if task == "loss_risk_classification":
        if "不代表因果关系" not in sanitized["risk_warning"]:
            sanitized["risk_warning"] = f"{sanitized['risk_warning']} 不代表因果关系。"
        if "不用于自动决策" not in sanitized["risk_warning"]:
            sanitized["risk_warning"] = f"{sanitized['risk_warning']} 不用于自动决策。"
    if task in {"sales_amount_forecast_baseline", "sales_amount_regression"}:
        if "不是生产级预测" not in sanitized["risk_warning"]:
            sanitized["risk_warning"] = f"{sanitized['risk_warning']} 不是生产级预测。"
        if "不用于自动决策" not in sanitized["risk_warning"]:
            sanitized["risk_warning"] = f"{sanitized['risk_warning']} 不用于自动决策。"
    if str(outcome.get("modeling_status")) == "weak_model" and "不建议直接用于复核排序" not in sanitized["risk_warning"]:
        sanitized["risk_warning"] = f"{sanitized['risk_warning']} 不建议直接用于复核排序。"
    sanitized["risk_warning"] = _clip_text(sanitized["risk_warning"], fallback["risk_warning"], limit=260)
    return sanitized


def _payload(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "Explain a unified modeling outcome for a sales analysis notebook.",
        "primary_modeling_task": outcome.get("primary_modeling_task"),
        "modeling_status": outcome.get("modeling_status"),
        "modeling_value_level": outcome.get("modeling_value_level"),
        "metrics_summary": outcome.get("metrics_summary"),
        "main_findings": outcome.get("main_findings"),
        "recommended_action": outcome.get("recommended_action"),
        "limitations": outcome.get("limitations"),
        "constraints": [
            "Do not claim causality.",
            "Do not claim production-grade forecasting.",
            "Do not suggest automatic decisions.",
            "For sales_amount_regression, return model_selection_takeaway, cv_stability_takeaway, prediction_fit_takeaway, feature_importance_takeaway, error_analysis_takeaway, and final_regression_synthesis.",
            "For sales_amount_regression model_selection_takeaway, if selection_basis exists, explain how holdout_mae_rank relates to cv_mae_rank and cv_stability_note.",
            "If selection_basis shows holdout is best but CV is unstable, downgrade the business conclusion to monitoring benchmark or follow-up comparison only.",
            "For sales_amount_regression cv_stability_takeaway, explain whether TimeSeriesSplit mean/std suggests stable cross-window performance or only a fragile final-window win.",
            "For sales_amount_regression error analysis, use error_direction exactly: actual < predicted means over-prediction, actual > predicted means under-prediction.",
            "For sales_amount_regression error analysis, distinguish the largest single-period error direction from the overall over/under-prediction tendency when both are mentioned.",
            "For sales_amount_regression takeaways, do not include section-title prefixes such as 模型选择解释：; these values are ordinary paragraphs under charts/tables.",
            "For sales_amount_regression takeaways, write at least two Chinese sentences within the 220-character limit; include metric judgment plus business meaning, usage limit, or next-step variables.",
            "For sales_amount_regression final_regression_synthesis, write 3-5 Chinese sentences with business value, usage limits, and next-step variables.",
            "Do not expose internal enum values such as holdout_and_cv_aligned or holdout_best_cv_not_stable; rewrite them as natural Chinese.",
            "Return concise Chinese JSON only.",
        ],
    }


def build_modeling_outcome_interpretation_with_trace(
    outcome: dict[str, Any],
    llm_client=None,
    output_language: str | None = None,
) -> tuple[dict[str, str], LLMStageTrace]:
    fallback = _deterministic_interpretation(outcome)
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return fallback, _trace(
            llm_client=llm_client,
            status="disabled",
            reason="LLM is disabled, so modeling outcome interpretation used deterministic text.",
        )
    suggest = getattr(llm_client, "suggest_modeling_outcome_interpretation", None)
    if suggest is None:
        return fallback, _trace(
            llm_client=llm_client,
            status="skipped",
            reason="LLM client does not expose suggest_modeling_outcome_interpretation.",
        )
    try:
        raw = suggest(
            _payload(outcome),
            language_instruction=user_facing_language_instruction(output_language),
        )
    except Exception as exc:
        return fallback, _trace(
            llm_client=llm_client,
            status="fallback_on_error",
            reason=f"Modeling outcome interpretation failed; used deterministic text. {describe_llm_error(exc)}",
            attempted=True,
        )
    sanitized = _sanitize(raw if isinstance(raw, dict) else {}, fallback, outcome)
    if sanitized is None:
        return fallback, _trace(
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason="Modeling outcome interpretation contained unsafe or out-of-scope language.",
            attempted=True,
        )
    return sanitized, _trace(
        llm_client=llm_client,
        status="llm_applied",
        reason="LLM modeling outcome interpretation was applied.",
        attempted=True,
        applied=True,
    )
