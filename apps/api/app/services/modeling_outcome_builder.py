from __future__ import annotations

from typing import Any


HIGH_MAPE_THRESHOLD = 100.0


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _module(report: Any, module_id: str) -> Any | None:
    modules = _get(report, "modules", []) or []
    return next((module for module in modules if _get(module, "module_id") == module_id), None)


def _summary(module: Any | None) -> dict[str, Any]:
    return dict(_get(module, "summary_metrics", {}) or {}) if module is not None else {}


def _tables(module: Any | None) -> dict[str, Any]:
    return dict(_get(module, "tables", {}) or {}) if module is not None else {}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_feature_names(module: Any | None, *, limit: int = 5) -> list[str]:
    features: list[str] = []
    for row in (_tables(module).get("feature_importance") or [])[:limit]:
        feature = str(_get(row, "feature") or "").strip()
        if feature:
            features.append(feature)
    return features


def _top_error_rows(module: Any | None, *, limit: int = 5) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in (_tables(module).get("worst_error_periods") or [])[:limit]:
        rows.append(
            {
                key: _get(row, key)
                for key in ("period", "actual", "predicted", "absolute_error", "percentage_error", "error_direction")
                if _get(row, key) is not None
            }
        )
    return rows


def _prediction_error_direction_counts(module: Any | None) -> tuple[int, int]:
    over_prediction_count = 0
    under_prediction_count = 0
    tables = _tables(module)
    source_rows = tables.get("regression_backtest") or tables.get("worst_error_periods") or []
    for row in source_rows:
        direction = str(_get(row, "error_direction") or "").strip().lower()
        if direction == "over":
            over_prediction_count += 1
            continue
        if direction == "under":
            under_prediction_count += 1
            continue
        actual = _float(_get(row, "actual"))
        predicted = _float(_get(row, "predicted"))
        if actual is None or predicted is None:
            continue
        if predicted > actual:
            over_prediction_count += 1
        elif predicted < actual:
            under_prediction_count += 1
    return over_prediction_count, under_prediction_count


def _is_loss_risk_runnable(module: Any | None) -> bool:
    if module is None:
        return False
    tables = _tables(module)
    summary = _summary(module)
    if str(summary.get("modeling_status") or summary.get("status") or "").lower() == "skipped":
        return False
    return bool(tables.get("model_comparison")) or bool(summary.get("best_model"))


def _is_forecast_evaluated(module: Any | None) -> bool:
    if module is None:
        return False
    summary = _summary(module)
    if summary.get("forecast_status") != "baseline_evaluated":
        return False
    tables = _tables(module)
    return bool(tables.get("baseline_metrics")) or summary.get("best_baseline_mape") is not None


def _is_sales_regression_evaluated(module: Any | None) -> bool:
    if module is None:
        return False
    summary = _summary(module)
    if summary.get("forecast_status") != "regression_evaluated":
        return False
    tables = _tables(module)
    return bool(tables.get("model_comparison")) and summary.get("best_model_mae") is not None


def _opportunity_text(
    modeling_opportunity_plan: dict[str, Any] | None,
    modeling_opportunity_decision: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    for payload in (modeling_opportunity_plan or {}, modeling_opportunity_decision or {}):
        if isinstance(payload, dict):
            parts.extend(str(value) for value in payload.values() if isinstance(value, str))
    return " ".join(parts)


def _forecast_supersedes_opportunity_only(
    *,
    modeling_opportunity_plan: dict[str, Any] | None,
    modeling_opportunity_decision: dict[str, Any] | None,
) -> bool:
    text = _opportunity_text(modeling_opportunity_plan, modeling_opportunity_decision)
    return any(term in text for term in ("仅记录机会", "不训练模型", "不训练新模型", "opportunity_only"))


def _loss_risk_outcome(dataset: str | None, module: Any) -> dict[str, Any]:
    summary = _summary(module)
    quality = str(summary.get("model_quality_status") or "usable").lower()
    is_weak = quality == "weak"
    metrics_summary = {
        key: summary.get(key)
        for key in (
            "model_quality_status",
            "best_model",
            "best_precision",
            "best_recall",
            "best_f1",
            "best_roc_auc",
            "target_positive_count",
            "target_positive_rate",
            "test_positive_count",
            "false_positive_count",
            "false_negative_count",
        )
        if summary.get(key) is not None
    }
    if is_weak:
        main_findings = [
            "亏损风险模型已跑通，但质量状态为 weak，只适合作为探索性风险线索。",
            "正例数量或测试集正例不足时，指标波动会很大。",
        ]
        recommended_action = "先补充亏损样本或调整目标口径；当前结果不建议直接用于复核排序。"
        limitations = [
            "不建议直接用于复核排序。",
            "不代表因果关系，不用于自动决策。",
        ]
        if summary.get("weak_reasons"):
            limitations.append("weak_reasons: " + ", ".join(str(item) for item in summary.get("weak_reasons") or []))
    else:
        main_findings = [
            "亏损风险分类已有可用建模证据，可作为人工复核优先级参考。",
            "阈值、混淆矩阵和特征重要性可共同解释复核量与漏判风险。",
        ]
        recommended_action = "可把模型结果作为人工复核优先级参考，并结合订单、折扣和利润明细复盘。"
        limitations = [
            "模型输出不代表因果关系，不用于自动决策。",
            "复核策略仍需结合业务规则和人工判断。",
        ]
    return {
        "dataset": dataset or "",
        "primary_modeling_task": "loss_risk_classification",
        "modeling_status": "weak_model" if is_weak else "strong_model",
        "modeling_value_level": "low" if is_weak else "high",
        "metrics_summary": metrics_summary,
        "main_findings": main_findings,
        "business_interpretation": (
            "当前模型识别的是亏损风险预测信号，而不是利润变化的因果解释。"
            if is_weak
            else "当前模型可以把亏损风险转化成更容易复核的优先级线索。"
        ),
        "recommended_action": recommended_action,
        "limitations": limitations,
        "should_show_in_notebook": True,
        "source_modules": ["loss_risk_modeling"],
        "conflict_resolution": {
            "primary_precedence": "loss_risk_modeling",
            "forecast_baseline_supersedes_opportunity_only": False,
        },
    }


def _forecast_outcome(
    dataset: str | None,
    module: Any,
    *,
    modeling_opportunity_plan: dict[str, Any] | None,
    modeling_opportunity_decision: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = _summary(module)
    mape = _float(summary.get("best_baseline_mape"))
    high_mape = mape is None or mape > HIGH_MAPE_THRESHOLD
    metrics_summary = {
        key: summary.get(key)
        for key in (
            "forecast_status",
            "best_baseline",
            "best_baseline_mae",
            "best_baseline_rmse",
            "best_baseline_mape",
            "time_split",
            "series_granularity",
            "day_span",
            "date_gap_ratio",
            "zero_or_missing_period_ratio",
        )
        if summary.get(key) is not None
    }
    if high_mape:
        main_findings = [
            "销售额预测 baseline 已完成时间顺序回测，但 MAPE 偏高。",
            "当前序列波动较大，预测难度高，更适合作为监控参照。",
        ]
        recommended_action = "仅作为监控参照和后续模型对照基线，暂不包装成稳定预测能力。"
        limitations = [
            "MAPE 偏高，预测难度高。",
            "这是 baseline，不是生产级预测，不用于自动决策。",
        ]
    else:
        main_findings = [
            "销售额预测 baseline 已完成时间顺序回测。",
            "当前误差水平可作为后续预测模型或业务监控的对照基线。",
        ]
        recommended_action = "把 baseline 作为监控参照和后续预测增强的比较基准。"
        limitations = [
            "这是 baseline，不是生产级预测，不用于自动决策。",
            "尚未纳入更复杂的节假日、促销或外部变量。",
        ]
    return {
        "dataset": dataset or "",
        "primary_modeling_task": "sales_amount_forecast_baseline",
        "modeling_status": "weak_baseline" if high_mape else "baseline_evaluated",
        "modeling_value_level": "low" if high_mape else "medium",
        "metrics_summary": metrics_summary,
        "main_findings": main_findings,
        "business_interpretation": (
            "当前 forecast 证据说明序列可被回测，但误差偏高时只能作为监控参照。"
            if high_mape
            else "当前 forecast baseline 可以帮助判断后续复杂预测模型是否真正带来增益。"
        ),
        "recommended_action": recommended_action,
        "limitations": limitations,
        "should_show_in_notebook": True,
        "source_modules": ["forecast_analysis"],
        "conflict_resolution": {
            "primary_precedence": "forecast_analysis",
            "forecast_baseline_supersedes_opportunity_only": _forecast_supersedes_opportunity_only(
                modeling_opportunity_plan=modeling_opportunity_plan,
                modeling_opportunity_decision=modeling_opportunity_decision,
            ),
        },
    }


def _sales_regression_outcome(dataset: str | None, module: Any) -> dict[str, Any]:
    summary = _summary(module)
    status = str(summary.get("regression_status") or "weak")
    is_weak = status != "usable"
    metrics_summary = {
        key: summary.get(key)
        for key in (
            "forecast_status",
            "regression_status",
            "best_model",
            "best_model_mae",
            "best_model_rmse",
            "best_model_mape",
            "best_model_r2",
            "baseline_model",
            "baseline_mae",
            "improvement_vs_baseline",
            "time_split",
            "regression_time_split",
            "regression_cv_strategy",
            "cv_mae_mean",
            "cv_mae_std",
            "cv_mape_mean",
            "cv_mape_std",
            "cv_r2_mean",
            "cv_r2_std",
            "series_granularity",
            "day_span",
            "observed_period_count",
            "train_period_count",
            "backtest_period_count",
            "regression_effective_sample_count",
            "regression_feature_leakage_guard",
            "selection_basis",
        )
        if summary.get(key) is not None
    }
    top_features = _top_feature_names(module)
    worst_error_periods = _top_error_rows(module)
    over_prediction_count, under_prediction_count = _prediction_error_direction_counts(module)
    if top_features:
        metrics_summary["top_features"] = top_features
    if worst_error_periods:
        metrics_summary["worst_error_periods"] = worst_error_periods
        metrics_summary["over_prediction_count"] = over_prediction_count
        metrics_summary["under_prediction_count"] = under_prediction_count
    if is_weak:
        reasons = summary.get("regression_weak_reasons") or []
        reason_text = "、".join(str(item) for item in reasons) if isinstance(reasons, list) else str(reasons)
        main_findings = [
            "销售额回归已完成轻量回测，但当前没有形成稳定预测能力。",
            "复杂模型没有明显超过 baseline 或误差偏高时，更适合作为预测难度评估。",
        ]
        recommended_action = "仅作为监控参照和后续模型对照；不建议直接用于销售计划、库存、补货或经营目标制定。"
        limitations = [
            "当前销售额回归模型没有形成稳定预测能力。",
            "不建议直接用于销售计划、库存、补货或经营目标制定。",
            "这是轻量回归尝试，不是生产级预测，不用于自动决策。",
        ]
        if reason_text:
            limitations.append("weak_reasons: " + reason_text)
    else:
        main_findings = [
            "销售额回归已完成时间顺序回测，并与简单 baseline 做了对比。",
            "当前最佳模型相对 baseline 有改善，可作为销售监控和后续预测增强的对照模型。",
        ]
        recommended_action = "把最佳模型作为销售监控参照和后续预测模型的比较基准，仍需结合业务规则复核。"
        limitations = [
            "这是轻量回归模型，不是生产级预测，不用于自动决策。",
            "尚未纳入促销、节假日、库存、门店活动等外部变量。",
        ]
    return {
        "dataset": dataset or "",
        "primary_modeling_task": "sales_amount_regression",
        "modeling_status": "regression_weak" if is_weak else "regression_usable",
        "modeling_value_level": "low" if is_weak else "medium",
        "metrics_summary": metrics_summary,
        "main_findings": main_findings,
        "business_interpretation": (
            "当前销售额回归主要说明预测难度较高，适合做监控参照。"
            if is_weak
            else "当前销售额回归可以帮助判断简单 baseline 之外是否存在稳定可利用的时间序列信号。"
        ),
        "recommended_action": recommended_action,
        "limitations": limitations,
        "should_show_in_notebook": True,
        "source_modules": ["forecast_analysis"],
        "conflict_resolution": {
            "primary_precedence": "forecast_analysis",
            "loss_risk_supersedes_sales_regression": True,
        },
    }


def _opportunity_only_outcome(
    dataset: str | None,
    modeling_opportunity_plan: dict[str, Any] | None,
    modeling_opportunity_decision: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = modeling_opportunity_decision or {}
    plan = modeling_opportunity_plan or {}
    notebook_message = str(decision.get("notebook_message") or plan.get("notebook_message") or "").strip()
    recommended = str(plan.get("recommended_modeling_task") or "no_modeling_descriptive_only")
    return {
        "dataset": dataset or "",
        "primary_modeling_task": "modeling_opportunity_only",
        "modeling_status": str(plan.get("decision_status") or "opportunity_only"),
        "modeling_value_level": "none",
        "metrics_summary": {},
        "main_findings": [
            notebook_message or "当前没有稳定的已训练模型或 forecast baseline 证据。",
        ],
        "business_interpretation": "本轮只保留建模机会判断，不把数据强行塞进低价值模型。",
        "recommended_action": f"后续可按 `{recommended}` 的字段条件继续评估。",
        "limitations": [
            "当前没有实际训练或回测证据。",
            str(decision.get("risk_warning") or "不用于自动决策。").strip(),
        ],
        "should_show_in_notebook": True,
        "source_modules": [],
        "conflict_resolution": {
            "primary_precedence": "modeling_opportunity",
            "forecast_baseline_supersedes_opportunity_only": False,
        },
    }


def build_modeling_outcome(
    report: Any,
    *,
    modeling_opportunity_plan: dict[str, Any] | None = None,
    modeling_opportunity_decision: dict[str, Any] | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Build one canonical modeling story from existing module evidence."""
    loss_module = _module(report, "loss_risk_modeling")
    if _is_loss_risk_runnable(loss_module):
        return _loss_risk_outcome(dataset, loss_module)

    forecast_module = _module(report, "forecast_analysis")
    if _is_sales_regression_evaluated(forecast_module):
        return _sales_regression_outcome(dataset, forecast_module)
    if _is_forecast_evaluated(forecast_module):
        return _forecast_outcome(
            dataset,
            forecast_module,
            modeling_opportunity_plan=modeling_opportunity_plan,
            modeling_opportunity_decision=modeling_opportunity_decision,
        )

    return _opportunity_only_outcome(dataset, modeling_opportunity_plan, modeling_opportunity_decision)
