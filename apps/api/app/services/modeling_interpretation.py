from __future__ import annotations

import re
from typing import Any

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.report import AnalysisReport, ModuleReport
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.output_language import user_facing_language_instruction

INTERPRETATION_KEYS = (
    "modeling_summary",
    "model_choice_takeaway",
    "threshold_takeaway",
    "feature_takeaway",
    "error_takeaway",
    "review_guidance",
    "business_use_warning",
)

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
)

LIMITATION_SENTENCE = "该模型只用于风险预警和人工复核优先级排序，不代表因果关系，不用于自动决策。"
WEAK_LIMITATION_SENTENCE = "该模型只适合作为探索性风险线索和人工复核参考，不建议直接用于复核排序，不代表因果关系，不用于自动决策。"
SUMMARY_FALLBACK_SENTENCE = "本节将亏损风险转化为人工复核优先级。"
SAFE_DECISION_LIMITS = (
    "不用于自动决策",
    "不能用于自动决策",
    "不可用于自动决策",
    "不是自动决策",
)


def _modeling_module(report: AnalysisReport) -> ModuleReport | None:
    return next(
        (module for module in report.modules if module.module_id == "loss_risk_modeling"),
        None,
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
        stage="modeling_interpretation",
        llm_client=llm_client,
        status=status,
        reason=reason,
        attempted=attempted,
        applied=applied,
    )


def _is_runnable_modeling_module(module: ModuleReport) -> bool:
    return bool(module.tables.get("model_comparison"))


def _top_rows(rows: list[dict[str, object]], limit: int) -> list[dict[str, object]]:
    return [dict(row) for row in rows[:limit] if isinstance(row, dict)]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _threshold_row(rows: list[dict[str, object]], target: float) -> dict[str, object] | None:
    for row in rows:
        threshold = _as_float(row.get("threshold"))
        if threshold is not None and abs(threshold - target) < 0.000001:
            return row
    return None


def _number_delta(
    base: dict[str, object] | None,
    comparison: dict[str, object] | None,
    key: str,
) -> float | None:
    if not base or not comparison:
        return None
    base_value = _as_float(base.get(key))
    comparison_value = _as_float(comparison.get(key))
    if base_value is None or comparison_value is None:
        return None
    return round(comparison_value - base_value, 6)


def _loss_risk_analysis_context(module: ModuleReport) -> dict[str, Any]:
    tables = module.tables or {}
    summary_metrics = dict(module.summary_metrics or {})
    thresholds = _top_rows(tables.get("threshold_analysis", []), 10)
    threshold_0_4 = _threshold_row(thresholds, 0.4)
    threshold_0_5 = _threshold_row(thresholds, 0.5)
    threshold_0_7 = _threshold_row(thresholds, 0.7)
    feature_importance_top_raw = _top_rows(tables.get("feature_importance", []), 8)
    feature_importance_grouped = _top_rows(tables.get("feature_importance_grouped", []), 8)
    return {
        "modeling_status": summary_metrics.get("modeling_status"),
        "model_quality_status": summary_metrics.get("model_quality_status"),
        "target_positive_rate": summary_metrics.get("target_positive_rate"),
        "train_positive_count": summary_metrics.get("train_positive_count"),
        "test_positive_count": summary_metrics.get("test_positive_count"),
        "weak_reasons": summary_metrics.get("weak_reasons") or [],
        "best_model": summary_metrics.get("best_model"),
        "best_recall": summary_metrics.get("best_recall"),
        "best_precision": summary_metrics.get("best_precision"),
        "best_f1": summary_metrics.get("best_f1"),
        "best_roc_auc": summary_metrics.get("best_roc_auc"),
        "model_comparison": _top_rows(tables.get("model_comparison", []), 10),
        "default_threshold": summary_metrics.get("threshold_default", 0.5),
        "threshold_analysis": thresholds,
        "threshold_0_4": threshold_0_4,
        "threshold_0_5": threshold_0_5,
        "threshold_0_7": threshold_0_7,
        "recall_delta_0_4_vs_0_5": _number_delta(threshold_0_5, threshold_0_4, "recall"),
        "precision_delta_0_4_vs_0_5": _number_delta(threshold_0_5, threshold_0_4, "precision"),
        "review_count_delta_0_4_vs_0_5": _number_delta(threshold_0_5, threshold_0_4, "predicted_loss_count"),
        "review_load_rate_delta_0_4_vs_0_5": _number_delta(threshold_0_5, threshold_0_4, "review_load_rate"),
        "recall_delta_0_7_vs_0_5": _number_delta(threshold_0_5, threshold_0_7, "recall"),
        "review_count_delta_0_7_vs_0_5": _number_delta(threshold_0_5, threshold_0_7, "predicted_loss_count"),
        "tp": summary_metrics.get("true_positive_count"),
        "fp": summary_metrics.get("false_positive_count"),
        "fn": summary_metrics.get("false_negative_count"),
        "tn": summary_metrics.get("true_negative_count"),
        "predicted_loss_count": (threshold_0_5 or {}).get("predicted_loss_count"),
        "review_load_rate": (threshold_0_5 or {}).get("review_load_rate"),
        "top_feature_groups": feature_importance_grouped,
        "top_raw_features": feature_importance_top_raw,
        "safe_features": _top_rows(tables.get("safe_features", []), 8),
        "leakage_blocked_fields": _top_rows(tables.get("blocked_features", []), 8),
        "high_risk_examples_summary": _top_rows(tables.get("high_risk_examples", []), 3),
        "missed_loss_examples_summary": _top_rows(tables.get("false_negative_examples", []), 3),
    }


def _build_modeling_payload(module: ModuleReport) -> dict[str, Any]:
    tables = module.tables or {}
    summary_metrics = dict(module.summary_metrics or {})
    feature_importance_top_raw = _top_rows(tables.get("feature_importance", []), 5)
    feature_importance_grouped = _top_rows(
        tables.get("feature_importance_grouped", []),
        5,
    )
    return {
        "task": "Interpret loss risk modeling results for a sales analysis notebook.",
        "target_definition": "is_loss = profit < 0",
        "summary_metrics": summary_metrics,
        "business_priority_metric": summary_metrics.get("business_priority_metric"),
        "threshold_default": summary_metrics.get("threshold_default"),
        "model_comparison": _top_rows(tables.get("model_comparison", []), 10),
        "threshold_analysis": _top_rows(tables.get("threshold_analysis", []), 5),
        "confusion_matrix": _top_rows(tables.get("confusion_matrix", []), 4),
        "feature_importance_grouped": feature_importance_grouped,
        "feature_importance_top_raw": feature_importance_top_raw,
        "loss_risk_analysis_context": _loss_risk_analysis_context(module),
        "top_raw_features": feature_importance_top_raw,
        "top_feature_groups": feature_importance_grouped,
        "false_positive_count": summary_metrics.get("false_positive_count"),
        "false_negative_count": summary_metrics.get("false_negative_count"),
        "warnings": list(module.warnings or []),
    }


def _deterministic_modeling_interpretation(module: ModuleReport) -> dict[str, str]:
    summary_metrics = module.summary_metrics or {}
    tables = module.tables or {}
    groups = tables.get("feature_importance_grouped", [])
    top_groups = ", ".join(
        str(row.get("feature_group"))
        for row in groups[:3]
        if isinstance(row, dict) and row.get("feature_group")
    )
    best_model = summary_metrics.get("best_model", "当前最佳模型")
    best_recall = summary_metrics.get("best_recall", "unknown")
    best_f1 = summary_metrics.get("best_f1", "unknown")
    best_roc_auc = summary_metrics.get("best_roc_auc", "unknown")
    target_positive_rate = summary_metrics.get("target_positive_rate", "unknown")
    threshold_default = summary_metrics.get("threshold_default", 0.5)
    false_positive_count = summary_metrics.get("false_positive_count", 0)
    false_negative_count = summary_metrics.get("false_negative_count", 0)
    model_quality_status = str(summary_metrics.get("model_quality_status") or "usable")
    weak_reasons = summary_metrics.get("weak_reasons") or []
    weak_reason_text = "、".join(str(item) for item in weak_reasons) if isinstance(weak_reasons, list) else str(weak_reasons)
    if model_quality_status == "weak":
        quality_sentence = (
            "当前模型质量状态为 weak，只适合作为探索性参考，不建议直接用于复核排序。"
            + (f"主要原因包括 {weak_reason_text}。" if weak_reason_text else "")
        )
        review_guidance = "建议先补充样本或调整建模口径；当前结果只适合探索性参考，不建议直接用于复核排序。"
        summary_limit_sentence = WEAK_LIMITATION_SENTENCE
        business_use_warning = WEAK_LIMITATION_SENTENCE
    else:
        quality_sentence = "当前模型可作为亏损风险预警和人工复核优先级排序参考。"
        review_guidance = (
            "建议将模型输出作为人工复核优先级，优先查看高风险和漏判亏损样本，"
            "并结合折扣、类目和订单明细判断是否需要调整业务规则。"
        )
        summary_limit_sentence = LIMITATION_SENTENCE
        business_use_warning = LIMITATION_SENTENCE
    feature_takeaway = (
        f"当前较突出的预测信号组是 {top_groups}，需结合业务明细复盘。该字段仅为预测信号，不代表因果关系。"
        if top_groups
        else "当前证据不足以进一步判断主要预测信号组。该字段仅为预测信号，不代表因果关系。"
    )
    summary_top_groups = top_groups or "暂无稳定特征组"
    return {
        "modeling_summary": _ensure_modeling_summary_limits(
            f"本节以 is_loss = profit < 0 定义亏损风险，当前亏损样本占比为 {target_positive_rate}。"
            f"{quality_sentence}"
            f"最佳模型为 {best_model}，Recall={best_recall}，F1={best_f1}，ROC AUC={best_roc_auc}；"
            f"默认阈值下 False Negative={false_negative_count}、False Positive={false_positive_count}。"
            f"主要预测信号组包括 {summary_top_groups}。"
            f"{summary_limit_sentence}",
            limitation_sentence=summary_limit_sentence,
        ),
        "model_choice_takeaway": (
            f"当前最佳模型为 {best_model}，Recall={best_recall}，"
            f"F1={best_f1}，ROC AUC={best_roc_auc}。{quality_sentence}"
        ),
        "threshold_takeaway": (
            f"默认阈值 {threshold_default} 下需在召回率和人工复核量之间取舍，"
            "低阈值通常会增加复核量。"
        ),
        "feature_takeaway": feature_takeaway,
        "error_takeaway": (
            f"False Negative={false_negative_count} 是漏判亏损样本，"
            f"False Positive={false_positive_count} 主要代表额外复核成本。"
        ),
        "review_guidance": review_guidance,
        "business_use_warning": business_use_warning,
    }


def _clip_text(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _trim_to_complete_sentences(text: str, max_len: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    sentences = re.findall(r"[^。；;]+[。；;]", normalized)
    if not sentences:
        return normalized.rstrip(" ，,;；") if len(normalized) <= max_len else ""

    selected: list[str] = []
    total = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if total + len(sentence) > max_len:
            break
        selected.append(sentence)
        total += len(sentence)
    return "".join(selected).strip()


def _summary_body_without_existing_limits(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    marker_positions = [
        normalized.find(marker)
        for marker in ("不代表因果关系", "不用于自动决策")
        if marker in normalized
    ]
    if marker_positions:
        normalized = normalized[: min(marker_positions)]
    return normalized.rstrip(" ，。；,;")


def _ensure_modeling_summary_limits(
    text: str,
    limit: int = 230,
    limitation_sentence: str = LIMITATION_SENTENCE,
) -> str:
    room = max(0, limit - len(limitation_sentence))
    body_source = _summary_body_without_existing_limits(text)
    body = _trim_to_complete_sentences(body_source, room)
    if not body:
        body = SUMMARY_FALLBACK_SENTENCE
    summary = f"{body}{limitation_sentence}"
    if len(summary) <= limit:
        return summary
    return f"{SUMMARY_FALLBACK_SENTENCE}{limitation_sentence}"[:limit]


def _has_banned_content(text: str) -> bool:
    if any(phrase in text for phrase in BANNED_PHRASES):
        return True
    if "自动决策" in text and not any(safe in text for safe in SAFE_DECISION_LIMITS):
        return True
    return False


def _sanitize_modeling_interpretation(
    result: dict[str, Any],
    fallback: dict[str, str],
) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    is_weak_fallback = "不建议直接用于复核排序" in fallback.get("modeling_summary", "")
    limitation_sentence = WEAK_LIMITATION_SENTENCE if is_weak_fallback else LIMITATION_SENTENCE
    for key in INTERPRETATION_KEYS:
        raw_value = str(result.get(key, "")).strip()
        if not raw_value:
            sanitized[key] = fallback[key]
            continue
        if _has_banned_content(raw_value):
            sanitized[key] = fallback[key]
            continue
        if key == "modeling_summary":
            sanitized[key] = _ensure_modeling_summary_limits(raw_value, limitation_sentence=limitation_sentence)
        elif key == "feature_takeaway":
            feature_text = raw_value
            if "不代表因果关系" not in feature_text:
                feature_text = f"{feature_text} 该字段仅为预测信号，不代表因果关系。"
            sanitized[key] = _clip_text(feature_text, limit=180)
        else:
            sanitized[key] = _clip_text(raw_value, limit=180)

    warning = sanitized["business_use_warning"]
    if "不代表因果关系" not in warning or "不用于自动决策" not in warning:
        warning = f"{warning} {limitation_sentence}".strip()
    if is_weak_fallback and "不建议直接用于复核排序" not in warning:
        warning = f"{warning} {WEAK_LIMITATION_SENTENCE}".strip()
    sanitized["business_use_warning"] = _clip_text(warning, limit=180)
    return sanitized


def build_modeling_interpretation_with_trace(
    report: AnalysisReport,
    llm_client=None,
    output_language: str | None = None,
) -> tuple[dict[str, str], LLMStageTrace]:
    module = _modeling_module(report)
    if module is None:
        return {}, _trace(
            llm_client=llm_client,
            status="not_applicable",
            reason="No loss_risk_modeling module was present in the report.",
        )
    if not _is_runnable_modeling_module(module):
        return {}, _trace(
            llm_client=llm_client,
            status="not_applicable",
            reason="loss_risk_modeling was skipped or has no model comparison table.",
        )

    fallback = _deterministic_modeling_interpretation(module)
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return fallback, _trace(
            llm_client=llm_client,
            status="disabled",
            reason="LLM client is disabled; using deterministic modeling interpretation.",
            attempted=False,
            applied=False,
        )
    if not hasattr(llm_client, "suggest_modeling_interpretation"):
        return fallback, _trace(
            llm_client=llm_client,
            status="skipped",
            reason="LLM client does not expose suggest_modeling_interpretation.",
            attempted=False,
            applied=False,
        )

    payload = _build_modeling_payload(module)
    try:
        raw_result = llm_client.suggest_modeling_interpretation(
            payload,
            language_instruction=user_facing_language_instruction(output_language),
        )
        sanitized = _sanitize_modeling_interpretation(raw_result, fallback)
        return sanitized, _trace(
            llm_client=llm_client,
            status="llm_applied",
            reason="LLM modeling interpretation was applied.",
            attempted=True,
            applied=True,
        )
    except Exception as exc:
        return fallback, _trace(
            llm_client=llm_client,
            status="fallback_on_error",
            reason=f"LLM modeling interpretation failed: {describe_llm_error(exc)}",
            attempted=True,
            applied=False,
        )
