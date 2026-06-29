from __future__ import annotations

import re
from typing import Any

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_client import complete_json_for_stage
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.final_synthesis_guard import guard_english_final_synthesis
from app.services.notebook.markdown_sanitizer import clean_business_text
from app.services.output_language import is_english_output, user_facing_language_instruction


STAGE = "final_synthesis"

PROCESS_LOG_MARKERS = (
    "发现 0 条时间缺失记录",
    "已完成",
    "共覆盖",
    "已按",
    "Top 商品为",
    "本节汇总最重要的发现",
    "一份分析 notebook 的最终价值",
    "建立复盘责任人和验收口径",
    "指标画像",
)

GENERIC_ACTION_MARKERS = (
    "加强管理",
    "持续关注",
    "优化经营",
    "建立机制",
    "提升效率",
    "补齐字段并复跑分析",
    "围绕",
    "建立复盘责任人",
)

LIMITATION_MARKERS = (
    "缺少",
    "补齐",
    "未提供",
    "不可用",
    "不支持",
    "missing",
    "unavailable",
)

UNSUPPORTED_RULES = (
    ({"discount"}, ("折扣", "discount")),
    ({"profit"}, ("利润", "利润率", "profit", "margin")),
    ({"customer_id", "order_id"}, ("复购", "客户生命周期", "订单结构")),
)

PROFILE_KEYS = (
    "row_count",
    "column_count",
    "order_count",
    "customer_count",
    "total_sales_amount",
    "total_profit_amount",
    "total_profit",
    "profit_margin",
    "negative_profit_rate",
    "monthly_volatility",
    "profit_margin_spread",
    "top_country_sales_share",
    "repeat_customer_rate",
    "line_per_order_avg",
    "top_category_sales_share",
    "country_count",
    "product_count",
)

MODELING_METRIC_KEYS = (
    "best_model",
    "model_quality_status",
    "best_precision",
    "best_recall",
    "best_f1",
    "best_roc_auc",
    "threshold_default",
    "false_positive_count",
    "false_negative_count",
    "target_positive_rate",
    "test_positive_count",
    "weak_reasons",
    "best_model_mae",
    "best_model_r2",
    "best_model_mape",
    "baseline_model",
    "baseline_mae",
    "improvement_vs_baseline",
    "worst_error_periods",
)


SYSTEM_PROMPT = """
你是销售分析 Notebook 的最终报告总编辑。你必须基于输入中的真实 EDA 证据、图表上下文、建模结果和 action_plan_rows 参考，重新综合生成最终摘要和结论。

只输出 JSON，字段必须是：
{
  "brief_findings": [{"finding": "...", "evidence": "...", "business_meaning": "..."}],
  "brief_actions": [{"action": "...", "linked_evidence": "...", "priority_reason": "..."}],
  "main_conclusions": [{"conclusion": "...", "evidence": "...", "business_meaning": "..."}],
  "recommended_actions": [{"action": "...", "linked_metric_or_segment": "...", "expected_use": "...", "display_text": "..."}]
}

recommended_actions 每一项必须提供 display_text。display_text 是一条完整、自然、可直接放进 Notebook 的行动建议，不要只是 action、linked_metric_or_segment、expected_use 的拼接；句子内部要自然连接动作、证据和目标，可以包含具体指标，但指标必须服务于建议，不要孤立堆数字。display_text 建议 1 到 2 句话，约 80 到 180 个中文字符。可以使用“由于……，建议……”“针对……，先……，再……” “不要直接……，应先……”等表达方向，但不要让所有建议套同一种句式。
display_text 禁止写成“动作；指标；用途”的三段式拼接，也禁止空话：加强管理、持续关注、优化经营、建立机制、提升效率。

不要写流程日志。禁止：发现 0 条时间缺失记录、已完成 N 个指标画像、共覆盖 N 个时间粒度、已按 xxx 完成拆解、Top 商品为 xxx、本节汇总最重要的发现、一份分析 notebook 的最终价值、建立复盘责任人和验收口径。
不要输出通用模板建议。禁止：加强管理、持续关注、优化经营、建立机制、提升效率、补齐字段并复跑分析、围绕 xxx 建立复盘责任人和验收口径。
每条建议必须绑定当前数据集里的一个证据，例如指标、切片、模型结果、风险区间或业务对象；没有足够证据就少写，不要硬凑。
不同数据集必须根据证据差异生成不同判断：strong loss-risk 突出折扣、利润、模型、阈值、误报漏判和人工复核；weak loss-risk 突出正例不足和回到 EDA/业务复盘；sales regression 突出最佳模型、baseline 改善、R2/误差、非生产级定位和需要补充节假日/促销/库存/价格/渠道字段。
不要声称模型可自动决策，不要把相关性写成因果。
""".strip()


def _clean(value: Any, limit: int = 220) -> str:
    text = " ".join(clean_business_text(value).split())
    text = text.replace("缺失", "缺少")
    text = text.replace("需新增字段", "缺少字段时需新增")
    text = text.replace("需补充", "缺少字段时先补充")
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _is_process_log(text: str) -> bool:
    if re.search(r"发现\s*\d+\s*条", text):
        return True
    return any(marker in text for marker in PROCESS_LOG_MARKERS)


def _is_generic_action(text: str) -> bool:
    return any(marker in text for marker in GENERIC_ACTION_MARKERS)


def _looks_like_mechanical_display_text(text: str) -> bool:
    return text.count("；") + text.count(";") >= 2


def _valid_display_text(value: Any) -> str:
    text = _clean(value, 360)
    if not text or len(text) < 20:
        return ""
    if _is_process_log(text) or _is_generic_action(text):
        return ""
    if _looks_like_mechanical_display_text(text):
        return ""
    return text


def _has_unsupported_claim(text: str, mapped_fields: set[str]) -> bool:
    lower = text.lower()
    if any(marker in lower or marker in text for marker in LIMITATION_MARKERS):
        return False
    for required_fields, terms in UNSUPPORTED_RULES:
        if required_fields & mapped_fields:
            continue
        if any(term.lower() in lower or term in text for term in terms):
            return True
    return False


def _compact_dict(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    source = payload or {}
    return {key: source[key] for key in keys if key in source and source[key] is not None}


def _compact_evidence_pack(evidence_pack: dict[str, Any] | None) -> dict[str, Any]:
    pack = evidence_pack or {}
    focus_evidence: dict[str, list[dict[str, str]]] = {}
    for focus, items in (pack.get("focus_evidence") or {}).items():
        if not isinstance(items, list):
            continue
        compact_items = []
        for item in items[:4]:
            if isinstance(item, dict):
                compact_items.append(
                    {
                        key: _clean(item.get(key), 180)
                        for key in ("finding", "evidence", "business_meaning", "risk", "text")
                        if item.get(key)
                    }
                )
            elif item:
                compact_items.append({"text": _clean(item, 180)})
        if compact_items:
            focus_evidence[str(focus)] = compact_items
    return {
        "dataset_signature": pack.get("dataset_signature") or {},
        "focus_evidence": focus_evidence,
        "risks": pack.get("risks") or pack.get("risk_items") or [],
        "findings": pack.get("findings") or pack.get("distinctive_facts") or [],
    }


def _compact_modules(report: AnalysisReport) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    for module in report.modules:
        module_id = str(module.module_id)
        if module_id not in {
            "sales_trends",
            "product_and_category",
            "segment_and_region",
            "order_structure",
            "discount_and_profit",
            "loss_risk_modeling",
            "forecast_analysis",
            "country_market",
        }:
            continue
        findings = [
            _clean(item, 180)
            for item in module.findings[:4]
            if _clean(item, 180) and not _is_process_log(_clean(item, 180))
        ]
        metrics = {
            key: value
            for key, value in (module.summary_metrics or {}).items()
            if key in MODELING_METRIC_KEYS
            or key.endswith("_rate")
            or key.endswith("_share")
            or key in {"total_sales_amount", "total_profit_amount", "monthly_volatility"}
        }
        modules.append(
            {
                "module_id": module_id,
                "title": module.title,
                "summary_metrics": metrics,
                "findings": findings,
            }
        )
    return modules[:8]


def _compact_modeling_outcome(modeling_outcome: dict[str, Any] | None) -> dict[str, Any]:
    outcome = modeling_outcome or {}
    metrics = outcome.get("metrics_summary") if isinstance(outcome.get("metrics_summary"), dict) else {}
    if not metrics and isinstance(outcome.get("summary_metrics"), dict):
        metrics = outcome.get("summary_metrics") or {}
    return {
        "primary_modeling_task": outcome.get("primary_modeling_task"),
        "modeling_status": outcome.get("modeling_status"),
        "modeling_value_level": outcome.get("modeling_value_level"),
        "metrics": {key: metrics.get(key) for key in MODELING_METRIC_KEYS if key in metrics},
        "main_findings": outcome.get("main_findings") or [],
        "limitations": outcome.get("limitations") or [],
    }


def _compact_chart_selection_plan(chart_selection_plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = chart_selection_plan or {}
    selected = plan.get("selected_charts") or plan.get("selected") or []
    compact = []
    if isinstance(selected, list):
        for item in selected[:8]:
            if not isinstance(item, dict):
                continue
            compact.append(
                {
                    key: item.get(key)
                    for key in ("chart_id", "section_id", "business_question", "reason", "title")
                    if item.get(key)
                }
            )
    return {"selected_charts": compact}


def build_final_synthesis_context(
    *,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
    modeling_outcome: dict[str, Any] | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    action_plan_rows: list[dict[str, Any]] | None,
    chart_selection_plan: dict[str, Any] | None,
    chart_contexts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    focus = analysis_focus or {}
    return {
        "dataset_type": report.dataset_type,
        "mapped_fields": sorted(set(schema_mapping.field_mapping.values())),
        "dataset_profile": _compact_dict(dataset_profile, PROFILE_KEYS),
        "analysis_focus": {
            "selected_focuses": focus.get("selected_focuses") or [],
            "dominant_story": focus.get("dominant_story"),
            "section_priority": focus.get("section_priority"),
        },
        "evidence_pack": _compact_evidence_pack(evidence_pack),
        "report_modules": _compact_modules(report),
        "modeling_outcome": _compact_modeling_outcome(modeling_outcome),
        "modeling_interpretation": modeling_outcome_interpretation or {},
        "action_plan_rows": (action_plan_rows or [])[:6],
        "chart_selection_plan": _compact_chart_selection_plan(chart_selection_plan),
        "chart_contexts": (chart_contexts or [])[:6],
    }


def _validate_items(
    items: Any,
    *,
    text_key: str,
    evidence_keys: tuple[str, ...],
    action: bool,
    counters: dict[str, int],
    limit: int,
    mapped_fields: set[str],
) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    cleaned_items: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        text = _clean(raw.get(text_key), 260)
        if not text:
            continue
        joined = " ".join(_clean(value, 180) for value in raw.values())
        if _is_process_log(joined):
            counters["dropped_process_log_bullets_count"] += 1
            continue
        if _has_unsupported_claim(joined, mapped_fields):
            counters["dropped_unsupported_bullets_count"] += 1
            continue
        if action and _is_generic_action(joined):
            counters["dropped_generic_bullets_count"] += 1
            continue
        evidence_present = any(_clean(raw.get(key), 220) for key in evidence_keys)
        if action and not evidence_present:
            counters["dropped_missing_evidence_actions_count"] += 1
            continue
        key = text.casefold()
        if key in seen:
            counters["dropped_duplicate_bullets_count"] += 1
            continue
        seen.add(key)
        item = {text_key: text}
        for key in evidence_keys:
            value = _clean(raw.get(key), 220)
            if value:
                item[key] = value
        if action and "linked_metric_or_segment" in evidence_keys:
            display_text = _valid_display_text(raw.get("display_text"))
            if display_text:
                item["display_text"] = display_text
                counters["final_action_display_text_used_count"] += 1
            else:
                if _clean(raw.get("display_text"), 360):
                    counters["final_action_display_text_rejected_count"] += 1
                else:
                    counters["final_action_display_text_missing_count"] += 1
                counters["final_action_display_text_fallback_count"] += 1
        cleaned_items.append(item)
        if len(cleaned_items) >= limit:
            break
    return cleaned_items


def _backfill_brief_actions_from_recommendations(cleaned: dict[str, Any]) -> None:
    if len(cleaned["brief_actions"]) >= 2:
        return
    seen = {item.get("action", "").casefold() for item in cleaned["brief_actions"]}
    for item in cleaned["recommended_actions"]:
        action = item.get("action", "")
        if not action or action.casefold() in seen:
            continue
        cleaned["brief_actions"].append(
            {
                "action": action,
                "linked_evidence": item.get("linked_metric_or_segment", ""),
                "priority_reason": item.get("expected_use", ""),
            }
        )
        seen.add(action.casefold())
        if len(cleaned["brief_actions"]) >= 2:
            break


def _metric(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _format_rate(value: Any) -> str:
    if isinstance(value, (int, float)):
        rate = float(value)
        if 0 < rate < 0.001:
            return "<0.1%"
        return f"{rate * 100:.2f}".rstrip("0").rstrip(".") + "%"
    return str(value)


def _contains_any(items: list[dict[str, str]], terms: tuple[str, ...]) -> bool:
    text = "\n".join(" ".join(item.values()) for item in items)
    return any(term in text for term in terms)


def _append_unique(items: list[dict[str, str]], item: dict[str, str], text_key: str, limit: int) -> bool:
    key = item.get(text_key, "").casefold()
    if not key or any(existing.get(text_key, "").casefold() == key for existing in items):
        return False
    if len(items) >= limit:
        return False
    items.append(item)
    return True


def _ensure_required_modeling_coverage(cleaned: dict[str, Any], context: dict[str, Any]) -> None:
    modeling = context.get("modeling_outcome") if isinstance(context.get("modeling_outcome"), dict) else {}
    metrics = modeling.get("metrics") if isinstance(modeling.get("metrics"), dict) else {}
    task = str(modeling.get("primary_modeling_task") or "")
    status = str(modeling.get("modeling_status") or metrics.get("model_quality_status") or "").lower()
    added = False
    if task == "loss_risk_classification" and "weak" in status:
        positives = _metric(metrics, "test_positive_count")
        positive_rate = _metric(metrics, "target_positive_rate")
        if not _contains_any(cleaned["main_conclusions"], ("正例", "探索性参考", "不建议直接用于复核排序")):
            added = _append_unique(
                cleaned["main_conclusions"],
                {
                    "conclusion": "亏损风险模型只能作为探索性参考，不建议直接用于复核排序或自动决策。",
                    "evidence": f"测试集正例={_format_number(positives) if positives is not None else '不足'}，亏损占比={_format_rate(positive_rate) if positive_rate is not None else '极低'}。",
                    "business_meaning": "经营判断应优先回到类目、国家市场和利润质量等 EDA 证据。",
                },
                "conclusion",
                5,
            ) or added
        if not _contains_any(cleaned["recommended_actions"], ("不建议直接用模型", "EDA", "业务复盘")):
            added = _append_unique(
                cleaned["recommended_actions"],
                {
                    "action": "不建议直接用于复核排序或自动决策，先复盘低利润类目、国家市场和利润质量异常切片。",
                    "linked_metric_or_segment": f"测试集正例={_format_number(positives) if positives is not None else '不足'}，模型质量={status or 'weak'}。",
                    "expected_use": "把人工精力放在更可靠的经营切片复盘上，等待正例标签更稳定后再评估模型。",
                    "display_text": f"由于测试集正例只有{_format_number(positives) if positives is not None else '不足'}且模型质量为 {status or 'weak'}，当前亏损风险模型只适合作为探索性参考，不建议直接用于复核排序或自动决策；应先回到低利润类目、国家市场和盈利质量异常切片做业务复盘。",
                },
                "action",
                5,
            ) or added
    elif task == "loss_risk_classification":
        fp = _metric(metrics, "false_positive_count")
        fn = _metric(metrics, "false_negative_count")
        recall = _metric(metrics, "best_recall")
        model = _metric(metrics, "best_model") or "亏损风险模型"
        if not _contains_any(cleaned["recommended_actions"], ("阈值", "误报", "漏判", "复核量")):
            added = _append_unique(
                cleaned["recommended_actions"],
                {
                    "action": "用模型高风险订单建立人工复核清单，并按多档阈值模拟复核量、误报和漏判后再确定上线口径。",
                    "linked_metric_or_segment": f"{model} recall={_format_number(recall) if recall is not None else '可用'}，误报={_format_number(fp) if fp is not None else '需复核'}，漏判={_format_number(fn) if fn is not None else '需复核'}。",
                    "expected_use": "把模型用作人工复核排序工具，而不是自动拦截或自动决策依据。",
                },
                "action",
                5,
            ) or added
    if task in {"sales_amount_forecast_baseline", "sales_amount_regression"}:
        worst = _metric(metrics, "worst_error_periods")
        period = ""
        if isinstance(worst, list) and worst and isinstance(worst[0], dict):
            period = str(worst[0].get("period") or "")
        if not _contains_any(cleaned["recommended_actions"], ("误差最大", "节假日", "促销", "库存", "价格")):
            added = _append_unique(
                cleaned["recommended_actions"],
                {
                    "action": "对预测误差最大的时间段回查促销、节假日、大单或异常订单，并补充价格、库存、渠道等字段后重新评估模型。",
                    "linked_metric_or_segment": f"最大误差周期={period or '回测误差 Top 周期'}，R2={_format_number(_metric(metrics, 'best_model_r2')) if _metric(metrics, 'best_model_r2') is not None else '偏低'}。",
                    "expected_use": "判断误差来自真实经营事件还是模型缺少解释变量，避免把当前模型包装成生产级预测。",
                },
                "action",
                5,
            ) or added
    if added:
        metadata = cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else {}
        metadata["final_synthesis_fallback_reason"] = "quality_gate_backfilled_required_modeling_coverage"
        metadata["quality_gate_backfilled_required_modeling_coverage"] = True
        cleaned["metadata"] = metadata


def _validate_llm_payload(payload: dict[str, Any], *, mapped_fields: set[str]) -> dict[str, Any]:
    counters = {
        "dropped_generic_bullets_count": 0,
        "dropped_process_log_bullets_count": 0,
        "dropped_missing_evidence_actions_count": 0,
        "dropped_duplicate_bullets_count": 0,
        "dropped_unsupported_bullets_count": 0,
        "final_action_display_text_missing_count": 0,
        "final_action_display_text_rejected_count": 0,
        "final_action_display_text_fallback_count": 0,
        "final_action_display_text_used_count": 0,
        "final_action_deduped_count": 0,
        "final_action_render_mode": "not_rendered",
    }
    cleaned = {
        "brief_findings": _validate_items(
            payload.get("brief_findings"),
            text_key="finding",
            evidence_keys=("evidence", "business_meaning"),
            action=False,
            counters=counters,
            limit=4,
            mapped_fields=mapped_fields,
        ),
        "brief_actions": _validate_items(
            payload.get("brief_actions"),
            text_key="action",
            evidence_keys=("linked_evidence", "priority_reason"),
            action=True,
            counters=counters,
            limit=4,
            mapped_fields=mapped_fields,
        ),
        "main_conclusions": _validate_items(
            payload.get("main_conclusions"),
            text_key="conclusion",
            evidence_keys=("evidence", "business_meaning"),
            action=False,
            counters=counters,
            limit=5,
            mapped_fields=mapped_fields,
        ),
        "recommended_actions": _validate_items(
            payload.get("recommended_actions"),
            text_key="action",
            evidence_keys=("linked_metric_or_segment", "expected_use"),
            action=True,
            counters=counters,
            limit=5,
            mapped_fields=mapped_fields,
        ),
        "metadata": counters,
    }
    used = counters["final_action_display_text_used_count"]
    fallback = counters["final_action_display_text_fallback_count"]
    if used and fallback:
        counters["final_action_render_mode"] = "mixed"
    elif used:
        counters["final_action_render_mode"] = "display_text"
    elif fallback:
        counters["final_action_render_mode"] = "fallback_natural"
    _backfill_brief_actions_from_recommendations(cleaned)
    return cleaned


def _empty_synthesis(status: str, reason: str, *, used_llm: bool = False) -> dict[str, Any]:
    return {
        "brief_findings": [],
        "brief_actions": [],
        "main_conclusions": [],
        "recommended_actions": [],
        "metadata": {
            "final_synthesis_llm_status": status,
            "final_synthesis_used_llm": used_llm,
            "final_synthesis_fallback_reason": reason,
            "dropped_generic_bullets_count": 0,
            "dropped_process_log_bullets_count": 0,
            "dropped_missing_evidence_actions_count": 0,
            "dropped_duplicate_bullets_count": 0,
            "dropped_unsupported_bullets_count": 0,
            "final_action_display_text_missing_count": 0,
            "final_action_display_text_rejected_count": 0,
            "final_action_display_text_fallback_count": 0,
            "final_action_display_text_used_count": 0,
            "final_action_deduped_count": 0,
            "final_action_render_mode": "not_rendered",
        },
    }


def build_final_synthesis_with_trace(
    *,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
    modeling_outcome: dict[str, Any] | None = None,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    action_plan_rows: list[dict[str, Any]] | None = None,
    chart_selection_plan: dict[str, Any] | None = None,
    chart_contexts: list[dict[str, Any]] | None = None,
    llm_client=None,
    output_language: str | None = None,
) -> tuple[dict[str, Any], LLMStageTrace]:
    if not (llm_client is not None and getattr(llm_client, "enabled", False)):
        reason = "Final synthesis used deterministic fallback because no LLM client was enabled."
        return _empty_synthesis("disabled", reason), build_llm_stage_trace(
            stage=STAGE,
            llm_client=llm_client,
            status="disabled",
            reason=reason,
            attempted=False,
            applied=False,
        )
    context = build_final_synthesis_context(
        report=report,
        schema_mapping=schema_mapping,
        dataset_profile=dataset_profile,
        analysis_focus=analysis_focus,
        evidence_pack=evidence_pack,
        modeling_outcome=modeling_outcome,
        modeling_outcome_interpretation=modeling_outcome_interpretation,
        action_plan_rows=action_plan_rows,
        chart_selection_plan=chart_selection_plan,
        chart_contexts=chart_contexts,
    )
    try:
        payload = complete_json_for_stage(
            llm_client,
            system_prompt=f"{SYSTEM_PROMPT}\n\n{user_facing_language_instruction(output_language)}",
            user_payload={
                "final_synthesis_context": context,
                "language_instruction": user_facing_language_instruction(output_language),
            },
            cache_stage=STAGE,
        )
    except Exception as exc:
        reason = describe_llm_error(exc)
        return _empty_synthesis("fallback_on_error", reason), build_llm_stage_trace(
            stage=STAGE,
            llm_client=llm_client,
            status="fallback_on_error",
            reason=reason,
            attempted=True,
            applied=False,
        )
    if not isinstance(payload, dict):
        reason = "Final synthesis LLM payload was not a JSON object."
        return _empty_synthesis("fallback_invalid_payload", reason), build_llm_stage_trace(
            stage=STAGE,
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason=reason,
            attempted=True,
            applied=False,
        )
    cleaned = _validate_llm_payload(
        payload,
        mapped_fields=set(context.get("mapped_fields") or []),
    )
    _ensure_required_modeling_coverage(cleaned, context)
    cleaned["metadata"].update(
        {
            "final_synthesis_llm_status": "llm_applied",
            "final_synthesis_used_llm": True,
            "final_synthesis_fallback_reason": cleaned["metadata"].get("final_synthesis_fallback_reason", ""),
        }
    )
    if is_english_output(output_language):
        cleaned = guard_english_final_synthesis(
            cleaned,
            report=report,
            schema_mapping=schema_mapping,
            dataset_profile=dataset_profile,
        )
    has_content = bool(
        cleaned["brief_findings"]
        or cleaned["brief_actions"]
        or cleaned["main_conclusions"]
        or cleaned["recommended_actions"]
    )
    if not has_content:
        reason = "Final synthesis LLM payload had no valid business bullets after quality gates."
        return _empty_synthesis("fallback_invalid_payload", reason), build_llm_stage_trace(
            stage=STAGE,
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason=reason,
            attempted=True,
            applied=False,
        )
    return cleaned, build_llm_stage_trace(
        stage=STAGE,
        llm_client=llm_client,
        status="llm_applied",
        reason="LLM generated final Notebook summary and conclusion; deterministic rules only validated structure and safety.",
        attempted=True,
        applied=True,
    )
