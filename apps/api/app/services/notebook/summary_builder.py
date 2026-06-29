from __future__ import annotations

import re
from typing import Any

from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping

from app.services.final_synthesis_guard import guard_english_final_synthesis
from app.services.notebook.action_plan_builder import (
    action_plan_rows,
    dedupe_action_texts,
    evidence_action_rows,
    focus_actions,
    focus_evidence_items,
    focus_risk_items,
    profile_evidence_items,
)
from app.services.notebook.capability_matrix import capability_rows
from app.services.notebook.evidence_formatter import (
    append_unique_text,
    collect_evidence_items,
    evidence_pack_findings,
    evidence_pack_risks,
    evidence_text_id,
    remove_loss_evidence_when_no_negative_profit,
)
from app.services.notebook.formatters import format_percent, format_scalar
from app.services.notebook.markdown_sanitizer import clean_business_text
from app.services.notebook.report_utils import first_metric
from app.services.notebook_narrative_guard import remove_unsupported_narrative
from app.services.output_language import english_safe_bullets, is_english_output


_PROCESS_LOG_MARKERS = (
    "发现 0 条",
    "发现 1 条",
    "已使用",
    "已完成",
    "已按",
    "已覆盖",
    "共覆盖",
    "指标画像",
    "时间缺失记录",
    "本节汇总最重要的发现",
    "一份分析 notebook 的最终价值",
    "Top 商品为",
    "Top商品为",
    "订单数为",
    "峰值日期",
    "谷值日期",
    "建立复盘责任人",
    "字段数",
    "待确认字段数",
    "模块",
    "流程",
    "FALLBACK",
)


def _clean_bullet_candidates(
    items: list[Any],
    *,
    limit: int = 5,
    mapped_fields: set[str] | None = None,
) -> list[str]:
    bullets: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = clean_business_text(str(item or "")).strip(" -\n\t")
        text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
        if mapped_fields is not None:
            text, _ = remove_unsupported_narrative(text, mapped_fields=mapped_fields)
            text = text.strip()
        if not text or len(text) < 8:
            continue
        if re.search(r"发现\s*\d+\s*条", text):
            continue
        if re.search(r"(^|[，,；;]\s*)[A-Za-z_][A-Za-z0-9_]*=", text):
            continue
        if any(marker in text for marker in _PROCESS_LOG_MARKERS):
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(text)
        if len(bullets) >= limit:
            break
    return bullets


def _summary_metrics(outcome: dict[str, Any]) -> dict[str, Any]:
    for key in ("summary_metrics", "metrics_summary"):
        value = outcome.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _metric_value(metrics: dict[str, Any], outcome: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metrics.get(key)
        if value is None:
            value = outcome.get(key)
        if value is not None:
            return value
    return None


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _format_rate_hint(value: Any) -> str:
    if isinstance(value, (int, float)):
        rate = float(value)
        if 0 < rate < 0.001:
            return "<0.1%"
        if 0 < rate < 0.01:
            return f"{rate * 100:.1f}%"
    return format_percent(value)


def _modeling_action_bullets(modeling_outcome: dict[str, Any] | None) -> list[str]:
    outcome = modeling_outcome or {}
    metrics = _summary_metrics(outcome)
    task_type = str(
        outcome.get("primary_modeling_task")
        or outcome.get("task_type")
        or outcome.get("modeling_task_type")
        or ""
    )
    status = str(
        outcome.get("modeling_status")
        or outcome.get("status")
        or outcome.get("quality_status")
        or metrics.get("model_quality_status")
        or ""
    ).lower()
    weak = "weak" in status
    actions: list[str] = []
    if task_type == "loss_risk_classification" or "loss_risk" in task_type:
        if weak:
            actions.append(
                "不要直接用当前亏损风险模型做复核排序，先回到 EDA 中的利润、折扣、类目或区域异常做业务复盘，并补充稳定标签后再评估模型。"
            )
        else:
            actions.append(
                "用模型高风险订单建立人工复核清单，先按不同阈值模拟复核量、误报和漏判，再确定运营可承受的复核规则。"
            )
    if task_type in {"sales_amount_forecast_baseline", "sales_amount_regression"} or "sales_amount" in task_type:
        actions.append(
            "对销售预测误差最大的时间段回查促销、节假日、大单或异常订单，并补充价格、库存、渠道等字段后重新评估模型。"
        )
    return actions


def _fallback_action_candidates(
    mapped: set[str],
    selected_focuses: list[str],
    profile: dict[str, Any],
    modeling_outcome: dict[str, Any] | None,
) -> list[str]:
    candidates = _modeling_action_bullets(modeling_outcome)
    if {"discount", "profit"} <= mapped:
        candidates.append("优先筛出高折扣且利润为负或利润率偏低的订单，逐笔核查审批、定价、成本和履约费用。")
    if {"category", "sales_amount", "profit"} <= mapped:
        candidates.append("对高销售低利润类目建立单独复盘表，比较销售额、利润率和折扣结构后再调整资源倾斜。")
    if {"product_name", "sales_amount", "profit"} <= mapped:
        candidates.append("对高销售低利润商品单独核查定价、折扣和成本，避免只按销售额 Top 分配资源。")
    if {"order_datetime", "sales_amount"} <= mapped or "trend_volatility_focus" in selected_focuses:
        candidates.append("对销售波动或预测误差最大的月份/周回查促销、节假日、大单和异常订单，确认波动是否可解释。")
    if "country_market_focus" in selected_focuses:
        candidates.append("按国家或区域拆分销售额、订单数和客单价，优先复盘高占比市场是否存在集中度风险。")
    if "customer_order_structure_focus" in selected_focuses:
        candidates.append("按复购层级和订单篮子大小建立客户清单，分别提升复购客户占比和高价值订单占比。")
    candidates.append("补充促销、库存、成本、履约或渠道等字段后重新运行分析，用新增证据验证当前经营判断。")
    return candidates


def _modeling_conclusion_bullets(
    modeling_outcome: dict[str, Any] | None,
    modeling_outcome_interpretation: dict[str, str] | None,
) -> list[str]:
    outcome = modeling_outcome or {}
    interpretation = modeling_outcome_interpretation or {}
    bullets: list[str] = []
    display_type = str(outcome.get("display_type") or outcome.get("modeling_display_type") or "")
    task_type = str(
        outcome.get("primary_modeling_task")
        or outcome.get("task_type")
        or outcome.get("modeling_task_type")
        or ""
    )
    status = str(
        outcome.get("modeling_status")
        or outcome.get("status")
        or outcome.get("quality_status")
        or ""
    )
    metrics = _summary_metrics(outcome)
    best_model = _metric_value(metrics, outcome, "best_model")
    status_text = f"{display_type} {task_type} {status} {metrics.get('model_quality_status') or ''}".lower()
    weak = "weak" in status_text
    if "loss_risk" in display_type or "loss_risk" in task_type:
        metric_bits = []
        for label, key in [("precision", "best_precision"), ("recall", "best_recall"), ("F1", "best_f1"), ("ROC AUC", "best_roc_auc")]:
            value = _metric_value(metrics, outcome, key)
            if value is not None:
                metric_bits.append(f"{label}={_format_metric_value(value)}")
        if weak:
            positives = _metric_value(metrics, outcome, "test_positive_count", "positive_count")
            positive_rate = _metric_value(metrics, outcome, "target_positive_rate", "positive_rate")
            sample_hint = []
            if positives is not None:
                sample_hint.append(f"测试集正例={_format_metric_value(positives)}")
            if positive_rate is not None:
                sample_hint.append(f"亏损占比={_format_rate_hint(positive_rate)}")
            hint = f"（{'、'.join(sample_hint)}）" if sample_hint else ""
            bullets.append(
                f"亏损风险模型{hint}显示亏损样本或正例不足，模型稳定性不够，只能作为探索性参考；不建议直接用于复核排序或自动决策，经营判断应主要依赖 EDA 和业务复盘。"
            )
        else:
            model_text = f"{best_model} " if best_model else ""
            metric_text = f"，关键指标为 {'、'.join(metric_bits[:4])}" if metric_bits else ""
            threshold = _metric_value(metrics, outcome, "threshold_default", "default_threshold")
            fp = _metric_value(metrics, outcome, "false_positive_count")
            fn = _metric_value(metrics, outcome, "false_negative_count")
            tradeoff_bits = []
            if threshold is not None:
                tradeoff_bits.append(f"当前阈值={_format_metric_value(threshold)}")
            if fp is not None:
                tradeoff_bits.append(f"误报={_format_metric_value(fp)}")
            if fn is not None:
                tradeoff_bits.append(f"漏判={_format_metric_value(fn)}")
            tradeoff = f"；{'、'.join(tradeoff_bits)}说明阈值选择会影响复核量、误报和漏判" if tradeoff_bits else "；阈值选择需要在复核量、误报和漏判之间取舍"
            bullets.append(
                f"亏损风险模型 {model_text}可用于识别高风险订单并辅助人工复核排序，不是自动决策依据{metric_text}{tradeoff}。"
            )
    if "sales_regression" in display_type or "sales_amount" in task_type or "forecast" in task_type:
        metric_bits = []
        for label, keys in [
            ("MAE", ("best_model_mae", "best_mae")),
            ("R2", ("best_model_r2", "best_r2")),
            ("baseline 改善", ("improvement_vs_baseline", "mae_improvement_vs_baseline")),
            ("MAPE", ("best_model_mape", "best_mape")),
        ]:
            value = _metric_value(metrics, outcome, *keys)
            if value is not None:
                suffix = "%" if label == "baseline 改善" else ""
                metric_bits.append(f"{label}={_format_metric_value(value)}{suffix}")
        baseline_model = _metric_value(metrics, outcome, "baseline_model")
        r2_value = _metric_value(metrics, outcome, "best_model_r2", "best_r2")
        weak_regression = weak or (isinstance(r2_value, (int, float)) and r2_value < 0.2)
        model_text = f"{best_model or '当前最佳模型'}"
        baseline_text = f"相对 {baseline_model} baseline " if baseline_model else "相对 baseline "
        limit_text = "当前解释力偏弱，只适合作为 baseline/监控参考，不应包装成生产级预测能力" if weak_regression else "可作为销售监控和后续模型增强的对照，但仍不是生产级预测能力"
        bullets.append(
            f"销售额预测建模选择 {model_text}，{baseline_text}的回测表现为 {'、'.join(metric_bits[:4]) or '指标需结合明细复核'}；{limit_text}，后续需要补充节假日、促销、价格、库存、渠道等业务变量。"
        )
    for key in ("synthesis", "overall_interpretation", "summary", "final_synthesis"):
        if interpretation.get(key):
            bullets.append(interpretation[key])
    return _clean_bullet_candidates(bullets, limit=3)


def _final_synthesis_used_llm(final_synthesis: dict[str, Any] | None) -> bool:
    metadata = final_synthesis.get("metadata") if isinstance(final_synthesis, dict) else {}
    return bool(isinstance(metadata, dict) and metadata.get("final_synthesis_used_llm"))


def _join_synthesis_item(item: dict[str, Any], text_key: str, evidence_keys: tuple[str, ...]) -> str:
    lead = clean_business_text(item.get(text_key, ""))
    details = [clean_business_text(item.get(key, "")) for key in evidence_keys]
    details = [detail for detail in details if detail]
    return _soften_metric_assignments("；".join([lead, *details]).strip("；"))


def _soften_metric_assignments(text: str) -> str:
    softened = re.sub(r"\b([A-Za-z_][A-Za-z0-9_]*)=", r"\1 为 ", text)
    return softened.replace("探索性使用", "探索性参考")


def _final_synthesis_brief_findings(final_synthesis: dict[str, Any] | None) -> list[str]:
    if not _final_synthesis_used_llm(final_synthesis):
        return []
    return _clean_bullet_candidates(
        [
            _join_synthesis_item(item, "finding", ("evidence", "business_meaning"))
            for item in final_synthesis.get("brief_findings", [])
            if isinstance(item, dict)
        ],
        limit=3,
    )


def _final_synthesis_brief_actions(final_synthesis: dict[str, Any] | None) -> list[str]:
    if not _final_synthesis_used_llm(final_synthesis):
        return []
    return _clean_bullet_candidates(
        [
            _join_synthesis_item(item, "action", ("linked_evidence", "priority_reason"))
            for item in final_synthesis.get("brief_actions", [])
            if isinstance(item, dict)
        ],
        limit=3,
    )


def _has_profit_field(schema_mapping: SchemaMapping | None) -> bool:
    mapped = set((schema_mapping.field_mapping or {}).values()) if schema_mapping else set()
    return bool({"profit", "cost", "margin", "gross_margin"} & mapped)


def _module(report: AnalysisReport, module_id: str) -> Any | None:
    return next((module for module in report.modules if module.module_id == module_id), None)


def _english_mapped_fields(schema_mapping: SchemaMapping | None) -> set[str]:
    return set((schema_mapping.field_mapping or {}).values()) if schema_mapping else set()


def _english_number(value: Any) -> float | None:
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _english_discount_risk_row(report: AnalysisReport) -> dict[str, Any]:
    module = _module(report, "discount_profit_analysis")
    rows = []
    if module is not None:
        rows = [
            dict(row)
            for row in module.tables.get("discount_threshold_candidates", [])
            if isinstance(row, dict)
        ]
    rows.sort(
        key=lambda row: (
            {"high": 0, "medium": 1, "low": 2}.get(str(row.get("risk_level")).lower(), 3),
            -(_english_number(row.get("negative_profit_rate")) or 0.0),
            _english_number(row.get("profit_margin")) if _english_number(row.get("profit_margin")) is not None else 999.0,
        )
    )
    return rows[0] if rows else {}


def _english_quality_fact(report: AnalysisReport) -> str | None:
    module = _module(report, "data_quality_check")
    metrics = module.summary_metrics if module else {}
    issues: list[str] = []
    for label, key in (
        ("missing order-date records", "missing_order_datetime"),
        ("negative sales records", "negative_sales_amount"),
        ("negative quantity records", "negative_quantity"),
        ("duplicate rows", "duplicate_rows"),
    ):
        value = _english_number(metrics.get(key))
        if value and value > 0:
            issues.append(f"{format_scalar(value)} {label}")
    if not issues:
        return None
    return "Data quality checks identified " + ", ".join(issues[:3]) + ", which should be resolved before deeper operational conclusions."


def _english_product_or_slice_fact(report: AnalysisReport) -> str | None:
    for module_id in ("product_contribution_analysis", "dimension_breakdown_analysis", "country_market_analysis"):
        module = _module(report, module_id)
        if module is None:
            continue
        metrics = module.summary_metrics or {}
        label = metrics.get("top_product") or metrics.get("top_category") or metrics.get("dimension")
        if label:
            return f"`{label}` is a priority review slice based on the current mapped fields."
        for rows in module.tables.values():
            if not rows:
                continue
            row = rows[0]
            if not isinstance(row, dict):
                continue
            label = (
                row.get("product_name")
                or row.get("category")
                or row.get("sub_category")
                or row.get("segment")
                or row.get("region")
                or row.get("country")
                or row.get("sku")
            )
            if label:
                return f"`{label}` is a priority review slice based on the current mapped fields."
    return None


def _english_fact_bullets(
    report: AnalysisReport,
    *,
    schema_mapping: SchemaMapping | None = None,
    dataset_profile: dict[str, Any] | None = None,
) -> list[str]:
    profile = dataset_profile or {}
    mapped = _english_mapped_fields(schema_mapping)
    total_sales = first_metric(report, "total_sales_amount") or profile.get("total_sales_amount")
    total_profit = (
        first_metric(report, "total_profit_amount", "total_profit")
        or profile.get("total_profit_amount")
        or profile.get("total_profit")
    )
    has_profit = "profit" in mapped if schema_mapping else total_profit is not None and profile.get("has_profit") is not False
    has_discount = "discount" in mapped if schema_mapping else bool(_english_discount_risk_row(report))
    facts: list[str] = []
    if total_sales is not None:
        facts.append(f"Total sales reached {format_scalar(total_sales)} based on the mapped sales field.")
    if has_profit and total_profit is not None and _english_number(total_sales):
        margin = (_english_number(total_profit) or 0.0) / (_english_number(total_sales) or 1.0)
        facts.append(f"Total profit was {format_scalar(total_profit)}, corresponding to a profit margin of {format_percent(margin)}.")
    risk_row = _english_discount_risk_row(report)
    if has_profit and has_discount and risk_row:
        bucket = risk_row.get("discount_bucket") or risk_row.get("threshold")
        loss_rate = risk_row.get("negative_profit_rate")
        margin = risk_row.get("profit_margin")
        if bucket:
            metric = f"loss rate {format_percent(loss_rate)}" if loss_rate is not None else f"profit margin {format_percent(margin)}"
            facts.append(f"The `{bucket}` discount tier was the main margin-risk area according to the current threshold analysis, with {metric}.")
    slice_fact = _english_product_or_slice_fact(report)
    if slice_fact:
        facts.append(slice_fact)
    quality_fact = _english_quality_fact(report)
    if quality_fact:
        facts.append(quality_fact)
    if not (has_profit and has_discount):
        missing = []
        if not has_profit:
            missing.append("profit")
        if not has_discount:
            missing.append("discount")
        if missing:
            facts.append(f"{' and '.join(item.capitalize() for item in missing)} fields were not mapped, so margin and discount-risk conclusions are out of scope.")
    return facts


def _is_generic_english_final_conclusion(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().split()).lower()
    if not normalized:
        return False
    if normalized.startswith("section purpose:"):
        return True
    if normalized.startswith("this section reviews"):
        return True
    if "mapped sales dataset" in normalized:
        return True
    if "available module evidence" in normalized or "available module outputs" in normalized:
        return True
    if "summarize key findings, risks, and next actions" in normalized:
        return True
    if (
        normalized.startswith("this section reviews")
        and "mapped" in normalized
        and "dataset" in normalized
        and ("available module evidence" in normalized or "available module outputs" in normalized)
    ):
        return True
    if normalized.startswith("this section provides supporting evidence"):
        return True
    if normalized.startswith("use the mapped dataset") and "module evidence" in normalized:
        return True
    if normalized.startswith("review the chart and table outputs before acting"):
        return True
    return False


def _english_action_bullets(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    modeling_outcome: dict[str, Any] | None = None,
) -> list[str]:
    mapped = _english_mapped_fields(schema_mapping)
    has_profit = "profit" in mapped
    has_discount = "discount" in mapped
    actions: list[str] = []
    if has_profit and has_discount and _english_discount_risk_row(report):
        actions.append("Review the supported high-risk discount tier against approval, pricing, cost, and fulfillment details.")
    elif has_profit:
        actions.append("Compare high-sales slices with profit outcomes before prioritizing growth actions.")
    else:
        actions.append("Add profit and discount fields before making margin, discount-risk, or what-if profit decisions.")
    if {"product_name", "category", "sub_category", "sku"} & mapped:
        actions.append("Review product, category, or SKU concentration to confirm whether sales are concentrated in a small set of items.")
    if {"segment", "region", "country", "city"} & mapped:
        actions.append("Compare available segment and location slices to separate structural sales patterns from one-off spikes.")
    if modeling_outcome:
        actions.append("Use modeling outputs only as human-review support and validate the boundary before operational use.")
    actions.append("Assign owners, review cadence, and acceptance metrics for each follow-up slice.")
    return actions[:5]


def _sanitize_no_profit_final_text(text: str, *, has_profit_field: bool) -> str:
    if has_profit_field:
        return text
    sanitized = text
    if re.search(r"销售-利润|销售与利润|利润质量分析|销售利润", sanitized):
        if "负销售额记录" in sanitized:
            count_match = re.search(r"(\d+)\s*条负销售额记录", sanitized)
            category_match = re.search(r"(?:基于|category_count\s*)(\d+)\s*个?类目", sanitized)
            evidence_note = ""
            if count_match or category_match:
                notes = []
                if count_match:
                    notes.append(f"{count_match.group(1)} 条")
                if category_match:
                    notes.append(f"{category_match.group(1)} 个类目")
                evidence_note = f"（{'、'.join(notes)}）"
            return (
                f"先核查负销售额记录的业务含义{evidence_note}，判断它们是退货、冲销、录入错误还是特殊交易；"
                "当前只能基于销售额、时间和类目/产品字段做销售结构和预测复盘；"
                "如需判断盈利表现，应先补充盈利相关、成本、退货等字段"
            )
        negative_sales_re = r"(?:当前数据存在\s*)?(\d+\s*条)?负销售额记录[^；。]*"
        sanitized = re.sub(
            negative_sales_re,
            "先核查负销售额记录的业务含义，判断它们是退货、冲销、录入错误还是特殊交易",
            sanitized,
        )
        sanitized = re.sub(r"类目级销售-利润(?:交叉)?分析", "类目级销售结构复盘", sanitized)
        sanitized = re.sub(r"销售-利润(?:交叉)?分析", "销售结构与异常销售记录复盘", sanitized)
        sanitized = re.sub(r"销售与利润(?:交叉)?分析", "销售结构与异常销售记录复盘", sanitized)
        sanitized = sanitized.replace("利润质量分析需要先补充利润、成本或毛利字段", "如需判断盈利表现，应先补充盈利相关、成本、退货等字段")
        sanitized = sanitized.replace("核查利润质量分析", "核查负销售额记录口径")
        sanitized = sanitized.replace("利润质量分析", "利润质量判断")
        sanitized = sanitized.replace("利润管理", "销售质量管理")
        if "盈利相关、成本、退货等字段" not in sanitized:
            sanitized = sanitized.rstrip("；。")
            sanitized = f"{sanitized}；如需判断盈利表现，应先补充盈利相关、成本、退货等字段"
    sanitized = re.sub(r"profit", "盈利相关字段", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"margin", "盈利率字段", sanitized, flags=re.IGNORECASE)
    sanitized = sanitized.replace("利润率", "盈利率")
    sanitized = sanitized.replace("利润质量", "盈利质量")
    sanitized = sanitized.replace("毛利", "盈利")
    sanitized = sanitized.replace("利润", "盈利")
    return sanitized


def _action_category(text: str) -> str | None:
    if any(marker in text for marker in ("高风险订单", "模型高风险", "人工复核", "复核清单", "复核队列", "阈值", "误报", "漏判", "复核量")):
        return "model_review"
    if any(marker in text for marker in ("预测模型", "销售额预测", "baseline", "MAE", "R2", "R²")):
        return "forecast_model_use"
    if any(marker in text for marker in ("补充", "补齐", "新增字段", "缺少字段", "价格", "库存", "渠道", "促销", "节假日", "利润字段", "成本字段")):
        return "field_supplement"
    if any(marker in text for marker in ("预测误差", "误差最大", "回查", "异常订单", "大单")):
        return "forecast_error_review"
    if any(marker in text for marker in ("折扣", "高折扣", "利润为负", "低利润", "亏损订单", "30%+")):
        return "discount_profit_review"
    return None


def _action_specificity_score(text: str, category: str | None) -> int:
    score = min(len(text), 180)
    score += 8 * len(re.findall(r"\d+(?:\.\d+)?%?|\d+(?:\.\d+)?/\d+(?:\.\d+)?", text))
    if category == "model_review":
        score += 18 * sum(marker in text for marker in ("阈值", "误报", "漏判", "复核量"))
    if category == "field_supplement":
        score += 8 * sum(marker in text for marker in ("节假日", "促销", "价格", "库存", "渠道", "退货", "成本"))
    if category == "forecast_model_use":
        score += 10 * sum(marker in text for marker in ("baseline", "MAE", "R2", "R²", "不用于实际业务决策", "监控"))
    if category == "forecast_error_review":
        score += 12 * sum(marker in text for marker in ("误差最大", "回查", "节假日", "促销", "大单", "异常订单"))
    if category == "discount_profit_review":
        score += 14 * sum(marker in text for marker in ("30%+", "利润为负", "审批", "定价", "成本", "Furniture"))
    return score


def _dedupe_final_recommended_actions(actions: list[str]) -> list[str]:
    selected_by_category: dict[str, tuple[int, int, str]] = {}
    passthrough: list[tuple[int, str]] = []
    for index, action in enumerate(actions):
        category = _action_category(action)
        if not category:
            passthrough.append((index, action))
            continue
        score = _action_specificity_score(action, category)
        current = selected_by_category.get(category)
        if current is None or score > current[0]:
            selected_by_category[category] = (score, index, action)

    indexed_actions = [(index, action) for _, index, action in selected_by_category.values()]
    indexed_actions.extend(passthrough)
    indexed_actions.sort(key=lambda item: item[0])
    return [action for _, action in indexed_actions]


def _strip_sentence_end(text: str) -> str:
    return str(text or "").strip().strip("；;。.")


def _looks_like_mechanical_action_join(text: str, item: dict[str, Any]) -> bool:
    semicolon_count = text.count("；") + text.count(";")
    if semicolon_count >= 2:
        return True
    parts = [
        clean_business_text(item.get("action", "")),
        clean_business_text(item.get("linked_metric_or_segment", "")),
        clean_business_text(item.get("expected_use", "")),
    ]
    matched = sum(bool(part and part in text) for part in parts)
    return matched >= 3 and semicolon_count >= 1


def _valid_recommended_action_display_text(
    item: dict[str, Any],
    *,
    has_profit_field: bool,
) -> str:
    display_text = _sanitize_no_profit_final_text(
        clean_business_text(item.get("display_text", "")),
        has_profit_field=has_profit_field,
    ).strip()
    if not display_text or len(display_text) < 20:
        return ""
    if re.search(r"发现\s*\d+\s*条", display_text) or any(marker in display_text for marker in _PROCESS_LOG_MARKERS):
        return ""
    if any(marker in display_text for marker in ("加强管理", "持续关注", "提升效率", "优化经营", "建立机制")):
        return ""
    if _looks_like_mechanical_action_join(display_text, item):
        return ""
    return display_text


def _fallback_recommended_action_text(
    item: dict[str, Any],
    *,
    has_profit_field: bool,
) -> str:
    action = _strip_sentence_end(
        _sanitize_no_profit_final_text(
            clean_business_text(item.get("action", "")),
            has_profit_field=has_profit_field,
        )
    )
    evidence = _strip_sentence_end(
        _sanitize_no_profit_final_text(
            clean_business_text(item.get("linked_metric_or_segment", "")),
            has_profit_field=has_profit_field,
        )
    )
    expected = _strip_sentence_end(
        _sanitize_no_profit_final_text(
            clean_business_text(item.get("expected_use", "")),
            has_profit_field=has_profit_field,
        )
    )
    if not action:
        return ""
    if action.startswith(("建议", "不要", "先", "优先")):
        text = action
    else:
        text = f"建议{action}"
    if evidence:
        text = f"{text}，依据是{evidence}"
    if expected:
        text = f"{text}，用于{expected}"
    return text.rstrip("，；。") + "。"


def _render_recommended_action_text(
    item: dict[str, Any],
    *,
    has_profit_field: bool,
) -> tuple[str, str]:
    if not isinstance(item, dict):
        return "", "empty"
    if clean_business_text(item.get("display_text", "")):
        display_text = _valid_recommended_action_display_text(
            item,
            has_profit_field=has_profit_field,
        )
        if display_text:
            return display_text, "display_text"
        return _fallback_recommended_action_text(item, has_profit_field=has_profit_field), "rejected_fallback"
    return _fallback_recommended_action_text(item, has_profit_field=has_profit_field), "missing_fallback"


def _update_final_action_render_trace(
    final_synthesis: dict[str, Any] | None,
    *,
    used_count: int,
    missing_count: int,
    fallback_count: int,
    rejected_count: int,
    deduped_count: int,
) -> None:
    if not isinstance(final_synthesis, dict):
        return
    metadata = final_synthesis.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        final_synthesis["metadata"] = metadata
    metadata["final_action_display_text_used_count"] = used_count
    metadata["final_action_display_text_missing_count"] = missing_count
    metadata["final_action_display_text_fallback_count"] = fallback_count
    metadata["final_action_display_text_rejected_count"] = rejected_count
    metadata["final_action_deduped_count"] = deduped_count
    if used_count and fallback_count:
        render_mode = "mixed"
    elif used_count:
        render_mode = "display_text"
    elif fallback_count:
        render_mode = "fallback_natural"
    else:
        render_mode = "empty"
    metadata["final_action_render_mode"] = render_mode


def _final_synthesis_sales_regression_metric_hint(
    final_synthesis: dict[str, Any] | None,
    *,
    has_profit_field: bool,
) -> str:
    if not isinstance(final_synthesis, dict):
        return ""
    candidates: list[str] = []
    for key in ("brief_findings", "brief_actions"):
        for item in final_synthesis.get(key, []):
            if not isinstance(item, dict):
                continue
            candidates.extend(str(item.get(field, "")) for field in ("evidence", "linked_evidence"))
    for candidate in candidates:
        text = _sanitize_no_profit_final_text(clean_business_text(candidate), has_profit_field=has_profit_field)
        if ("R²" in text or "R2" in text) and ("MAE" in text or "baseline" in text):
            return text.strip("；。")
    return ""


def _modeling_sales_regression_metric_hint(modeling_outcome: dict[str, Any] | None) -> str:
    outcome = modeling_outcome or {}
    task = str(outcome.get("primary_modeling_task") or "")
    if task not in {"sales_amount_forecast_baseline", "sales_amount_regression"}:
        return ""
    metrics = _summary_metrics(outcome)
    r2 = _metric_value(metrics, outcome, "best_model_r2", "best_r2")
    mae = _metric_value(metrics, outcome, "best_model_mae", "best_mae")
    baseline = _metric_value(metrics, outcome, "baseline_model")
    baseline_mae = _metric_value(metrics, outcome, "baseline_mae")
    improvement = _metric_value(metrics, outcome, "improvement_vs_baseline")
    parts: list[str] = []
    if r2 is not None:
        r2_text = format_percent(r2) if isinstance(r2, (int, float)) and abs(float(r2)) <= 1 else _format_metric_value(r2)
        parts.append(f"R²={r2_text}")
    if mae is not None:
        parts.append(f"MAE={_format_metric_value(mae)}")
    if baseline:
        baseline_text = f"{baseline} baseline"
        if baseline_mae is not None:
            baseline_text = f"{baseline_text} MAE={_format_metric_value(baseline_mae)}"
        parts.append(baseline_text)
    if improvement is not None:
        parts.append(f"相对 baseline 改善 {_format_metric_value(improvement)}%")
    return "，".join(parts)


def _attach_sales_regression_metric_hint(
    actions: list[str],
    final_synthesis: dict[str, Any] | None,
    *,
    has_profit_field: bool,
    modeling_outcome: dict[str, Any] | None = None,
) -> list[str]:
    combined = "\n".join(actions)
    if ("R²" in combined or "R2" in combined) and "MAE" in combined and "baseline" in combined:
        return actions
    hint = _final_synthesis_sales_regression_metric_hint(
        final_synthesis,
        has_profit_field=has_profit_field,
    ) or _modeling_sales_regression_metric_hint(modeling_outcome)
    if not hint:
        return actions
    updated = list(actions)
    for index, action in enumerate(updated):
        if _action_category(action) == "forecast_model_use":
            updated[index] = f"{action.rstrip('；。')}，模型指标参考为{hint}。"
            break
    return updated


def _render_final_synthesis_conclusion(
    final_synthesis: dict[str, Any] | None,
    *,
    schema_mapping: SchemaMapping | None = None,
    modeling_outcome: dict[str, Any] | None = None,
) -> str:
    has_profit = _has_profit_field(schema_mapping)
    conclusions = _clean_bullet_candidates(
        [
            _sanitize_no_profit_final_text(
                _join_synthesis_item(item, "conclusion", ("evidence", "business_meaning")),
                has_profit_field=has_profit,
            )
            for item in (final_synthesis or {}).get("main_conclusions", [])
            if isinstance(item, dict)
        ],
        limit=5,
    )
    rendered_actions: list[str] = []
    used_count = 0
    missing_count = 0
    fallback_count = 0
    rejected_count = 0
    for item in (final_synthesis or {}).get("recommended_actions", []):
        if not isinstance(item, dict):
            continue
        action_text, mode = _render_recommended_action_text(item, has_profit_field=has_profit)
        if mode == "display_text":
            used_count += 1
        elif mode == "missing_fallback":
            missing_count += 1
            fallback_count += 1
        elif mode == "rejected_fallback":
            rejected_count += 1
            fallback_count += 1
        if action_text:
            rendered_actions.append(action_text)
    actions = _clean_bullet_candidates(rendered_actions, limit=10)
    before_dedupe_count = len(actions)
    actions = _dedupe_final_recommended_actions(actions)[:5]
    deduped_count = max(0, before_dedupe_count - len(actions))
    actions = _attach_sales_regression_metric_hint(
        actions,
        final_synthesis,
        has_profit_field=has_profit,
        modeling_outcome=modeling_outcome,
    )
    _update_final_action_render_trace(
        final_synthesis,
        used_count=used_count,
        missing_count=missing_count,
        fallback_count=fallback_count,
        rejected_count=rejected_count,
        deduped_count=deduped_count,
    )
    lines = ["## 结论与行动建议", "", "### 主要结论", ""]
    for item in conclusions:
        lines.extend([f"- {item}", ""])
    lines.extend(["### 行动建议", ""])
    for item in actions:
        lines.extend([f"- {item}", ""])
    return "\n".join(lines).rstrip()


def _english_join_synthesis_item(
    item: dict[str, Any],
    primary_key: str,
    detail_keys: tuple[str, ...],
    *,
    prefer_display_text: bool = False,
) -> str:
    if prefer_display_text and item.get("display_text"):
        return clean_business_text(item.get("display_text", ""))
    parts = [clean_business_text(item.get(primary_key, ""))]
    parts.extend(clean_business_text(item.get(key, "")) for key in detail_keys)
    parts = [part for part in parts if part]
    text = " ".join(parts)
    if text and not re.search(r"[.!?]$", text):
        text += "."
    return text


def _render_english_final_synthesis_conclusion(
    final_synthesis: dict[str, Any],
    *,
    fallback_findings: list[str],
    fallback_actions: list[str],
) -> str:
    conclusions = [
        _english_join_synthesis_item(item, "conclusion", ("evidence", "business_meaning"))
        for item in final_synthesis.get("main_conclusions", [])
        if isinstance(item, dict)
    ]
    if not conclusions:
        conclusions = [
            _english_join_synthesis_item(item, "finding", ("evidence", "business_meaning"))
            for item in final_synthesis.get("brief_findings", [])
            if isinstance(item, dict)
        ]
    conclusions = english_safe_bullets(
        [item for item in conclusions if not _is_generic_english_final_conclusion(item)],
        fallback_findings,
        limit=5,
    )
    actions = [
        _english_join_synthesis_item(
            item,
            "action",
            ("linked_metric_or_segment", "expected_use"),
            prefer_display_text=True,
        )
        for item in final_synthesis.get("recommended_actions", [])
        if isinstance(item, dict)
    ]
    if not actions:
        actions = [
            _english_join_synthesis_item(item, "action", ("linked_evidence", "priority_reason"))
            for item in final_synthesis.get("brief_actions", [])
            if isinstance(item, dict)
        ]
    actions = english_safe_bullets(
        [item for item in actions if not _is_generic_english_final_conclusion(item)],
        fallback_actions,
        limit=5,
    )
    return "\n".join(
        [
            "## Conclusions and Recommendations",
            "",
            "### Main Conclusions",
            *[f"- {item}" for item in conclusions[:5]],
            "",
            "### Recommendations",
            *[f"- {item}" for item in actions[:5]],
            "",
            "### Usage Limits",
            "- Use this Notebook for business review and human judgment support; it does not replace experiments or controlled validation.",
            "- Keep raw dataset fields unchanged and validate metric definitions before operational use.",
        ]
    )


def _numbered_markdown_lines(items: list[str], *, isolate_items: bool = False) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item}")
        if isolate_items:
            lines.append("")
    if lines and not lines[-1]:
        lines.pop()
    return lines


def _bullet_markdown_lines(items: list[str], *, isolate_items: bool = False) -> list[str]:
    lines: list[str] = []
    for item in items:
        lines.append(f"- {item}")
        if isolate_items:
            lines.append("")
    if lines and not lines[-1]:
        lines.pop()
    return lines


def build_final_conclusion_markdown(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    analysis_focus: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
    *,
    narrative_intro: str | None = None,
    narrative_observations: list[str] | None = None,
    narrative_takeaway: str | None = None,
    content_blocks: list[str] | None = None,
    modeling_outcome: dict[str, Any] | None = None,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    final_synthesis: dict[str, Any] | None = None,
    output_language: str | None = None,
) -> str:
    if is_english_output(output_language):
        structured_facts = _english_fact_bullets(
            report,
            schema_mapping=schema_mapping,
            dataset_profile=dataset_profile,
        )
        fallback_findings = structured_facts or [
            f"The dataset was analyzed as `{report.dataset_type}` with {report.module_count} modules.",
            "Use chart-level evidence in the preceding sections before making operational decisions.",
        ]
        english_candidates = [
            item
            for item in [narrative_intro, *(narrative_observations or []), narrative_takeaway, *list(report.summary)]
            if not _is_generic_english_final_conclusion(str(item or ""))
        ]
        main_bullets = english_safe_bullets(
            english_candidates,
            fallback_findings,
            limit=5,
        )
        main_bullets = [
            item for item in main_bullets if not _is_generic_english_final_conclusion(item)
        ]
        main_bullets = (structured_facts + [item for item in main_bullets if item not in structured_facts])[:5]
        action_bullets = _english_action_bullets(
            report,
            schema_mapping,
            dataset_profile=dataset_profile,
            modeling_outcome=modeling_outcome,
        )
        if _final_synthesis_used_llm(final_synthesis):
            guarded_synthesis = guard_english_final_synthesis(
                final_synthesis,
                report=report,
                schema_mapping=schema_mapping,
                dataset_profile=dataset_profile,
            )
            if (
                guarded_synthesis.get("main_conclusions")
                or guarded_synthesis.get("recommended_actions")
                or guarded_synthesis.get("brief_findings")
                or guarded_synthesis.get("brief_actions")
            ):
                return _render_english_final_synthesis_conclusion(
                    guarded_synthesis,
                    fallback_findings=fallback_findings,
                    fallback_actions=action_bullets,
                )
        return "\n".join(
            [
                "## Conclusions and Recommendations",
                "",
                "### Main Conclusions",
                *[f"- {item}" for item in main_bullets[:5]],
                "",
                "### Recommendations",
                *[f"- {item}" for item in action_bullets[:5]],
                "",
                "### Usage Limits",
                "- Use this Notebook for business review and human judgment support; it does not replace experiments or controlled validation.",
                "- Keep raw dataset fields unchanged and validate metric definitions before operational use.",
            ]
        )
    if _final_synthesis_used_llm(final_synthesis):
        return _render_final_synthesis_conclusion(
            final_synthesis,
            schema_mapping=schema_mapping,
            modeling_outcome=modeling_outcome,
        )
    profile = dataset_profile or {}
    mapped = set(schema_mapping.field_mapping.values())
    selected_focuses = list((analysis_focus or {}).get("selected_focuses", []))
    evidence = remove_loss_evidence_when_no_negative_profit(collect_evidence_items(report), profile)
    report_summary = remove_loss_evidence_when_no_negative_profit(list(report.summary), profile)
    conclusion_candidates: list[Any] = []
    if narrative_intro:
        conclusion_candidates.append(narrative_intro)
    conclusion_candidates.extend(narrative_observations or [])
    if narrative_takeaway:
        conclusion_candidates.append(narrative_takeaway)
    conclusion_candidates.extend(content_blocks or [])
    modeling_bullets = _modeling_conclusion_bullets(modeling_outcome, modeling_outcome_interpretation)
    conclusion_candidates.extend(modeling_bullets)
    conclusion_candidates.extend(evidence_pack_findings(evidence_pack)[:4])
    conclusion_candidates.extend(evidence_pack_risks(evidence_pack)[:3])
    conclusion_candidates.extend(evidence[:5])
    conclusion_candidates.extend(report_summary[:5])

    main_bullets = _clean_bullet_candidates(conclusion_candidates, limit=5, mapped_fields=mapped)
    while len(main_bullets) < 3:
        fallback = "当前数据已经支持基础销售画像，但关键经营判断仍需要结合明细和业务口径复核。"
        if fallback in main_bullets:
            break
        main_bullets.append(fallback)

    rows = action_plan_rows(report, schema_mapping, dataset_profile, analysis_focus, evidence_pack)
    action_candidates = _modeling_action_bullets(modeling_outcome) + [
        f"{row.get('issue', '')}：{row.get('action', '')}".strip("：")
        for row in rows
    ]
    if selected_focuses:
        action_candidates.extend(focus_actions(profile, selected_focuses, evidence or report.summary))
    action_bullets = _clean_bullet_candidates(action_candidates, limit=5, mapped_fields=mapped)
    action_bullets.extend(
        item
        for item in _clean_bullet_candidates(
            _fallback_action_candidates(mapped, selected_focuses, profile, modeling_outcome),
            limit=5,
            mapped_fields=mapped,
        )
        if item not in action_bullets
    )
    while len(action_bullets) < 3:
        fallback = "补充促销、库存、成本、履约或渠道等字段后重新运行分析，用新增证据验证当前经营判断。"
        if fallback in action_bullets:
            break
        action_bullets.append(fallback)

    return "\n".join(
        [
            "## 结论与行动建议",
            "",
            "### 主要结论",
            *[f"- {item}" for item in main_bullets[:5]],
            "",
            "### 行动建议",
            "以下动作需结合已映射字段执行；缺少字段时先补齐口径，再进入业务复盘。",
            *[f"- {item}" for item in action_bullets[:5]],
        ]
    )


def build_kaggle_analysis_brief_markdown(
    report: AnalysisReport,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
    final_synthesis: dict[str, Any] | None = None,
    output_language: str | None = None,
) -> str:
    profile = dataset_profile or {}
    focus = analysis_focus or {}
    selected_focuses = [str(item) for item in focus.get("selected_focuses", []) if str(item).strip()]
    pack_findings = evidence_pack_findings(evidence_pack)
    total_sales = first_metric(report, "total_sales_amount") or profile.get("total_sales_amount")
    total_profit = (
        first_metric(report, "total_profit_amount", "total_profit")
        or profile.get("total_profit_amount")
        or profile.get("total_profit")
    )
    profit_margin = None
    if isinstance(total_sales, (int, float)) and isinstance(total_profit, (int, float)) and total_sales:
        profit_margin = float(total_profit) / float(total_sales)
    has_profit = profile.get("has_profit") is not False
    if is_english_output(output_language):
        kpis = [
            f"Total Sales: {format_scalar(total_sales) if total_sales is not None else 'not available from mapped fields'}",
            f"Order Count: {format_scalar(profile.get('order_count')) if profile.get('order_count') is not None else 'not available from mapped fields'}",
            f"Customer Count: {format_scalar(profile.get('customer_count')) if profile.get('customer_count') is not None else 'not available from mapped fields'}",
        ]
        if total_profit is not None:
            kpis[1:1] = [
                f"Total Profit: {format_scalar(total_profit) if total_profit is not None else 'not available from mapped fields'}",
                f"Profit Margin: {format_percent(profit_margin) if profit_margin is not None else 'not available from mapped fields'}",
            ]
        structured_facts = _english_fact_bullets(report, dataset_profile=profile)
        findings = english_safe_bullets(
            [*evidence_pack_findings(evidence_pack), *list(report.summary)],
            structured_facts
            or [
                f"The analysis completed {report.module_count} modules for the `{report.dataset_type}` dataset.",
                "The available mapped fields support a first-pass sales performance review.",
                "Additional business context should be used before turning insights into operating decisions.",
            ],
            limit=3,
        )
        findings = (structured_facts + [item for item in findings if item not in structured_facts])[:3]
        return "\n".join(
            [
                "## Notebook Analysis Summary",
                "",
                "### KPI",
                *_bullet_markdown_lines(kpis),
                "",
                "### Key Findings",
                "",
                *_numbered_markdown_lines(findings[:3]),
            ]
        ).rstrip()
    kpis = [
        f"总销售额：{format_scalar(total_sales) if total_sales is not None else '数据限制说明：缺少可验证字段'}",
        f"订单数：{format_scalar(profile.get('order_count')) if profile.get('order_count') is not None else '数据限制说明：缺少可验证字段'}",
        f"客户数：{format_scalar(profile.get('customer_count')) if profile.get('customer_count') is not None else '数据限制说明：缺少可验证字段'}",
    ]
    if has_profit:
        negative_profit_rate = profile.get("negative_profit_rate")
        negative_profit_text = (
            "0" if negative_profit_rate == 0 else format_percent(negative_profit_rate)
        )
        kpis[1:1] = [
            f"总利润：{format_scalar(total_profit) if total_profit is not None else '数据限制说明：缺少可验证字段'}",
            f"利润率：{format_percent(profit_margin) if profit_margin is not None else '数据限制说明：缺少可验证字段'}",
            f"负利润记录占比为 {negative_profit_text}",
        ]

    llm_findings = _final_synthesis_brief_findings(final_synthesis)
    findings = llm_findings[:3] if llm_findings else pack_findings[:3]
    report_evidence = remove_loss_evidence_when_no_negative_profit(collect_evidence_items(report), profile)
    if not llm_findings:
        for item in report_evidence:
            if len(findings) >= 3:
                break
            if item not in findings:
                findings.append(item)
        while len(findings) < 3:
            findings.append("数据限制说明：当前发现需要更多利润、折扣或切片字段继续验证。")

    focus_labels = {
        "discount_erosion_focus": "折扣侵蚀",
        "profit_quality_focus": "利润质量",
        "segment_region_focus": "客群区域切片",
        "product_concentration_focus": "商品集中度",
        "trend_volatility_focus": "月度波动",
        "customer_order_structure_focus": "订单与客户结构",
        "country_market_focus": "市场切片",
    }
    focus_text = "、".join(focus_labels.get(item, item) for item in selected_focuses) or "基础销售经营分析"
    story = str((evidence_pack or {}).get("dataset_signature", {}).get("dominant_story") or "").strip()
    if not story:
        story = "以字段画像和证据包为基础的销售表现诊断"

    lines = [
        "## Notebook 分析摘要",
        "",
        "### KPI",
        *_bullet_markdown_lines(kpis),
        "",
        "### 核心发现",
        "",
        *(
            _bullet_markdown_lines(findings[:3], isolate_items=True)
            if llm_findings
            else _numbered_markdown_lines(findings[:3])
        ),
    ]
    return "\n".join(lines).rstrip()


def build_management_summary_markdown(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    analysis_focus: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
) -> str:
    profile = dataset_profile or {}
    selected_focuses = list((analysis_focus or {}).get("selected_focuses", []))
    evidence = remove_loss_evidence_when_no_negative_profit(collect_evidence_items(report), profile)
    capability_status_rows = capability_rows(schema_mapping)
    disabled_limits = [
        row["data_limit"] for row in capability_status_rows if row["status"] == "disabled"
    ]

    pack_findings = evidence_pack_findings(evidence_pack)
    profile_evidence = profile_evidence_items(profile, selected_focuses)
    focus_evidence = [*profile_evidence, *focus_evidence_items(report, profile, selected_focuses)]
    findings = [*pack_findings[:3]]
    for item in focus_evidence if focus_evidence else evidence:
        if len(findings) >= 3:
            break
        if item not in findings:
            findings.append(item)
    while len(findings) < 3:
        findings.append(
            disabled_limits[len(findings) % len(disabled_limits)]
            if disabled_limits
            else "数据限制说明：当前可用字段支持基础销售分析，但缺少更多切片证据。"
        )

    risks = evidence_pack_risks(evidence_pack)[:3]
    seen_risk_ids = {evidence_text_id(item) for item in risks if evidence_text_id(item)}
    for item in focus_risk_items(report, profile, selected_focuses):
        if len(risks) >= 3:
            break
        append_unique_text(risks, item, seen_risk_ids)
    for row in capability_status_rows:
        if len(risks) >= 3:
            break
        if row["status"] == "disabled":
            append_unique_text(risks, row["data_limit"], seen_risk_ids)
    fallback_risks = [
        f"[profit_margin_spread] 利润率跨度为 {format_percent(profile.get('profit_margin_spread'))}，需要下钻利润质量差异。",
        "[high_sales_low_profit_products] 高销售低利润商品需要单独复盘，避免规模掩盖利润压力。",
        "[weak_segment_region_slice] 弱势组合切片需要继续拆到商品、类目和折扣策略。",
        f"[monthly_volatility] 月度波动为 {format_percent(profile.get('monthly_volatility'))}，需要复核经营节奏是否稳定。",
    ]
    for item in fallback_risks:
        if len(risks) >= 3:
            break
        append_unique_text(risks, item, seen_risk_ids)
    while len(risks) < 3:
        risks.append(
            "数据限制说明：需要结合利润、折扣、商品或切片字段复核当前发现的经营风险。"
        )

    action_evidence = evidence or findings
    actions = [
        row["action"]
        for row in evidence_action_rows(evidence_pack, selected_focuses)[:3]
    ]
    if len(actions) < 3:
        actions.extend(focus_actions(profile, selected_focuses, action_evidence))
    if not actions:
        actions = [
        f"复盘证据项：{action_evidence[0]}，明确负责人和修正时间表。",
        f"调整高风险商品、折扣或切片的经营策略，并用 {action_evidence[min(1, len(action_evidence) - 1)]} 作为验收基线。",
        f"建立周度追踪看板，持续检查 {action_evidence[min(2, len(action_evidence) - 1)]} 是否改善。",
        ]
    actions = dedupe_action_texts(actions, profile)

    return "\n".join(
        [
            "## Management Summary",
            "",
            "### 核心发现",
            *[f"{index}. {item}" for index, item in enumerate(findings[:3], start=1)],
            "",
            "### 主要风险",
            *[f"{index}. {item}" for index, item in enumerate(risks[:3], start=1)],
            "",
            "### 优先行动",
            *[f"{index}. {item}" for index, item in enumerate(actions[:3], start=1)],
        ]
    )


_kaggle_analysis_brief_markdown = build_kaggle_analysis_brief_markdown
_management_summary_markdown = build_management_summary_markdown
