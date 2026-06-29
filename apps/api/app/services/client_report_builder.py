from __future__ import annotations

import json
import re
from html import escape
from math import isfinite
from typing import Any

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.evidence_guard import downgrade_strong_inference_claims
from app.services.final_synthesis_guard import (
    build_scoped_evidence_rows,
    guard_english_final_synthesis,
    violates_scoped_evidence,
)
from app.services.llm_client import complete_json_for_stage
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.notebook.action_plan_builder import action_plan_rows
from app.services.output_language import (
    contains_cjk,
    english_safe_bullets,
    englishize_payload_strings,
    is_english_output,
    user_facing_language_instruction,
)

SCENARIO_LIMITATION = "这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。"
MODEL_LIMITATION = "模型只用于人工复核优先级排序，不代表因果关系，不用于自动决策。"


def _module(report: AnalysisReport, module_id: str) -> ModuleReport | None:
    return next((module for module in report.modules if module.module_id == module_id), None)


def _clean_text(value: object, fallback: str = "暂无") -> str:
    text = str(value).strip() if value is not None else ""
    blocked = {"", "none", "nan", "unknown"}
    if text.lower() in blocked:
        return fallback
    return text.replace("%%", "%")


def _guard_text(value: object, fallback: str = "暂无") -> str:
    return downgrade_strong_inference_claims(_clean_text(value, fallback))


def _guard_payload_text(payload: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(payload)
    for key in ("business_theme",):
        if key in guarded:
            guarded[key] = _guard_text(guarded.get(key))
    for key in ("executive_summary", "limitations"):
        value = guarded.get(key)
        if isinstance(value, list):
            guarded[key] = [_guard_text(item) for item in value]
    findings = guarded.get("key_findings")
    if isinstance(findings, dict):
        guarded["key_findings"] = {
            section: [_guard_text(item) for item in items]
            for section, items in findings.items()
            if isinstance(items, list)
        }
    actions = guarded.get("priority_actions")
    if isinstance(actions, list):
        guarded["priority_actions"] = [
            {
                **row,
                "issue": _guard_text(row.get("issue", "")),
                "evidence": _guard_text(row.get("evidence", "")),
                "action": _guard_text(row.get("action", "")),
            }
            if isinstance(row, dict)
            else row
            for row in actions
        ]
    return guarded


def _number(value: object) -> float | None:
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _format_number(value: object, *, percent: bool = False, currency: bool = False) -> str:
    numeric = _number(value)
    if numeric is None:
        return "暂无"
    if percent:
        return f"{numeric:.2%}"
    if currency:
        return f"{numeric:,.2f}"
    if abs(numeric) >= 1000:
        return f"{numeric:,.2f}"
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _normalize_negative_zero_display(value: object) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"-0(?:\.0+)?(%?)", text)
    if match:
        return f"0{match.group(1)}"
    return text


def _format_display_number(value: object, *, percent: bool = False, currency: bool = False) -> str:
    return _normalize_negative_zero_display(_format_number(value, percent=percent, currency=currency))


CLIENT_REPORT_ENGLISH_REPLACEMENTS = {
    "最终Conclusion": "Business Observation",
    "经营观察": "Business Observation",
    "核心摘要": "Executive Summary",
    "核心指标": "Core Metrics",
    "高风险区间": "High-risk discount tier",
    "亏损率": "Loss Rate",
    "亏损复核召回": "Loss review recall",
    "模型召回率仪表盘": "Model recall gauge",
    "可以辅助复核排序，但不能自动决策": "The model can support review prioritization, but it must not be used for automatic decisions.",
    "可以辅助复核排序": "can support review prioritization",
    "误报亏损数": "False positives",
    "漏判亏损数": "False negatives",
    "受影响Record Count": "Affected records",
    "受影响Sales": "Affected sales",
    "受影响记录数": "Affected records",
    "受影响销售额": "Affected sales",
    "受影响": "Affected",
    "影响记录": "Affected records",
    "Profit率": "Profit margin",
    "利润率": "Profit margin",
    "平均Profit": "Average profit",
    "平均利润": "Average profit",
    "Discount区间": "discount tier",
    "折扣区间": "discount tier",
    "预测基线": "Forecast baseline",
    "平均绝对误差": "Mean absolute error",
    "平均百分比误差": "Mean absolute percentage error",
    "预测边界": "Forecast boundary",
    "模型边界": "Model boundary",
    "模型复核": "Model review",
    "主要经营观察": "Business observation",
    "暂无": "Not available",
    "记录数": "records",
    "折扣": "discount",
    "利润": "profit",
    "亏损": "loss",
    "区间": "tier",
}

OVERSTRONG_ENGLISH_REPLACEMENTS = {
    "catastrophic": "high",
    "confirms": "indicates",
    "confirms ": "indicates ",
    "systemic issue": "persistent margin risk",
    "must be addressed": "should be reviewed",
    "proves": "suggests",
    "proved": "suggested",
    "causality": "root cause",
    "causal": "directional",
}


def _english_client_text(value: object, fallback: str = "Business observation") -> str:
    text = englishize_payload_strings(value)
    for source, replacement in CLIENT_REPORT_ENGLISH_REPLACEMENTS.items():
        text = str(text).replace(source, replacement)
    for source, replacement in OVERSTRONG_ENGLISH_REPLACEMENTS.items():
        text = re.sub(re.escape(source), replacement, str(text), flags=re.IGNORECASE)
    text = _safe_english_visible_text(str(text))
    text = text.replace("30%+ discount tier tier", "30%+ discount tier")
    if not text or contains_cjk(text):
        return fallback
    return text


def _safe_english_visible_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).replace("\ufffd", "")).strip()
    if "…" not in cleaned:
        return cleaned
    prefix = cleaned.split("…", 1)[0].rstrip(" ,;:-")
    prefix = re.sub(r"\b[A-Za-z]{1,6}$", "", prefix).rstrip(" ,;:-")
    if not prefix:
        return ""
    if not re.search(r"[.!?]$", prefix):
        prefix += "."
    return prefix


def _englishize_client_payload_strings(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _englishize_client_payload_strings(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_englishize_client_payload_strings(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(_englishize_client_payload_strings(item) for item in payload)
    if isinstance(payload, str):
        return _english_client_text(payload, "Business observation")
    return payload


_DYNAMIC_PLACEHOLDER_PUBLIC_KEYS = (
    "executive_summary",
    "key_findings",
    "priority_actions",
    "final_synthesis",
    "modeling_outcome",
    "modeling_outcome_interpretation",
)


def _is_dynamic_business_observation_placeholder(value: object) -> bool:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    return normalized in {"", "business observation"}


def _drop_dynamic_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {key: _drop_dynamic_placeholders(item) for key, item in value.items()}
        return {key: item for key, item in cleaned.items() if item not in ("", [], {})}
    if isinstance(value, list):
        return [item for item in (_drop_dynamic_placeholders(item) for item in value) if item not in ("", [], {})]
    if isinstance(value, str):
        return "" if _is_dynamic_business_observation_placeholder(value) else value
    return value


def _remove_dynamic_business_observation_placeholders(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    for key in _DYNAMIC_PLACEHOLDER_PUBLIC_KEYS:
        if key not in cleaned:
            continue
        value = _drop_dynamic_placeholders(cleaned[key])
        if value in ("", [], {}):
            cleaned.pop(key, None)
        else:
            cleaned[key] = value
    return cleaned


def _violates_scoped_evidence(text: str, scoped_rows: list[dict[str, Any]]) -> bool:
    return violates_scoped_evidence(text, scoped_rows)


def _scope_guard_value(value: Any, scoped_rows: list[dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        guarded = {key: _scope_guard_value(item, scoped_rows) for key, item in value.items()}
        return {key: item for key, item in guarded.items() if item not in ("", [], {})}
    if isinstance(value, list):
        return [item for item in (_scope_guard_value(item, scoped_rows) for item in value) if item not in ("", [], {})]
    if isinstance(value, str):
        return "" if _violates_scoped_evidence(value, scoped_rows) else value
    return value


def _fallback_empty_after_scope_guard(guarded: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    merged = dict(guarded)
    for key in ("business_theme", "executive_summary", "key_findings", "priority_actions", "final_synthesis"):
        if not merged.get(key):
            merged[key] = base.get(key)
    for key in ("executive_summary", "limitations"):
        if isinstance(merged.get(key), list) and isinstance(base.get(key), list):
            existing = [str(item) for item in merged[key]]
            for item in base[key]:
                text = str(item)
                if text and not any(_text_similar(text, old) for old in existing):
                    merged[key].append(item)
                    existing.append(text)
                if len(merged[key]) >= 4:
                    break
    return merged


def _english_evidence_scope_guard(
    payload: dict[str, Any],
    *,
    base_payload: dict[str, Any],
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
) -> dict[str, Any]:
    scoped_rows = build_scoped_evidence_rows(report, schema_mapping)
    if not scoped_rows:
        return payload
    guarded = _scope_guard_value(payload, scoped_rows)
    return _fallback_empty_after_scope_guard(guarded if isinstance(guarded, dict) else payload, base_payload)


def _format_compact_number(
    value: object,
    *,
    currency: bool = False,
    percent: bool = False,
    output_language: str | None = None,
) -> str:
    if isinstance(value, str) and "%" in value:
        return _clean_text(value, "")
    if percent:
        return _format_number(value, percent=True)
    numeric = _number(value)
    if numeric is None:
        return _clean_text(value, "")
    if is_english_output(output_language):
        absolute = abs(numeric)
        if absolute >= 1_000_000_000:
            return f"{numeric / 1_000_000_000:.2f}B"
        if absolute >= 1_000_000:
            return f"{numeric / 1_000_000:.2f}M"
        if absolute >= 1_000:
            return f"{numeric / 1_000:.2f}K"
        return _format_number(numeric, currency=currency)
    if abs(numeric) >= 100000000:
        return f"{numeric / 100000000:.2f}亿"
    if abs(numeric) >= 10000:
        return f"{numeric / 10000:,.2f}万"
    return _format_number(numeric, currency=currency)


def _first_table_row(module: ModuleReport | None, table_name: str) -> dict[str, object]:
    if module is None:
        return {}
    rows = module.tables.get(table_name, [])
    if rows and isinstance(rows[0], dict):
        return dict(rows[0])
    return {}


def _risk_rank(value: object) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(str(value).lower(), 3)


def _highest_risk_discount_row(discount_module: ModuleReport | None) -> dict[str, object]:
    if discount_module is None:
        return {}
    rows = [
        dict(row)
        for row in discount_module.tables.get("discount_threshold_candidates", [])
        if isinstance(row, dict)
    ]
    rows.sort(
        key=lambda row: (
            _risk_rank(row.get("risk_level")),
            -(_number(row.get("negative_profit_rate")) or 0.0),
            _number(row.get("profit_margin")) if _number(row.get("profit_margin")) is not None else 999.0,
            _number(row.get("avg_profit")) if _number(row.get("avg_profit")) is not None else 999.0,
        )
    )
    return rows[0] if rows else {}


def _dataset_stat(dataset_profile: dict[str, Any] | None, *keys: str) -> object:
    profile = dataset_profile or {}
    for key in keys:
        if key in profile and profile[key] not in (None, ""):
            return profile[key]
    return None


def _field_count(dataset_profile: dict[str, Any] | None, schema_mapping: SchemaMapping) -> object:
    value = _dataset_stat(dataset_profile, "column_count", "field_count", "columns")
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    if value is not None:
        return value
    return len(schema_mapping.field_mapping)


def _negative_profit_rate_value(
    discount_module: ModuleReport | None,
    dataset_profile: dict[str, Any] | None,
) -> object:
    metrics = discount_module.summary_metrics if discount_module else {}
    explicit_rate = metrics.get("negative_profit_rate") if metrics else None
    if _number(explicit_rate) is not None:
        return explicit_rate
    negative_count = _number(metrics.get("negative_profit_order_count") if metrics else None)
    row_count = _number(_dataset_stat(dataset_profile, "row_count", "total_rows", "rows", "record_count"))
    if negative_count is None or row_count is None or row_count <= 0:
        return None
    return negative_count / row_count


def _discount_payload(discount_module: ModuleReport | None) -> dict[str, object]:
    threshold = _highest_risk_discount_row(discount_module)
    scenario = _first_table_row(discount_module, "discount_cap_what_if")
    high_risk_bucket = (
        scenario.get("high_risk_bucket")
        or scenario.get("current_threshold")
        or threshold.get("discount_bucket")
        or threshold.get("threshold")
    )
    return {
        "high_risk_bucket": _clean_text(high_risk_bucket),
        "risk_level": _clean_text(threshold.get("risk_level")),
        "negative_profit_rate": threshold.get("negative_profit_rate"),
        "profit_margin": threshold.get("profit_margin"),
        "avg_profit": threshold.get("avg_profit"),
        "target_discount_cap": scenario.get("target_discount_cap"),
        "affected_row_count": scenario.get("affected_row_count"),
        "affected_sales_amount": scenario.get("affected_sales_amount"),
        "estimated_profit_delta": scenario.get("estimated_profit_delta"),
        "current_negative_profit_rate": scenario.get("current_negative_profit_rate"),
        "estimated_negative_profit_rate_after_cap": scenario.get(
            "estimated_negative_profit_rate_after_cap"
        ),
        "limitation": SCENARIO_LIMITATION,
    }


def _modeling_payload(modeling_module: ModuleReport | None) -> dict[str, object] | None:
    if modeling_module is None:
        return None
    metrics = modeling_module.summary_metrics or {}
    if not metrics.get("best_model"):
        return None
    return {
        "best_model": _clean_text(metrics.get("best_model")),
        "recall": metrics.get("best_recall"),
        "f1": metrics.get("best_f1"),
        "roc_auc": metrics.get("best_roc_auc"),
        "false_negative_count": metrics.get("false_negative_count"),
        "false_positive_count": metrics.get("false_positive_count"),
        "limitation": MODEL_LIMITATION,
    }


def _module_has_evidence(module: ModuleReport | None) -> bool:
    if module is None:
        return False
    if module.summary_metrics:
        return True
    if any(rows for rows in module.tables.values()):
        return True
    return bool(module.findings)


def _english_capabilities(report: AnalysisReport, schema_mapping: SchemaMapping) -> dict[str, bool]:
    mapped = set(schema_mapping.field_mapping.values())
    discount_module = _module(report, "discount_profit_analysis")
    modeling_module = _module(report, "loss_risk_modeling")
    quality_module = _module(report, "data_quality_check")
    has_profit_and_discount = "profit" in mapped and "discount" in mapped
    discount_risk_row = _highest_risk_discount_row(discount_module)
    return {
        "has_sales": "sales_amount" in mapped,
        "has_profit": "profit" in mapped,
        "has_discount": "discount" in mapped,
        "has_profit_and_discount": has_profit_and_discount,
        "has_segment_or_region": bool({"segment", "region", "country", "city"} & mapped),
        "has_product_or_category": bool({"product_name", "category", "sub_category", "sku"} & mapped),
        "has_quality_evidence": _module_has_evidence(quality_module),
        "has_modeling_support": _modeling_payload(modeling_module) is not None,
        "has_discount_risk_evidence": has_profit_and_discount
        and _module_has_evidence(discount_module)
        and bool(discount_risk_row),
    }


def _english_business_theme(capabilities: dict[str, bool]) -> str:
    if capabilities["has_profit_and_discount"] and capabilities["has_discount_risk_evidence"]:
        return "Identify priority review targets across sales contribution, profit quality, and discount risk."
    if capabilities["has_profit"]:
        return "Identify priority review targets across sales contribution and profit quality."
    return "Identify priority review targets across data quality, sales structure, time patterns, and available operational dimensions."


def _english_quality_fact(report: AnalysisReport) -> str | None:
    quality = _module(report, "data_quality_check")
    metrics = quality.summary_metrics if quality else {}
    issues: list[str] = []
    for label, key in (
        ("missing order-date records", "missing_order_datetime"),
        ("negative sales records", "negative_sales_amount"),
        ("negative quantity records", "negative_quantity"),
        ("duplicate rows", "duplicate_rows"),
    ):
        value = _number(metrics.get(key))
        if value and value > 0:
            issues.append(f"{_format_number(value)} {label}")
    if not issues:
        return None
    return "Data quality checks identified " + ", ".join(issues[:3]) + ", which should be resolved before deeper operational conclusions."


def _english_product_fact(report: AnalysisReport) -> str | None:
    product = _module(report, "product_contribution_analysis")
    if product is None:
        return None
    metrics = product.summary_metrics or {}
    top_product = _clean_text(metrics.get("top_product") or metrics.get("top_category"), "")
    if top_product:
        return f"`{top_product}` is the leading product or category signal in the current mapped fields and should be reviewed for concentration risk."
    for rows in product.tables.values():
        if not rows:
            continue
        row = rows[0]
        if not isinstance(row, dict):
            continue
        label = _clean_text(
            row.get("product_name")
            or row.get("category")
            or row.get("sub_category")
            or row.get("sku")
            or row.get("Product Name")
            or row.get("Category"),
            "",
        )
        if label:
            return f"`{label}` is the top available product or category slice and should be reviewed for sales concentration."
    return None


def _english_summary_facts(
    report: AnalysisReport,
    *,
    total_sales: object,
    total_profit: object,
    discount: dict[str, object],
    capabilities: dict[str, bool],
) -> list[str]:
    facts: list[str] = []
    if capabilities["has_sales"] and _number(total_sales) is not None:
        facts.append(f"Total sales reached {_format_number(total_sales, currency=True)} based on the mapped sales field.")
    if capabilities["has_profit"] and _number(total_profit) is not None and _number(total_sales):
        margin = (_number(total_profit) or 0.0) / (_number(total_sales) or 1.0)
        facts.append(
            f"Total profit was {_format_number(total_profit, currency=True)}, corresponding to a profit margin of {_format_number(margin, percent=True)}."
        )
    if capabilities["has_discount_risk_evidence"]:
        bucket = _clean_text(discount.get("high_risk_bucket"), "")
        loss_rate = _format_number(discount.get("negative_profit_rate"), percent=True)
        margin = _format_number(discount.get("profit_margin"), percent=True)
        if bucket and bucket != "暂无":
            metric = f"loss rate {loss_rate}" if loss_rate != "暂无" else f"profit margin {margin}"
            facts.append(f"The `{bucket}` discount tier had the weakest profit quality, with {metric} in the current threshold analysis.")
    product_fact = _english_product_fact(report)
    if product_fact and capabilities["has_product_or_category"]:
        facts.append(product_fact)
    quality_fact = _english_quality_fact(report)
    if quality_fact:
        facts.append(quality_fact)
    if not capabilities["has_profit_and_discount"]:
        missing = []
        if not capabilities["has_profit"]:
            missing.append("profit")
        if not capabilities["has_discount"]:
            missing.append("discount")
        limitation = _english_missing_field_limitation(capabilities)
        if limitation:
            facts.append(limitation)
    return facts


def _english_priority_actions(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    discount: dict[str, object],
    capabilities: dict[str, bool],
) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    if capabilities["has_discount_risk_evidence"]:
        bucket = _clean_text(discount.get("high_risk_bucket"), "")
        loss_rate = _format_number(discount.get("negative_profit_rate"), percent=True)
        actions.append(
            {
                "priority": "P1",
                "issue": "Review discount-profit risk tier",
                "evidence": f"The `{bucket}` tier has loss rate {loss_rate}." if bucket and loss_rate != "暂无" else "Discount threshold analysis found a supported risk tier.",
                "action": "Review the supported high-risk discount tier against available order and pricing details. Add cost, fulfillment, or approval data before diagnosing root causes.",
            }
        )
    elif capabilities["has_profit"]:
        actions.append(
            {
                "priority": "P1",
                "issue": "Review profit quality slices",
                "evidence": "Profit is mapped, so sales contribution can be compared with profit outcomes.",
                "action": "Review high-sales slices against profit outcomes before prioritizing growth actions.",
            }
        )
    else:
        quality_fact = _english_quality_fact(report)
        actions.append(
            {
                "priority": "P1",
                "issue": "Confirm field coverage before profitability decisions",
                "evidence": quality_fact or f"Mapped field count: {_field_count(dataset_profile, schema_mapping)}.",
                "action": "Add profit and discount fields before making deeper profitability, pricing, or scenario decisions.",
            }
        )
    if capabilities["has_product_or_category"]:
        actions.append(
            {
                "priority": "P2",
                "issue": "Review product or SKU concentration",
                "evidence": _english_product_fact(report) or "Product, category, or SKU fields are mapped.",
                "action": "Check whether sales are concentrated in a small set of products, categories, or SKUs before changing commercial focus.",
            }
        )
    if capabilities["has_segment_or_region"]:
        actions.append(
            {
                "priority": "P3",
                "issue": "Compare operational dimensions",
                "evidence": "Segment, region, country, or city fields are mapped.",
                "action": "Compare available segment and location slices to separate structural sales patterns from one-off spikes.",
            }
        )
    else:
        actions.append(
            {
                "priority": "P3",
                "issue": "Use time and status patterns for follow-up",
                "evidence": "The current mapped fields support sales, time, status, or product-structure review.",
                "action": "Review sales trend, order status, volatility, and SKU structure while treating unsupported profitability and pricing questions as out of scope.",
            }
        )
    return actions[:5]


_ENGLISH_PROFIT_DEPENDENT_TERMS = (
    "profit quality",
    "margin risk",
    "low-margin",
)

_ENGLISH_DISCOUNT_EXCLUSIVE_TERMS = (
    "high-discount",
    "discount risk",
    "discount tier",
    "scoped discount tier",
    "discount-profit relationship",
    "highest-risk discount",
    "discount approval",
    "what-if profit lift",
)


def _english_missing_field_limitation(capabilities: dict[str, bool]) -> str | None:
    has_profit = capabilities["has_profit"]
    has_discount = capabilities["has_discount"]
    if not has_profit and not has_discount:
        return "Profit and discount fields were not mapped, so profit-margin and discount-risk conclusions are out of scope."
    if not has_profit:
        return "Profit field was not mapped, so profit-margin and discount-risk conclusions are out of scope."
    if not has_discount:
        return "Discount field was not mapped, so discount-risk conclusions are out of scope."
    return None


def _english_capability_guard(payload: dict[str, Any], capabilities: dict[str, bool]) -> dict[str, Any]:
    guarded = dict(payload)

    def supported_text(value: object) -> bool:
        text = _clean_text(value, "").lower()
        if not text:
            return False
        if not capabilities["has_profit"] and any(
            term in text for term in (*_ENGLISH_PROFIT_DEPENDENT_TERMS, *_ENGLISH_DISCOUNT_EXCLUSIVE_TERMS)
        ):
            return False
        if not capabilities["has_discount"] and any(term in text for term in _ENGLISH_DISCOUNT_EXCLUSIVE_TERMS):
            return False
        if not capabilities["has_discount_risk_evidence"] and any(term in text for term in _ENGLISH_DISCOUNT_EXCLUSIVE_TERMS):
            return False
        return True

    kpis = guarded.get("kpi_cards") if isinstance(guarded.get("kpi_cards"), list) else []
    filtered_kpis: list[dict[str, Any]] = []
    for card in kpis:
        if not isinstance(card, dict):
            continue
        label = _clean_text(card.get("label"), "").lower()
        if "profit" in label and not capabilities["has_profit"]:
            continue
        if "discount" in label and not capabilities["has_discount_risk_evidence"]:
            continue
        if "what-if" in label and not capabilities["has_discount_risk_evidence"]:
            continue
        filtered_kpis.append(card)
    guarded["kpi_cards"] = filtered_kpis

    for key in ("executive_summary", "limitations"):
        items = guarded.get(key) if isinstance(guarded.get(key), list) else []
        guarded[key] = [item for item in items if supported_text(item)]
    if not guarded["executive_summary"]:
        missing_field_limitation = _english_missing_field_limitation(capabilities)
        guarded["executive_summary"] = [
            missing_field_limitation or _english_business_theme(capabilities)
        ]

    actions = guarded.get("priority_actions") if isinstance(guarded.get("priority_actions"), list) else []
    filtered_actions = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        combined = " ".join(_clean_text(action.get(key), "") for key in ("issue", "evidence", "action"))
        if supported_text(combined):
            filtered_actions.append(action)
    guarded["priority_actions"] = filtered_actions

    key_findings = guarded.get("key_findings")
    if isinstance(key_findings, dict):
        guarded["key_findings"] = {
            section: [item for item in items if supported_text(item)]
            for section, items in key_findings.items()
            if isinstance(items, list)
        }
    final_synthesis = guarded.get("final_synthesis")
    if isinstance(final_synthesis, dict):
        cleaned = dict(final_synthesis)
        for key in ("brief_findings", "brief_actions", "main_conclusions", "recommended_actions"):
            value = cleaned.get(key)
            if isinstance(value, list):
                cleaned[key] = [item for item in value if supported_text(json.dumps(item, ensure_ascii=False))]
        metadata = cleaned.get("metadata")
        if isinstance(metadata, dict):
            cleaned["metadata"] = {
                key: value
                for key, value in metadata.items()
                if supported_text(json.dumps(value, ensure_ascii=False))
            }
        guarded["final_synthesis"] = {
            key: value for key, value in cleaned.items() if value not in ("", [], {})
        }
    if not capabilities["has_discount_risk_evidence"]:
        guarded["discount_what_if"] = {}
    return guarded


_CHINESE_UNSUPPORTED_CURRENT_DISCOUNT_TERMS = (
    "折扣策略",
    "折扣政策",
    "高折扣订单",
    "折扣审批",
    "审批阈值",
    "折扣阈值",
    "按折扣区间",
    "折扣力度过大",
    "折扣收紧",
    "折扣回收",
    "折扣是否真正带来销售增长",
    "折扣风险",
    "最高风险折扣阈值",
    "what-if 估算利润改善",
)

_CHINESE_ALLOWED_MISSING_DISCOUNT_TERMS = (
    "缺少 discount 字段",
    "未映射折扣字段",
    "当前无法进行折扣分析",
    "当前无法开展折扣区间",
    "补充折扣字段后再验证",
)

_CHINESE_PROFIT_QUALITY_THEME = "围绕销售贡献和利润质量，识别优先复盘对象并沉淀行动清单。"
_CHINESE_MISSING_DISCOUNT_LIMITATION = "缺少 discount 字段，当前无法开展折扣区间和 what-if 情景分析。"


def _is_allowed_missing_discount_statement(text: str) -> bool:
    return any(term in text for term in _CHINESE_ALLOWED_MISSING_DISCOUNT_TERMS)


def _is_unsupported_current_discount_claim(text: str) -> bool:
    if _is_allowed_missing_discount_statement(text):
        return False
    return any(term in text for term in _CHINESE_UNSUPPORTED_CURRENT_DISCOUNT_TERMS)


def _zh_no_discount_value_guard(value: Any) -> Any:
    if isinstance(value, dict):
        combined = " ".join(str(item) for item in value.values())
        if _is_unsupported_current_discount_claim(combined):
            return None
        guarded = {key: _zh_no_discount_value_guard(item) for key, item in value.items()}
        return {key: item for key, item in guarded.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        guarded_items = [_zh_no_discount_value_guard(item) for item in value]
        return [item for item in guarded_items if item not in (None, "", [], {})]
    if isinstance(value, str) and _is_unsupported_current_discount_claim(value):
        return None
    return value


def _zh_no_discount_payload_guard(payload: dict[str, Any], capabilities: dict[str, bool]) -> dict[str, Any]:
    if capabilities["has_discount"]:
        return payload
    guarded = dict(payload)
    if _is_unsupported_current_discount_claim(str(guarded.get("business_theme", ""))):
        guarded["business_theme"] = _CHINESE_PROFIT_QUALITY_THEME
    guarded["discount_what_if"] = {}

    kpis = guarded.get("kpi_cards") if isinstance(guarded.get("kpi_cards"), list) else []
    guarded["kpi_cards"] = [
        card
        for card in kpis
        if isinstance(card, dict)
        and not _is_unsupported_current_discount_claim(str(card.get("label", "")))
    ]

    for key in ("executive_summary", "limitations", "priority_actions"):
        value = guarded.get(key)
        if isinstance(value, list):
            guarded[key] = _zh_no_discount_value_guard(value)

    key_findings = guarded.get("key_findings")
    if isinstance(key_findings, dict):
        guarded["key_findings"] = {
            section: _zh_no_discount_value_guard(items)
            for section, items in key_findings.items()
            if isinstance(items, list)
        }
    final_synthesis = guarded.get("final_synthesis")
    if isinstance(final_synthesis, dict):
        cleaned = dict(final_synthesis)
        for key in ("brief_findings", "brief_actions", "main_conclusions", "recommended_actions"):
            if isinstance(cleaned.get(key), list):
                cleaned[key] = _zh_no_discount_value_guard(cleaned[key])
        guarded["final_synthesis"] = {
            key: value for key, value in cleaned.items() if value not in (None, "", [], {})
        }

    limitations = guarded.get("limitations") if isinstance(guarded.get("limitations"), list) else []
    if not any(_is_allowed_missing_discount_statement(str(item)) for item in limitations):
        limitations = [_CHINESE_MISSING_DISCOUNT_LIMITATION, *limitations]
    guarded["limitations"] = limitations
    return guarded


def _key_findings(report: AnalysisReport) -> dict[str, list[str]]:
    mapping = {
        "product_category": ["product_contribution_analysis"],
        "segment_region": ["dimension_breakdown_analysis", "country_market_analysis"],
        "discount_profit": ["discount_profit_analysis"],
        "modeling": ["loss_risk_modeling"],
    }
    findings: dict[str, list[str]] = {}
    for section, module_ids in mapping.items():
        section_findings: list[str] = []
        for module_id in module_ids:
            module = _module(report, module_id)
            if module:
                section_findings.extend(_clean_text(item) for item in module.findings[:2])
        findings[section] = section_findings[:3]
    return findings


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _process_log_like(summary_item: object) -> bool:
    text = _clean_text(summary_item, "").strip()
    return _contains_any(
        text,
        (
            "时间缺失",
            "缺失记录",
            "指标画像",
            "完成 ",
            "发现 0 条",
            "数据清洗",
            "字段类型",
        ),
    )


def _executive_summary_guard(payload: dict[str, Any]) -> dict[str, Any]:
    guarded = dict(payload)
    raw_summary = payload.get("executive_summary", [])
    summary = [_clean_text(item) for item in raw_summary if _clean_text(item)] if isinstance(raw_summary, list) else []
    combined = "\n".join(summary)
    kpis = payload.get("kpi_cards", []) if isinstance(payload.get("kpi_cards"), list) else []
    kpi_map = {
        str(card.get("label")): str(card.get("value"))
        for card in kpis
        if isinstance(card, dict) and card.get("label") and _clean_text(card.get("value"), "")
    }
    discount = payload.get("discount_what_if") if isinstance(payload.get("discount_what_if"), dict) else {}
    actions = payload.get("priority_actions") if isinstance(payload.get("priority_actions"), list) else []
    deterministic: list[str] = []

    kpi_parts = [
        f"{label}为 {value}"
        for label, value in (
            ("总销售额", kpi_map.get("总销售额")),
            ("总利润", kpi_map.get("总利润")),
            ("负利润记录占比", kpi_map.get("负利润记录占比")),
        )
        if value and value != "暂无"
    ]
    if kpi_parts and not _contains_any(combined, ("总销售额", "总利润", "负利润记录占比")):
        deterministic.append("核心 KPI：" + "，".join(kpi_parts) + "。")

    high_risk_bucket = _clean_text(discount.get("high_risk_bucket"), "")
    if high_risk_bucket and high_risk_bucket != "暂无" and not _contains_any(combined, ("最高风险折扣阈值",)):
        deterministic.append(f"最高风险折扣阈值为 {high_risk_bucket}，需要优先复盘折扣审批和利润质量。")

    profit_delta = _format_number(discount.get("estimated_profit_delta"), currency=True)
    if profit_delta != "暂无" and not _contains_any(combined, ("what-if",)):
        deterministic.append(f"折扣收紧 what-if 估算利润改善为 {profit_delta}，仅作为情景估算。")

    p1_action = next(
        (
            row
            for row in actions
            if isinstance(row, dict) and str(row.get("priority", "")).upper() == "P1"
        ),
        None,
    )
    if p1_action and not _contains_any(combined, ("P1", "优先", "行动")):
        action_text = _clean_text(p1_action.get("action") or p1_action.get("issue"), "")
        if action_text and action_text != "暂无":
            deterministic.append(f"P1 行动方向：{action_text}")

    non_process = [item for item in summary if not _process_log_like(item)]
    process = [item for item in summary if _process_log_like(item)]
    guarded["executive_summary"] = (deterministic + non_process + process)[:4]
    if not guarded["executive_summary"]:
        guarded["executive_summary"] = ["本次分析已形成客户报告，但当前证据不足以生成核心经营摘要。"]
    return guarded


def _fallback_payload(
    *,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
    section_priority: dict[str, Any] | None,
    final_synthesis: dict[str, Any] | None = None,
    modeling_outcome: dict[str, Any] | None = None,
    modeling_outcome_interpretation: dict[str, Any] | None = None,
    output_language: str | None = None,
) -> dict[str, Any]:
    discount_module = _module(report, "discount_profit_analysis")
    trend_module = _module(report, "sales_trend_analysis")
    modeling_module = _module(report, "loss_risk_modeling")
    discount = _discount_payload(discount_module)
    modeling = _modeling_payload(modeling_module)
    capabilities = _english_capabilities(report, schema_mapping)
    total_sales = (trend_module.summary_metrics or {}).get("total_sales_amount") if trend_module else None
    total_profit = (discount_module.summary_metrics or {}).get("total_profit_amount") if discount_module else None
    negative_profit_rate = _negative_profit_rate_value(discount_module, dataset_profile)
    executive_summary = [_clean_text(item) for item in report.summary[:3]]
    if not executive_summary:
        executive_summary = [
            f"本次共执行 {report.module_count} 个分析模块，重点围绕销售、利润、折扣和风险切片形成经营复盘。"
        ]
    rows = action_plan_rows(report, schema_mapping, dataset_profile, analysis_focus, evidence_pack)
    english = is_english_output(output_language)
    final_synthesis_payload = final_synthesis or {}
    if english:
        final_synthesis_payload = guard_english_final_synthesis(
            final_synthesis,
            report=report,
            schema_mapping=schema_mapping,
            dataset_profile=dataset_profile,
        )
        if not capabilities["has_discount_risk_evidence"]:
            discount = {}
        structured_facts = _english_summary_facts(
            report,
            total_sales=total_sales,
            total_profit=total_profit,
            discount=discount,
            capabilities=capabilities,
        )
        safe_summary = english_safe_bullets(report.summary, [], limit=2)
        executive_summary = (structured_facts + safe_summary)[:4] or [
            "The current mapped fields support a limited sales review, and unsupported profit or discount-risk conclusions are out of scope."
        ]
        rows = _english_priority_actions(report, schema_mapping, dataset_profile, discount, capabilities)
        if capabilities["has_discount_risk_evidence"]:
            discount = {**discount, "limitation": "This what-if estimate is based on discount recovery only; it does not represent real demand, volume, or customer behavior changes."}
        if modeling:
            modeling = {**modeling, "limitation": "The model is only for human review prioritization; it does not prove root causes and must not be used for automatic decisions."}
        key_findings = {
            "executive_summary": executive_summary[:3],
            "modeling": ["Model output is bounded by the available features and labels."] if modeling else [],
        }
    else:
        key_findings = _key_findings(report)
    payload = _guard_payload_text({
        "title": "Client-ready Sales Analysis Report" if english else "客户经营分析简报",
        "output_language": "en" if english else "zh-CN",
        "dataset_type": _clean_text(report.dataset_type),
        "sample_size": _dataset_stat(dataset_profile, "row_count", "total_rows", "rows", "record_count"),
        "field_count": _field_count(dataset_profile, schema_mapping),
        "business_theme": (
            _english_business_theme(capabilities)
            if english
            else "围绕销售贡献、利润质量和折扣风险，识别优先复盘对象并沉淀行动清单。"
            if capabilities["has_discount"]
            else _CHINESE_PROFIT_QUALITY_THEME
        ),
        "executive_summary": executive_summary,
        "kpi_cards": (
            [
                *(
                    [{"label": "Total Sales", "value": _format_number(total_sales, currency=True)}]
                    if capabilities["has_sales"] and _number(total_sales) is not None
                    else []
                ),
                *(
                    [{"label": "Total Profit", "value": _format_number(total_profit, currency=True)}]
                    if capabilities["has_profit"] and _number(total_profit) is not None
                    else []
                ),
                *(
                    [{"label": "Negative Profit Record Rate", "value": _format_number(negative_profit_rate, percent=True)}]
                    if capabilities["has_profit"] and _number(negative_profit_rate) is not None
                    else []
                ),
                *(
                    [{"label": "Highest-Risk Discount Threshold", "value": _clean_text(discount.get("high_risk_bucket"))}]
                    if capabilities["has_discount_risk_evidence"]
                    else []
                ),
                *(
                    [{"label": "Estimated What-if Profit Lift", "value": _format_number(discount.get("estimated_profit_delta"), currency=True)}]
                    if capabilities["has_discount_risk_evidence"] and _number(discount.get("estimated_profit_delta")) is not None
                    else []
                ),
            ]
            if english
            else [
                {"label": "总销售额", "value": _format_number(total_sales, currency=True)},
                *(
                    [{"label": "总利润", "value": _format_number(total_profit, currency=True)}]
                    if capabilities["has_profit"]
                    else []
                ),
                *(
                    [{"label": "负利润记录占比", "value": _format_number(negative_profit_rate, percent=True)}]
                    if capabilities["has_profit"]
                    else []
                ),
                *(
                    [
                        {"label": "最高风险折扣阈值", "value": _clean_text(discount.get("high_risk_bucket"))},
                        {
                            "label": "what-if 估算利润改善",
                            "value": _format_number(discount.get("estimated_profit_delta"), currency=True),
                        },
                    ]
                    if capabilities["has_discount_risk_evidence"]
                    else []
                ),
            ]
        ),
        "discount_what_if": discount if english or capabilities["has_discount_risk_evidence"] else {},
        "key_findings": key_findings,
        "priority_actions": rows[:5],
        "modeling_support": modeling,
        "limitations": (
            [
                "Field gaps may limit the depth of some business questions; validate decisions with additional business fields.",
                *(
                    [missing_field_limitation]
                    if (missing_field_limitation := _english_missing_field_limitation(capabilities))
                    else []
                ),
                *(
                    ["The what-if estimate is scenario-based and does not represent real demand, volume, or customer behavior changes."]
                    if capabilities["has_discount_risk_evidence"] and _number(discount.get("estimated_profit_delta")) is not None
                    else []
                ),
                *(
                    ["The model is only for human review prioritization and must not be used for automatic decisions."]
                    if modeling
                    else []
                ),
                "Break P1/P2/P3 actions into owners, review cycles, and acceptance metrics before execution.",
            ]
            if english
            else [
                "字段缺口会限制部分经营问题的验证深度，需结合业务字段继续补齐验证。",
                SCENARIO_LIMITATION if capabilities["has_discount_risk_evidence"] else _CHINESE_MISSING_DISCOUNT_LIMITATION,
                MODEL_LIMITATION if modeling else "当前未生成建模支持，后续可在字段和样本满足条件后补充风险复核模型。",
                "下一步建议把 P1/P2/P3 行动拆解为负责人、周期和验收指标。",
            ]
        ),
        "chart_section": {
            "title": "Notebook Chart Reference" if english else "Notebook 图表参考",
            "note": "See the Notebook for the full chart set." if english else "完整图表请查看 Notebook。",
        },
        "section_priority": section_priority or {},
        "final_synthesis": _englishize_client_payload_strings(final_synthesis_payload) if english else final_synthesis_payload,
        "modeling_outcome": (
            _englishize_client_payload_strings(modeling_outcome or {})
            if english
            else modeling_outcome or {}
        ),
        "modeling_outcome_interpretation": (
            _englishize_client_payload_strings(modeling_outcome_interpretation or {})
            if english
            else modeling_outcome_interpretation or {}
        ),
    })
    if english:
        payload = _english_capability_guard(payload, capabilities)
        payload = _englishize_client_payload_strings(payload)
        return _remove_dynamic_business_observation_placeholders(payload)
    return _zh_no_discount_payload_guard(payload, capabilities)


def _compact_prompt_payload(
    payload: dict[str, Any],
    report: AnalysisReport,
    section_priority: dict[str, Any] | None,
    llm_evidence_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    discount_module = _module(report, "discount_profit_analysis")
    modeling_module = _module(report, "loss_risk_modeling")
    return {
        "fallback_payload": payload,
        "evidence_pack": llm_evidence_pack
        or {
            "report_summary": report.summary[:5],
            "discount_evidence": {
                "threshold_candidates": (
                    (discount_module.tables or {}).get("discount_threshold_candidates", [])[:5]
                    if discount_module
                    else []
                ),
                "what_if": (
                    (discount_module.tables or {}).get("discount_cap_what_if", [])[:2]
                    if discount_module
                    else []
                ),
            },
            "modeling_evidence": {
                "metrics": modeling_module.summary_metrics if modeling_module else {},
            },
        },
        "section_priority": section_priority or {},
    }


def _merge_llm_payload(base: dict[str, Any], llm_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ("business_theme", "executive_summary", "key_findings", "limitations"):
        value = llm_payload.get(key)
        if key == "business_theme" and isinstance(value, str) and value.strip():
            merged[key] = _guard_text(value)
        elif key in {"executive_summary", "limitations"} and isinstance(value, list):
            merged[key] = [_guard_text(item) for item in value[:4] if _clean_text(item)]
        elif key == "key_findings" and isinstance(value, dict):
            merged[key] = {
                str(section): [_guard_text(item) for item in items[:3]]
                for section, items in value.items()
                if isinstance(items, list)
            }
    return merged


def build_client_report_payload_with_trace(
    *,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
    section_priority: dict[str, Any] | None,
    llm_client,
    llm_evidence_pack: dict[str, Any] | None = None,
    final_synthesis: dict[str, Any] | None = None,
    modeling_outcome: dict[str, Any] | None = None,
    modeling_outcome_interpretation: dict[str, Any] | None = None,
    output_language: str | None = None,
) -> tuple[dict[str, Any], LLMStageTrace]:
    base_payload = _fallback_payload(
        report=report,
        schema_mapping=schema_mapping,
        dataset_profile=dataset_profile,
        analysis_focus=analysis_focus,
        evidence_pack=evidence_pack,
        section_priority=section_priority,
        final_synthesis=final_synthesis,
        modeling_outcome=modeling_outcome,
        modeling_outcome_interpretation=modeling_outcome_interpretation,
        output_language=output_language,
    )
    capabilities = _english_capabilities(report, schema_mapping)
    if not is_english_output(output_language):
        base_payload = _executive_summary_guard(base_payload)
    if not (llm_client is not None and getattr(llm_client, "enabled", False)):
        return base_payload, build_llm_stage_trace(
            stage="client_report",
            llm_client=llm_client,
            status="disabled",
            reason="Client report used deterministic fallback because no LLM client was enabled.",
            attempted=False,
            applied=False,
        )

    system_prompt = (
        "你是面向业务方的销售分析报告编辑。只输出 JSON，不输出 HTML。"
        "中文输出，整体控制在 800-1200 中文字以内。只能使用输入证据，不要夸大 what-if，"
        "不要声称模型存在因果关系。除非证据包包含审批记录、实验设计、因果识别或明确规则，"
        "不要写“审批失控”“核心原因”“必然导致”“证明”等强因果措辞；"
        "优先使用“风险信号”“复核重点”“可能加剧”“需要结合明细验证”。"
        "返回字段可包含 business_theme、executive_summary、"
        "key_findings、limitations。"
    )
    if is_english_output(output_language):
        system_prompt = (
            "You are a client-facing sales analysis report editor. Return JSON only, not HTML. "
            "Use only the provided evidence, do not overstate what-if estimates, and do not claim causality for models. "
            "Use concise business English. Do not create long headings. Do not output truncated headlines. "
            "Do not include Chinese text. Prefer short noun phrases for card labels and complete short sentences for explanations. "
            "Avoid overclaiming causality. "
            "Returned fields may include business_theme, executive_summary, key_findings, and limitations. "
            f"{user_facing_language_instruction(output_language)}"
        )
    try:
        llm_payload = complete_json_for_stage(
            llm_client,
            system_prompt=system_prompt,
            user_payload={
                **_compact_prompt_payload(base_payload, report, section_priority, llm_evidence_pack),
                "language_instruction": user_facing_language_instruction(output_language),
            },
            cache_stage="client_report",
        )
    except Exception as exc:
        return base_payload, build_llm_stage_trace(
            stage="client_report",
            llm_client=llm_client,
            status="fallback_on_error",
            reason=describe_llm_error(exc),
            attempted=True,
            applied=False,
        )

    if not isinstance(llm_payload, dict):
        return base_payload, build_llm_stage_trace(
            stage="client_report",
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason="LLM client report payload was not a JSON object.",
            attempted=True,
            applied=False,
        )
    merged_payload = _guard_payload_text(_merge_llm_payload(base_payload, llm_payload))
    if not is_english_output(output_language):
        merged_payload = _executive_summary_guard(merged_payload)
        merged_payload = _zh_no_discount_payload_guard(merged_payload, capabilities)
    else:
        merged_payload = _englishize_client_payload_strings(merged_payload)
        merged_payload = _english_evidence_scope_guard(
            merged_payload,
            base_payload=_englishize_client_payload_strings(base_payload),
            report=report,
            schema_mapping=schema_mapping,
        )
        merged_payload = _english_capability_guard(merged_payload, capabilities)
    return merged_payload, build_llm_stage_trace(
        stage="client_report",
        llm_client=llm_client,
        status="llm_applied",
        reason="LLM generated structured client report prose; fixed template rendered HTML.",
        attempted=True,
        applied=True,
    )


def _html_list(items: list[object], empty: str = "暂无可展示内容。") -> str:
    cleaned = [_clean_text(item) for item in items if _clean_text(item)]
    if not cleaned:
        cleaned = [empty]
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in cleaned) + "</ul>"


def _risk_badge(level: object) -> str:
    clean = _clean_text(level, "low").lower()
    if clean not in {"high", "medium", "low"}:
        clean = "low"
    return f"<span class='risk risk-{clean}'>{escape(clean)}</span>"


def _public_text(value: object, fallback: str = "") -> str:
    text = _guard_text(value, fallback).strip()
    if text == "暂无":
        return fallback
    replacements = {
        "Capability Matrix": "分析覆盖范围",
        "capability matrix": "分析覆盖范围",
        "Charts v1": "图表",
        "v1": "",
        "debug": "",
        "prototype": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    return text or fallback


def _valid_display_value(value: object) -> bool:
    return _clean_text(value, "").strip().lower() not in {"", "暂无", "none", "nan", "unknown"}


def _clip_percent(value: object) -> float:
    numeric = _number(value)
    if numeric is None:
        return 0.0
    if abs(numeric) <= 1:
        numeric *= 100
    return max(0.0, min(100.0, numeric))


def _format_rate_label(value: object) -> str:
    formatted = _format_number(value, percent=True)
    return formatted if formatted != "暂无" else _format_number(value)


def _safe_headline(
    text: str,
    *,
    max_chars: int = 26,
    fallback: str = "本次销售数据已完成核心经营复盘",
) -> str:
    cleaned = _public_text(text, fallback)
    first = re.split(r"[。！？!?\n]", cleaned, maxsplit=1)[0].strip() or fallback
    if len(first) <= max_chars:
        return first
    cut_marks = "，,、；;：: "
    candidates = [first.rfind(mark, 0, max_chars + 1) for mark in cut_marks]
    breakpoint = max(candidates)
    if breakpoint >= 8:
        headline = first[:breakpoint].strip()
    else:
        headline = first[: max_chars - 1].strip()
    bad_tails = ("或使", "以及", "因为", "通过", "如果", "但是", "同时", "并且", "或者", "或", "和", "与", "及", "并", "但", "而")
    while headline and any(headline.endswith(tail) for tail in bad_tails):
        matched = next(tail for tail in bad_tails if headline.endswith(tail))
        headline = headline[: -len(matched)].rstrip()
    return (headline or fallback) + "…"


def _compress_report_title(text: str, *, fallback: str = "核心经营信号") -> str:
    cleaned = _public_text(text, fallback)
    lowered = cleaned.lower()
    rules: list[tuple[tuple[str, ...], str]] = [
        (("折扣", "利润"), "高折扣侵蚀利润质量"),
        (("类目", "分散"), "类目分散削弱经营聚焦"),
        (("类目", "集中"), "类目集中风险突出"),
        (("类目", "偏差"), "类目利润质量偏弱"),
        (("类目", "利润率"), "类目利润质量偏弱"),
        (("预测模型", "有限"), "预测模型仅供监控参考"),
        (("预测模型", "初步"), "预测模型仅供监控参考"),
        (("预测模型", "不具备"), "预测模型仅供监控参考"),
        (("销售增长", "波动"), "销售增长波动明显"),
        (("趋势", "下滑"), "销售趋势下滑明显"),
        (("近期增长率", "下滑"), "销售趋势下滑明显"),
        (("利润质量", "分化"), "利润质量分化明显"),
        (("客户粘性", "增长乏力"), "存量客户增长乏力"),
    ]
    if ("亏损风险建模" in cleaned or "亏损风险模型" in cleaned or "亏损模型" in cleaned or "loss_risk" in lowered) and (
        "正例不足" in cleaned or ("正例" in cleaned and "不足" in cleaned) or "偏弱" in cleaned or "weak" in lowered
    ):
        return "亏损模型仅作探索线索"
    for needles, title in rules:
        if all(needle in cleaned for needle in needles):
            return title
    first = re.split(r"[。！？!?\n]", cleaned, maxsplit=1)[0].strip()
    if 4 <= len(first) <= 18 and not first.endswith("…"):
        return first
    return _safe_headline(first, max_chars=18, fallback=fallback)


def _chinese_insight_card_title(text: str, *, fallback: str) -> str:
    compressed = _compress_report_title(text, fallback=fallback)
    cleaned = _public_text(text, fallback)
    if "…" not in compressed:
        if compressed == "利润质量分化明显" and "、" in cleaned:
            first = re.split(r"[。！？!?\n]", cleaned, maxsplit=1)[0].strip() or fallback
            pieces = re.split(r"([，,；;：:])", first)
            prefix = ""
            for index in range(0, len(pieces) - 1, 2):
                prefix += pieces[index]
                if index + 1 < len(pieces):
                    prefix += pieces[index + 1]
                candidate = prefix.rstrip("，,；;：: ").strip()
                if 10 <= len(candidate) <= 30:
                    return candidate
        return compressed

    first = re.split(r"[。！？!?\n]", cleaned, maxsplit=1)[0].strip() or fallback
    pieces = re.split(r"([，,；;：:])", first)
    prefix = ""
    for index in range(0, len(pieces) - 1, 2):
        prefix += pieces[index]
        if index + 1 < len(pieces):
            prefix += pieces[index + 1]
        candidate = prefix.rstrip("，,；;：: ").strip()
        if 10 <= len(candidate) <= 30:
            return candidate
    return first


def _build_english_report_headline(text: str, *, fallback: str = "Key business finding") -> str:
    cleaned = _english_client_text(text, fallback)
    lowered = cleaned.lower()
    if "add profit and discount fields" in lowered or "field coverage" in lowered:
        return "Field coverage limits deeper analysis"
    if "out of scope" in lowered and ("not mapped" in lowered or "fields" in lowered):
        return "Field coverage limits deeper risk analysis"
    if any(starter in lowered for starter in ("the primary", "a strong", "overall,", "this analysis", "the business")):
        if any(token in lowered for token in ("discount", "margin", "loss", "profit risk", "negative profit")):
            return "Margin erosion is concentrated in high-discount orders"
        if any(token in lowered for token in ("model", "recall", "auc", "prediction", "forecast", "regression")):
            return "Loss-risk model supports review prioritization"
    if any(token in lowered for token in ("discount", "margin", "loss", "profit risk")):
        return "High-discount margin risk requires review"
    if any(token in lowered for token in ("model", "recall", "auc", "prediction")):
        return "Loss-risk model supports review prioritization"
    if any(token in lowered for token in ("negative profit", "loss-making")):
        return "Negative-profit records require margin review"
    if any(token in lowered for token in ("product", "category", "furniture", "sku")):
        return "Product mix requires concentration review"
    words = re.findall(r"[A-Za-z0-9%+\-]+", cleaned)
    if 6 <= len(words) <= 12 and not cleaned.endswith("…"):
        return cleaned.rstrip(".")
    return fallback


def _build_english_insight_title(text: str, *, fallback: str = "Business observation") -> str:
    return _build_english_report_headline(text, fallback=fallback)


def _english_model_title_if_needed(*parts: str) -> str | None:
    combined = " ".join(part for part in parts if part).lower()
    model_tokens = (
        "model",
        "logisticregression",
        "recall",
        "precision",
        "auc",
        "score",
        "review prioritization",
        "prediction",
        "classification",
    )
    if any(token in combined for token in model_tokens):
        return "Loss-risk model supports review prioritization"
    return None


def _trim_scope_label(value: str) -> str:
    label = " ".join(value.strip(" .,:;").split())
    label = re.sub(r"^(review|check|compare|the|scoped)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+(review|slice|order|orders|structure|patterns?)$", "", label, flags=re.IGNORECASE)
    return label.strip(" .,:;")


def _english_scope_title_if_needed(*parts: str) -> str | None:
    combined = " ".join(part for part in parts if part)
    if not combined:
        return None
    model_title = _english_model_title_if_needed(combined)
    if model_title:
        return model_title
    pair_match = re.search(
        r"\b([A-Z][A-Za-z0-9%+\- &/]{1,48}?)\s*(?:×| x | X | / )\s*([A-Z0-9][A-Za-z0-9%+\- &/]{1,48}?)(?=\s+(?:margin|order|orders|slice|review|profit|loss|discount|tier)|[.,;]|$)",
        combined,
    )
    if pair_match:
        left = _trim_scope_label(pair_match.group(1))
        right = _trim_scope_label(pair_match.group(2))
        if left and right:
            return f"{left} × {right} margin requires review"
    discount_slice = re.search(
        r"\b([A-Z][A-Za-z0-9 &/\-]{1,48}?)\s+(?:in|within)\s+the\s+([0-9]+%?\+?\s+discount\s+tier)\b",
        combined,
        flags=re.IGNORECASE,
    )
    if discount_slice:
        left = _trim_scope_label(discount_slice.group(1))
        tier = _trim_scope_label(discount_slice.group(2))
        if left and tier:
            return f"{left} × {tier} requires review"
    tier_match = re.search(r"\b([0-9]+%?\+?\s+discount\s+tier)\b", combined, flags=re.IGNORECASE)
    if tier_match:
        return f"{_trim_scope_label(tier_match.group(1))} requires review"
    sku_match = re.search(r"\b(SKU\s+[A-Z0-9][A-Z0-9._-]{1,32})\b", combined)
    if sku_match:
        return f"{_trim_scope_label(sku_match.group(1))} requires review"
    product_match = re.search(r"\b(Product\s+[A-Z0-9][A-Za-z0-9 ._\-]{1,48}?)(?=\s+(?:margin|review|profit|loss|channel|slice)|[.,;]|$)", combined)
    if product_match:
        return f"{_trim_scope_label(product_match.group(1))} requires review"
    if re.search(r"\bsegment\s*(?:×|x|and)\s*region\b", combined, flags=re.IGNORECASE):
        return "Segment × Region slice requires review"
    if re.search(r"\bproduct\s*(?:×|x|and)\s*channel\b", combined, flags=re.IGNORECASE):
        return "Product and channel slice requires review"
    return None


def _english_action_card_title(*, issue: str, display_text: str, action: str, expected_use: str, evidence: str, fallback: str) -> str:
    title = _english_scope_title_if_needed(issue, display_text, action, expected_use, evidence)
    if title:
        return title
    if "discount" in issue.lower() and "driving" in issue.lower() and "margin risk" in issue.lower():
        return "High-discount margin risk requires review"
    if issue:
        return issue
    return _build_english_insight_title(action or display_text, fallback=fallback)


def _first_sentence(text: str, *, fallback: str = "主要经营观察") -> str:
    return _safe_headline(text, max_chars=34, fallback=fallback)


def _headline_like(text: str) -> bool:
    return bool(text) and not any(
        marker in text for marker in ("核心 KPI", "what-if", "P1 行动", "完成 ", "字段类型")
    )


def _extract_metric(text: str) -> str:
    return _extract_best_metric(text)


def _extract_best_metric(text: str) -> str:
    cleaned = _clean_text(text, "")
    matches = list(re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?", cleaned))
    if not matches:
        return ""
    metric_keywords = ("利润率", "占比", "亏损率", "R²", "R2", "MAPE", "波动率", "增长率", "Recall", "F1", "ROC", "AUC")

    def score(match: re.Match[str]) -> tuple[int, int]:
        value = match.group(0)
        start, end = match.span()
        window = cleaned[max(0, start - 18) : min(len(cleaned), end + 18)]
        points = 0
        if "%" in value:
            points += 30
        if value.strip().startswith("-"):
            points += 12
        if any(keyword in window for keyword in metric_keywords):
            points += 10
        if "," in value:
            points += 2
        return points, -start

    best = max(matches, key=score).group(0)
    return re.sub(r"\s+%", "%", best.strip())


def _pick_insight_metric(*texts: str) -> str:
    for text in texts:
        metric = _extract_best_metric(text)
        if metric:
            return metric
    return ""


def _text_similar(a: str, b: str) -> bool:
    left = re.sub(r"[\s，。！？!?,.;；:：、]+", "", _clean_text(a, ""))
    right = re.sub(r"[\s，。！？!?,.;；:：、]+", "", _clean_text(b, ""))
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    overlap = len(set(left) & set(right)) / max(len(set(left) | set(right)), 1)
    return overlap >= 0.88


def _short_evidence(text: str, *, max_chars: int = 86) -> str:
    cleaned = _public_text(text, "")
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    return _safe_headline(cleaned, max_chars=max_chars, fallback="")


def _short_chinese_insight_evidence(text: str, max_chars: int = 86) -> str:
    cleaned = _public_text(text, "")
    if not cleaned or len(cleaned) <= max_chars:
        return cleaned
    natural_delimiters = "。！？!?，,；;：:"
    candidates = [
        index
        for index, char in enumerate(cleaned[: max_chars + 1])
        if char in natural_delimiters
    ]
    if candidates:
        return cleaned[: candidates[-1]].rstrip("。！？!?，,；;：: ").strip()
    sentence_match = re.split(r"[。！？!?\n]", cleaned, maxsplit=1)[0].strip()
    return sentence_match or cleaned


def _insight_evidence(*, title: str, description: str, evidence: str, english: bool = False) -> str:
    evidence_text = _short_evidence(evidence) if english else _short_chinese_insight_evidence(evidence)
    if not evidence_text:
        return ""
    if _text_similar(evidence_text, title) or _text_similar(evidence_text, description):
        return ""
    return evidence_text


def _insight_description(text: str, *, max_chars: int = 95) -> str:
    cleaned = _public_text(text, "")
    if len(cleaned) <= max_chars:
        return cleaned
    return _safe_headline(cleaned, max_chars=max_chars, fallback="")


def _has_real_what_if(payload: dict[str, Any]) -> bool:
    discount = payload.get("discount_what_if") if isinstance(payload.get("discount_what_if"), dict) else {}
    what_if_keys = (
        "estimated_profit_delta",
        "target_discount_cap",
        "affected_row_count",
        "current_negative_profit_rate",
        "estimated_negative_profit_rate_after_cap",
    )
    return any(_valid_display_value(discount.get(key)) for key in what_if_keys)


def _is_what_if_limitation(text: str) -> bool:
    lowered = text.lower()
    return "what-if" in lowered or "情景估算" in text or "折扣回收" in text


def _is_classification_review_limitation(text: str) -> bool:
    return any(token in text for token in ("亏损风险模型", "风险复核模型", "人工复核优先级排序", "复核排序", "亏损风险订单"))


def _kpi_class(label: str) -> str:
    lowered = label.lower()
    if any(token in label for token in ("风险", "负利润", "折扣", "亏损")) or any(token in lowered for token in ("risk", "negative profit", "discount", "loss")):
        return "risk"
    if any(token in label for token in ("模型", "Recall", "F1", "ROC")) or any(token in lowered for token in ("model", "recall", "auc")):
        return "model"
    if any(token in label for token in ("利润", "改善")) or any(token in lowered for token in ("profit", "lift", "margin")):
        return "good"
    return ""


def _tag_for_section(section: str, *, english: bool = False, text: str = "") -> str:
    if english:
        lowered = f"{section} {text}".lower()
        if "out of scope" in lowered and ("not mapped" in lowered or "fields" in lowered):
            return "Field Coverage"
        if any(token in lowered for token in ("discount", "margin", "profit", "loss")):
            return "Margin Risk" if "model" not in lowered else "Model Signal"
        if any(token in lowered for token in ("model", "recall", "auc", "prediction", "forecast")):
            return "Model Signal"
        if any(token in lowered for token in ("action", "priority", "review")):
            return "Action Priority"
        if any(token in lowered for token in ("category", "product")):
            return "Profit Quality"
        return "Business Observation"
    return {
        "final_synthesis": "最终结论",
        "product_category": "商品结构",
        "segment_region": "区域切片",
        "discount_profit": "折扣利润",
        "modeling": "模型边界",
        "executive_summary": "核心摘要",
    }.get(section, "经营观察")


def _collect_finding_texts(payload: dict[str, Any]) -> list[tuple[str, str]]:
    final_synthesis = payload.get("final_synthesis") if isinstance(payload.get("final_synthesis"), dict) else {}
    main_conclusions = final_synthesis.get("main_conclusions")
    if isinstance(main_conclusions, list):
        collected_from_final: list[tuple[str, str]] = []
        for item in main_conclusions:
            if not isinstance(item, dict):
                continue
            text = _public_text(
                item.get("conclusion")
                or item.get("business_meaning")
                or item.get("evidence")
            )
            if text:
                collected_from_final.append(("final_synthesis", text))
        if collected_from_final:
            collected = collected_from_final
        else:
            collected = []
    else:
        collected = []
    findings = payload.get("key_findings") if isinstance(payload.get("key_findings"), dict) else {}
    for section, items in findings.items():
        if not isinstance(items, list):
            continue
        for item in items:
            text = _public_text(item)
            if text:
                collected.append((str(section), text))
    summary = payload.get("executive_summary") if isinstance(payload.get("executive_summary"), list) else []
    for item in summary:
        text = _public_text(item)
        if text:
            collected.append(("executive_summary", text))
    return collected


def _metric_from_mapping(mapping: dict[str, Any], *keys: str) -> object:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _modeling_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("modeling_outcome")
    return value if isinstance(value, dict) else {}


def _modeling_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    outcome = _modeling_outcome(payload)
    metrics = outcome.get("metrics_summary")
    merged = dict(metrics) if isinstance(metrics, dict) else {}
    for key, value in outcome.items():
        if key not in merged:
            merged[key] = value
    return merged


def _modeling_task(payload: dict[str, Any]) -> str:
    outcome = _modeling_outcome(payload)
    task = _clean_text(outcome.get("primary_modeling_task"), "")
    status = _clean_text(outcome.get("modeling_status"), "")
    metrics = _modeling_metrics(payload)
    combined = " ".join([task, status, " ".join(str(key) for key in metrics.keys())]).lower()
    has_regression_model = any(key in metrics for key in ("best_model", "best_model_r2", "best_r2", "r2"))
    if "forecast_baseline" in task.lower() or ("best_baseline" in metrics and not has_regression_model):
        return "forecast_baseline"
    if any(token in combined for token in ("regression", "forecast")):
        return "regression"
    if "loss_risk" in combined or "classification" in combined:
        return "classification"
    return ""


def _is_weak_model(payload: dict[str, Any]) -> bool:
    outcome = _modeling_outcome(payload)
    metrics = _modeling_metrics(payload)
    text = " ".join(
        _clean_text(outcome.get(key), "")
        for key in ("modeling_status", "modeling_value_level", "model_quality_status", "regression_status")
    ).lower()
    metrics_text = " ".join(
        _clean_text(metrics.get(key), "")
        for key in ("model_quality_status", "regression_status")
    ).lower()
    r2 = _number(_metric_from_mapping(metrics, "best_model_r2", "best_r2", "r2"))
    mape = _number(_metric_from_mapping(metrics, "best_model_mape", "best_baseline_mape", "best_mape", "mape"))
    return "weak" in text or "weak" in metrics_text or (r2 is not None and r2 < 0.2) or (mape is not None and mape >= 50)


def _metric_display(value: object) -> str:
    formatted = _format_number(value)
    if _valid_display_value(formatted):
        return formatted
    return _clean_text(value, "")


def _metric_bar(label: str, value: object) -> dict[str, object] | None:
    display = _metric_display(value)
    if not _valid_display_value(display):
        return None
    return {"label": label, "value_label": display, "percent": _clip_percent(value)}


def _append_model_kpis(payload: dict[str, Any], kpis: list[dict[str, Any]], *, english: bool = False) -> None:
    metrics = _modeling_metrics(payload)
    task = _modeling_task(payload)
    candidates: list[tuple[str, object, str]] = []
    if task == "forecast_baseline":
        candidates = [
            ("MAPE", _metric_from_mapping(metrics, "best_baseline_mape", "best_model_mape", "best_mape", "mape"), "baseline percentage error" if english else "baseline 百分比误差"),
            ("MAE", _metric_from_mapping(metrics, "best_baseline_mae", "best_model_mae", "best_mae", "mae"), "baseline mean absolute error" if english else "baseline 平均绝对误差"),
            ("RMSE", _metric_from_mapping(metrics, "best_baseline_rmse", "best_model_rmse", "rmse"), "baseline root mean squared error" if english else "baseline 均方根误差"),
            ("baseline", _metric_from_mapping(metrics, "best_model", "best_baseline", "baseline_model"), "Forecast baseline" if english else "预测基线"),
        ]
    elif task == "regression":
        candidates = [
            ("R²", _metric_from_mapping(metrics, "best_model_r2", "best_r2", "r2"), "Regression explanatory power" if english else "回归解释力"),
            ("MAE", _metric_from_mapping(metrics, "best_model_mae", "best_mae", "mae"), "Mean absolute error" if english else "平均绝对误差"),
            ("MAPE", _metric_from_mapping(metrics, "best_model_mape", "best_mape", "mape"), "Mean absolute percentage error" if english else "平均百分比误差"),
            ("Baseline lift" if english else "baseline 改善", _metric_from_mapping(metrics, "improvement_vs_baseline"), "Improvement versus baseline" if english else "相对基线改善"),
        ]
    elif task == "classification":
        candidates = [
            ("Recall", _metric_from_mapping(metrics, "best_recall", "recall"), "Loss review recall" if english else "亏损复核召回"),
            ("F1", _metric_from_mapping(metrics, "best_f1", "f1"), "Classification balance metric" if english else "分类综合指标"),
            ("ROC AUC", _metric_from_mapping(metrics, "best_roc_auc", "roc_auc"), "Ranking separation ability" if english else "排序区分能力"),
        ]
    for label, value, sub in candidates:
        if len(kpis) >= 6:
            return
        formatted = _metric_display(value)
        if english:
            formatted = _normalize_negative_zero_display(formatted)
        if _valid_display_value(formatted) and not any(kpi["label"] == label for kpi in kpis):
            kpis.append(
                {
                    "label": label,
                    "value": formatted,
                    "sub": sub,
                    "class": "model",
                    "accent": "rgba(79,70,229,.11)",
                }
            )


def _build_kpi_view_model(payload: dict[str, Any]) -> list[dict[str, Any]]:
    english = is_english_output(str(payload.get("output_language") or ""))
    raw_kpis = payload.get("kpi_cards") if isinstance(payload.get("kpi_cards"), list) else []
    kpis: list[dict[str, Any]] = []
    for card in raw_kpis:
        if not isinstance(card, dict):
            continue
        label = _english_client_text(card.get("label"), "Core Metric") if english else _public_text(card.get("label"), "核心指标")
        raw_value = card.get("value")
        value = _clean_text(raw_value, "")
        if not _valid_display_value(value):
            continue
        compact_value = (
            value
            if "%" in value or any(token in label for token in ("R²", "R2", "F1", "Recall", "ROC", "AUC", "率"))
            else _format_compact_number(raw_value, output_language="en" if english else None)
        )
        if not _valid_display_value(compact_value):
            compact_value = value
        kpis.append(
            {
                "label": label,
                "value": compact_value,
                "full_value": value,
                "sub": label,
                "class": _kpi_class(label),
                "accent": (
                    "rgba(239,68,68,.12)"
                    if _kpi_class(label) == "risk"
                    else "rgba(22,163,74,.11)"
                    if _kpi_class(label) == "good"
                    else "rgba(79,70,229,.11)"
                    if _kpi_class(label) == "model"
                    else "rgba(37,99,235,.10)"
                ),
            }
        )
    _append_model_kpis(payload, kpis, english=english)
    sample_size = _format_number(payload.get("sample_size"))
    field_count = _format_number(payload.get("field_count"))
    fillers = [
        {"label": "Sample Size" if english else "样本量", "value": sample_size, "sub": "Records analyzed" if english else "本次分析记录数", "class": "", "accent": "rgba(37,99,235,.10)"},
        {"label": "Field Count" if english else "字段数", "value": field_count, "sub": "Available field coverage" if english else "可用字段规模", "class": "", "accent": "rgba(37,99,235,.10)"},
        {
            "label": "Dataset Type" if english else "数据类型",
            "value": _clean_text(payload.get("dataset_type"), "sales_transaction"),
            "sub": "Business dataset type" if english else "业务数据类型",
            "class": "",
            "accent": "rgba(37,99,235,.10)",
        },
    ]
    existing_labels = {item["label"] for item in kpis}
    for filler in fillers:
        if len(kpis) >= 3:
            break
        if filler["label"] not in existing_labels and _valid_display_value(filler["value"]):
            kpis.append(filler)
    return kpis[:6]


def _build_spotlight_view_model(payload: dict[str, Any]) -> dict[str, Any]:
    english = is_english_output(str(payload.get("output_language") or ""))
    discount = payload.get("discount_what_if") if isinstance(payload.get("discount_what_if"), dict) else {}
    bucket = _clean_text(discount.get("high_risk_bucket"), "")
    negative_profit_rate = discount.get("negative_profit_rate") or discount.get("current_negative_profit_rate")
    if bucket and bucket != "暂无" and _number(negative_profit_rate) is not None:
        affected_rows = _format_number(discount.get("affected_row_count"))
        affected_sales = (
            _format_compact_number(discount.get("affected_sales_amount"), currency=True, output_language="en")
            if english
            else _format_number(discount.get("affected_sales_amount"), currency=True)
        )
        profit_margin = _format_rate_label(discount.get("profit_margin"))
        avg_profit = (
            _format_compact_number(discount.get("avg_profit"), currency=True, output_language="en")
            if english
            else _format_number(discount.get("avg_profit"), currency=True)
        )
        if english:
            facts = [
                f"Affected records {affected_rows}" if affected_rows != "暂无" else "",
                f"Affected sales {affected_sales}" if affected_sales != "暂无" else "",
                f"Profit margin {profit_margin}" if profit_margin != "暂无" else "",
                f"Average profit {avg_profit}" if avg_profit != "暂无" else "",
            ]
            return {
                "kind": "discount_profit",
                "pill": f"High-risk discount tier: {bucket}",
                "headline_metric": _format_rate_label(negative_profit_rate),
                "metric_label": "Loss Rate",
                "title": f"The {bucket} discount tier should be reviewed first",
                "description": " · ".join(fact for fact in facts if fact)
                + ". This is not a normal discount fluctuation; it is a margin-risk signal that should be reviewed first.",
                "percent": _clip_percent(negative_profit_rate),
                "bars": [
                    {"label": "Loss Rate", "value_label": _format_rate_label(negative_profit_rate), "percent": _clip_percent(negative_profit_rate)},
                    {"label": "Profit margin", "value_label": profit_margin, "percent": _clip_percent(abs(_number(discount.get("profit_margin")) or 0.0))},
                    {"label": "Affected records", "value_label": affected_rows, "percent": min(100.0, (_number(discount.get("affected_row_count")) or 0.0) / 20.0)},
                ],
            }
        facts = [
            f"受影响记录数 {affected_rows}" if affected_rows != "暂无" else "",
            f"受影响销售额 {affected_sales}" if affected_sales != "暂无" else "",
            f"利润率 {profit_margin}" if profit_margin != "暂无" else "",
            f"平均利润 {avg_profit}" if avg_profit != "暂无" else "",
        ]
        return {
            "kind": "discount_profit",
            "pill": f"高风险区间：{bucket}",
            "headline_metric": _format_rate_label(negative_profit_rate),
            "metric_label": "亏损率",
            "title": f"{bucket} 折扣区间是本次优先复盘对象",
            "description": "，".join(fact for fact in facts if fact) + "。它不是普通折扣波动，而是需要优先复核的利润风险信号。",
            "percent": _clip_percent(negative_profit_rate),
            "bars": [
                {"label": "亏损率", "value_label": _format_rate_label(negative_profit_rate), "percent": _clip_percent(negative_profit_rate)},
                {"label": "利润率", "value_label": profit_margin, "percent": _clip_percent(abs(_number(discount.get("profit_margin")) or 0.0))},
                {"label": "影响记录", "value_label": affected_rows, "percent": min(100.0, (_number(discount.get("affected_row_count")) or 0.0) / 20.0)},
            ],
        }

    task = _modeling_task(payload)
    metrics = _modeling_metrics(payload)
    if task == "forecast_baseline":
        mae = _metric_from_mapping(metrics, "best_model_mae", "best_baseline_mae", "best_mae", "mae")
        rmse = _metric_from_mapping(metrics, "best_model_rmse", "best_baseline_rmse", "rmse")
        mape = _metric_from_mapping(metrics, "best_model_mape", "best_baseline_mape", "best_mape", "mape")
        baseline = _metric_from_mapping(metrics, "best_model", "best_baseline", "baseline_model")
        headline_value = mape if _number(mape) is not None else mae
        headline = _metric_display(headline_value)
        facts = [
            f"MAE {_metric_display(mae)}" if _valid_display_value(_metric_display(mae)) else "",
            f"RMSE {_metric_display(rmse)}" if _valid_display_value(_metric_display(rmse)) else "",
            f"MAPE {_metric_display(mape)}" if _valid_display_value(_metric_display(mape)) else "",
            f"baseline {_metric_display(baseline)}" if _valid_display_value(_metric_display(baseline)) else "",
        ]
        bars = [
            bar
            for bar in (
                _metric_bar("MAPE", mape),
                _metric_bar("MAE", mae),
                _metric_bar("RMSE", rmse),
            )
            if bar is not None
        ]
        return {
            "kind": "forecast_boundary",
            "pill": "Forecast boundary" if english else "预测边界",
            "headline_metric": headline if _valid_display_value(headline) else "监控",
            "metric_label": "MAPE" if _number(mape) is not None else "MAE" if _number(mae) is not None else "模型边界",
            "title": "Forecast baseline should only be used as a monitoring reference" if english else "当前预测 baseline 仅适合作为监控参照",
            "description": (
                "; ".join(item for item in facts if item)
                + ". This chronological backtest baseline can support sales monitoring and future model comparison, but it is not a production forecast."
                if english
                else "，".join(item for item in facts if item) + "。这是时间顺序回测得到的 baseline 结果，可用于销售监控和后续模型对照，不应包装成生产级自动预测。"
            ),
            "percent": _clip_percent(headline_value),
            "bars": bars,
        }

    if task == "regression":
        r2 = _metric_from_mapping(metrics, "best_model_r2", "best_r2", "r2")
        mape = _metric_from_mapping(metrics, "best_model_mape", "best_mape", "mape")
        mae = _metric_from_mapping(metrics, "best_model_mae", "best_mae", "mae")
        improvement = _metric_from_mapping(metrics, "improvement_vs_baseline")
        weak = _is_weak_model(payload)
        formatted_improvement = _format_display_number(improvement) if english else _format_number(improvement)
        headline = _format_display_number(r2) if english else _format_number(r2)
        if headline == "暂无":
            headline = _format_display_number(mape) if english else _format_number(mape)
        r2_display = _format_display_number(r2) if english else _format_number(r2)
        mae_display = _format_display_number(mae) if english else _format_number(mae)
        mape_display = _format_display_number(mape) if english else _format_number(mape)
        details = [
            f"R² {r2_display}" if r2_display != "暂无" else "",
            f"MAE {mae_display}" if mae_display != "暂无" else "",
            f"MAPE {mape_display}" if mape_display != "暂无" else "",
            (
                f"Baseline improvement {formatted_improvement}"
                if english
                else f"相对基线改善 {formatted_improvement}"
            )
            if formatted_improvement != "暂无"
            else "",
        ]
        return {
            "kind": "forecast_boundary",
            "pill": "Forecast boundary" if english else "预测边界",
            "headline_metric": headline if headline != "暂无" else "参考",
            "metric_label": "R²" if r2_display != "暂无" else "MAPE",
            "title": (
                "Regression output should only support sales monitoring"
                if english and weak
                else "Regression output can support sales monitoring comparison"
                if english
                else "当前预测模型只适合作为监控参考" if weak else "预测模型可作为销售监控对照基线"
            ),
            "description": (
                "; ".join(item for item in details if item)
                + ". Use this result for sales monitoring and review, not production-grade automatic forecasting."
                if english
                else "，".join(item for item in details if item) + "。预测结果用于销售监控和复盘，不应包装成生产级自动预测。"
            ),
            "percent": _clip_percent(r2 if _number(r2) is not None else mape),
            "bars": [
                {"label": "R²", "value_label": r2_display, "percent": _clip_percent(r2)},
                {"label": "MAPE", "value_label": mape_display, "percent": _clip_percent(mape)},
                {
                    "label": "Baseline lift" if english else "基线改善",
                    "value_label": formatted_improvement,
                    "percent": _clip_percent(improvement),
                },
            ],
        }

    modeling = payload.get("modeling_support") if isinstance(payload.get("modeling_support"), dict) else None
    classification_metrics = _modeling_metrics(payload) if _modeling_task(payload) == "classification" else {}
    if modeling and _valid_display_value(modeling.get("best_model")):
        model_source = modeling
        recall = model_source.get("recall")
        f1 = model_source.get("f1")
        roc_auc = model_source.get("roc_auc")
    elif classification_metrics and _valid_display_value(_metric_from_mapping(classification_metrics, "best_model")):
        model_source = classification_metrics
        recall = _metric_from_mapping(classification_metrics, "best_recall", "recall")
        f1 = _metric_from_mapping(classification_metrics, "best_f1", "f1")
        roc_auc = _metric_from_mapping(classification_metrics, "best_roc_auc", "roc_auc")
    else:
        model_source = {}
    if model_source:
        weak = _is_weak_model(payload)
        headline_value = f1 if weak and _number(f1) is not None else recall if _number(recall) is not None else f1
        headline = _format_number(headline_value)
        return {
            "kind": "model_review",
            "pill": "Model boundary" if english and weak else "Model Signal" if english else "模型边界" if weak else "模型复核",
            "headline_metric": headline,
            "metric_label": "F1" if weak and _number(f1) is not None else "Recall" if _number(recall) is not None else "Model boundary" if english else "模型边界",
            "title": (
                "Model output is only an exploratory risk signal"
                if english and weak
                else "Loss-risk model supports review prioritization"
                if english
                else "亏损风险模型仅适合作为探索性线索" if weak else "模型可辅助人工复核排序"
            ),
            "description": (
                "Current model quality is weak, so it should be validated before review prioritization or automatic decisions."
                if english and weak
                else _english_client_text(
                    modeling.get("limitation") if isinstance(modeling, dict) else None,
                    "The model can support review prioritization, but it must not be used for automatic decisions.",
                )
                if english
                else "当前模型质量偏弱，不建议直接用于复核排序或自动决策。"
                if weak
                else _public_text(
                    modeling.get("limitation") if isinstance(modeling, dict) else None,
                    MODEL_LIMITATION,
                )
            ),
            "percent": _clip_percent(headline_value),
            "bars": [
                {"label": "Recall", "value_label": _format_number(recall), "percent": _clip_percent(recall)},
                {"label": "F1", "value_label": _format_number(f1), "percent": _clip_percent(f1)},
                {"label": "ROC AUC", "value_label": _format_number(roc_auc), "percent": _clip_percent(roc_auc)},
            ],
        }

    findings = _collect_finding_texts(payload)
    first = next((text for _section, text in findings if _headline_like(text)), "")
    if not first:
        first = _public_text(payload.get("business_theme"), "本次销售数据已完成核心经营复盘。")
    metric = _extract_metric(first)
    if english:
        kind = "general"
        pill = "Business Observation"
        if _contains_any(first, ("volatility", "decline", "growth", "monthly", "trend")):
            kind = "trend_volatility"
            pill = "Trend Volatility"
        elif _contains_any(first, ("Top", "top", "share", "concentration", "category", "product")):
            kind = "concentration"
            pill = "Concentration Signal"
        return {
            "kind": kind,
            "pill": pill,
            "headline_metric": metric or "Observation",
            "metric_label": "Core Signal",
            "title": _build_english_report_headline(first, fallback="Business observation"),
            "description": _english_client_text(first, "Review this business signal with supporting evidence."),
            "percent": _clip_percent(metric) if metric else 68.0,
            "bars": [],
        }
    kind = "general"
    pill = "主要经营观察"
    if _contains_any(first, ("波动", "下滑", "增长率", "月度", "趋势")):
        kind = "trend_volatility"
        pill = "趋势波动"
    elif _contains_any(first, ("Top", "top", "占比", "集中度", "头部", "类目", "商品数")):
        kind = "concentration"
        pill = "集中度观察"
    return {
        "kind": kind,
        "pill": pill,
        "headline_metric": metric or "观察",
        "metric_label": "核心信号",
        "title": _first_sentence(first),
        "description": first,
        "percent": _clip_percent(metric) if metric else 68.0,
        "bars": [],
    }


def _kpi_card_by_label(payload: dict[str, Any], *labels: str) -> dict[str, Any] | None:
    wanted = {label.lower() for label in labels}
    raw_kpis = payload.get("kpi_cards") if isinstance(payload.get("kpi_cards"), list) else []
    for card in raw_kpis:
        if not isinstance(card, dict):
            continue
        label = _english_client_text(card.get("label"), "")
        if label.lower() in wanted and _valid_display_value(card.get("value")):
            return {"label": label, "value": _clean_text(card.get("value"), "")}
    return None


def _build_english_structured_insight_cards(payload: dict[str, Any]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []

    total_sales = _kpi_card_by_label(payload, "Total Sales")
    if total_sales:
        cards.append(
            {
                "tag": "Sales Scale",
                "metric": str(total_sales["value"]),
                "title": "Total Sales",
                "description": f"Sales reached {total_sales['value']} based on the mapped sales field.",
                "evidence": "Mapped sales field",
            }
        )

    total_profit = _kpi_card_by_label(payload, "Total Profit")
    if total_profit:
        cards.append(
            {
                "tag": "Profitability",
                "metric": str(total_profit["value"]),
                "title": "Total Profit",
                "description": "Profit was measured from the mapped profit field.",
                "evidence": "Mapped profit field",
            }
        )

    discount = payload.get("discount_what_if") if isinstance(payload.get("discount_what_if"), dict) else {}
    bucket = _clean_text(discount.get("high_risk_bucket"), "")
    negative_profit_rate = discount.get("negative_profit_rate") or discount.get("current_negative_profit_rate")
    if bucket and bucket != "暂无" and _number(negative_profit_rate) is not None:
        cards.append(
            {
                "tag": "Discount Risk",
                "metric": _format_rate_label(negative_profit_rate),
                "title": f"{bucket} discount tier requires review",
                "description": "The tier has the weakest profit quality in the current threshold analysis.",
                "evidence": "Discount threshold analysis",
            }
        )

    actions = payload.get("priority_actions") if isinstance(payload.get("priority_actions"), list) else []
    for action in actions:
        if len(cards) >= 3:
            break
        if not isinstance(action, dict):
            continue
        issue = _english_client_text(action.get("issue"), "")
        body = _english_client_text(action.get("action"), "")
        evidence = _english_client_text(action.get("evidence"), "")
        if not issue or any(_text_similar(issue, card["title"]) for card in cards):
            continue
        cards.append(
            {
                "tag": "Action Priority",
                "metric": _clean_text(action.get("priority"), ""),
                "title": issue,
                "description": body,
                "evidence": evidence,
            }
        )

    return cards[:3]


def _build_insight_cards(payload: dict[str, Any]) -> list[dict[str, str]]:
    english = is_english_output(str(payload.get("output_language") or ""))
    if english:
        structured_cards = _build_english_structured_insight_cards(payload)
        if structured_cards:
            return structured_cards
    cards: list[dict[str, str]] = []
    seen: set[str] = set()
    final_synthesis = payload.get("final_synthesis") if isinstance(payload.get("final_synthesis"), dict) else {}
    main_conclusions = final_synthesis.get("main_conclusions")
    if isinstance(main_conclusions, list):
        for item in main_conclusions:
            if not isinstance(item, dict):
                continue
            conclusion = _public_text(item.get("conclusion"), "")
            evidence = _public_text(item.get("evidence"), "")
            business_meaning = _public_text(item.get("business_meaning"), "")
            title_source = conclusion or business_meaning or evidence
            title = (
                _build_english_insight_title(title_source, fallback="Business observation")
                if english
                else _chinese_insight_card_title(title_source, fallback="主要经营观察")
            )
            if title in seen:
                continue
            description = business_meaning or evidence or conclusion
            if english:
                description = _english_client_text(description, "This finding should be reviewed with the supporting evidence.")
            if _text_similar(title, description):
                if conclusion and not _text_similar(title, conclusion):
                    description = conclusion
                elif evidence and not _text_similar(title, evidence):
                    description = evidence
                else:
                    description = ""
            if _text_similar(title, description):
                description = ""
            description = _insight_description(description) if not english else description
            evidence_text = _insight_evidence(
                title=title,
                description=description,
                evidence=_english_client_text(evidence or business_meaning, "") if english else evidence or business_meaning,
                english=english,
            )
            seen.add(title)
            cards.append(
                {
                    "tag": _tag_for_section("final_synthesis", english=english, text=title_source),
                    "metric": _pick_insight_metric(evidence, business_meaning, conclusion),
                    "title": title,
                    "description": description,
                    "evidence": evidence_text,
                }
            )
            if len(cards) >= 3:
                break
    if len(cards) >= 3:
        return cards[:3]
    for section, text in _collect_finding_texts(payload):
        title = (
            _build_english_insight_title(text, fallback="Business observation")
            if english
            else _chinese_insight_card_title(text, fallback="主要经营观察")
        )
        if title in seen:
            continue
        description = _english_client_text(text, "Review this business signal with the supporting evidence.") if english else _insight_description(text)
        if _text_similar(title, description):
            description = ""
        evidence_text = _insight_evidence(
            title=title,
            description=description,
            evidence=_english_client_text(text, "") if english else text,
            english=english,
        )
        seen.add(title)
        cards.append(
            {
                "tag": _tag_for_section(section, english=english, text=text),
                "metric": _extract_best_metric(text),
                "title": title,
                "description": description,
                "evidence": evidence_text,
            }
        )
        if len(cards) >= 3:
            break
    if not cards:
        cards.append(
            {
                "tag": "Business Observation" if english else "核心摘要",
                "metric": "",
                "title": "Business review summary is available" if english else "已形成经营复盘摘要",
                "description": (
                    _english_client_text(payload.get("business_theme"), "The report summarizes business findings and recommended actions.")
                    if english
                    else _public_text(payload.get("business_theme"), "本次分析已形成可供业务复盘的摘要和行动建议。")
                ),
                "evidence": "",
            }
        )
    return cards[:3]


def _build_model_panel(payload: dict[str, Any]) -> dict[str, Any] | None:
    english = is_english_output(str(payload.get("output_language") or ""))
    task = _modeling_task(payload)
    if task == "forecast_baseline":
        metrics = _modeling_metrics(payload)
        mae = _metric_from_mapping(metrics, "best_model_mae", "best_baseline_mae", "best_mae", "mae")
        rmse = _metric_from_mapping(metrics, "best_model_rmse", "best_baseline_rmse", "rmse")
        mape = _metric_from_mapping(metrics, "best_model_mape", "best_baseline_mape", "best_mape", "mape")
        baseline = _metric_from_mapping(metrics, "best_model", "best_baseline", "baseline_model")
        best_model = _metric_from_mapping(metrics, "best_model", "best_baseline")
        headline_value = mape if _number(mape) is not None else mae
        weak = _is_weak_model(payload)
        scores = [
            {"label": "MAE", "value": _metric_display(mae)},
            {"label": "RMSE", "value": _metric_display(rmse)},
            {"label": "MAPE", "value": _metric_display(mape)},
            {"label": "baseline", "value": _metric_display(baseline)},
        ]
        return {
            "kind": "forecast_baseline",
            "best_model": _clean_text(best_model, "forecast baseline"),
            "headline_metric": _metric_display(headline_value),
            "metric_label": "MAPE" if _number(mape) is not None else "MAE",
            "gauge_percent": _clip_percent(headline_value),
            "title": (
                "Forecast baseline should only be used as a monitoring reference"
                if english and weak
                else "Forecast baseline can support future model comparison"
                if english
                else "当前预测 baseline 仅适合作为监控参照" if weak else "预测 baseline 可作为后续模型增强的对照基线"
            ),
            "description": (
                "This chronological backtest baseline can support sales monitoring and future model comparison, but it is not a production forecast."
                if english
                else "这是时间顺序回测得到的 baseline 结果，可用于销售监控和后续模型对照，不应包装成生产级自动预测。"
            ),
            "scores": [score for score in scores if _valid_display_value(score["value"])],
        }

    if task == "regression":
        metrics = _modeling_metrics(payload)
        r2 = _metric_from_mapping(metrics, "best_model_r2", "best_r2", "r2")
        mae = _metric_from_mapping(metrics, "best_model_mae", "best_mae", "mae")
        mape = _metric_from_mapping(metrics, "best_model_mape", "best_mape", "mape")
        baseline = _metric_from_mapping(metrics, "baseline_model")
        improvement = _metric_from_mapping(metrics, "improvement_vs_baseline")
        weak = _is_weak_model(payload)
        r2_display = _format_display_number(r2) if english else _format_number(r2)
        mae_display = _format_display_number(mae) if english else _format_number(mae)
        mape_display = _format_display_number(mape) if english else _format_number(mape)
        improvement_display = _format_display_number(improvement) if english else _format_number(improvement)
        return {
            "kind": "regression",
            "best_model": _clean_text(_metric_from_mapping(metrics, "best_model"), "sales regression"),
            "headline_metric": r2_display,
            "metric_label": "R²",
            "gauge_percent": _clip_percent(r2),
            "title": (
                "Regression output should only support sales monitoring"
                if english and weak
                else "Regression output can support sales monitoring comparison"
                if english
                else "仅适合作为销售监控参考，不应包装成生产级预测" if weak else "可作为销售监控和模型增强的对照基线"
            ),
            "description": (
                "Use the regression output for monitoring and review only; add promotion, holiday, price, inventory, and channel variables before stronger decisions."
                if english
                else "预测结果用于销售监控和复盘，不是生产级自动预测；后续应补充节假日、促销、价格、库存、渠道等变量。"
            ),
            "scores": [
                {"label": "MAE", "value": mae_display},
                {"label": "MAPE", "value": mape_display},
                {"label": "baseline", "value": _clean_text(baseline, "")},
                {"label": "Baseline lift" if english else "baseline 改善", "value": improvement_display},
            ],
        }

    modeling = payload.get("modeling_support") if isinstance(payload.get("modeling_support"), dict) else None
    classification_metrics = _modeling_metrics(payload) if task == "classification" else {}
    if modeling and _valid_display_value(modeling.get("best_model")):
        model_source = modeling
        best_model = modeling.get("best_model")
        recall = modeling.get("recall")
        f1 = modeling.get("f1")
        roc_auc = modeling.get("roc_auc")
        false_positive_count = modeling.get("false_positive_count")
        false_negative_count = modeling.get("false_negative_count")
        limitation = modeling.get("limitation")
    elif classification_metrics and _valid_display_value(_metric_from_mapping(classification_metrics, "best_model")):
        model_source = classification_metrics
        best_model = _metric_from_mapping(classification_metrics, "best_model")
        recall = _metric_from_mapping(classification_metrics, "best_recall", "recall")
        f1 = _metric_from_mapping(classification_metrics, "best_f1", "f1")
        roc_auc = _metric_from_mapping(classification_metrics, "best_roc_auc", "roc_auc")
        false_positive_count = _metric_from_mapping(classification_metrics, "false_positive_count")
        false_negative_count = _metric_from_mapping(classification_metrics, "false_negative_count")
        limitation = MODEL_LIMITATION
    else:
        return None
    weak = _is_weak_model(payload)
    return {
        "kind": "classification",
        "best_model": _clean_text(best_model),
        "headline_metric": _format_number(recall),
        "metric_label": "Recall",
        "gauge_percent": _clip_percent(recall),
        "title": (
            "Model output is only an exploratory risk signal"
            if english and weak
            else "Loss-risk model supports review prioritization"
            if english
            else "仅适合作为探索性风险线索，不建议直接用于复核排序" if weak else "可以辅助复核排序，但不能自动决策"
        ),
        "description": (
            "Current model quality is weak, so it should be treated as an exploratory signal until sample size, positive labels, and target definitions are validated."
            if english and weak
            else _english_client_text(limitation, "The model can support review prioritization, but it must not be used for automatic decisions.")
            if english
            else "当前亏损风险模型已跑通，但质量状态偏弱，指标可能受正例数量、样本规模或测试集分布影响。建议先补充样本和目标口径，再评估是否用于复核排序。"
            if weak
            else _public_text(limitation, MODEL_LIMITATION)
        ),
        "scores": [
            {"label": "F1", "value": _format_number(f1)},
            {"label": "ROC AUC", "value": _format_number(roc_auc)},
            {"label": "False positives" if english else "误报亏损数", "value": _format_number(false_positive_count)},
            {"label": "False negatives" if english else "漏判亏损数", "value": _format_number(false_negative_count)},
        ],
    }


_ENGLISH_ACTION_PRIORITY_RANK = {"P1": 1, "P2": 2, "P3": 3}


def _english_action_priority(value: object, *, allow_missing: bool) -> str | None:
    raw = _clean_text(value, "").upper()
    if raw in _ENGLISH_ACTION_PRIORITY_RANK:
        return raw
    if allow_missing and not raw:
        return "P3"
    return None


def _sort_english_action_cards(cards: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed = list(enumerate(cards))
    indexed.sort(key=lambda item: (_ENGLISH_ACTION_PRIORITY_RANK.get(item[1].get("priority", ""), 99), item[0]))
    return [card for _, card in indexed]


def _english_structured_action_card(row: dict[str, Any], *, index: int) -> dict[str, str] | None:
    priority = _english_action_priority(row.get("priority"), allow_missing=False)
    if priority is None:
        return None
    issue = _english_client_text(_public_text(row.get("issue") or row.get("action"), ""), "")
    body = _english_client_text(_public_text(row.get("action"), ""), "Review this priority before execution.")
    evidence = _english_client_text(_public_text(row.get("evidence"), ""), "")
    title = _english_action_card_title(
        issue=issue,
        display_text=body,
        action=body,
        expected_use="",
        evidence=evidence,
        fallback=f"Recommended action {index}",
    )
    if not title or not body:
        return None
    return {"priority": priority, "title": title, "body": body, "evidence": evidence}


def _english_final_synthesis_action_card(row: dict[str, Any], *, index: int) -> dict[str, str] | None:
    priority = _english_action_priority(row.get("priority"), allow_missing=True)
    if priority is None:
        return None
    issue = _english_client_text(_public_text(row.get("issue"), ""), "")
    display_text = _english_client_text(
        _public_text(row.get("display_text"), ""),
        "Review the priority business signal before action.",
    )
    action = _english_client_text(_public_text(row.get("action"), ""), "")
    expected_use = _english_client_text(_public_text(row.get("expected_use"), ""), "")
    evidence = _english_client_text(_public_text(row.get("linked_metric_or_segment") or row.get("evidence"), ""), "")
    title = _english_action_card_title(
        issue=issue,
        display_text=display_text,
        action=action,
        expected_use=expected_use,
        evidence=evidence,
        fallback=f"Recommended action {index}",
    )
    body = display_text or " ".join(part for part in (action, expected_use) if part)
    if not title or not body:
        return None
    return {"priority": priority, "title": title, "body": body, "evidence": evidence}


def _build_action_cards(payload: dict[str, Any]) -> list[dict[str, str]]:
    english = is_english_output(str(payload.get("output_language") or ""))
    final_synthesis = payload.get("final_synthesis") if isinstance(payload.get("final_synthesis"), dict) else {}
    final_actions = final_synthesis.get("recommended_actions")
    if english:
        structured_actions = payload.get("priority_actions") if isinstance(payload.get("priority_actions"), list) else []
        cards = [
            card
            for index, row in enumerate(structured_actions, start=1)
            if isinstance(row, dict)
            for card in [_english_structured_action_card(row, index=index)]
            if card is not None
        ]
        if cards:
            return _sort_english_action_cards(cards)[:3]
        if isinstance(final_actions, list):
            fallback_cards = [
                card
                for index, row in enumerate(final_actions, start=1)
                if isinstance(row, dict)
                for card in [_english_final_synthesis_action_card(row, index=index)]
                if card is not None
            ]
            if fallback_cards:
                return _sort_english_action_cards(fallback_cards)[:3]
        return [
            {
                "priority": "P1",
                "title": "Review priority business signals",
                "body": _english_client_text(
                    payload.get("business_theme"),
                    "Review the report's core metrics, risks, and recommended actions.",
                ),
                "evidence": "",
            }
        ]
    if isinstance(final_actions, list):
        cards: list[dict[str, str]] = []
        for index, row in enumerate(final_actions[:4], start=1):
            if not isinstance(row, dict):
                continue
            issue = _public_text(row.get("issue"), "")
            display_text = _public_text(row.get("display_text"), "")
            action = _public_text(row.get("action"), "")
            expected_use = _public_text(row.get("expected_use"), "")
            if english:
                issue = _english_client_text(issue, "")
                display_text = _english_client_text(display_text, "Review the priority business signal before action.")
                action = _english_client_text(action, "")
                expected_use = _english_client_text(expected_use, "")
            evidence = _public_text(row.get("linked_metric_or_segment") or row.get("evidence"), "")
            if english:
                evidence = _english_client_text(evidence, "")
            title = (
                _english_action_card_title(
                    issue=issue,
                    display_text=display_text,
                    action=action,
                    expected_use=expected_use,
                    evidence=evidence,
                    fallback=f"Recommended action {index}",
                )
                if english
                else action or _first_sentence(display_text, fallback=f"建议行动 {index}")
            )
            body = display_text or (" ".join(part for part in (action, expected_use) if part) if english else "。".join(part for part in (action, expected_use) if part))
            priority = _clean_text(row.get("priority"), f"P{index}").upper()
            if priority not in {"P1", "P2", "P3", "P4"}:
                priority = f"P{index}"
            if title and body:
                cards.append({"priority": priority, "title": title, "body": body, "evidence": evidence})
        if cards:
            return cards

    actions = payload.get("priority_actions") if isinstance(payload.get("priority_actions"), list) else []
    cards: list[dict[str, str]] = []
    for index, row in enumerate(actions[:4], start=1):
        if not isinstance(row, dict):
            continue
        priority = _clean_text(row.get("priority"), f"P{index}").upper()
        if priority not in {"P1", "P2", "P3", "P4"}:
            priority = f"P{index}"
        title = _public_text(row.get("issue") or row.get("action"), f"建议行动 {index}")
        body = _public_text(row.get("action"), title)
        evidence = _public_text(row.get("evidence"), "")
        if english:
            body = _english_client_text(body, "Review this priority before execution.")
            evidence = _english_client_text(evidence, "")
            title = (
                _english_action_card_title(
                    issue=_english_client_text(title, ""),
                    display_text=body,
                    action=body,
                    expected_use="",
                    evidence=evidence,
                    fallback=f"Recommended action {index}",
                )
            )
        cards.append({"priority": priority, "title": title, "body": body, "evidence": evidence})
    if not cards:
        cards.append(
            {
                "priority": "P1",
                "title": "Review priority business signals" if english else "复盘核心经营信号",
                "body": (
                    _english_client_text(payload.get("business_theme"), "Review the report's core metrics, risks, and recommended actions.")
                    if english
                    else _public_text(payload.get("business_theme"), "围绕本次报告的核心指标继续复盘经营动作。")
                ),
                "evidence": "",
            }
        )
    return cards[:4]


def _build_limitations(payload: dict[str, Any], *, has_model: bool) -> list[str]:
    english = is_english_output(str(payload.get("output_language") or ""))
    raw = payload.get("limitations") if isinstance(payload.get("limitations"), list) else []
    cleaned: list[str] = []
    task = _modeling_task(payload)
    has_what_if = _has_real_what_if(payload)
    weak_classification = task == "classification" and _is_weak_model(payload)
    for item in raw:
        text = _public_text(item)
        if english:
            text = _english_client_text(text, "")
        if not text or text == "暂无":
            continue
        if "分析覆盖范围" in text:
            text = "字段缺口会限制部分经营问题的验证深度，需结合业务字段继续补齐验证。"
        if text not in cleaned:
            cleaned.append(text)
    outcome = _modeling_outcome(payload)
    outcome_limitations = outcome.get("limitations")
    if has_model and isinstance(outcome_limitations, list):
        for item in outcome_limitations:
            text = _public_text(item)
            if english:
                text = _english_client_text(text, "")
            if text and text not in cleaned:
                cleaned.append(text)

    filtered: list[str] = []
    for item in cleaned:
        if not has_what_if and _is_what_if_limitation(item):
            continue
        if not has_model and "模型" in item:
            continue
        if weak_classification and _is_classification_review_limitation(item) and "不建议" not in item and "质量偏弱" not in item:
            continue
        if task in {"regression", "forecast_baseline"} and _is_classification_review_limitation(item):
            continue
        if task not in {"regression", "forecast_baseline"} and "预测结果用于销售监控" in item:
            continue
        if item not in filtered:
            filtered.append(item)
    cleaned = filtered

    if weak_classification:
        weak_classification_limit = (
            "Current model quality is weak, so it should not be used directly for review prioritization or automatic decisions."
            if english
            else "当前亏损风险模型质量偏弱，不建议直接用于复核排序或自动决策。"
        )
        cleaned = [weak_classification_limit] + [item for item in cleaned if item != weak_classification_limit]
    if task in {"regression", "forecast_baseline"}:
        regression_limit = (
            "Use forecast outputs for sales monitoring and review only; add promotion, holiday, price, inventory, and channel variables before stronger decisions."
            if english
            else "预测结果用于销售监控和复盘，不是生产级自动预测，需补充促销、节假日、价格、库存、渠道等变量后再评估。"
        )
        if regression_limit not in cleaned:
            cleaned.append(regression_limit)
    if not cleaned:
        cleaned = [
            "Use this report for business review and human judgment support; it does not replace experiments or controlled validation."
            if english
            else "本报告用于经营复盘和人工判断辅助，不替代业务审批、实验设计或因果验证。"
        ]
    return cleaned[:4]


def build_client_report_view_model(payload: dict[str, Any]) -> dict[str, Any]:
    english = is_english_output(str(payload.get("output_language") or ""))
    kpis = _build_kpi_view_model(payload)
    spotlight = _build_spotlight_view_model(payload)
    insights = _build_insight_cards(payload)
    model_panel = _build_model_panel(payload)
    summary = payload.get("executive_summary") if isinstance(payload.get("executive_summary"), list) else []
    summary_texts = [_public_text(item) for item in summary if _public_text(item)]
    final_synthesis = payload.get("final_synthesis") if isinstance(payload.get("final_synthesis"), dict) else {}
    main_conclusions = final_synthesis.get("main_conclusions")
    first_conclusion = main_conclusions[0] if isinstance(main_conclusions, list) and main_conclusions and isinstance(main_conclusions[0], dict) else {}
    hero_title_source = _public_text(first_conclusion.get("conclusion") or first_conclusion.get("business_meaning"), "")
    if not hero_title_source:
        hero_title_source = summary_texts[0] if summary_texts else ""
    if not _headline_like(hero_title_source):
        hero_title_source = str(spotlight.get("title") or payload.get("business_theme") or "本次销售数据已完成核心经营复盘。")
    lead_parts = [
        _public_text(first_conclusion.get("conclusion"), ""),
        _public_text(first_conclusion.get("evidence"), ""),
        _public_text(first_conclusion.get("business_meaning"), ""),
    ]
    lead = " ".join(part for part in lead_parts if part) or " ".join(summary_texts[:2]) or _public_text(payload.get("business_theme"), "本次分析已形成核心经营指标、风险观察和行动建议。")
    if english:
        hero_title_source = _english_client_text(hero_title_source, "Key business finding")
        lead = _english_client_text(
            lead,
            "This report summarizes key metrics, margin risks, model signals, recommended actions, and usage limits.",
        )
    tags = ["Sales Contribution" if english else "销售贡献"]
    if any(("Profit" if english else "利润") in str(kpi.get("label")) for kpi in kpis):
        tags.append("Profit Quality" if english else "利润质量")
    if spotlight.get("kind") == "discount_profit":
        tags.append("Discount Risk" if english else "折扣风险")
    if model_panel:
        tags.append("Model Review" if english else "模型复核")
    view_model = {
        "title": _clean_text(payload.get("title"), "客户经营分析简报"),
        "hero": {
            "headline": (
                _english_client_text(spotlight.get("title"), "")
                or _build_english_report_headline(hero_title_source, fallback="Key business finding")
                if english
                else _compress_report_title(hero_title_source, fallback="本次销售数据已完成核心经营复盘")
            ),
            "lead": lead,
            "tags": tags[:4],
            "dataset_type": _clean_text(payload.get("dataset_type"), "sales_transaction"),
            "sample_size": _format_number(payload.get("sample_size")),
            "field_count": _format_number(payload.get("field_count")),
            "signal_label": spotlight.get("pill", "Business observation" if english else "主要经营观察"),
            "signal_value": spotlight.get("headline_metric", ""),
            "signal_note": spotlight.get("metric_label", "Core signal" if english else "核心信号"),
        },
        "kpis": kpis,
        "spotlight": spotlight,
        "insight_cards": insights,
        "model_panel": model_panel,
        "action_cards": _build_action_cards(payload),
        "limitations": _build_limitations(payload, has_model=model_panel is not None),
    }
    return _englishize_client_payload_strings(view_model) if english else view_model


def render_client_report_html(
    payload: dict[str, Any],
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language or str(payload.get("output_language") or ""))
    labels = {
        "html_lang": "en" if english else "zh-CN",
        "eyebrow": "Sales Analysis Report" if english else "销售经营分析报告",
        "primary_risk": "View Key Risks" if english else "查看主要风险",
        "actions": "View Actions" if english else "查看建议行动",
        "dataset_overview": "Dataset Overview" if english else "数据集概览",
        "data_scale": "Data Scale" if english else "数据规模",
        "rows": "rows" if english else "行",
        "fields": "fields" if english else "字段",
        "dataset_type": "Dataset type" if english else "数据类型",
        "analysis_line": "Analysis Focus" if english else "分析主线",
        "analysis_line_value": "Find risks, then define actions" if english else "先定位风险，再形成动作",
        "analysis_line_note": "Focus on key metrics, risks, findings, and usage limits in the current dataset" if english else "聚焦当前数据里的关键指标、风险、发现与使用边界",
        "risk_spotlight": "Key Risk Spotlight" if english else "主要风险 Spotlight",
        "insights": "Key Findings" if english else "关键发现",
        "insight_count": "Core insights" if english else "核心洞察",
        "recommended_actions": "Recommended Actions" if english else "建议行动",
        "usage_limits": "Usage Limits" if english else "使用边界",
        "model_assist": "Model-Assisted Judgment" if english else "模型辅助判断",
        "model_recall_gauge": "Model recall gauge" if english else "模型召回率仪表盘",
        "core_metrics": "Core Metrics" if english else "核心指标",
        "evidence": "Evidence:" if english else "依据：",
        "signal_strength": "Signal Strength" if english else "信号强度",
        "observation": "Observation" if english else "观察",
    }
    vm = build_client_report_view_model(payload)
    hero = vm["hero"]
    spotlight = vm["spotlight"]
    kpi_html = "".join(
        "<article class='kpi {klass}' style='--accent-soft: {accent};'>"
        "<span class='label'>{label}</span>"
        "<span class='value' title='{full_value}'>{value}</span>"
        "<span class='sub'>{sub}</span>"
        "</article>".format(
            klass=escape(str(kpi.get("class", ""))),
            accent=escape(str(kpi.get("accent", "rgba(37,99,235,.10)"))),
            label=escape(str(kpi.get("label", ""))),
            full_value=escape(str(kpi.get("full_value", kpi.get("value", "")))),
            value=escape(str(kpi.get("value", ""))),
            sub=escape(str(kpi.get("sub", ""))),
        )
        for kpi in vm["kpis"]
    )
    bars_html = "".join(
        "<div class='bar-row'><span>{label}</span><div class='bar-track'><div class='bar-fill' style='--w: {percent:.2f}%;'></div></div><b>{value}</b></div>".format(
            label=escape(str(bar.get("label", ""))),
            percent=max(0.0, min(100.0, float(bar.get("percent", 0.0) or 0.0))),
            value=escape(str(bar.get("value_label", ""))),
        )
        for bar in spotlight.get("bars", [])
        if isinstance(bar, dict) and _valid_display_value(bar.get("value_label"))
    )
    if not bars_html:
        bars_html = (
            f"<div class='bar-row'><span>{labels['signal_strength']}</span><div class='bar-track'>"
            f"<div class='bar-fill' style='--w: {max(0.0, min(100.0, float(spotlight.get('percent', 0.0) or 0.0))):.2f}%;'></div>"
            f"</div><b>{escape(str(spotlight.get('headline_metric', labels['observation'])))}</b></div>"
        )
    insight_html = "".join(
        "<article class='insight-card' style='--glow: {glow};'>"
        "<span class='small-label'>{tag}</span>"
        "{metric}"
        "<h3>{title}</h3>"
        "<p>{description}</p>"
        "{evidence}"
        "</article>".format(
            glow=("rgba(239,68,68,.10)" if index == 0 else "rgba(249,115,22,.12)" if index == 1 else "rgba(37,99,235,.10)"),
            tag=escape(card["tag"]),
            metric=f"<div class='metric-xl {('red' if index == 0 else 'orange' if index == 1 else 'blue')}'>{escape(card['metric'])}</div>"
            if card.get("metric")
            else "",
            title=escape(card["title"]),
            description=escape(card["description"]),
            evidence=f"<div class='insight-evidence'><b>{labels['evidence']}</b>{escape(card['evidence'])}</div>" if card.get("evidence") else "",
        )
        for index, card in enumerate(vm["insight_cards"])
    )
    model_panel = vm["model_panel"]
    model_html = ""
    if isinstance(model_panel, dict):
        scores_html = "".join(
            f"<div class='score'><span>{escape(str(score.get('label', '')))}</span><b>{escape(str(score.get('value', '')))}</b></div>"
            for score in model_panel.get("scores", [])
            if isinstance(score, dict) and _valid_display_value(score.get("value"))
        )
        gauge_percent = max(0.0, min(100.0, float(model_panel.get("gauge_percent", 0.0) or 0.0)))
        gauge_offset = 100.0 - gauge_percent
        model_html = f"""
    <div class="section-head">
      <div><h2>{labels["model_assist"]}</h2></div>
      <span class="mini-pill">{escape(str(model_panel.get("best_model", "")))}</span>
    </div>
    <section class="panel model-panel">
      <div class="gauge-wrap" aria-label="{labels['model_recall_gauge']}">
        <svg viewBox="0 0 260 160" role="img" aria-label="{escape(str(model_panel.get("metric_label", "Model")))} {escape(str(model_panel.get("headline_metric", "")))}">
          <path d="M40 125 A90 90 0 0 1 220 125" fill="none" stroke="#e5e7eb" stroke-width="20" stroke-linecap="round"/>
          <path class="gauge-arc" d="M40 125 A90 90 0 0 1 220 125" fill="none" stroke="url(#gaugeGradient)" stroke-width="20" stroke-linecap="round" pathLength="100" stroke-dasharray="100" stroke-dashoffset="{gauge_offset:.2f}"/>
          <defs><linearGradient id="gaugeGradient" x1="0" x2="1"><stop offset="0" stop-color="#06b6d4"/><stop offset="1" stop-color="#2563eb"/></linearGradient></defs>
          <text x="130" y="103" text-anchor="middle" font-size="42" font-weight="900" fill="#111827">{escape(str(model_panel.get("headline_metric", "")))}</text>
          <text x="130" y="132" text-anchor="middle" font-size="15" fill="#667085">{escape(str(model_panel.get("metric_label", "")))}</text>
        </svg>
      </div>
      <div class="model-copy">
        <h3>{escape(str(model_panel.get("title", "")))}</h3>
        <p>{escape(str(model_panel.get("description", "")))}</p>
        <div class="score-grid">{scores_html}</div>
      </div>
    </section>"""
    priority_styles = {
        "P1": "#ef4444;--priority-color-2:#f97316",
        "P2": "#2563eb;--priority-color-2:#06b6d4",
        "P3": "#475569;--priority-color-2:#111827",
        "P4": "#7c3aed;--priority-color-2:#2563eb",
    }
    actions_html = "".join(
        "<article class='action-card' style='--priority-color:{style};'>"
        "<div class='priority'>{priority}</div>"
        "<div class='action-body'><h3>{title}</h3><p>{body}</p>{evidence}</div>"
        "</article>".format(
            style=priority_styles.get(action["priority"], "#475569;--priority-color-2:#111827"),
            priority=escape(action["priority"]),
            title=escape(action["title"]),
            body=escape(action["body"]),
            evidence=f"<div class='evidence-ribbon'><b>{labels['evidence']}</b>{escape(action['evidence'])}</div>" if action.get("evidence") else "",
        )
        for action in vm["action_cards"]
    )
    limitations_html = "".join(f"<li>{escape(item)}</li>" for item in vm["limitations"])
    tags_html = "".join(f"<span class='tag'>{escape(tag)}</span>" for tag in hero["tags"])
    html = f"""<!doctype html>
<html lang="{labels["html_lang"]}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(str(vm["title"]))}</title>
  <style>
    :root {{ --ink:#111827; --muted:#667085; --card:rgba(255,255,255,.9); --line:rgba(15,23,42,.1); --shadow:0 26px 70px rgba(15,23,42,.14); --soft-shadow:0 16px 42px rgba(15,23,42,.08); --blue:#2563eb; --cyan:#06b6d4; --red:#ef4444; --red-dark:#b91c1c; --orange:#f97316; --green:#16a34a; --max:1200px; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); font-family:Inter,"Microsoft YaHei","PingFang SC",Arial,sans-serif; background:radial-gradient(circle at 8% 4%,rgba(37,99,235,.22),transparent 26%),radial-gradient(circle at 92% 10%,rgba(249,115,22,.18),transparent 26%),radial-gradient(circle at 50% 34%,rgba(6,182,212,.12),transparent 30%),linear-gradient(180deg,#edf4ff 0%,#f8fafc 42%,#f3f6fb 100%); overflow-x:hidden; }}
    body::before {{ content:""; position:fixed; inset:0; pointer-events:none; background-image:linear-gradient(rgba(15,23,42,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(15,23,42,.035) 1px,transparent 1px); background-size:42px 42px; mask-image:linear-gradient(180deg,rgba(0,0,0,.78),transparent 70%); z-index:-1; }}
    main {{ width:min(var(--max),calc(100% - 36px)); margin:0 auto; padding:28px 0 72px; }}
    .report-shell {{ display:grid; gap:22px; }}
    .hero {{ position:relative; overflow:hidden; min-height:380px; padding:clamp(24px,3.2vw,36px); color:#fff; border-radius:42px; background:radial-gradient(circle at 88% 24%,rgba(255,255,255,.16),transparent 28%),linear-gradient(135deg,#07111f 0%,#0f1f3f 48%,#2563eb 100%); box-shadow:var(--shadow); isolation:isolate; }}
    .hero::before {{ content:""; position:absolute; width:680px; height:680px; right:-260px; top:-260px; border-radius:50%; background:radial-gradient(circle,rgba(255,255,255,.18),rgba(255,255,255,.03) 42%,transparent 66%); z-index:-1; }}
    .hero-grid {{ display:grid; grid-template-columns:minmax(0,1.12fr) 360px; gap:24px; align-items:stretch; position:relative; z-index:1; }}
    .eyebrow-row {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; }}
    .eyebrow,.tag {{ display:inline-flex; align-items:center; gap:7px; padding:8px 12px; border-radius:999px; font-size:13px; line-height:1; white-space:nowrap; }}
    .eyebrow {{ color:#dbeafe; background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.18); }}
    .tag {{ color:#bfdbfe; background:rgba(37,99,235,.22); border:1px solid rgba(147,197,253,.24); }}
    h1 {{ margin:22px 0 16px; max-width:720px; font-size:clamp(34px,4.9vw,58px); line-height:1.04; letter-spacing:-.045em; }}
    .hero-lead {{ max-width:720px; margin:0; color:#d7e6ff; font-size:clamp(16px,1.35vw,18px); line-height:1.78; }}
    .hero-lead strong {{ color:#fff7ed; font-weight:900; }}
    .hero-actions {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:24px; }}
    .ghost-button {{ display:inline-flex; align-items:center; gap:9px; padding:12px 15px; border-radius:999px; color:#eef6ff; text-decoration:none; background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.18); backdrop-filter:blur(18px); }}
    .hero-side {{ display:grid; gap:14px; align-content:stretch; }}
    .hero-metric-card {{ position:relative; overflow:hidden; padding:18px; min-height:118px; border-radius:24px; border:1px solid rgba(255,255,255,.16); background:rgba(255,255,255,.12); backdrop-filter:blur(18px); }}
    .hero-metric-card span {{ display:block; color:#c7d2fe; font-size:13px; }}
    .hero-metric-card b {{ display:block; margin-top:8px; font-size:27px; line-height:1.05; letter-spacing:-.035em; }}
    .hero-metric-card small {{ display:block; margin-top:9px; color:#dbeafe; line-height:1.5; }}
    .hero-metric-card.hot::after {{ content:""; position:absolute; right:-54px; bottom:-54px; width:170px; height:170px; border-radius:50%; background:rgba(239,68,68,.34); filter:blur(2px); }}
    html[lang="en"] h1 {{ font-size:clamp(30px,2.65rem,48px); line-height:1.08; letter-spacing:0; }}
    html[lang="en"] .insight-card h3 {{ font-size:18px; line-height:1.25; letter-spacing:0; }}
    html[lang="en"] .spotlight h3 {{ font-size:clamp(21px,1.65rem,28px); line-height:1.24; letter-spacing:0; }}
    .kpi-dock {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin:-56px 24px 4px; position:relative; z-index:5; }}
    .kpi {{ position:relative; overflow:hidden; min-height:142px; padding:20px; border-radius:26px; background:var(--card); border:1px solid rgba(255,255,255,.72); box-shadow:var(--soft-shadow); backdrop-filter:blur(18px); transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease; }}
    .kpi:hover {{ transform:translateY(-5px); box-shadow:0 24px 58px rgba(15,23,42,.13); border-color:rgba(37,99,235,.18); }}
    .kpi::after {{ content:""; position:absolute; right:-36px; top:-38px; width:110px; height:110px; border-radius:50%; background:var(--accent-soft,rgba(37,99,235,.1)); }}
    .kpi .label {{ display:flex; align-items:center; justify-content:space-between; gap:8px; color:var(--muted); font-size:13px; }}
    .kpi .value {{ display:block; margin-top:16px; font-size:clamp(25px,2.5vw,34px); font-weight:900; letter-spacing:-.06em; }}
    .kpi .sub {{ display:block; margin-top:8px; color:#7a8597; font-size:13px; line-height:1.45; }}
    .kpi.risk .value {{ color:var(--red-dark); }} .kpi.good .value {{ color:var(--green); }} .kpi.model .value {{ color:var(--blue); }}
    .section-head {{ display:flex; align-items:end; justify-content:space-between; gap:18px; margin:42px 0 18px; }}
    .section-head h2 {{ margin:0; font-size:clamp(26px,3vw,36px); letter-spacing:-.05em; }}
    .mini-pill {{ display:inline-flex; align-items:center; gap:8px; padding:9px 13px; border-radius:999px; border:1px solid var(--line); background:rgba(255,255,255,.72); color:#64748b; font-size:13px; white-space:nowrap; box-shadow:0 10px 28px rgba(15,23,42,.05); }}
    .panel {{ border:1px solid var(--line); border-radius:34px; background:rgba(255,255,255,.88); box-shadow:var(--soft-shadow); backdrop-filter:blur(16px); }}
    .spotlight {{ position:relative; overflow:hidden; display:grid; grid-template-columns:220px minmax(0,1fr); gap:34px; padding:28px; min-height:250px; background:rgba(255,255,255,.92); transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease; }}
    .spotlight:hover,.model-panel:hover {{ transform:translateY(-5px) scale(1.003); box-shadow:0 28px 64px rgba(15,23,42,.13); border-color:rgba(37,99,235,.22); }}
    .ring-card {{ display:grid; place-items:center; min-height:210px; }}
    .risk-ring {{ width:190px; height:190px; display:grid; place-items:center; border-radius:50%; background:radial-gradient(circle 67px,#fff 98%,transparent 100%),conic-gradient(#e5242a var(--value),#fee2e2 0); }}
    .risk-ring strong {{ display:block; color:#e5242a; font-size:38px; line-height:1; letter-spacing:-.06em; text-align:center; }}
    .risk-ring span {{ display:block; margin-top:7px; color:var(--muted); font-size:13px; text-align:center; }}
    .spotlight-copy {{ position:relative; z-index:2; display:grid; align-content:center; gap:16px; padding-right:10px; }}
    .spotlight h3 {{ margin:0; font-size:clamp(22px,2.1vw,30px); line-height:1.2; letter-spacing:-.035em; }}
    .spotlight p {{ margin:0; color:#667085; line-height:1.65; font-size:16px; }}
    .bar-stack {{ display:grid; gap:12px; margin-top:2px; }}
    .bar-row {{ display:grid; grid-template-columns:96px minmax(0,1fr) 74px; align-items:center; gap:12px; color:#64748b; font-size:14px; }}
    .bar-track {{ height:12px; overflow:hidden; border-radius:999px; background:#e5e7eb; }}
    .bar-fill {{ height:100%; width:var(--w); border-radius:inherit; background:linear-gradient(90deg,#fb923c,#e5242a); }}
    .insight-layout {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; }}
    .insight-card {{ position:relative; overflow:hidden; padding:24px 24px 28px; min-height:260px; border:1px solid var(--line); border-radius:26px; background:rgba(255,255,255,.9); box-shadow:var(--soft-shadow); transition:transform .25s ease,box-shadow .25s ease; }}
    .insight-card:hover {{ transform:translateY(-4px); box-shadow:0 24px 56px rgba(15,23,42,.11); }}
    .insight-card::after {{ content:""; position:absolute; right:-40px; bottom:-42px; width:142px; height:142px; border-radius:999px; background:var(--glow,rgba(37,99,235,.1)); }}
    .small-label {{ display:inline-flex; width:fit-content; padding:7px 10px; border-radius:999px; background:#eef2ff; color:#4f46e5; font-size:13px; font-weight:700; }}
    .metric-xl {{ margin:18px 0 14px; font-size:clamp(30px,3vw,40px); line-height:1; font-weight:1000; letter-spacing:-.05em; }}
    .red {{ color:var(--red-dark); }} .orange {{ color:var(--orange); }} .blue {{ color:var(--blue); }}
    .insight-card h3 {{ position:relative; z-index:2; margin:0 0 8px; font-size:19px; letter-spacing:-.03em; }}
    .insight-card p {{ position:relative; z-index:2; margin:0; color:var(--muted); line-height:1.65; }}
    .insight-evidence {{ position:relative; z-index:2; margin-top:14px; padding:10px 12px; border-left:3px solid rgba(37,99,235,.35); border-radius:12px; background:rgba(239,246,255,.72); color:#475569; font-size:13px; line-height:1.55; }}
    .model-panel {{ padding:26px; display:grid; grid-template-columns:300px minmax(0,1fr); gap:24px; align-items:center; transition:transform .25s ease,box-shadow .25s ease,border-color .25s ease; }}
    .gauge-wrap {{ position:relative; display:grid; place-items:center; min-height:248px; border-radius:30px; background:radial-gradient(circle at 50% 20%,rgba(37,99,235,.12),transparent 48%),linear-gradient(145deg,#f8fbff,#fff); border:1px solid var(--line); overflow:hidden; }}
    .gauge-wrap::before {{ content:""; position:absolute; width:178px; height:178px; border-radius:50%; background:radial-gradient(circle,rgba(6,182,212,.16),transparent 66%); }}
    .gauge-wrap svg {{ width:245px; max-width:100%; height:158px; }}
    .gauge-arc {{ filter:drop-shadow(0 8px 12px rgba(37,99,235,.14)); }}
    .model-copy h3 {{ margin:0 0 10px; font-size:26px; letter-spacing:-.05em; }}
    .model-copy p {{ margin:0; color:var(--muted); line-height:1.78; }}
    .score-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; margin-top:18px; }}
    .score {{ padding:14px; border-radius:18px; border:1px solid #edf2f7; background:#f8fafc; }}
    .score span {{ display:block; color:#64748b; font-size:13px; }}
    .score b {{ display:block; margin-top:6px; font-size:22px; letter-spacing:-.04em; }}
    .action-grid {{ display:grid; gap:16px; }}
    .action-card {{ position:relative; display:grid; grid-template-columns:74px minmax(0,1fr); gap:17px; padding:22px; border-radius:28px; border:1px solid var(--line); background:rgba(255,255,255,.92); box-shadow:var(--soft-shadow); overflow:hidden; transition:transform .25s ease,box-shadow .25s ease; }}
    .action-card:hover {{ transform:translateY(-4px); box-shadow:0 28px 62px rgba(15,23,42,.12); }}
    .action-card::after {{ content:""; position:absolute; inset:0 auto 0 0; width:5px; background:var(--priority-color,var(--red)); }}
    .priority {{ width:58px; height:58px; display:grid; place-items:center; border-radius:19px; color:#fff; font-weight:950; background:linear-gradient(135deg,var(--priority-color,#ef4444),var(--priority-color-2,#f97316)); box-shadow:0 14px 28px rgba(239,68,68,.22); }}
    .action-body h3 {{ margin:0 0 9px; font-size:20px; letter-spacing:-.03em; }}
    .action-body p {{ margin:0; color:var(--muted); line-height:1.72; }}
    .evidence-ribbon {{ position:relative; z-index:2; margin-top:16px; padding:12px 14px 12px 15px; border-left:4px solid var(--evidence,#f97316); border-radius:14px; background:var(--evidence-bg,#fff7ed); color:var(--evidence-ink,#9a3412); font-size:14px; line-height:1.58; transition:transform .2s ease,filter .2s ease; }}
    .evidence-ribbon:hover {{ transform:translateX(3px); filter:saturate(1.08); }}
    .footer-limit {{ margin-top:26px; padding:24px; border-radius:30px; color:#d7e1f2; background:radial-gradient(circle at 92% 0%,rgba(37,99,235,.22),transparent 34%),linear-gradient(135deg,#0f172a,#111827); box-shadow:var(--soft-shadow); }}
    .footer-limit h2 {{ margin:0 0 12px; color:#fff; font-size:22px; letter-spacing:-.03em; }}
    .footer-limit ul {{ margin:0; padding-left:20px; line-height:1.82; }}
    @media (max-width:980px) {{ main {{ width:min(100% - 26px,var(--max)); }} .hero-grid,.spotlight,.insight-layout,.model-panel {{ grid-template-columns:1fr; }} .kpi-dock {{ grid-template-columns:repeat(2,minmax(0,1fr)); margin:16px 0 0; }} .hero {{ min-height:auto; }} }}
    @media (max-width:620px) {{ main {{ width:min(100% - 18px,var(--max)); padding-top:12px; }} .hero {{ border-radius:28px; padding:24px; }} h1 {{ font-size:clamp(32px,10vw,44px); }} .kpi-dock,.score-grid {{ grid-template-columns:1fr; }} .section-head {{ align-items:start; flex-direction:column; }} .action-card {{ grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:82px 1fr 64px; font-size:13px; }} }}
    @media (prefers-reduced-motion: reduce) {{ *,*::before,*::after {{ animation:none !important; transition:none !important; scroll-behavior:auto !important; }} }}
  </style>
</head>
<body>
<main>
  <div class="report-shell">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <div class="eyebrow-row"><span class="eyebrow">{labels["eyebrow"]}</span>{tags_html}</div>
          <h1>{escape(str(hero["headline"]))}</h1>
          <p class="hero-lead">{escape(str(hero["lead"]))}</p>
          <div class="hero-actions"><a class="ghost-button" href="#risk">{labels["primary_risk"]}</a><a class="ghost-button" href="#actions">{labels["actions"]}</a></div>
        </div>
        <aside class="hero-side" aria-label="{labels["dataset_overview"]}">
          <div class="hero-metric-card hot"><span>{escape(str(hero["signal_label"]))}</span><b>{escape(str(hero["signal_value"]))}</b><small>{escape(str(hero["signal_note"]))}</small></div>
          <div class="hero-metric-card"><span>{labels["data_scale"]}</span><b>{escape(str(hero["sample_size"]))} {labels["rows"]} · {escape(str(hero["field_count"]))} {labels["fields"]}</b><small>{labels["dataset_type"]}: {escape(str(hero["dataset_type"]))}</small></div>
          <div class="hero-metric-card"><span>{labels["analysis_line"]}</span><b>{labels["analysis_line_value"]}</b><small>{labels["analysis_line_note"]}</small></div>
        </aside>
      </div>
    </section>
    <section class="kpi-dock" aria-label="{labels["core_metrics"]}">{kpi_html}</section>
    <div id="risk" class="section-head"><div><h2>{labels["risk_spotlight"]}</h2></div><span class="mini-pill">{escape(str(spotlight.get("pill", "Business Observation" if english else "主要经营观察")))}</span></div>
    <section class="panel spotlight">
      <div class="ring-card"><div class="risk-ring" style="--value: {max(0.0, min(100.0, float(spotlight.get("percent", 0.0) or 0.0))):.2f}%;"><div><strong>{escape(str(spotlight.get("headline_metric", "")))}</strong><span>{escape(str(spotlight.get("metric_label", "")))}</span></div></div></div>
      <div class="spotlight-copy"><h3>{escape(str(spotlight.get("title", "")))}</h3><p>{escape(str(spotlight.get("description", "")))}</p><div class="bar-stack" aria-label="{labels["risk_spotlight"]}">{bars_html}</div></div>
    </section>
    <div class="section-head"><div><h2>{labels["insights"]}</h2></div><span class="mini-pill">{labels["insight_count"]} × {len(vm["insight_cards"])}</span></div>
    <section class="insight-layout">{insight_html}</section>
    {model_html}
    <div id="actions" class="section-head"><div><h2>{labels["recommended_actions"]}</h2></div><span class="mini-pill">P1 / P2 / P3</span></div>
    <section class="action-grid">{actions_html}</section>
    <section class="footer-limit"><h2>{labels["usage_limits"]}</h2><ul>{limitations_html}</ul></section>
  </div>
</main>
<script>
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!prefersReduced) {{
    for (const node of document.querySelectorAll("[data-count]")) {{
      const target = Number(node.dataset.count || 0);
      if (!Number.isFinite(target)) continue;
      const start = performance.now();
      const duration = 700;
      function tick(now) {{
        const progress = Math.min((now - start) / duration, 1);
        node.textContent = Math.round(target * progress).toLocaleString("en-US");
        if (progress < 1) requestAnimationFrame(tick);
      }}
      requestAnimationFrame(tick);
    }}
  }}
</script>
</body>
</html>"""
    return html
