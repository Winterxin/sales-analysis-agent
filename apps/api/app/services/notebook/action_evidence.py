from __future__ import annotations

import re
from typing import Any

from app.schemas.report import AnalysisReport
from app.services.notebook.field_utils import FIELD_LABELS
from app.services.notebook.formatters import format_percent, format_scalar
from app.services.notebook.markdown_sanitizer import clean_business_text


MEASURE_FIELDS_FOR_ACTION = {
    "sales_amount",
    "sales_share",
    "profit",
    "profit_margin",
    "negative_profit_rate",
    "loss_rate",
    "avg_profit",
    "profit_margin_spread",
    "order_count",
    "customer_count",
    "quantity",
    "avg_order_value",
    "line_count",
    "repeat_customer_rate",
    "line_per_order_avg",
    "avg_basket_size",
}


CONCRETE_EVIDENCE_FIELDS = {
    "product": ("product_name",),
    "category": ("category", "sub_category"),
    "slice": ("segment", "region", "category", "sub_category"),
}


ROW_KEY_ALIASES = {
    "sales": "sales_amount",
    "sales_amount": "sales_amount",
    "profit": "profit",
    "discount": "discount",
    "threshold": "threshold",
    "product_name": "product_name",
    "product": "product_name",
    "category": "category",
    "sub_category": "sub_category",
    "subcategory": "sub_category",
    "segment": "segment",
    "region": "region",
    "country": "country",
    "quantity": "quantity",
    "order_id": "order_id",
    "customer_id": "customer_id",
    "productline": "productline",
    "deal_size": "deal_size",
    "dealsize": "deal_size",
    "status": "order_status",
    "order_status": "order_status",
}


ACTION_EVIDENCE_RULES: dict[str, dict[str, Any]] = {
    "discount": {
        "focuses": ("discount_erosion_focus", "profit_quality_focus"),
        "modules": ("discount_profit_analysis",),
        "tables": ("discount", "bucket", "profit", "loss", "risk", "worst"),
        "row_fields": (
            "discount_bucket",
            "threshold",
            "discount",
            "negative_profit_rate",
            "loss_rate",
            "profit_margin",
            "avg_profit",
            "risk_level",
            "profit_margin_spread",
        ),
        "metrics": ("negative_profit_rate", "profit_margin_spread", "avg_profit", "loss_rate"),
        "label": "折扣利润",
    },
    "category": {
        "focuses": ("product_concentration_focus", "profit_quality_focus"),
        "modules": ("product_contribution_analysis", "discount_profit_analysis"),
        "tables": ("category", "subcategory", "risk", "loss", "profit"),
        "row_fields": (
            "category",
            "sub_category",
            "sales_amount",
            "profit",
            "avg_profit",
            "profit_margin",
            "negative_profit_rate",
            "loss_rate",
            "sales_share",
        ),
        "metrics": ("top_category_sales_share", "category_count", "profit_margin_spread"),
        "label": "类目",
    },
    "product": {
        "focuses": ("product_concentration_focus", "profit_quality_focus"),
        "modules": ("product_contribution_analysis", "discount_profit_analysis"),
        "tables": ("product", "high_sales_low_profit", "loss", "worst", "profit"),
        "row_fields": (
            "product_name",
            "sales_amount",
            "profit",
            "avg_profit",
            "profit_margin",
            "negative_profit_rate",
            "loss_rate",
            "sales_share",
        ),
        "metrics": ("top_product_sales_share", "product_count", "top_product"),
        "label": "商品",
    },
    "country": {
        "focuses": ("country_market_focus",),
        "modules": ("country_market_analysis",),
        "tables": ("country", "market"),
        "row_fields": ("country", "sales_amount", "sales_share", "order_count", "avg_order_value", "customer_count"),
        "metrics": ("country_count", "top_country_sales_share", "country_order_count"),
        "label": "国家",
    },
    "productline": {
        "focuses": ("productline_performance_focus",),
        "modules": ("product_contribution_analysis", "order_structure_analysis"),
        "tables": ("productline",),
        "row_fields": ("productline", "sales_amount", "sales_share", "quantity", "order_count"),
        "metrics": ("productline_count", "productline_sales_share", "productline_quantity"),
        "label": "产品线",
    },
    "deal_size": {
        "focuses": ("deal_size_focus",),
        "modules": ("order_structure_analysis",),
        "tables": ("deal_size", "deal"),
        "row_fields": ("deal_size", "sales_amount", "sales_share", "order_count", "avg_order_value"),
        "metrics": ("distinct_deal_size_count", "deal_size_sales_share", "avg_order_value"),
        "label": "交易规模",
    },
    "status": {
        "focuses": ("order_status_focus",),
        "modules": ("order_structure_analysis",),
        "tables": ("status", "order_status"),
        "row_fields": ("order_status", "sales_amount", "sales_share", "order_count", "line_count"),
        "metrics": ("distinct_order_status_count", "status_order_count", "order_count"),
        "label": "订单状态",
    },
    "customer_order": {
        "focuses": ("customer_order_structure_focus",),
        "modules": ("order_structure_analysis",),
        "tables": ("order", "basket", "customer"),
        "row_fields": ("order_id", "customer_id", "order_count", "repeat_customer_rate", "line_per_order_avg", "avg_basket_size"),
        "metrics": ("repeat_customer_rate", "line_per_order_avg", "avg_basket_size", "order_count", "customer_count"),
        "label": "客户订单",
    },
    "slice": {
        "focuses": ("segment_region_focus", "profit_quality_focus"),
        "modules": ("dimension_breakdown_analysis",),
        "tables": ("weak", "segment", "region", "dimension", "risk", "loss", "profit"),
        "row_fields": (
            "segment",
            "region",
            "category",
            "sales_amount",
            "profit",
            "avg_profit",
            "profit_margin",
            "negative_profit_rate",
            "loss_rate",
        ),
        "metrics": ("segment_count", "region_count", "profit_margin_spread"),
        "label": "组合切片",
    },
}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _canonical_key(key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
    return ROW_KEY_ALIASES.get(normalized, normalized)


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in row.items():
        canonical_key = _canonical_key(key)
        if canonical_key not in canonical or not _has_business_value(canonical.get(canonical_key)):
            canonical[canonical_key] = value
    return canonical


def _format_metric_value(key: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.strip().endswith("%"):
        return clean_business_text(value)
    if any(token in key for token in ("rate", "share", "margin")) and isinstance(value, (int, float)):
        return format_percent(value)
    return format_scalar(value)


def _metric_label(key: str) -> str:
    labels = {
        "sales_amount": "销售额",
        "sales_share": "销售占比",
        "profit": "利润",
        "profit_margin": "利润率",
        "negative_profit_rate": "亏损率",
        "loss_rate": "亏损率",
        "avg_profit": "平均利润",
        "order_count": "订单数",
        "customer_count": "客户数",
        "quantity": "销量",
        "avg_order_value": "平均订单金额",
        "line_per_order_avg": "平均每单行数",
        "avg_basket_size": "平均篮子大小",
        "repeat_customer_rate": "复购客户占比",
        "risk_level": "风险等级",
    }
    return labels.get(key, FIELD_LABELS.get(key, key))


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100
            except ValueError:
                return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _discount_depth(value: Any) -> float:
    if value is None:
        return 0.0
    number = _as_number(value)
    if number is not None:
        return number * 100 if abs(number) <= 1 else number
    text = str(value)
    numbers = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return 0.0
    return max(numbers)


def _low_risk_discount_row(row: dict[str, Any]) -> bool:
    discount_depth = max(_discount_depth(row.get("discount_bucket")), _discount_depth(row.get("discount")))
    loss_rate = _as_number(row.get("negative_profit_rate"))
    if loss_rate is None:
        loss_rate = _as_number(row.get("loss_rate"))
    avg_profit = _as_number(row.get("avg_profit"))
    return discount_depth <= 10 and (loss_rate or 0) <= 0 and avg_profit is not None and avg_profit > 0


def _risk_level_rank(value: Any) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value or "").strip().lower(), 0)


def _discount_threshold_sort_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    loss_rate = _as_number(row.get("negative_profit_rate"))
    if loss_rate is None:
        loss_rate = _as_number(row.get("loss_rate"))
    profit_margin = _as_number(row.get("profit_margin"))
    avg_profit = _as_number(row.get("avg_profit"))
    return (
        _risk_level_rank(row.get("risk_level")),
        loss_rate if loss_rate is not None else -1.0,
        -(profit_margin if profit_margin is not None else 999999.0),
        -(avg_profit if avg_profit is not None else 999999.0),
    )


def _has_business_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"none", "nan", "unknown", "null"}


def _object_specificity_score(action_type: str, row: dict[str, Any]) -> float:
    fields = CONCRETE_EVIDENCE_FIELDS.get(action_type)
    if not fields:
        return 0.0
    concrete_count = sum(1 for field in fields if _has_business_value(row.get(field)))
    if concrete_count <= 0:
        return -220.0
    return 80.0 + min(40.0, concrete_count * 20.0)


def evidence_score(action_type: str, table_name: str, row: dict[str, Any]) -> float:
    table = table_name.lower()
    score = 0.0
    if any(term in table for term in ("risk", "loss", "loss_making", "worst", "high_sales_low_profit")):
        score += 45
    if action_type in {"discount", "category", "product", "slice"}:
        loss_rate = _as_number(row.get("negative_profit_rate"))
        if loss_rate is None:
            loss_rate = _as_number(row.get("loss_rate"))
        if loss_rate is not None:
            score += max(0.0, loss_rate) * 120
        avg_profit = _as_number(row.get("avg_profit"))
        if avg_profit is None:
            avg_profit = _as_number(row.get("profit"))
        if avg_profit is not None:
            score += 50 if avg_profit < 0 else max(0.0, 15 - min(avg_profit, 15))
            if avg_profit < 0:
                score += min(50, abs(avg_profit) / 2)
        margin = _as_number(row.get("profit_margin"))
        if margin is not None:
            score += 45 if margin < 0 else max(0.0, (0.08 - margin) * 200)
        score += min(40, max(_discount_depth(row.get("discount_bucket")), _discount_depth(row.get("discount"))))
        sales = _as_number(row.get("sales_amount"))
        if sales is not None and sales > 0:
            score += min(15, sales / 10000)
    if action_type == "discount" and _low_risk_discount_row(row):
        score -= 250
    score += _object_specificity_score(action_type, row)
    return score


def row_to_business_evidence(action_type: str, row: dict[str, Any]) -> str:
    rule = ACTION_EVIDENCE_RULES[action_type]
    preferred_fields = tuple(rule["row_fields"])
    label = str(rule["label"])
    dimension_keys: tuple[str, ...] = ()
    if action_type == "slice":
        dimension_keys = tuple(
            key
            for key in ("segment", "region", "category", "sub_category")
            if _has_business_value(row.get(key))
        )
    dimension_key = next((key for key in preferred_fields if key in row and key not in MEASURE_FIELDS_FOR_ACTION), "")
    parts: list[str] = []
    if dimension_keys:
        dimension_label = " / ".join(format_scalar(row.get(key)) for key in dimension_keys)
        parts.append(f"{label}「{dimension_label}」")
    elif dimension_key:
        parts.append(f"{label}「{format_scalar(row.get(dimension_key))}」")
    else:
        parts.append(f"{label}维度")
    for key in preferred_fields:
        if key == dimension_key or key in dimension_keys or key not in row:
            continue
        formatted = _format_metric_value(key, row.get(key))
        if formatted:
            parts.append(f"{_metric_label(key)}为 {formatted}")
        if len(parts) >= 4:
            break
    if action_type == "discount" and "risk_level" in row and not any("风险等级" in part for part in parts):
        formatted = _format_metric_value("risk_level", row.get("risk_level"))
        if formatted:
            parts.append(f"{_metric_label('risk_level')}为 {formatted}")
    return clean_business_text("，".join(parts) + "。")


def select_action_evidence(
    action_type: str,
    report: AnalysisReport,
    profile: dict[str, Any],
    evidence_pack: dict[str, Any] | None,
    used_evidence: set[str] | None = None,
) -> str:
    rule = ACTION_EVIDENCE_RULES[action_type]
    used = used_evidence if used_evidence is not None else set()
    candidates: list[tuple[float, str]] = []

    module_ids = set(rule["modules"])
    table_terms = tuple(rule["tables"])
    row_fields = set(rule["row_fields"])
    for module in report.modules:
        if module.module_id not in module_ids:
            continue
        if action_type == "discount":
            threshold_rows = [
                _canonical_row(row)
                for row in module.tables.get("discount_threshold_candidates", []) or []
                if isinstance(row, dict)
            ]
            threshold_rows = [
                row
                for row in threshold_rows
                if row_fields.intersection(row) and not _low_risk_discount_row(row)
            ]
            if threshold_rows:
                threshold_row = max(threshold_rows, key=_discount_threshold_sort_key)
                text = row_to_business_evidence(action_type, threshold_row)
                if text and text not in used:
                    used.add(text)
                    return text
        for table_name, rows in module.tables.items():
            if not _contains_any(table_name, table_terms):
                continue
            for index, row in enumerate(rows):
                if isinstance(row, dict):
                    normalized_row = _canonical_row(row)
                    if not row_fields.intersection(normalized_row):
                        continue
                    text = row_to_business_evidence(action_type, normalized_row)
                    if text and text not in used:
                        candidates.append((evidence_score(action_type, table_name, normalized_row) - index * 0.01, text))
        metric_parts: list[str] = []
        for key in rule["metrics"]:
            if key in module.summary_metrics:
                formatted = _format_metric_value(key, module.summary_metrics.get(key))
                if formatted:
                    metric_parts.append(f"{_metric_label(key)}为 {formatted}")
            if len(metric_parts) >= 2:
                break
        if metric_parts:
            text = clean_business_text(f"{rule['label']}维度：" + "，".join(metric_parts) + "。")
            if text not in used:
                candidates.append((1.0, text))

    if candidates:
        _, text = max(candidates, key=lambda item: item[0])
        used.add(text)
        return text

    if isinstance(evidence_pack, dict):
        focus_evidence = evidence_pack.get("focus_evidence") or {}
        for focus in rule["focuses"]:
            for item in focus_evidence.get(focus, []) or []:
                if not isinstance(item, dict):
                    continue
                evidence_id = str(item.get("evidence_id", ""))
                meaning = clean_business_text(item.get("business_meaning", ""))
                if not meaning:
                    continue
                if not (
                    _contains_any(evidence_id, tuple(rule["metrics"]) + tuple(rule["row_fields"]))
                    or _contains_any(meaning, (str(rule["label"]),))
                ):
                    continue
                if meaning not in used:
                    used.add(meaning)
                    return meaning

    profile_parts: list[str] = []
    for key in rule["metrics"]:
        if key in profile:
            formatted = _format_metric_value(key, profile.get(key))
            if formatted:
                profile_parts.append(f"{_metric_label(key)}为 {formatted}")
        if len(profile_parts) >= 2:
            break
    if profile_parts:
        text = clean_business_text(f"{rule['label']}维度：" + "，".join(profile_parts) + "。")
        if text not in used:
            used.add(text)
            return text
    return f"数据限制说明：当前缺少更细的{rule['label']}证据，仅能作为后续补充方向。"


def action_type_for_row(row: dict[str, str]) -> str | None:
    text = f"{row.get('issue', '')} {row.get('action', '')} {row.get('required_fields', '')}"
    checks = [
        ("discount", ("折扣", "审批")),
        ("country", ("国家", "市场")),
        ("productline", ("PRODUCTLINE", "产品线")),
        ("deal_size", ("DEALSIZE", "交易规模")),
        ("status", ("STATUS", "订单状态")),
        ("customer_order", ("客户", "复购", "订单篮子")),
        ("product", ("商品", "SKU")),
        ("category", ("类目",)),
        ("slice", ("组合切片", "切片", "客群", "区域")),
    ]
    return next((action_type for action_type, terms in checks if any(term in text for term in terms)), None)


_evidence_score = evidence_score
_select_action_evidence = select_action_evidence
_action_type_for_row = action_type_for_row
