from __future__ import annotations

import math
from typing import Any

from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook.action_plan_builder import action_plan_rows


SECTION_MODULE_IDS = {
    "data_cleaning": ["data_quality_check"],
    "sales_trends": ["sales_trend_analysis"],
    "product_and_category": ["product_contribution_analysis"],
    "segment_and_region": ["dimension_breakdown_analysis"],
    "discount_and_profit": ["discount_profit_analysis"],
    "modeling": ["loss_risk_modeling"],
    "forecast": ["forecast_analysis"],
    "conclusions": [
        "sales_trend_analysis",
        "product_contribution_analysis",
        "dimension_breakdown_analysis",
        "discount_profit_analysis",
        "forecast_analysis",
        "loss_risk_modeling",
    ],
}

DISCOUNT_TABLES = {
    "discount_threshold_candidates",
    "discount_cap_what_if",
    "discount_profit_risk_buckets",
    "discount_buckets",
}

MODELING_TABLES = {
    "threshold_analysis",
    "feature_importance_grouped",
    "confusion_matrix",
    "high_risk_examples",
}

KEY_TABLES_BY_MODULE = {
    "metric_distribution_analysis": [
        "metric_quantiles",
        "metric_outliers",
        "relationship_candidates",
    ],
    "sales_trend_analysis": [
        "monthly_totals",
        "weekday_profile",
        "year_month_totals",
    ],
    "product_contribution_analysis": [
        "top_products",
        "category_sales",
        "high_sales_low_profit_products",
    ],
    "dimension_breakdown_analysis": [
        "segment_region_matrix",
        "top_performance_cuts",
        "weak_performance_cuts",
    ],
    "discount_profit_analysis": [
        "discount_threshold_candidates",
        "discount_cap_what_if",
        "discount_profit_risk_buckets",
        "high_sales_low_profit_items",
    ],
    "loss_risk_modeling": [
        "model_comparison",
        "threshold_analysis",
        "feature_importance_grouped",
        "confusion_matrix",
        "high_risk_examples",
    ],
    "forecast_analysis": ["forecast"],
}


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "unknown", "none", "nan", "null"}:
        return True
    return False


def _clean_value(value: Any) -> Any:
    if _is_missing_value(value):
        return None
    if isinstance(value, dict):
        cleaned = {
            str(key): cleaned_value
            for key, item in value.items()
            if (cleaned_value := _clean_value(item)) is not None
            and str(key) != "chart_payload"
        }
        return cleaned or None
    if isinstance(value, list):
        cleaned_items = [
            cleaned_item
            for item in value
            if (cleaned_item := _clean_value(item)) is not None
        ]
        return cleaned_items
    return value


def _clean_mapping(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cleaned = _clean_value(payload)
    return cleaned if isinstance(cleaned, dict) else {}


def _table_rows(rows: Any, max_rows: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        cleaned = _clean_mapping(row)
        if cleaned:
            cleaned_rows.append(cleaned)
        if len(cleaned_rows) >= max_rows:
            break
    return cleaned_rows


def _key_tables(
    module: ModuleReport,
    max_table_rows: int,
    max_tables: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    key_tables: dict[str, list[dict[str, Any]]] = {}
    preferred = KEY_TABLES_BY_MODULE.get(module.module_id)
    table_items = (
        [(table_name, module.tables.get(table_name, [])) for table_name in preferred if table_name in module.tables]
        if preferred
        else list(module.tables.items())[:max_tables]
    )
    if not table_items:
        table_items = list(module.tables.items())[:max_tables]
    for table_name, rows in table_items[:max_tables]:
        preview = _table_rows(rows, max_table_rows)
        if preview:
            key_tables[str(table_name)] = preview
    return key_tables


def _module_evidence(
    module: ModuleReport,
    *,
    max_table_rows: int,
    max_findings_per_module: int,
) -> dict[str, Any]:
    return {
        "module_id": module.module_id,
        "title": module.title,
        "chart_type": module.chart_type,
        "summary_metrics": _clean_mapping(module.summary_metrics),
        "top_findings": [str(item) for item in module.findings[:max_findings_per_module] if str(item).strip()],
        "key_tables": _key_tables(module, max_table_rows),
        "warnings": [str(item) for item in module.warnings[:3] if str(item).strip()],
    }


def _module(report: AnalysisReport, module_id: str) -> ModuleReport | None:
    return next((module for module in report.modules if module.module_id == module_id), None)


def _first_metric(report: AnalysisReport, keys: tuple[str, ...]) -> Any:
    for module in report.modules:
        metrics = module.summary_metrics or {}
        for key in keys:
            value = metrics.get(key)
            if not _is_missing_value(value):
                return value
    return None


def _business_kpis(report: AnalysisReport) -> dict[str, Any]:
    return _clean_mapping(
        {
            "total_sales": _first_metric(report, ("total_sales", "total_sales_amount", "sales_amount")),
            "total_profit": _first_metric(report, ("total_profit", "total_profit_amount", "profit_amount")),
            "profit_margin": _first_metric(report, ("profit_margin", "overall_profit_margin")),
            "order_count": _first_metric(report, ("order_count", "distinct_order_count", "total_orders")),
            "negative_profit_rate": _first_metric(report, ("negative_profit_rate", "loss_rate")),
        }
    )


def _discount_evidence(report: AnalysisReport, max_table_rows: int) -> dict[str, Any]:
    module = _module(report, "discount_profit_analysis")
    if module is None:
        return {}
    tables = module.tables or {}
    return _clean_mapping(
        {
            "threshold_candidates": _table_rows(tables.get("discount_threshold_candidates"), max_table_rows),
            "what_if": _table_rows(tables.get("discount_cap_what_if"), max_table_rows),
            "risk_buckets": _table_rows(
                tables.get("discount_profit_risk_buckets") or tables.get("discount_buckets"),
                max_table_rows,
            ),
        }
    )


def _modeling_evidence(report: AnalysisReport, max_table_rows: int) -> dict[str, Any]:
    module = _module(report, "loss_risk_modeling")
    if module is None:
        return {}
    metrics = _clean_mapping(module.summary_metrics)
    tables = module.tables or {}
    return _clean_mapping(
        {
            "best_model": metrics.get("best_model"),
            "metrics": metrics,
            "threshold_analysis": _table_rows(tables.get("threshold_analysis"), max_table_rows),
            "feature_importance_grouped": _table_rows(tables.get("feature_importance_grouped"), max_table_rows),
            "confusion_matrix": _table_rows(tables.get("confusion_matrix"), max_table_rows),
            "high_risk_examples": _table_rows(tables.get("high_risk_examples"), max_table_rows),
        }
    )


def _action_evidence(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
) -> dict[str, list[dict[str, str]]]:
    rows = action_plan_rows(report, schema_mapping, dataset_profile, analysis_focus, evidence_pack)
    p1 = [row for row in rows if str(row.get("priority", "")).upper() == "P1"]
    p2 = [row for row in rows if str(row.get("priority", "")).upper() == "P2"]
    return {
        "p1_actions": p1[:5],
        "p2_actions": p2[:5],
    }


def _chart_evidence(chart_selection_plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(chart_selection_plan, dict):
        return []
    candidates = chart_selection_plan.get("selected_charts")
    if not isinstance(candidates, list):
        candidates = chart_selection_plan.get("charts")
    if not isinstance(candidates, list):
        return []
    chart_items: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        chart = _clean_mapping(
            {
                "section_id": item.get("section_id"),
                "chart_id": item.get("chart_id"),
                "chart_title": item.get("chart_title") or item.get("title"),
                "chart_kind": item.get("chart_kind") or item.get("chart_type"),
                "business_question": item.get("business_question"),
                "selection_reason": item.get("selection_reason") or item.get("reason"),
                "chart_summary": item.get("chart_summary") or item.get("evidence_summary"),
            }
        )
        if chart:
            chart_items.append(chart)
    return chart_items[:12]


def _limitations(
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    report: AnalysisReport,
) -> list[str]:
    limitations: list[str] = []
    if schema_mapping.missing_required_fields:
        limitations.append("缺失字段：" + ", ".join(schema_mapping.missing_required_fields[:8]))
    if schema_mapping.uncertain_fields:
        limitations.append("不确定字段：" + ", ".join(schema_mapping.uncertain_fields[:8]))
    for module in report.modules:
        limitations.extend(str(item) for item in module.warnings[:2] if str(item).strip())
    if isinstance(dataset_profile, dict):
        for item in dataset_profile.get("limitations", []) or []:
            if str(item).strip():
                limitations.append(str(item))
    return limitations[:8]


def build_llm_evidence_pack(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    analysis_focus: dict[str, Any] | None = None,
    chart_selection_plan: dict[str, Any] | None = None,
    max_table_rows: int = 3,
    max_findings_per_module: int = 5,
    action_source_evidence_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = dataset_profile or {}
    pack = {
        "dataset": {
            "dataset_type": report.dataset_type,
            "row_count": profile.get("row_count") or profile.get("total_rows") or profile.get("rows"),
            "column_count": profile.get("column_count") or profile.get("field_count") or profile.get("columns"),
            "mapped_fields": [
                {"original": original, "canonical": canonical}
                for original, canonical in schema_mapping.field_mapping.items()
            ],
            "missing_required_fields": list(schema_mapping.missing_required_fields),
            "uncertain_fields": list(schema_mapping.uncertain_fields),
        },
        "business_kpis": _business_kpis(report),
        "module_evidence": [
            _module_evidence(
                module,
                max_table_rows=max_table_rows,
                max_findings_per_module=max_findings_per_module,
            )
            for module in report.modules
        ],
        "discount_evidence": _discount_evidence(report, max_table_rows),
        "modeling_evidence": _modeling_evidence(report, max_table_rows),
        "action_evidence": _action_evidence(
            report,
            schema_mapping,
            dataset_profile,
            analysis_focus,
            action_source_evidence_pack,
        ),
        "chart_evidence": _chart_evidence(chart_selection_plan),
        "limitations": _limitations(schema_mapping, dataset_profile, report),
    }
    cleaned = _clean_mapping(pack)
    return cleaned


def build_section_evidence_slice(evidence_pack: dict[str, Any], section_id: str) -> dict[str, Any]:
    module_ids = set(SECTION_MODULE_IDS.get(section_id, []))
    module_evidence = [
        item
        for item in evidence_pack.get("module_evidence", []) or []
        if isinstance(item, dict) and str(item.get("module_id", "")) in module_ids
    ]
    chart_evidence = [
        item
        for item in evidence_pack.get("chart_evidence", []) or []
        if isinstance(item, dict) and str(item.get("section_id", "")) == section_id
    ]
    result = {
        "dataset": evidence_pack.get("dataset", {}),
        "business_kpis": evidence_pack.get("business_kpis", {}),
        "module_evidence": module_evidence,
        "limitations": evidence_pack.get("limitations", []),
    }
    if section_id == "discount_and_profit":
        result["discount_evidence"] = evidence_pack.get("discount_evidence", {})
    if section_id == "modeling":
        result["modeling_evidence"] = evidence_pack.get("modeling_evidence", {})
    if section_id == "conclusions":
        result["discount_evidence"] = evidence_pack.get("discount_evidence", {})
        result["modeling_evidence"] = evidence_pack.get("modeling_evidence", {})
        result["action_evidence"] = evidence_pack.get("action_evidence", {})
    if chart_evidence:
        result["chart_evidence"] = chart_evidence
    return _clean_mapping(result)
