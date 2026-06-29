from __future__ import annotations

from copy import deepcopy
from typing import Any


HARD_REJECTION_PREFIXES = (
    "missing required fields",
    "skipped section",
    "country field has no useful market slice",
    "too many categories",
    "too many category combinations",
    "limited by data availability",
)

BAR_FAMILY_KINDS = {"bar", "stacked_bar"}
NON_BAR_RESTORE_PRIORITY = {
    "heatmap": 0,
    "scatter": 1,
    "box": 2,
    "boxplot": 2,
    "histogram": 3,
    "line": 4,
    "pareto": 5,
    "donut": 6,
}

RENDERABLE_CHART_IDS = {
    "basket_size_distribution",
    "category_concentration_bar",
    "category_profit_margin_bar",
    "country_avg_order_value_bar",
    "country_concentration_pareto",
    "country_customer_count_bar",
    "country_monthly_trend",
    "country_order_count_bar",
    "country_productline_heatmap",
    "country_sales_bar",
    "country_sales_donut",
    "deal_size_avg_order_value_bar",
    "deal_size_order_count_bar",
    "deal_size_order_value_boxplot",
    "deal_size_sales_bar",
    "deal_size_sales_donut",
    "discount_category_heatmap",
    "discount_profit_quality_bar",
    "discount_region_heatmap",
    "discount_segment_heatmap",
    "discount_vs_profit_scatter",
    "invoice_value_distribution",
    "metric_sales_distribution",
    "product_category_bar",
    "productline_deal_size_heatmap",
    "productline_deal_size_stacked_bar",
    "productline_monthly_trend",
    "productline_quantity_bar",
    "productline_quantity_sales_combo",
    "productline_sales_bar",
    "productline_sales_donut",
    "profit_distribution_if_available",
    "profit_margin_by_discount_box",
    "quantity_distribution",
    "recent_growth_bar",
    "region_sales_profit_bar",
    "rolling_volatility_line",
    "sales_amount_log_distribution",
    "sales_trends_month_heatmap",
    "sales_trends_monthly_line",
    "segment_region_low_margin_table_or_bar",
    "segment_region_margin_heatmap",
    "status_deal_size_stacked_bar",
    "status_order_count_bar",
    "status_order_line_count_bar",
    "subcategory_sales_profit_matrix",
    "top_product_profit_gap_bar",
    "unit_price_quantity_scatter",
}


def _is_hard_rejected(candidate: dict[str, Any]) -> bool:
    reason = str(candidate.get("rejection_reason") or "")
    return any(reason.startswith(prefix) for prefix in HARD_REJECTION_PREFIXES)


def _chart_id(item: dict[str, Any]) -> str:
    return str(item.get("chart_id") or "")


def _chart_kind(item: dict[str, Any]) -> str:
    return str(item.get("chart_kind") or "unknown")


def _is_bar_family(item: dict[str, Any]) -> bool:
    return _chart_kind(item) in BAR_FAMILY_KINDS


def _bar_family_ratio(selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    return round(sum(1 for item in selected if _is_bar_family(item)) / len(selected), 4)


def _candidate_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _chart_id(item): item
        for item in plan.get("candidate_charts", []) or []
        if isinstance(item, dict) and _chart_id(item)
    }


def _selected_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "chart_id": candidate.get("chart_id"),
        "section_id": candidate.get("section_id"),
        "chart_kind": candidate.get("chart_kind"),
        "title": candidate.get("title"),
        "reason": "Portfolio diversity gate restored a safe non-bar chart to reduce bar-family pressure.",
        "business_question": candidate.get("business_question"),
        "selection_source": "portfolio_diversity_gate",
        "evidence_ids": candidate.get("evidence_ids", []),
        "evidence_summary": "Restored by portfolio diversity gate.",
        "score": candidate.get("score"),
        "ranking_factors": candidate.get("ranking_factors", []),
        "insight_type": candidate.get("insight_type") or "structure",
        "dataset_specificity_score": candidate.get("dataset_specificity_score"),
        "intent_support_score": candidate.get("intent_support_score", 0),
        "matched_intent_ids": candidate.get("matched_intent_ids", []),
        "best_intent_match_quality": candidate.get("best_intent_match_quality"),
    }


def _selected_with_candidate_data(item: dict[str, Any], candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = candidates.get(_chart_id(item), {})
    return {**candidate, **item}


def _safe_non_bar_candidates(
    *,
    plan: dict[str, Any],
    selected_ids: set[str],
    mapped_fields: set[str],
    renderable_chart_ids: set[str],
) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    for candidate in plan.get("candidate_charts", []) or []:
        if not isinstance(candidate, dict):
            continue
        chart_id = _chart_id(candidate)
        required = {str(field) for field in candidate.get("required_fields", []) or []}
        if not chart_id or chart_id in selected_ids:
            continue
        if _is_bar_family(candidate):
            continue
        if chart_id not in renderable_chart_ids:
            continue
        if not required <= mapped_fields:
            continue
        if str(candidate.get("section_role") or "") == "skipped":
            continue
        if _is_hard_rejected(candidate):
            continue
        safe.append(candidate)
    safe.sort(
        key=lambda item: (
            NON_BAR_RESTORE_PRIORITY.get(_chart_kind(item), 9),
            -(float(item.get("intent_support_score") or 0)),
            -(float(item.get("score") or 0)),
            -float(item.get("dataset_specificity_score") or 0),
            _chart_id(item),
        )
    )
    return safe


def _redundant_bar_pairs(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(selected):
        for right in selected[index + 1 :]:
            if not (_is_bar_family(left) and _is_bar_family(right)):
                continue
            if left.get("section_id") != right.get("section_id"):
                continue
            pairs.append({"chart_id_a": _chart_id(left), "chart_id_b": _chart_id(right), "section_id": left.get("section_id")})
    return pairs


def _pick_bar_to_remove(selected: list[dict[str, Any]], candidates: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    enriched = [_selected_with_candidate_data(item, candidates) for item in selected]
    duplicate_ids = {
        pair["chart_id_a"]
        for pair in _redundant_bar_pairs(enriched)
    } | {
        pair["chart_id_b"]
        for pair in _redundant_bar_pairs(enriched)
    }
    bar_items = [item for item in enriched if _is_bar_family(item)]
    if not bar_items:
        return None
    return sorted(
        bar_items,
        key=lambda item: (
            _chart_id(item) not in duplicate_ids,
            float(item.get("score") or 0),
            float(item.get("intent_support_score") or 0),
            float(item.get("dataset_specificity_score") or 0),
            _chart_id(item),
        ),
    )[0]


def _trace(plan: dict[str, Any]) -> dict[str, Any]:
    trace = plan.get("intent_guided_selection_trace")
    if isinstance(trace, dict):
        return trace
    trace = plan.get("llm_selection_trace")
    if isinstance(trace, dict):
        return trace
    return {}


def _store_trace(plan: dict[str, Any], trace: dict[str, Any]) -> None:
    if "intent_guided_selection_trace" in plan:
        plan["intent_guided_selection_trace"] = trace
    else:
        plan["llm_selection_trace"] = trace


def apply_portfolio_diversity_gate(
    plan: dict[str, Any],
    *,
    mapped_fields: set[str],
    renderable_chart_ids: set[str] | None = None,
) -> dict[str, Any]:
    updated = deepcopy(plan)
    selected = [item for item in updated.get("selected_charts", []) or [] if isinstance(item, dict)]
    trace = dict(_trace(updated))
    candidates = _candidate_map(updated)
    renderable = renderable_chart_ids or RENDERABLE_CHART_IDS
    before_ratio = _bar_family_ratio(selected)
    redundant_before = _redundant_bar_pairs([_selected_with_candidate_data(item, candidates) for item in selected])
    status = "not_needed"
    removed: list[str] = []
    restored: list[str] = []
    unresolved_reason = ""

    needs_gate = len(selected) >= 4 and (before_ratio >= 0.7 or len(redundant_before) >= 2)
    if needs_gate:
        selected_ids = {_chart_id(item) for item in selected}
        restore_pool = _safe_non_bar_candidates(
            plan=updated,
            selected_ids=selected_ids,
            mapped_fields=mapped_fields,
            renderable_chart_ids=renderable,
        )
        remove_item = _pick_bar_to_remove(selected, candidates)
        if restore_pool and remove_item:
            remove_id = _chart_id(remove_item)
            restore = restore_pool[0]
            restore_id = _chart_id(restore)
            selected = [item for item in selected if _chart_id(item) != remove_id]
            selected.append(_selected_from_candidate(restore))
            removed.append(remove_id)
            restored.append(restore_id)
            status = "improved"
        else:
            status = "unresolved"
            unresolved_reason = "no safe renderable non-bar candidate available" if not restore_pool else "no removable bar chart"

    after_ratio = _bar_family_ratio(selected)
    selected_ids_after = [_chart_id(item) for item in selected]
    restored_set = set(restored)
    removed_set = set(removed)
    for key in ("remove_only_chart_ids", "base_removed_chart_ids"):
        values = trace.get(key)
        if not isinstance(values, list):
            values = []
        value_set = {str(value) for value in values}
        value_set -= restored_set
        value_set |= removed_set
        trace[key] = sorted(value_set)

    trace.update(
        {
            "portfolio_diversity_status": status,
            "bar_family_ratio_before": before_ratio,
            "bar_family_ratio_after": after_ratio,
            "redundant_pairs_before": redundant_before,
            "redundant_pairs_after": _redundant_bar_pairs(
                [_selected_with_candidate_data(item, candidates) for item in selected]
            ),
            "removed_for_diversity_chart_ids": removed,
            "restored_for_diversity_chart_ids": restored,
            "non_bar_challenger_ids": [_chart_id(item) for item in _safe_non_bar_candidates(
                plan=updated,
                selected_ids=set(selected_ids_after),
                mapped_fields=mapped_fields,
                renderable_chart_ids=renderable,
            )],
            "portfolio_diversity_unresolved_reason": unresolved_reason,
            "final_selected_chart_ids": selected_ids_after,
        }
    )
    updated["selected_charts"] = selected
    updated["llm_sanitized_selected_chart_ids"] = selected_ids_after
    _store_trace(updated, trace)
    for candidate in updated.get("candidate_charts", []) or []:
        if isinstance(candidate, dict) and _chart_id(candidate) in set(selected_ids_after):
            candidate["availability_status"] = "selected"
            candidate.pop("rejection_reason", None)
    return updated
