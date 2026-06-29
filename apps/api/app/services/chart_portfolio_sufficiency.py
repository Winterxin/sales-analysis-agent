from __future__ import annotations

from copy import deepcopy
from typing import Any


FIELD_GROUPS: tuple[tuple[str, set[str]], ...] = (
    ("date", {"order_datetime"}),
    ("sales_amount", {"sales_amount"}),
    ("quantity", {"quantity"}),
    ("unit_price", {"unit_price"}),
    ("product/category/sku", {"product_name", "category", "sub_category", "productline", "sku"}),
    ("order_status", {"order_status"}),
    ("customer/order", {"customer_id", "order_id"}),
    ("discount", {"discount"}),
    ("country/region/store", {"country", "region", "state", "city", "store"}),
)


HARD_REJECTION_PREFIXES = (
    "missing required fields",
    "skipped section",
    "country field has no useful market slice",
    "too many categories",
    "too many category combinations",
    "limited by data availability",
)


def available_sales_field_groups(mapped_fields: set[str]) -> list[str]:
    return [name for name, fields in FIELD_GROUPS if fields & mapped_fields]


def _is_hard_rejected(candidate: dict[str, Any]) -> bool:
    reason = str(candidate.get("rejection_reason") or "")
    return any(reason.startswith(prefix) for prefix in HARD_REJECTION_PREFIXES)


def _is_safe_candidate(candidate: dict[str, Any], mapped_fields: set[str]) -> bool:
    required = {str(field) for field in candidate.get("required_fields", []) or []}
    if not required <= mapped_fields:
        return False
    if str(candidate.get("section_role") or "") == "skipped":
        return False
    if _is_hard_rejected(candidate):
        return False
    return True


def _selected_chart(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "chart_id": candidate.get("chart_id"),
        "section_id": candidate.get("section_id"),
        "chart_kind": candidate.get("chart_kind"),
        "title": candidate.get("title"),
        "reason": "Portfolio sufficiency restored a safe removed chart for a rich sales dataset.",
        "business_question": candidate.get("business_question"),
        "selection_source": "portfolio_sufficiency_gate",
        "evidence_ids": candidate.get("evidence_ids", []),
        "evidence_summary": "Restored by portfolio sufficiency gate.",
        "score": candidate.get("score"),
        "ranking_factors": candidate.get("ranking_factors", []),
        "insight_type": candidate.get("insight_type") or "structure",
        "dataset_specificity_score": candidate.get("dataset_specificity_score"),
        "intent_support_score": candidate.get("intent_support_score", 0),
        "matched_intent_ids": candidate.get("matched_intent_ids", []),
        "best_intent_match_quality": candidate.get("best_intent_match_quality"),
    }


def apply_portfolio_sufficiency_gate(
    plan: dict[str, Any],
    *,
    mapped_fields: set[str],
) -> dict[str, Any]:
    updated = deepcopy(plan)
    trace = updated.get("intent_guided_selection_trace")
    if not isinstance(trace, dict):
        trace = updated.get("llm_selection_trace")
    if not isinstance(trace, dict):
        trace = {}

    selected = [item for item in updated.get("selected_charts", []) or [] if isinstance(item, dict)]
    before_count = len(selected)
    base_selected = [
        item
        for item in updated.get("deterministic_selected_charts", []) or []
        if isinstance(item, dict)
    ]
    if not base_selected:
        base_selected = selected
    base_count = len(base_selected)
    field_groups = available_sales_field_groups(mapped_fields)
    min_count = 0
    if len(field_groups) >= 7 and base_count >= 5:
        min_count = 5
    elif len(field_groups) >= 5 and base_count >= 4:
        min_count = 4
    rich = min_count > 0

    selected_ids = {str(item.get("chart_id")) for item in selected}
    candidate_by_id = {
        str(item.get("chart_id")): item
        for item in updated.get("candidate_charts", []) or []
        if isinstance(item, dict) and item.get("chart_id")
    }
    restored: list[str] = []
    unresolved_reason = ""
    if rich and before_count < min_count:
        restore_pool = [
            candidate_by_id.get(str(item.get("chart_id"))) or item
            for item in base_selected
            if str(item.get("chart_id")) not in selected_ids
        ]
        restore_pool = [
            item for item in restore_pool if isinstance(item, dict) and _is_safe_candidate(item, mapped_fields)
        ]
        restore_pool.sort(
            key=lambda item: (
                str(item.get("section_id") or "") in {str(selected.get("section_id")) for selected in selected},
                -(float(item.get("score") or 0)),
                str(item.get("chart_id") or ""),
            )
        )
        for candidate in restore_pool:
            if len(selected) >= min_count or len(selected) >= base_count:
                break
            selected.append(_selected_chart(candidate))
            restored.append(str(candidate.get("chart_id")))
            selected_ids.add(str(candidate.get("chart_id")))
        if len(selected) < min_count:
            unresolved_reason = "no safe removed base chart available for portfolio sufficiency"

    if restored:
        restored_ids = set(restored)
        for removed_key in ("remove_only_chart_ids", "base_removed_chart_ids"):
            removed_ids = trace.get(removed_key)
            if isinstance(removed_ids, list):
                trace[removed_key] = [chart_id for chart_id in removed_ids if str(chart_id) not in restored_ids]

    status = "not_needed"
    if rich and before_count < min_count:
        status = "restored" if restored else "unresolved"
    trace.update(
        {
            "portfolio_sufficiency_status": status,
            "rich_sales_dataset_detected": rich,
            "available_sales_field_groups": field_groups,
            "min_recommended_chart_count": min_count,
            "restored_chart_ids": restored,
            "restore_reason": (
                "restored safe removed base charts for rich sales dataset"
                if restored
                else ""
            ),
            "portfolio_sufficiency_unresolved": bool(unresolved_reason),
            "unresolved_reason": unresolved_reason,
            "before_final_chart_count": before_count,
            "after_final_chart_count": len(selected),
        }
    )
    updated["selected_charts"] = selected
    updated["llm_sanitized_selected_chart_ids"] = [str(item.get("chart_id")) for item in selected]
    trace["final_selected_chart_ids"] = [str(item.get("chart_id")) for item in selected]
    if "intent_guided_selection_trace" in updated:
        updated["intent_guided_selection_trace"] = trace
    else:
        updated["llm_selection_trace"] = trace
    return updated
