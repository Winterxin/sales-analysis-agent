from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_value_audit import build_chart_value_audit
from app.services.llm_client import complete_json_for_stage


SYSTEM_PROMPT = """You are choosing chart IDs for a sales-analysis notebook.
Return JSON only with keys: selected_chart_ids, base_chart_audit, kept_for_unique_value, remove_as_redundant, replace, remove_only, redundancy_summary, overall_strategy, rejected_high_specific_notes.
Rules:
1. Choose the charts that best fit the current data structure and business story, not mechanical diversity.
2. Sparse datasets may use generic charts.
3. Rich datasets should prefer charts that use dataset-specific fields when those charts serve the business question.
4. Do not only choose safe bar/line/heatmap charts, but do not choose unsuitable charts for novelty.
5. Chart-kind diversity is a supporting constraint, not the primary goal.
6. Only choose chart_id values from candidate_charts. Do not invent chart IDs or chart code.
7. Do not choose hard-rejected charts: missing required fields, skipped section, too many categories, too many category combinations, limited by data availability, or country field has no useful market slice.
8. You may reconsider soft-rejected high-specific charts: core budget exceeded, support budget exceeded, redundant insight type, lower evidence score.
9. The final combination must stay within field, section-budget, chart-shape, and safety rules.
10. When chart intent planning context is provided, prefer exact/strong intent-supported candidates when they fit the business story.
11. Do not mechanically choose every intent-supported candidate.
12. weak_match candidates are fallback references only; false_friend_match candidates must not be treated as intent support evidence.
13. Your task is not to preserve every base selected chart.
14. Review base_selected_charts for redundancy, weak incremental value, and replaceable charts.
15. If two charts answer similar questions, keep the one that best supports the business story.
16. If a challenger is better than a base chart, replace the base chart.
17. If a base chart has low incremental value, remove it.
18. Do not select a chart just because it is base selected or intent supported.
19. Do not select charts to meet a quantity target; chart count is an outcome, not a goal.
20. Aim for compact but complete: fewer sharper charts are better than every useful chart.
21. If the final set is identical to the base set, explain why no chart can be removed or replaced.
22. Every base chart should be audited as keep, remove, or replace.
23. A challenger can enter the final set only through one-for-one replacement of a base chart.
24. Do not standalone-add challengers.
25. The final selected_chart_ids count must not exceed base_selected_charts count.
26. If you include a challenger, replace must name the removed base chart and the added challenger.
27. selected_chart_ids must match keep/remove/replace decisions: removed charts should be absent, replacement adds should be present, replacement removes should be absent.
28. If you only remove charts and add no challenger, report remove-only rather than pretending a replacement happened.
29. Avoid visually complex charts with weak incremental information gain.
30. Do not judge chart value only by title. You must use chart_value_audit and base_chart_value_review.
31. low_visual_gain_risk charts should be removed or replaced unless they have irreplaceable business value.
32. low_information_gain charts should be removed or replaced first.
33. In a section with multiple bar charts, keep the chart with the strongest information gain and clearest business value.
34. Strong intent support matters, but weak visual_gain_score or information_gain_score can still justify rejection.
35. If keeping a low-value-risk chart, explain the unique value in kept_for_unique_value.
"""

HARD_REJECTION_PREFIXES = (
    "missing required fields",
    "skipped section",
    "country field has no useful market slice",
    "too many categories",
    "too many category combinations",
    "limited by data availability",
)


SAFE_CANDIDATE_KEYS = {
    "chart_id",
    "title",
    "section_id",
    "chart_kind",
    "insight_type",
    "business_question",
    "required_fields",
    "dataset_specificity_score",
    "dataset_specificity_factors",
    "score",
    "ranking_factors",
    "availability_status",
    "rejection_reason",
    "intent_support_score",
    "matched_intent_ids",
    "matched_intent_priorities",
    "best_intent_match_quality",
    "intent_match_reasons",
    "intent_guided_candidate",
    "intent_support_type",
    "evidence_ids",
    "evidence_summary",
    "section_role",
}


def _is_hard_rejection(reason: str | None) -> bool:
    return any(str(reason or "").startswith(prefix) for prefix in HARD_REJECTION_PREFIXES)


def _enabled(llm_client: Any, llm_profile: str | None, llm_chart_selection_enabled: bool | None) -> tuple[bool, str]:
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return False, "LLM client is disabled."
    if llm_chart_selection_enabled is False:
        return False, "LLM chart selection is disabled by settings or profile policy."
    profile = str(llm_profile or "full").strip().lower()
    if llm_chart_selection_enabled is True or profile == "full":
        return True, "LLM chart selection enabled."
    return False, f"LLM chart selection is disabled for llm_profile={profile}."


def _safe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: candidate.get(key) for key in SAFE_CANDIDATE_KEYS if key in candidate}


def _field_set(item: dict[str, Any]) -> set[str]:
    return {str(field) for field in item.get("required_fields", []) or []}


def _safe_for_payload(candidate: dict[str, Any], fields: set[str]) -> bool:
    return (
        _field_set(candidate) <= fields
        and not _is_hard_rejection(str(candidate.get("rejection_reason") or ""))
        and str(candidate.get("section_role") or "") != "skipped"
    )


def _keywords(text: str | None) -> set[str]:
    stop = {"the", "and", "for", "with", "what", "how", "does", "this", "that", "chart", "sales"}
    words = {
        token.strip(" ?.,:;()[]{}").lower()
        for token in str(text or "").replace("_", " ").split()
    }
    return {word for word in words if len(word) >= 4 and word not in stop}


def _group_same(items: list[dict[str, Any]], *, keys: tuple[str, ...], label_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[str]] = {}
    for item in items:
        group_key = tuple(str(item.get(key) or "") for key in keys)
        if all(group_key):
            grouped.setdefault(group_key, []).append(str(item.get("chart_id")))
    groups: list[dict[str, Any]] = []
    for group_key, chart_ids in grouped.items():
        if len(chart_ids) < 2:
            continue
        entry = {label_key: value for label_key, value in zip(label_keys, group_key, strict=False)}
        entry["chart_ids"] = chart_ids
        entry["reason"] = "Multiple charts share this section/kind or insight lens and may have overlapping value."
        groups.append(entry)
    return groups


def _redundancy_groups(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    same_section_same_kind = _group_same(
        items,
        keys=("section_id", "chart_kind"),
        label_keys=("section_id", "chart_kind"),
    )
    same_section_same_insight_type = _group_same(
        items,
        keys=("section_id", "insight_type"),
        label_keys=("section_id", "insight_type"),
    )
    overlapping_required_fields: list[dict[str, Any]] = []
    similar_business_question_keywords: list[dict[str, Any]] = []
    for index, first in enumerate(items):
        first_id = str(first.get("chart_id") or "")
        first_fields = _field_set(first)
        first_keywords = _keywords(first.get("business_question"))
        for second in items[index + 1 :]:
            second_id = str(second.get("chart_id") or "")
            if not first_id or not second_id:
                continue
            overlap = sorted(first_fields & _field_set(second))
            if overlap:
                overlapping_required_fields.append(
                    {
                        "chart_ids": [first_id, second_id],
                        "overlapping_fields": overlap,
                        "reason": "The charts use overlapping required fields and may explain similar evidence.",
                    }
                )
            keyword_overlap = sorted(first_keywords & _keywords(second.get("business_question")))
            if keyword_overlap:
                similar_business_question_keywords.append(
                    {
                        "chart_ids": [first_id, second_id],
                        "overlapping_keywords": keyword_overlap[:6],
                        "reason": "The business questions share keywords and should be compared for incremental value.",
                    }
                )
    return {
        "same_section_same_kind": same_section_same_kind,
        "same_section_same_insight_type": same_section_same_insight_type,
        "overlapping_required_fields": overlapping_required_fields[:12],
        "similar_business_question_keywords": similar_business_question_keywords[:12],
    }


def _redundancy_pressure(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    bar_count = sum(1 for item in items if str(item.get("chart_kind") or "") == "bar")
    section_counts: dict[str, int] = {}
    for item in items:
        section_id = str(item.get("section_id") or "")
        if section_id:
            section_counts[section_id] = section_counts.get(section_id, 0) + 1
    groups = _redundancy_groups(items)
    same_insight_groups = groups["same_section_same_insight_type"]
    return {
        "bar_chart_count": bar_count,
        "bar_chart_ratio": round(bar_count / total, 4) if total else 0,
        "max_charts_per_section": max(section_counts.values(), default=0),
        "sections_with_3_or_more_charts": sorted(
            section_id for section_id, count in section_counts.items() if count >= 3
        ),
        "same_section_same_kind_group_count": len(groups["same_section_same_kind"]),
        "same_insight_type_group_count": len(same_insight_groups),
    }


def _base_selected_charts(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [_safe_candidate(chart) for chart in plan.get("selected_charts", []) or [] if isinstance(chart, dict)]


def _safe_diagnostics(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = diagnostics or {}
    keys = [
        "generic_chart_ratio",
        "dataset_specific_chart_ratio",
        "chart_kind_distribution",
        "section_budget_pressure",
        "hard_rejection_distribution",
        "soft_rejection_distribution",
        "top_unselected_dataset_specific_charts",
    ]
    return {key: diagnostics.get(key) for key in keys if key in diagnostics}


def _base_chart_value_review(base_selected: list[dict[str, Any]], value_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chart in base_selected:
        chart_id = str(chart.get("chart_id") or "")
        value = value_by_id.get(chart_id, {})
        rows.append(
            {
                "chart_id": chart_id,
                "overall_chart_value_score": value.get("overall_chart_value_score"),
                "information_gain_score": value.get("information_gain_score"),
                "visual_gain_score": value.get("visual_gain_score"),
                "redundancy_risk_score": value.get("redundancy_risk_score"),
                "value_flags": value.get("value_flags", []),
                "value_reason": value.get("value_reason", ""),
            }
        )
    return rows


def _chart_value_trace(
    *,
    audit: dict[str, Any],
    base_selected: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    result: dict[str, Any],
) -> dict[str, Any]:
    value_by_id = audit.get("chart_value_by_id", {}) if isinstance(audit, dict) else {}
    low_visual = set(audit.get("low_visual_gain_risk_chart_ids", []) or [])
    low_info = set(audit.get("low_information_gain_chart_ids", []) or [])
    high_value = list(audit.get("high_value_challenger_ids", []) or [])
    base_ids = _chart_ids(base_selected)
    final_ids = _chart_ids(selected)

    def avg(ids: list[str], key: str) -> float:
        values = [
            float(value_by_id.get(chart_id, {}).get(key) or 0)
            for chart_id in ids
            if chart_id in value_by_id
        ]
        return round(sum(values) / len(values), 4) if values else 0.0

    selected_low_visual = [chart_id for chart_id in final_ids if chart_id in low_visual]
    removed_low_visual = [chart_id for chart_id in base_ids if chart_id in low_visual and chart_id not in final_ids]
    kept_reasons = _dict_value(result, "kept_for_unique_value") or _dict_value(result, "selection_rationale")
    return {
        "chart_value_audit_summary": audit.get("summary", {}),
        "low_visual_gain_risk_chart_ids": sorted(low_visual),
        "low_information_gain_chart_ids": sorted(low_info),
        "high_value_challenger_ids": high_value[:8],
        "selected_low_visual_gain_chart_ids": selected_low_visual,
        "removed_low_visual_gain_chart_ids": removed_low_visual,
        "average_value_score_before": avg(base_ids, "overall_chart_value_score"),
        "average_value_score_after": avg(final_ids, "overall_chart_value_score"),
        "average_redundancy_risk_before": avg(base_ids, "redundancy_risk_score"),
        "average_redundancy_risk_after": avg(final_ids, "redundancy_risk_score"),
        "selected_chart_value_scores": {
            chart_id: value_by_id.get(chart_id, {}).get("overall_chart_value_score")
            for chart_id in final_ids
            if chart_id in value_by_id
        },
        "charts_kept_despite_low_visual_gain": {
            chart_id: kept_reasons.get(chart_id, "")
            for chart_id in selected_low_visual
        },
        "high_value_challengers_not_selected": [
            chart_id for chart_id in high_value if chart_id not in set(final_ids)
        ],
    }


def _payload(
    *,
    schema_mapping: SchemaMapping,
    analysis_focus: dict[str, Any],
    section_priority: dict[str, Any],
    evidence_pack: dict[str, Any] | None,
    dataset_profile: dict[str, Any] | None,
    deterministic_chart_selection_plan: dict[str, Any],
    diagnostics: dict[str, Any] | None,
    candidate_charts: list[dict[str, Any]],
    intent_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signature = (evidence_pack or {}).get("dataset_signature") or {}
    fields = set(schema_mapping.field_mapping.values())
    base_selected = _base_selected_charts(deterministic_chart_selection_plan)
    selected_ids = {str(item.get("chart_id")) for item in base_selected if item.get("chart_id")}
    safe_unselected = [
        candidate
        for candidate in candidate_charts
        if isinstance(candidate, dict)
        and str(candidate.get("chart_id") or "") not in selected_ids
        and str(candidate.get("availability_status") or "") != "selected"
        and _safe_for_payload(candidate, fields)
    ]
    intent_supported_challengers = [
        _safe_candidate(candidate) for candidate in safe_unselected if candidate.get("intent_guided_candidate")
    ]
    high_specificity_unselected = sorted(
        [
            candidate
            for candidate in safe_unselected
            if candidate.get("score") is not None and float(candidate.get("dataset_specificity_score") or 0) >= 0.55
        ],
        key=lambda item: (
            -float(item.get("dataset_specificity_score") or 0),
            -float(item.get("score") or 0),
            str(item.get("chart_id") or ""),
        ),
    )[:8]
    non_bar_challengers = [
        _safe_candidate(candidate)
        for candidate in safe_unselected
        if str(candidate.get("chart_kind") or "") != "bar"
    ][:8]
    redundancy_groups = _redundancy_groups(base_selected)
    redundancy_pressure = _redundancy_pressure(base_selected)
    chart_value_audit = build_chart_value_audit(
        candidate_charts=candidate_charts,
        schema_mapping=schema_mapping,
        dataset_profile=dataset_profile or {},
        evidence_pack=evidence_pack or {},
    )
    value_by_id = chart_value_audit.get("chart_value_by_id", {})
    high_value_challengers = sorted(
        [
            candidate
            for candidate in safe_unselected
            if (
                float(value_by_id.get(str(candidate.get("chart_id") or ""), {}).get("overall_chart_value_score") or 0)
                >= 0.52
                or candidate.get("intent_guided_candidate")
            )
        ],
        key=lambda item: (
            -float(value_by_id.get(str(item.get("chart_id") or ""), {}).get("overall_chart_value_score") or 0),
            str(item.get("chart_id") or ""),
        ),
    )[:8]
    payload = {
        "task": "Review the base chart set, remove redundant low-increment charts, and choose final chart_id values from the safe candidate pool.",
        "dataset_signature": signature,
        "dominant_story": diagnostics.get("dominant_story") if isinstance(diagnostics, dict) else signature.get("dominant_story"),
        "available_field_groups": (diagnostics or {}).get("available_field_groups", []),
        "selected_focuses": list(analysis_focus.get("selected_focuses", []) or []),
        "support_focuses": list(analysis_focus.get("support_focuses", []) or []),
        "section_priority": {
            "core_sections": list(section_priority.get("core_sections", []) or []),
            "support_sections": list(section_priority.get("support_sections", []) or []),
            "skipped_sections": list(section_priority.get("skipped_sections", []) or []),
        },
        "chart_budget": deterministic_chart_selection_plan.get("chart_budget", {}),
        "base_selected_charts": base_selected,
        "intent_supported_challengers": intent_supported_challengers,
        "high_specificity_unselected_candidates": [_safe_candidate(candidate) for candidate in high_specificity_unselected],
        "non_bar_challengers": non_bar_challengers,
        "redundancy_groups": redundancy_groups,
        "redundancy_pressure": redundancy_pressure,
        "chart_value_audit": chart_value_audit,
        "chart_value_by_id": value_by_id,
        "base_chart_value_review": _base_chart_value_review(base_selected, value_by_id),
        "low_visual_gain_risk_candidates": chart_value_audit.get("low_visual_gain_risk_chart_ids", []),
        "low_information_gain_candidates": chart_value_audit.get("low_information_gain_chart_ids", []),
        "high_value_challengers": [_safe_candidate(candidate) for candidate in high_value_challengers],
        "candidate_charts": [_safe_candidate(candidate) for candidate in candidate_charts],
        "diagnostics": _safe_diagnostics(diagnostics),
        "output_contract": {
            "selected_chart_ids": ["chart_id"],
            "base_chart_audit": {
                "base_chart_id": {
                    "decision": "keep | remove | replace",
                    "reason": "why",
                    "replacement_chart_id": "optional challenger id",
                }
            },
            "kept_for_unique_value": {"chart_id": "why this chart has unique business value"},
            "remove_as_redundant": {"chart_id": "why this base chart is redundant or lower-value"},
            "replace": [{"remove": "base_chart_id", "add": "challenger_chart_id", "reason": "why challenger is better"}],
            "remove_only": {"base_chart_id": "removed without replacement because..."},
            "redundancy_summary": "how the final chart set was compressed or de-duplicated",
            "overall_strategy": "overall chart combination logic",
            "rejected_high_specific_notes": {"chart_id": "why a high-specific candidate was not selected"},
        },
    }
    if intent_context:
        payload.update(intent_context)
        payload["intent_guidance_rules"] = [
            "A chart intent plan is available for this dataset.",
            "Prefer exact_match or strong_match supported candidates when they fit the business story.",
            "Do not mechanically select every intent-supported chart.",
            "If an intent-supported chart is worse than a base chart, explain why.",
            "weak_match candidates are fallback references only.",
            "false_friend_match candidates cannot be used as intent support evidence.",
            "Sparse fields or weak intent support may still justify generic charts.",
            "The final target is the best chart combination for the data structure and business story.",
        ]
    return payload


def _call_llm(llm_client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    method = getattr(llm_client, "suggest_llm_chart_selection", None)
    if method is not None:
        result = method(payload)
    else:
        result = complete_json_for_stage(
            llm_client,
            system_prompt=SYSTEM_PROMPT,
            user_payload=payload,
            cache_stage="llm_chart_selection",
        )
    return result if isinstance(result, dict) else {}


def _dict_value(result: dict[str, Any], key: str) -> dict[str, Any]:
    value = result.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _replacement_decisions(result: dict[str, Any]) -> list[dict[str, str]]:
    value = result.get("replace")
    if not isinstance(value, list):
        return []
    decisions: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        remove = str(item.get("remove") or "").strip()
        add = str(item.get("add") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if remove or add:
            decisions.append({"remove": remove, "add": add, "reason": reason})
    return decisions


def _safe_candidate_reason(candidate: dict[str, Any] | None, fields: set[str]) -> str | None:
    if candidate is None:
        return "missing_add_candidate"
    required = {str(field) for field in candidate.get("required_fields", []) or []}
    if not required <= fields:
        return "unsafe_add_candidate"
    if _is_hard_rejection(str(candidate.get("rejection_reason") or "")):
        return "unsafe_add_candidate"
    if str(candidate.get("section_role") or "") == "skipped":
        return "unsafe_add_candidate"
    return None


def _replacement_consistency(
    *,
    selected: list[dict[str, Any]],
    result: dict[str, Any],
    candidate_charts: list[dict[str, Any]],
    schema_mapping: SchemaMapping,
    base_selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol_keys = {"base_chart_audit", "remove_only", "replace", "remove_as_redundant"}
    if not any(key in result for key in protocol_keys):
        final_ids = _chart_ids(selected)
        return selected, {
            "base_chart_audit": {},
            "remove_only": {},
            "true_replacements": [],
            "suggested_replacements_not_applied": [],
            "invalid_replacements": [],
            "standalone_added_candidate_ids": [],
            "selected_non_base_without_valid_replace": [],
            "remove_only_chart_ids": [],
            "replacement_consistency_status": "ok",
            "replacement_consistency_reason": "legacy_selection_without_audit_fields",
            "replacement_consistency_warnings": ["missing_audit_fields"],
            "final_chart_count_le_base": len(final_ids) <= len(base_selected),
            "base_chart_count": len(base_selected),
            "final_chart_count": len(final_ids),
        }
    candidate_by_id = {str(candidate.get("chart_id")): candidate for candidate in candidate_charts}
    base_ids = _chart_ids(base_selected)
    base_set = set(base_ids)
    fields = set(schema_mapping.field_mapping.values())
    selected_ids_before = _chart_ids(selected)
    selected_set_before = set(selected_ids_before)
    replacements = _replacement_decisions(result)
    valid_replacement_adds: set[str] = set()
    invalid_replacements: list[dict[str, str]] = []
    true_replacements: list[dict[str, str]] = []
    suggested_replacements_not_applied: list[dict[str, str]] = []

    for replacement in replacements:
        remove_id = replacement.get("remove", "")
        add_id = replacement.get("add", "")
        invalid_reason = None
        if remove_id not in base_set:
            invalid_reason = "remove_not_base_chart"
        elif not add_id:
            invalid_reason = "missing_add_candidate"
        else:
            invalid_reason = _safe_candidate_reason(candidate_by_id.get(add_id), fields)
        if invalid_reason:
            invalid_replacements.append({**replacement, "invalid_reason": invalid_reason})
            continue
        valid_replacement_adds.add(add_id)
        if add_id in selected_set_before:
            true_replacements.append(replacement)
        elif remove_id not in selected_set_before and add_id not in selected_set_before:
            suggested_replacements_not_applied.append(replacement)

    remove_as_redundant = set(_dict_value(result, "remove_as_redundant"))
    remove_only_declared = set(_dict_value(result, "remove_only"))
    replacement_remove_ids = {item.get("remove", "") for item in replacements if item.get("remove")}
    true_replacement_remove_ids = {item["remove"] for item in true_replacements}
    suggested_replacement_remove_ids = {item["remove"] for item in suggested_replacements_not_applied}
    ids_to_remove = set(remove_as_redundant)
    ids_to_remove.update(true_replacement_remove_ids)

    standalone_added = [
        chart_id
        for chart_id in selected_ids_before
        if chart_id not in base_set and chart_id not in valid_replacement_adds
    ]
    ids_to_remove.update(standalone_added)

    filtered = [chart for chart in selected if str(chart.get("chart_id")) not in ids_to_remove]
    if len(filtered) > len(base_ids):
        overflow = len(filtered) - len(base_ids)
        for chart in list(reversed(filtered)):
            chart_id = str(chart.get("chart_id"))
            if overflow <= 0:
                break
            if chart_id not in base_set:
                filtered.remove(chart)
                standalone_added.append(chart_id)
                overflow -= 1

    final_ids = _chart_ids(filtered)
    final_set = set(final_ids)
    remove_only_chart_ids = sorted(
        chart_id
        for chart_id in base_set - final_set
        if chart_id not in true_replacement_remove_ids
    )
    consistency_warnings: list[str] = []
    base_chart_audit = _dict_value(result, "base_chart_audit")
    if not base_chart_audit:
        consistency_warnings.append("missing_base_chart_audit")
    if not isinstance(result.get("remove_only"), dict):
        consistency_warnings.append("missing_remove_only")
    for chart_id, audit in base_chart_audit.items():
        if not isinstance(audit, dict):
            continue
        decision = str(audit.get("decision") or "")
        replacement_id = str(audit.get("replacement_chart_id") or "")
        if decision == "remove" and chart_id in final_set:
            consistency_warnings.append(f"audit_remove_still_selected:{chart_id}")
        if decision == "replace":
            if chart_id in final_set:
                consistency_warnings.append(f"audit_replace_remove_still_selected:{chart_id}")
            if replacement_id and replacement_id not in final_set:
                consistency_warnings.append(f"audit_replace_add_not_selected:{replacement_id}")

    final_ids_before_status = _chart_ids(filtered)
    status = "ok"
    reason = "replacement_consistency_ok"
    if standalone_added or final_ids_before_status != selected_ids_before or len(final_ids_before_status) > len(base_ids):
        status = "fixed"
        reason = "removed_standalone_or_inconsistent_selected_charts"
    if not filtered:
        status = "fallback"
        reason = "replacement_consistency_removed_all_charts"

    trace = {
        "base_chart_audit": base_chart_audit,
        "remove_only": _dict_value(result, "remove_only"),
        "true_replacements": true_replacements,
        "suggested_replacements_not_applied": suggested_replacements_not_applied,
        "invalid_replacements": invalid_replacements,
        "standalone_added_candidate_ids": list(dict.fromkeys(standalone_added)),
        "selected_non_base_without_valid_replace": list(dict.fromkeys(standalone_added)),
        "remove_only_chart_ids": remove_only_chart_ids,
        "replacement_consistency_status": status,
        "replacement_consistency_reason": reason,
        "replacement_consistency_warnings": consistency_warnings,
        "final_chart_count_le_base": len(final_ids) <= len(base_ids),
        "base_chart_count": len(base_ids),
        "final_chart_count": len(final_ids),
    }
    return filtered, trace


def _raw_ids(result: dict[str, Any]) -> list[str]:
    raw = result.get("selected_chart_ids")
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        chart_id = str(item).strip()
        if chart_id and chart_id not in ids:
            ids.append(chart_id)
    return ids


def _chart_ids(charts: list[dict[str, Any]]) -> list[str]:
    return [str(chart.get("chart_id")) for chart in charts if isinstance(chart, dict) and chart.get("chart_id")]


def _bar_count(charts: list[dict[str, Any]]) -> int:
    return sum(1 for chart in charts if str(chart.get("chart_kind") or "") == "bar")


def _bar_ratio(charts: list[dict[str, Any]]) -> float:
    return round(_bar_count(charts) / len(charts), 4) if charts else 0.0


_BAR_DIMENSION_FIELDS = {
    "category",
    "sub_category",
    "product_name",
    "productline",
    "segment",
    "region",
    "country",
    "order_status",
    "deal_size",
    "discount",
}
_BAR_METRIC_FIELDS = {"sales_amount", "profit", "quantity", "discount", "unit_price", "order_id"}
_STRUCTURE_INSIGHT_TYPES = {"mix", "concentration", "structure"}


def _apply_bar_dedup_guard(
    *,
    selected: list[dict[str, Any]],
    candidate_charts: list[dict[str, Any]],
    schema_mapping: SchemaMapping,
    chart_value_audit: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = set(schema_mapping.field_mapping.values())
    candidate_by_id = {
        str(candidate.get("chart_id")): candidate
        for candidate in candidate_charts
        if isinstance(candidate, dict) and candidate.get("chart_id")
    }
    value_by_id = chart_value_audit.get("chart_value_by_id", {}) if isinstance(chart_value_audit, dict) else {}
    low_visual = set(chart_value_audit.get("low_visual_gain_risk_chart_ids", []) or []) if isinstance(chart_value_audit, dict) else set()
    low_info = set(chart_value_audit.get("low_information_gain_chart_ids", []) or []) if isinstance(chart_value_audit, dict) else set()
    before_count = len(selected)
    bars = [
        {**candidate_by_id.get(str(chart.get("chart_id") or ""), {}), **chart}
        for chart in selected
        if str(chart.get("chart_kind") or "") == "bar"
    ]
    trace = {
        "bar_dedup_guard_applied": False,
        "bar_dedup_before_count": before_count,
        "bar_dedup_after_count": before_count,
        "removed_duplicate_bar_chart_ids": [],
        "retained_bar_chart_ids": _chart_ids(bars),
        "bar_dedup_reason": "not_needed",
        "bar_dedup_groups": [],
        "final_bar_chart_count": len(bars),
        "final_bar_chart_ratio": _bar_ratio(selected),
    }
    if not bars:
        trace["bar_dedup_reason"] = "no_bar"
        return selected, trace
    if len(bars) == 1:
        trace["bar_dedup_reason"] = "only_one_bar"
        return selected, trace

    def text(chart: dict[str, Any]) -> str:
        return " ".join(
            str(chart.get(key) or "")
            for key in ("chart_id", "title", "business_question", "insight_type")
        ).lower()

    def dimensions(chart: dict[str, Any]) -> set[str]:
        required = {str(field) for field in chart.get("required_fields", []) or []}
        optional = {str(field) for field in chart.get("optional_fields", []) or []}
        direct = required & _BAR_DIMENSION_FIELDS
        if direct:
            return direct
        optional_available = optional & fields & _BAR_DIMENSION_FIELDS
        return optional_available

    def metrics(chart: dict[str, Any]) -> set[str]:
        required = {str(field) for field in chart.get("required_fields", []) or []}
        metric_fields = required & _BAR_METRIC_FIELDS
        if "order_id" in metric_fields:
            metric_fields = (metric_fields - {"order_id"}) | {"order_count"}
        return metric_fields

    def family(chart: dict[str, Any]) -> str:
        chart_text = text(chart)
        insight = str(chart.get("insight_type") or "")
        if insight == "risk" or any(token in chart_text for token in ("risk", "loss", "profit", "margin", "discount", "亏损", "利润", "折扣")):
            return "risk_quality"
        if insight in _STRUCTURE_INSIGHT_TYPES or any(
            token in chart_text
            for token in ("contribution", "concentration", "structure", "mix", "sales", "贡献", "集中", "结构", "销售额")
        ):
            return "structure_contribution"
        return insight or "other"

    def uses_fallback_dimension(chart: dict[str, Any]) -> bool:
        required = {str(field) for field in chart.get("required_fields", []) or []}
        optional = {str(field) for field in chart.get("optional_fields", []) or []}
        return not (required & _BAR_DIMENSION_FIELDS) and bool(optional & fields & _BAR_DIMENSION_FIELDS)

    def duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_dims = dimensions(left)
        right_dims = dimensions(right)
        if not left_dims or left_dims != right_dims:
            return False
        left_metrics = metrics(left)
        right_metrics = metrics(right)
        if left_metrics and right_metrics and not (left_metrics & right_metrics):
            return False
        left_family = family(left)
        right_family = family(right)
        return left_family == right_family == "structure_contribution"

    duplicate_groups: list[set[str]] = []
    for index, left in enumerate(bars):
        for right in bars[index + 1 :]:
            if not duplicate(left, right):
                continue
            pair = {str(left.get("chart_id")), str(right.get("chart_id"))}
            merged = False
            for group in duplicate_groups:
                if group & pair:
                    group.update(pair)
                    merged = True
                    break
            if not merged:
                duplicate_groups.append(pair)
    if not duplicate_groups:
        trace["bar_dedup_reason"] = "bars_not_semantically_duplicate"
        return selected, trace

    bar_by_id = {str(chart.get("chart_id")): chart for chart in bars}

    def value_score(chart_id: str) -> float:
        return float(value_by_id.get(chart_id, {}).get("overall_chart_value_score") or 0)

    def rank(chart: dict[str, Any]) -> tuple[Any, ...]:
        chart_id = str(chart.get("chart_id") or "")
        return (
            not uses_fallback_dimension(chart),
            chart_id not in low_info,
            chart_id not in low_visual,
            value_score(chart_id),
            bool(chart.get("evidence_ids")),
            float(chart.get("dataset_specificity_score") or 0),
            float(chart.get("score") or 0),
            len(dimensions(chart)),
            chart_id,
        )

    removed_ids: list[str] = []
    groups_trace: list[dict[str, Any]] = []
    for group in duplicate_groups:
        charts = [bar_by_id[chart_id] for chart_id in group if chart_id in bar_by_id]
        if len(charts) < 2:
            continue
        retained = max(charts, key=rank)
        retained_id = str(retained.get("chart_id") or "")
        removed = sorted(str(chart.get("chart_id") or "") for chart in charts if str(chart.get("chart_id") or "") != retained_id)
        removed_ids.extend(removed)
        groups_trace.append(
            {
                "chart_ids": sorted(str(chart.get("chart_id") or "") for chart in charts),
                "retained_chart_id": retained_id,
                "removed_chart_ids": removed,
                "semantic_dimensions": sorted(dimensions(retained)),
                "semantic_metric_fields": sorted(metrics(retained)),
                "reason": "same_category_dimension_metric_and_structure_question",
            }
        )
    removed_set = set(removed_ids)
    if not removed_set:
        trace["bar_dedup_reason"] = "bars_not_semantically_duplicate"
        return selected, trace

    deduped = [chart for chart in selected if str(chart.get("chart_id") or "") not in removed_set]
    trace.update(
        {
            "bar_dedup_guard_applied": True,
            "bar_dedup_after_count": len(deduped),
            "removed_duplicate_bar_chart_ids": sorted(removed_set),
            "retained_bar_chart_ids": [
                str(chart.get("chart_id") or "")
                for chart in deduped
                if str(chart.get("chart_kind") or "") == "bar"
            ],
            "bar_dedup_reason": "removed_semantically_duplicate_bar_charts",
            "bar_dedup_groups": groups_trace,
            "final_bar_chart_count": _bar_count(deduped),
            "final_bar_chart_ratio": _bar_ratio(deduped),
        }
    )
    return deduped, trace


def _redundancy_trace(
    *,
    result: dict[str, Any],
    base_selected: list[dict[str, Any]],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    base_ids = _chart_ids(base_selected)
    final_ids = _chart_ids(selected)
    base_set = set(base_ids)
    final_set = set(final_ids)
    before_groups = _redundancy_groups(base_selected)
    after_groups = _redundancy_groups(selected)
    before_pressure = _redundancy_pressure(base_selected)
    after_pressure = _redundancy_pressure(selected)
    return {
        "kept_for_unique_value": _dict_value(result, "kept_for_unique_value") or _dict_value(result, "selection_rationale"),
        "remove_as_redundant": _dict_value(result, "remove_as_redundant"),
        "replacement_decisions": _replacement_decisions(result),
        "redundancy_summary": str(result.get("redundancy_summary") or ""),
        "base_selected_chart_ids": base_ids,
        "final_selected_chart_ids": final_ids,
        "base_retained_chart_ids": [chart_id for chart_id in base_ids if chart_id in final_set],
        "base_removed_chart_ids": [chart_id for chart_id in base_ids if chart_id not in final_set],
        "newly_selected_challenger_ids": [chart_id for chart_id in final_ids if chart_id not in base_set],
        "base_retention_ratio": round(len(base_set & final_set) / len(base_ids), 4) if base_ids else 0,
        "final_bar_chart_count": after_pressure["bar_chart_count"],
        "final_bar_chart_ratio": after_pressure["bar_chart_ratio"],
        "redundancy_pressure_before": before_pressure,
        "redundancy_pressure_after": after_pressure,
        "same_section_same_kind_groups_before": before_groups["same_section_same_kind"],
        "same_section_same_kind_groups_after": after_groups["same_section_same_kind"],
    }


def _budget_for(candidate: dict[str, Any], chart_budget: dict[str, Any]) -> int:
    role = str(candidate.get("section_role") or "skipped")
    if role == "core":
        return int(chart_budget.get("core_max_per_section") or 0)
    if role == "support":
        return int(chart_budget.get("support_max_per_section") or 0)
    return int(chart_budget.get("skipped_max_per_section") or 0)


def _selected_chart(
    candidate: dict[str, Any],
    rationale: dict[str, Any],
    base_selected_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chart_id = str(candidate.get("chart_id"))
    base_chart = (base_selected_by_id or {}).get(chart_id, {})
    llm_reason = str(rationale.get(chart_id) or "").strip()
    reason = llm_reason or str(base_chart.get("reason") or "LLM selected this chart from the safe candidate pool.")
    return {
        "chart_id": chart_id,
        "section_id": candidate.get("section_id"),
        "chart_kind": candidate.get("chart_kind"),
        "title": candidate.get("title"),
        "reason": reason,
        "business_question": candidate.get("business_question"),
        "selection_source": "llm_sanitized",
        "evidence_ids": list(candidate.get("evidence_ids") or base_chart.get("evidence_ids") or []),
        "evidence_summary": candidate.get("evidence_summary") or base_chart.get("evidence_summary") or "",
        "llm_selection_reason": llm_reason,
        "score": candidate.get("score"),
        "ranking_factors": candidate.get("ranking_factors") or base_chart.get("ranking_factors", []),
        "insight_type": candidate.get("insight_type") or "structure",
        "dataset_specificity_score": candidate.get("dataset_specificity_score") or base_chart.get("dataset_specificity_score"),
        "intent_support_score": candidate.get("intent_support_score", 0),
        "matched_intent_ids": candidate.get("matched_intent_ids", []),
        "best_intent_match_quality": candidate.get("best_intent_match_quality"),
    }


def _sanitize(
    *,
    raw_ids: list[str],
    result: dict[str, Any],
    candidate_charts: list[dict[str, Any]],
    schema_mapping: SchemaMapping,
    chart_budget: dict[str, Any],
    base_selected_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    candidate_by_id = {str(candidate.get("chart_id")): candidate for candidate in candidate_charts}
    fields = set(schema_mapping.field_mapping.values())
    trace_lists = {
        "invalid_chart_ids": [],
        "hard_rejected_chart_ids": [],
        "budget_rejected_chart_ids": [],
        "chart_kind_rejected_chart_ids": [],
    }
    selected: list[dict[str, Any]] = []
    count_by_section: dict[str, int] = {}
    kinds_by_section: dict[str, set[str]] = {}
    rationale = {}
    if isinstance(result.get("kept_for_unique_value"), dict):
        rationale.update(result.get("kept_for_unique_value") or {})
    if isinstance(result.get("selection_rationale"), dict):
        rationale.update(result.get("selection_rationale") or {})

    def is_safe_candidate(candidate: dict[str, Any] | None) -> bool:
        if candidate is None:
            return False
        required = {str(field) for field in candidate.get("required_fields", []) or []}
        return required <= fields and not _is_hard_rejection(str(candidate.get("rejection_reason") or ""))

    def later_different_kind_available(index: int, section_id: str, chart_kind: str) -> bool:
        for later_chart_id in raw_ids[index + 1 :]:
            later = candidate_by_id.get(later_chart_id)
            if (
                is_safe_candidate(later)
                and str(later.get("section_id") or "") == section_id
                and str(later.get("chart_kind") or "") != chart_kind
            ):
                return True
        return False

    for index, chart_id in enumerate(raw_ids):
        candidate = candidate_by_id.get(chart_id)
        if candidate is None:
            trace_lists["invalid_chart_ids"].append(chart_id)
            continue
        if not is_safe_candidate(candidate):
            trace_lists["hard_rejected_chart_ids"].append(chart_id)
            continue
        section_id = str(candidate.get("section_id") or "")
        chart_kind = str(candidate.get("chart_kind") or "")
        current_count = count_by_section.get(section_id, 0)
        if current_count >= _budget_for(candidate, chart_budget):
            trace_lists["budget_rejected_chart_ids"].append(chart_id)
            continue
        section_kinds = kinds_by_section.get(section_id, set())
        if (
            current_count > 0
            and section_kinds == {chart_kind}
            and later_different_kind_available(index, section_id, chart_kind)
        ):
            trace_lists["chart_kind_rejected_chart_ids"].append(chart_id)
            continue
        selected.append(_selected_chart(candidate, rationale, base_selected_by_id))
        count_by_section[section_id] = current_count + 1
        kinds_by_section.setdefault(section_id, set()).add(chart_kind)
    return selected, trace_lists


def _apply_chart_count_guard(
    *,
    selected: list[dict[str, Any]],
    raw_ids: list[str],
    result: dict[str, Any],
    candidate_charts: list[dict[str, Any]],
    schema_mapping: SchemaMapping,
    chart_budget: dict[str, Any],
    deterministic_chart_selection_plan: dict[str, Any],
    base_selected_by_id: dict[str, dict[str, Any]],
    disable_mechanical_restore: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del raw_ids, result, chart_budget, base_selected_by_id
    before_count = len(selected)
    base_selected = [
        dict(chart)
        for chart in deterministic_chart_selection_plan.get("selected_charts", []) or []
        if isinstance(chart, dict) and chart.get("chart_id")
    ]
    base_count = len(base_selected)
    selected_ids = set(_chart_ids(selected))
    selected_insight_types = {str(chart.get("insight_type") or "") for chart in selected if chart.get("insight_type")}
    rejected_ids = {
        str(chart.get("chart_id"))
        for chart in deterministic_chart_selection_plan.get("rejected_charts", []) or []
        if isinstance(chart, dict) and chart.get("chart_id")
    }
    candidate_by_id = {
        str(candidate.get("chart_id")): candidate
        for candidate in candidate_charts
        if isinstance(candidate, dict) and candidate.get("chart_id")
    }
    fields = set(schema_mapping.field_mapping.values())
    before_bar_count = _bar_count(selected)
    guard_trace = {
        "minimum_chart_count": 0,
        "chart_count_guard_applied": False,
        "chart_count_before_guard": before_count,
        "chart_count_after_guard": before_count,
        "restored_base_chart_ids": [],
        "chart_count_guard_status": "not_applied",
        "chart_count_guard_reason": "guard_not_needed",
        "final_bar_chart_count": before_bar_count,
        "final_bar_chart_ratio": _bar_ratio(selected),
    }

    if disable_mechanical_restore:
        guard_trace["chart_count_guard_status"] = "disabled"
        guard_trace["chart_count_guard_reason"] = "disabled_no_mechanical_restoration"
        return selected, guard_trace

    if base_count < 4:
        guard_trace["chart_count_guard_reason"] = "base_chart_count_below_guard_threshold"
        return selected, guard_trace

    base_has_bar = any(str(chart.get("chart_kind") or "") == "bar" for chart in base_selected)
    needs_minimum_count = before_count < 3
    needs_bar_restore = base_has_bar and before_bar_count == 0
    if not needs_minimum_count and not needs_bar_restore:
        guard_trace["chart_count_guard_reason"] = (
            "final_already_has_bar_chart" if before_bar_count > 0 else "guard_not_needed"
        )
        return selected, guard_trace

    def safe_base_candidate(chart: dict[str, Any]) -> bool:
        chart_id = str(chart.get("chart_id") or "")
        candidate = candidate_by_id.get(chart_id, chart)
        if not chart_id or chart_id in selected_ids or chart_id in rejected_ids:
            return False
        if _is_hard_rejection(str(candidate.get("rejection_reason") or chart.get("rejection_reason") or "")):
            return False
        if str(candidate.get("section_role") or chart.get("section_role") or "") == "skipped":
            return False
        required = {
            str(field)
            for field in (candidate.get("required_fields") or chart.get("required_fields") or [])
        }
        return required <= fields

    def business_question_priority(chart: dict[str, Any]) -> int:
        text = f"{chart.get('business_question') or ''} {chart.get('title') or ''}".lower()
        keywords = ("category", "segment", "region", "country", "product", "market", "structure", "mix", "concentration")
        return sum(1 for keyword in keywords if keyword in text)

    candidates = [chart for chart in base_selected if safe_base_candidate(chart)]
    if needs_bar_restore:
        candidates = [chart for chart in candidates if str(chart.get("chart_kind") or "") == "bar"]
    if not candidates:
        guard_trace["chart_count_guard_reason"] = "no_safe_base_chart_to_restore"
        return selected, guard_trace

    candidates.sort(
        key=lambda chart: (
            0 if str(chart.get("chart_kind") or "") == "bar" else 1,
            0 if chart.get("evidence_ids") else 1,
            -float(chart.get("dataset_specificity_score") or 0),
            -float(chart.get("score") or 0),
            str(chart.get("insight_type") or "") in selected_insight_types,
            -business_question_priority(chart),
            str(chart.get("chart_id") or ""),
        )
    )
    restored = dict(candidates[0])
    restored["selection_source"] = "llm_sanitized_guard"
    restored["reason"] = str(restored.get("reason") or "Restored from deterministic base selection by chart count guard.")
    selected = [*selected, restored]
    after_bar_count = _bar_count(selected)
    if needs_minimum_count and needs_bar_restore:
        reason = "restored_one_safe_base_chart_for_minimum_count_and_bar_coverage"
    elif needs_bar_restore:
        reason = "restored_one_safe_base_bar_chart"
    else:
        reason = "restored_one_safe_base_chart_for_minimum_count"
    guard_trace.update(
        {
            "chart_count_guard_applied": True,
            "chart_count_after_guard": len(selected),
            "restored_base_chart_ids": [str(restored.get("chart_id"))],
            "chart_count_guard_status": "applied",
            "chart_count_guard_reason": reason,
            "final_bar_chart_count": after_bar_count,
            "final_bar_chart_ratio": _bar_ratio(selected),
        }
    )
    return selected, guard_trace


def build_llm_chart_selection_plan(
    *,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any],
    section_priority: dict[str, Any],
    evidence_pack: dict[str, Any] | None,
    deterministic_chart_selection_plan: dict[str, Any],
    diagnostics: dict[str, Any],
    candidate_charts: list[dict[str, Any]],
    llm_client: Any,
    llm_profile: str | None,
    llm_chart_selection_enabled: bool | None,
    rebuild_diagnostics: Callable[[list[dict[str, Any]]], dict[str, Any]],
    intent_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = dict(deterministic_chart_selection_plan)
    base_selected_by_id = {
        str(chart.get("chart_id")): dict(chart)
        for chart in deterministic_chart_selection_plan.get("selected_charts", []) or []
        if isinstance(chart, dict) and chart.get("chart_id")
    }
    can_attempt, reason = _enabled(llm_client, llm_profile, llm_chart_selection_enabled)
    if not can_attempt:
        plan["llm_selection_trace"] = {
            **plan.get("llm_selection_trace", {}),
            "attempted": False,
            "applied": False,
            "status": "skipped",
            "reason": reason,
            "invalid_chart_ids": [],
            "hard_rejected_chart_ids": [],
            "budget_rejected_chart_ids": [],
            "chart_kind_rejected_chart_ids": [],
            "minimum_chart_count": 0,
            "chart_count_guard_applied": False,
            "chart_count_before_guard": 0,
            "chart_count_after_guard": 0,
            "restored_base_chart_ids": [],
            "chart_count_guard_status": "disabled",
            "chart_count_guard_reason": "disabled_no_mechanical_restoration",
            "bar_dedup_guard_applied": False,
            "bar_dedup_before_count": 0,
            "bar_dedup_after_count": 0,
            "removed_duplicate_bar_chart_ids": [],
            "retained_bar_chart_ids": [],
            "bar_dedup_reason": "disabled",
            "bar_dedup_groups": [],
            "fallback_reason": None,
        }
        return plan

    payload = _payload(
        schema_mapping=schema_mapping,
        analysis_focus=analysis_focus,
        section_priority=section_priority,
        evidence_pack=evidence_pack,
        dataset_profile=dataset_profile,
        deterministic_chart_selection_plan=deterministic_chart_selection_plan,
        diagnostics=diagnostics,
        candidate_charts=candidate_charts,
        intent_context=intent_context,
    )
    trace = {
        "attempted": True,
        "applied": False,
        "status": "fallback",
        "reason": "LLM chart selection attempted.",
        "invalid_chart_ids": [],
        "hard_rejected_chart_ids": [],
        "budget_rejected_chart_ids": [],
        "chart_kind_rejected_chart_ids": [],
        "minimum_chart_count": 0,
        "chart_count_guard_applied": False,
        "chart_count_before_guard": 0,
        "chart_count_after_guard": 0,
        "restored_base_chart_ids": [],
        "chart_count_guard_status": "disabled",
        "chart_count_guard_reason": "disabled_no_mechanical_restoration",
        "bar_dedup_guard_applied": False,
        "bar_dedup_before_count": 0,
        "bar_dedup_after_count": 0,
        "removed_duplicate_bar_chart_ids": [],
        "retained_bar_chart_ids": [],
        "bar_dedup_reason": "not_run",
        "bar_dedup_groups": [],
        "fallback_reason": None,
    }
    try:
        result = _call_llm(llm_client, payload)
    except Exception as exc:
        trace["fallback_reason"] = f"llm_error: {exc}"
        plan["selection_mode"] = "llm_fallback"
        plan["llm_selection_trace"] = trace
        return plan

    raw_selected_ids = _raw_ids(result)
    selected, trace_lists = _sanitize(
        raw_ids=raw_selected_ids,
        result=result,
        candidate_charts=candidate_charts,
        schema_mapping=schema_mapping,
        chart_budget=dict(deterministic_chart_selection_plan.get("chart_budget", {})),
        base_selected_by_id=base_selected_by_id,
    )
    trace.update(trace_lists)
    base_selected = [
        dict(chart)
        for chart in deterministic_chart_selection_plan.get("selected_charts", []) or []
        if isinstance(chart, dict)
    ]
    chart_value_audit = payload.get("chart_value_audit", {}) if isinstance(payload.get("chart_value_audit"), dict) else {}
    plan["llm_selected_chart_ids"] = raw_selected_ids
    plan["llm_sanitized_selected_chart_ids"] = [chart["chart_id"] for chart in selected]
    plan["llm_selection_rationale"] = {
        "selection_rationale": result.get("selection_rationale", {}),
        "base_chart_audit": result.get("base_chart_audit", {}),
        "kept_for_unique_value": result.get("kept_for_unique_value", {}),
        "remove_as_redundant": result.get("remove_as_redundant", {}),
        "replace": result.get("replace", []),
        "remove_only": result.get("remove_only", {}),
        "redundancy_summary": str(result.get("redundancy_summary") or ""),
        "overall_strategy": str(result.get("overall_strategy") or ""),
        "rejected_high_specific_notes": result.get("rejected_high_specific_notes", {}),
    }

    fallback_reasons: list[str] = []
    if not raw_selected_ids:
        fallback_reasons.append("empty_selection")
    if trace["invalid_chart_ids"]:
        fallback_reasons.append("invalid_chart_ids")
    if trace["hard_rejected_chart_ids"]:
        fallback_reasons.append("hard_rejected_chart_ids")
    if not selected:
        fallback_reasons.append("no_sanitized_charts")
    if fallback_reasons:
        trace["fallback_reason"] = ", ".join(dict.fromkeys(fallback_reasons))
        plan["selection_mode"] = "llm_fallback"
        plan["llm_selection_trace"] = trace
        return plan

    selected, consistency_trace = _replacement_consistency(
        selected=selected,
        result=result,
        candidate_charts=candidate_charts,
        schema_mapping=schema_mapping,
        base_selected=base_selected,
    )
    trace.update(consistency_trace)
    if not selected:
        trace["fallback_reason"] = "replacement_consistency_removed_all_charts"
        plan["selection_mode"] = "llm_fallback"
        plan["llm_selection_trace"] = trace
        return plan
    selected, guard_trace = _apply_chart_count_guard(
        selected=selected,
        raw_ids=raw_selected_ids,
        result=result,
        candidate_charts=candidate_charts,
        schema_mapping=schema_mapping,
        chart_budget=dict(deterministic_chart_selection_plan.get("chart_budget", {})),
        deterministic_chart_selection_plan=deterministic_chart_selection_plan,
        base_selected_by_id=base_selected_by_id,
        disable_mechanical_restore=any(key in result for key in {"base_chart_audit", "remove_only", "replace", "remove_as_redundant"}),
    )
    trace.update(guard_trace)
    selected, bar_dedup_trace = _apply_bar_dedup_guard(
        selected=selected,
        candidate_charts=candidate_charts,
        schema_mapping=schema_mapping,
        chart_value_audit=chart_value_audit,
    )
    trace.update(bar_dedup_trace)
    plan["llm_sanitized_selected_chart_ids"] = [chart["chart_id"] for chart in selected]
    trace.update(_redundancy_trace(result=result, base_selected=base_selected, selected=selected))
    trace.update(consistency_trace)
    trace.update(
        _chart_value_trace(
            audit=chart_value_audit,
            base_selected=base_selected,
            selected=selected,
            result=result,
        )
    )

    trace["applied"] = True
    trace["status"] = "llm_sanitized"
    trace["reason"] = "LLM selected chart IDs were sanitized and applied."
    plan["selection_mode"] = "llm_sanitized"
    plan["selected_charts"] = selected
    plan["diagnostics"] = rebuild_diagnostics(selected)
    plan["llm_selection_trace"] = trace
    return plan
