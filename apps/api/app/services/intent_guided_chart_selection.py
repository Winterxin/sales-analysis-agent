from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_portfolio_diversity import apply_portfolio_diversity_gate
from app.services.chart_portfolio_sufficiency import apply_portfolio_sufficiency_gate


HARD_REJECTION_PREFIXES = (
    "missing required fields",
    "skipped section",
    "country field has no useful market slice",
    "too many categories",
    "too many category combinations",
    "limited by data availability",
)

QUALITY_RANK = {"exact_match": 0, "strong_match": 1, "weak_match": 2, "false_friend_match": 3}


def _is_hard_rejection(reason: str | None) -> bool:
    return any(str(reason or "").startswith(prefix) for prefix in HARD_REJECTION_PREFIXES)


def _priority_bonus(priority: str | None) -> float:
    value = str(priority or "").strip().lower()
    if value == "high":
        return 0.05
    if value == "medium":
        return 0.03
    if value == "low":
        return 0.01
    return 0.0


def _quality_bonus(quality: str) -> float:
    if quality == "exact_match":
        return 0.3
    if quality == "strong_match":
        return 0.2
    return 0.0


def _budget_for(candidate: dict[str, Any], chart_budget: dict[str, Any]) -> int:
    role = str(candidate.get("section_role") or "skipped")
    if role == "core":
        return int(chart_budget.get("core_max_per_section") or 0)
    if role == "support":
        return int(chart_budget.get("support_max_per_section") or 0)
    return int(chart_budget.get("skipped_max_per_section") or 0)


def _selected_chart(candidate: dict[str, Any], *, source: str, reason: str) -> dict[str, Any]:
    return {
        "chart_id": candidate.get("chart_id"),
        "section_id": candidate.get("section_id"),
        "chart_kind": candidate.get("chart_kind"),
        "title": candidate.get("title"),
        "reason": reason,
        "business_question": candidate.get("business_question"),
        "selection_source": source,
        "evidence_ids": [],
        "evidence_summary": reason,
        "score": candidate.get("score"),
        "ranking_factors": candidate.get("ranking_factors", []),
        "insight_type": candidate.get("insight_type") or "structure",
        "dataset_specificity_score": candidate.get("dataset_specificity_score"),
        "intent_support_score": candidate.get("intent_support_score", 0),
        "matched_intent_ids": candidate.get("matched_intent_ids", []),
        "best_intent_match_quality": candidate.get("best_intent_match_quality"),
    }


def _empty_trace(base_plan: dict[str, Any], *, attempted: bool, status: str, fallback_reason: str | None = None) -> dict[str, Any]:
    base_ids = [str(item.get("chart_id")) for item in base_plan.get("selected_charts", []) if isinstance(item, dict)]
    return {
        "attempted": attempted,
        "applied": False,
        "status": status,
        "fallback_reason": fallback_reason,
        "base_selected_chart_ids": base_ids,
        "intent_supported_candidate_ids": [],
        "exact_supported_candidate_ids": [],
        "strong_supported_candidate_ids": [],
        "selected_due_to_intent": [],
        "intent_supported_not_selected": [],
        "intent_supported_but_hard_rejected_candidate_ids": [],
        "intent_supported_hard_rejection_reasons": {},
        "not_selected_reasons": {},
        "weak_match_ignored_candidate_ids": [],
        "false_friend_ignored_candidate_ids": [],
        "sanitizer_rejected_candidate_ids": [],
        "final_selected_chart_ids": base_ids,
    }


def _intent_matches(chart_intent_plan: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, int]]:
    support: dict[str, dict[str, Any]] = {}
    ignored = {"weak": set(), "false_friend": set()}
    counts = {"valid_supported": 0, "exact": 0, "strong": 0, "weak": 0, "false_friend": 0}
    for intent in chart_intent_plan.get("validated_intents", []) or []:
        if not isinstance(intent, dict):
            continue
        if intent.get("validation_status") != "valid_supported":
            continue
        counts["valid_supported"] += 1
        intent_id = str(intent.get("intent_id") or "")
        priority = str(intent.get("priority") or "")
        for match in intent.get("candidate_matches", []) or []:
            if not isinstance(match, dict):
                continue
            chart_id = str(match.get("chart_id") or "")
            quality = str(match.get("match_quality") or "")
            if quality == "weak_match":
                counts["weak"] += 1
                ignored["weak"].add(chart_id)
                continue
            if quality == "false_friend_match":
                counts["false_friend"] += 1
                ignored["false_friend"].add(chart_id)
                continue
            if quality not in {"exact_match", "strong_match"} or not chart_id:
                continue
            counts["exact" if quality == "exact_match" else "strong"] += 1
            increment = _quality_bonus(quality) + _priority_bonus(priority)
            entry = support.setdefault(
                chart_id,
                {
                    "score": 0.0,
                    "intent_ids": [],
                    "priorities": [],
                    "qualities": [],
                    "reasons": [],
                },
            )
            entry["score"] += increment
            entry["intent_ids"].append(intent_id)
            entry["priorities"].append(priority)
            entry["qualities"].append(quality)
            entry["reasons"].append(str(match.get("match_reason") or "Intent-supported candidate."))
    return support, ignored, counts


def _best_quality(qualities: list[str]) -> str | None:
    if not qualities:
        return None
    return sorted(qualities, key=lambda item: QUALITY_RANK.get(item, 9))[0]


def _enhance_candidates(
    base_plan: dict[str, Any],
    chart_intent_plan: dict[str, Any],
    fields: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int], dict[str, str]]:
    support, ignored, counts = _intent_matches(chart_intent_plan)
    enhanced: list[dict[str, Any]] = []
    hard_rejected_support: dict[str, str] = {}
    for candidate in base_plan.get("candidate_charts", []) or []:
        if not isinstance(candidate, dict):
            continue
        item = dict(candidate)
        entry = support.get(str(item.get("chart_id") or ""))
        if entry:
            required = {str(field) for field in item.get("required_fields", []) or []}
            hard_reason = ""
            if not required <= fields:
                hard_reason = "missing required fields: " + ", ".join(sorted(required - fields))
            elif str(item.get("section_role") or "") == "skipped":
                hard_reason = "skipped section"
            elif _is_hard_rejection(str(item.get("rejection_reason") or "")):
                hard_reason = str(item.get("rejection_reason") or "hard rejection")
            if hard_reason:
                hard_rejected_support[str(item.get("chart_id") or "")] = hard_reason
                item.update(
                    {
                        "intent_support_score": 0,
                        "matched_intent_ids": [],
                        "matched_intent_priorities": [],
                        "best_intent_match_quality": None,
                        "intent_match_reasons": [],
                        "intent_guided_candidate": False,
                        "intent_support_type": "none",
                    }
                )
            else:
                best = _best_quality(entry["qualities"])
                item.update(
                    {
                        "intent_support_score": round(float(entry["score"]), 4),
                        "matched_intent_ids": entry["intent_ids"],
                        "matched_intent_priorities": entry["priorities"],
                        "best_intent_match_quality": best,
                        "intent_match_reasons": entry["reasons"],
                        "intent_guided_candidate": True,
                        "intent_support_type": "exact" if best == "exact_match" else "strong",
                    }
                )
        else:
            item.update(
                {
                    "intent_support_score": 0,
                    "matched_intent_ids": [],
                    "matched_intent_priorities": [],
                    "best_intent_match_quality": None,
                    "intent_match_reasons": [],
                    "intent_guided_candidate": False,
                    "intent_support_type": "none",
                }
            )
        enhanced.append(item)
    return enhanced, ignored, counts, hard_rejected_support


def _is_safe_candidate(candidate: dict[str, Any], fields: set[str]) -> bool:
    required = {str(field) for field in candidate.get("required_fields", []) or []}
    if not required <= fields:
        return False
    if _is_hard_rejection(str(candidate.get("rejection_reason") or "")):
        return False
    return True


def _deterministic_intent_selection(
    *,
    enhanced_candidates: list[dict[str, Any]],
    base_plan: dict[str, Any],
    fields: set[str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, str]]:
    chart_budget = dict(base_plan.get("chart_budget", {}))
    selected: list[dict[str, Any]] = []
    rejected: list[str] = []
    not_selected_reasons: dict[str, str] = {}
    count_by_section: dict[str, int] = {}
    insight_by_section: dict[str, set[str]] = {}
    candidates = sorted(
        enhanced_candidates,
        key=lambda item: (
            str(item.get("section_id") or ""),
            -(float(item.get("score") or 0) + float(item.get("intent_support_score") or 0)),
            str(item.get("chart_id") or ""),
        ),
    )
    for candidate in candidates:
        chart_id = str(candidate.get("chart_id") or "")
        if not _is_safe_candidate(candidate, fields):
            rejected.append(chart_id)
            not_selected_reasons[chart_id] = str(candidate.get("rejection_reason") or "unsafe_candidate")
            continue
        if candidate.get("score") is None:
            not_selected_reasons[chart_id] = "no_base_score"
            continue
        section_id = str(candidate.get("section_id") or "")
        budget = _budget_for(candidate, chart_budget)
        if count_by_section.get(section_id, 0) >= budget:
            not_selected_reasons[chart_id] = "section budget exceeded"
            continue
        insight_type = str(candidate.get("insight_type") or "structure")
        section_insights = insight_by_section.setdefault(section_id, set())
        if insight_type in section_insights:
            not_selected_reasons[chart_id] = "redundant insight type"
            continue
        reason = (
            "Intent-guided selection: exact/strong chart intent support."
            if candidate.get("intent_guided_candidate")
            else "Intent-guided selection retained a base evidence-ranked chart."
        )
        selected.append(_selected_chart(candidate, source="intent_guided_deterministic", reason=reason))
        count_by_section[section_id] = count_by_section.get(section_id, 0) + 1
        section_insights.add(insight_type)
    return selected, rejected, not_selected_reasons


def _summary(counts: dict[str, int], selected: list[dict[str, Any]]) -> dict[str, Any]:
    selected_supported = [
        item for item in selected if float(item.get("intent_support_score") or 0) > 0
    ]
    total = len(selected)
    return {
        "valid_supported_intent_count": counts["valid_supported"],
        "exact_match_count": counts["exact"],
        "strong_match_count": counts["strong"],
        "weak_match_count": counts["weak"],
        "false_friend_match_count": counts["false_friend"],
        "final_selected_intent_supported_count": len(selected_supported),
        "final_selected_intent_supported_ratio": round(len(selected_supported) / total, 4) if total else 0.0,
    }


def _mark_final_candidate_status(candidates: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    selected_ids = {str(item.get("chart_id")) for item in selected if isinstance(item, dict)}
    for candidate in candidates:
        if str(candidate.get("chart_id")) in selected_ids:
            candidate["availability_status"] = "selected"
            candidate.pop("rejection_reason", None)


def _intent_context(chart_intent_plan: dict[str, Any], enhanced_candidates: list[dict[str, Any]], ignored: dict[str, set[str]]) -> dict[str, Any]:
    supported = [candidate for candidate in enhanced_candidates if candidate.get("intent_guided_candidate")]
    return {
        "chart_intent_strategy": chart_intent_plan.get("overall_chart_strategy", ""),
        "validated_chart_intents": [
            item for item in chart_intent_plan.get("validated_intents", []) if isinstance(item, dict)
        ],
        "intent_supported_candidates": [
            {
                "chart_id": item.get("chart_id"),
                "matched_intent_ids": item.get("matched_intent_ids", []),
                "best_intent_match_quality": item.get("best_intent_match_quality"),
                "intent_support_score": item.get("intent_support_score"),
            }
            for item in supported
        ],
        "unsupported_intents": chart_intent_plan.get("unsupported_intents", []),
        "exact_strong_candidate_ids": [str(item.get("chart_id")) for item in supported],
        "weak_only_intents": [],
        "false_friend_matches": sorted(ignored["false_friend"]),
    }


def apply_intent_guided_chart_selection(
    *,
    base_chart_selection_plan: dict[str, Any],
    chart_intent_plan: dict[str, Any],
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any],
    section_priority: dict[str, Any],
    evidence_pack: dict[str, Any] | None,
    llm_client: Any,
    llm_profile: str | None,
    enabled: bool | None,
) -> dict[str, Any]:
    del dataset_profile
    plan = deepcopy(base_chart_selection_plan)
    fields = set(schema_mapping.field_mapping.values())
    enhanced_candidates, ignored, counts, hard_rejected_support = _enhance_candidates(
        base_chart_selection_plan, chart_intent_plan, fields
    )
    plan["candidate_charts"] = enhanced_candidates
    trace = _empty_trace(plan, attempted=bool(enabled), status="skipped" if not enabled else "attempted")
    trace["weak_match_ignored_candidate_ids"] = sorted(ignored["weak"])
    trace["false_friend_ignored_candidate_ids"] = sorted(ignored["false_friend"])
    trace["intent_supported_but_hard_rejected_candidate_ids"] = sorted(hard_rejected_support)
    trace["intent_supported_hard_rejection_reasons"] = hard_rejected_support
    trace["intent_supported_candidate_ids"] = sorted(
        str(item.get("chart_id")) for item in enhanced_candidates if item.get("intent_guided_candidate")
    )
    trace["exact_supported_candidate_ids"] = sorted(
        str(item.get("chart_id"))
        for item in enhanced_candidates
        if item.get("best_intent_match_quality") == "exact_match"
    )
    trace["strong_supported_candidate_ids"] = sorted(
        str(item.get("chart_id"))
        for item in enhanced_candidates
        if item.get("best_intent_match_quality") == "strong_match"
    )
    if not enabled:
        trace["fallback_reason"] = "intent guided chart selection disabled"
        plan["intent_guided_selection_trace"] = trace
        plan["intent_guided_summary"] = _summary(counts, plan.get("selected_charts", []))
        return plan

    if llm_client is not None and getattr(llm_client, "enabled", False):
        from app.services.llm_chart_selector import build_llm_chart_selection_plan
        from app.services.chart_selection_planner import _build_diagnostics

        llm_plan = build_llm_chart_selection_plan(
            schema_mapping=schema_mapping,
            dataset_profile={},
            analysis_focus=analysis_focus,
            section_priority=section_priority,
            evidence_pack=evidence_pack,
            deterministic_chart_selection_plan={**plan, "candidate_charts": enhanced_candidates},
            diagnostics=plan.get("diagnostics", {}),
            candidate_charts=enhanced_candidates,
            llm_client=llm_client,
            llm_profile=llm_profile,
            llm_chart_selection_enabled=True,
            rebuild_diagnostics=lambda final_selected: _build_diagnostics(
                final_selected, plan.get("rejected_charts", []), enhanced_candidates, fields, evidence_pack
            ),
            intent_context=_intent_context(chart_intent_plan, enhanced_candidates, ignored),
        )
        if llm_plan.get("selection_mode") == "llm_sanitized":
            selected = list(llm_plan.get("selected_charts", []))
            trace.update(llm_plan.get("llm_selection_trace", {}))
            trace["attempted"] = True
            trace["applied"] = True
            trace["status"] = "intent_guided_llm_sanitized"
            plan = llm_plan
            plan["selection_mode"] = "intent_guided_llm_sanitized"
        else:
            trace["attempted"] = True
            trace["applied"] = False
            trace["status"] = "intent_guided_fallback"
            trace["fallback_reason"] = (llm_plan.get("llm_selection_trace") or {}).get("fallback_reason") or "llm_selection_failed"
            selected = list(base_chart_selection_plan.get("selected_charts", []))
            plan["selection_mode"] = "intent_guided_fallback"
            trace["final_selected_chart_ids"] = [
                str(item.get("chart_id")) for item in selected if isinstance(item, dict)
            ]
            plan["selected_charts"] = selected
            plan["intent_guided_selection_trace"] = trace
            plan["intent_guided_summary"] = _summary(counts, selected)
            return plan
    else:
        from app.services.chart_selection_planner import _build_diagnostics

        selected, sanitizer_rejected, not_selected_reasons = _deterministic_intent_selection(
            enhanced_candidates=enhanced_candidates,
            base_plan=base_chart_selection_plan,
            fields=fields,
        )
        trace["attempted"] = True
        trace["applied"] = True
        trace["status"] = "intent_guided_deterministic"
        trace["sanitizer_rejected_candidate_ids"] = sorted(set(sanitizer_rejected))
        trace["not_selected_reasons"] = not_selected_reasons
        plan["selection_mode"] = "intent_guided_deterministic"
        plan["selected_charts"] = selected
        plan["diagnostics"] = _build_diagnostics(
            selected, plan.get("rejected_charts", []), enhanced_candidates, fields, evidence_pack
        )

    final_ids = [str(item.get("chart_id")) for item in selected if isinstance(item, dict)]
    intent_supported = set(trace["intent_supported_candidate_ids"])
    trace["final_selected_chart_ids"] = final_ids
    trace["selected_due_to_intent"] = sorted(chart_id for chart_id in final_ids if chart_id in intent_supported)
    trace["intent_supported_not_selected"] = sorted(intent_supported - set(final_ids))
    trace.setdefault("not_selected_reasons", {})
    for chart_id in trace["intent_supported_not_selected"]:
        trace["not_selected_reasons"].setdefault(chart_id, "not selected after budget, redundancy, or LLM judgment")
    _mark_final_candidate_status(enhanced_candidates, selected)
    plan["candidate_charts"] = enhanced_candidates
    plan["intent_guided_selection_trace"] = trace
    plan["intent_guided_summary"] = _summary(counts, selected)
    plan = apply_portfolio_sufficiency_gate(plan, mapped_fields=fields)
    plan = apply_portfolio_diversity_gate(plan, mapped_fields=fields)
    return plan
