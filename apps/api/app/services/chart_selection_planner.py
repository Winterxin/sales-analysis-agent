from __future__ import annotations

from typing import Any

from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_candidate_registry import iter_chart_candidates


CHART_BUDGET = {
    "core_max_per_section": 3,
    "support_max_per_section": 1,
    "skipped_max_per_section": 0,
}

DATASET_SPECIFIC_THRESHOLD = 0.55

HARD_REJECTION_PREFIXES = (
    "missing required fields",
    "skipped section",
    "country field has no useful market slice",
    "too many categories",
    "too many category combinations",
    "limited by data availability",
)

SOFT_REJECTION_REASONS = {
    "lower evidence score",
    "redundant insight type",
    "budget exceeded",
    "support budget exceeded",
    "core budget exceeded",
}


def _mapped_fields(schema_mapping: SchemaMapping) -> set[str]:
    return set(schema_mapping.field_mapping.values())


def _section_role(section_id: str, section_priority: dict[str, Any]) -> str:
    if section_id in set(section_priority.get("core_sections", [])):
        return "core"
    if section_id in set(section_priority.get("support_sections", [])):
        return "support"
    return "skipped"


def _budget_for_role(role: str) -> int:
    if role == "core":
        return CHART_BUDGET["core_max_per_section"]
    if role == "support":
        return CHART_BUDGET["support_max_per_section"]
    return CHART_BUDGET["skipped_max_per_section"]


def _focuses(analysis_focus: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("selected_focuses", "support_focuses"):
        values.update(str(item) for item in analysis_focus.get(key, []) if str(item).strip())
    return values


def _selected_focuses(analysis_focus: dict[str, Any]) -> set[str]:
    return {
        str(item)
        for item in analysis_focus.get("selected_focuses", [])
        if str(item).strip()
    }


def _is_country_slice_useful(dataset_profile: dict[str, Any]) -> bool:
    country_count = dataset_profile.get("country_count")
    top_share = dataset_profile.get("top_country_sales_share")
    return (
        isinstance(country_count, int)
        and country_count >= 2
        and (top_share is None or float(top_share) < 0.95)
    )


_CATEGORY_COUNT_METRICS = {
    "country": "country_count",
    "productline": "productline_count",
    "deal_size": "distinct_deal_size_count",
    "order_status": "distinct_order_status_count",
    "category": "category_count",
    "sub_category": "sub_category_count",
    "segment": "segment_count",
    "region": "region_count",
}

_MEASURE_FIELDS = {
    "sales_amount",
    "profit",
    "discount",
    "quantity",
    "order_id",
    "customer_id",
    "order_datetime",
    "unit_price",
}


def _category_count(profile: dict[str, Any], field: str) -> int | None:
    value = profile.get(_CATEGORY_COUNT_METRICS.get(field, ""))
    if isinstance(value, int):
        return value
    return None


def _categorical_required_fields(candidate: dict[str, Any]) -> list[str]:
    return [
        str(field)
        for field in candidate.get("required_fields", []) or []
        if str(field) not in _MEASURE_FIELDS
    ]


def _reject_for_chart_shape(candidate: dict[str, Any], profile: dict[str, Any]) -> str | None:
    chart_kind = str(candidate.get("chart_kind") or "")
    dimensions = _categorical_required_fields(candidate)
    counts = [count for field in dimensions if (count := _category_count(profile, field)) is not None]
    if chart_kind in {"pie", "donut"} and counts and max(counts) > 8:
        return "too many categories for pie/donut"
    if chart_kind in {"heatmap", "stacked_bar"} and counts:
        if any(count > 12 for count in counts):
            return "too many categories for heatmap/stacked"
        if len(counts) >= 2 and counts[0] * counts[1] > 80:
            return "too many category combinations for heatmap/stacked"
    return None


def _evidence_by_id(evidence_pack: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(evidence_pack, dict):
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for items in (evidence_pack.get("focus_evidence") or {}).values():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict) and item.get("evidence_id"):
                by_id[str(item["evidence_id"])] = item
    for item in evidence_pack.get("distinctive_facts", []) or []:
        if isinstance(item, dict) and item.get("fact_id"):
            by_id.setdefault(str(item["fact_id"]), item)
    return by_id


def _limitation_fields(evidence_pack: dict[str, Any] | None) -> set[str]:
    if not isinstance(evidence_pack, dict):
        return set()
    return {
        str(item.get("field"))
        for item in evidence_pack.get("limitations", []) or []
        if isinstance(item, dict) and item.get("field")
    }


def _signature_tokens(evidence_pack: dict[str, Any] | None) -> set[str]:
    if not isinstance(evidence_pack, dict):
        return set()
    signature = evidence_pack.get("dataset_signature") or {}
    tokens = {
        str(signature.get("shape_type", "")),
        str(signature.get("dominant_story", "")),
    }
    for item in signature.get("available_fields", []) or []:
        tokens.add(str(item))
    return {token for token in tokens if token}


def _dataset_signature(evidence_pack: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(evidence_pack, dict):
        return {}
    signature = evidence_pack.get("dataset_signature")
    return dict(signature) if isinstance(signature, dict) else {}


def _available_field_groups(fields: set[str]) -> list[str]:
    groups: list[str] = []
    group_rules = [
        ("generic_sales_time", {"order_datetime", "sales_amount"}),
        ("generic_product_sales", {"product_name", "sales_amount"}),
        ("productline_mix", {"productline"}),
        ("deal_size_structure", {"deal_size"}),
        ("order_status_structure", {"order_status"}),
        ("country_market", {"country"}),
        ("customer_identity", {"customer_id"}),
        ("discount_profit", {"discount", "profit"}),
        ("basket_structure", {"order_id"}),
        ("price_quantity", {"unit_price", "quantity"}),
    ]
    for name, required in group_rules:
        if required <= fields:
            groups.append(name)
    return groups


def _dataset_specificity_score(
    candidate: dict[str, Any],
    fields: set[str],
    evidence_pack: dict[str, Any] | None,
) -> tuple[float, list[str]]:
    required = set(candidate.get("required_fields") or [])
    optional = set(candidate.get("optional_fields") or []) & fields
    all_candidate_fields = required | optional
    signature = _dataset_signature(evidence_pack)
    dominant_story = str(signature.get("dominant_story") or "")
    shape_type = str(signature.get("shape_type") or "")
    best_when = {str(item) for item in candidate.get("best_when", []) or []}

    if required <= {"order_datetime", "sales_amount"}:
        score = 0.18
        factors = ["generic_sales_time"]
    elif required <= {"product_name", "sales_amount"}:
        score = 0.24
        factors = ["generic_product_sales"]
    else:
        score = 0.32
        factors = ["base_candidate_structure"]

    distinctive_weights = {
        "productline": 0.3,
        "deal_size": 0.28,
        "order_status": 0.28,
        "country": 0.22,
        "customer_id": 0.18,
    }
    for field, weight in distinctive_weights.items():
        if field in required:
            score += weight
            factors.append(f"distinctive_field:{field}")
        elif field in optional:
            score += weight * 0.35
            factors.append(f"optional_distinctive_field:{field}")

    if {"discount", "profit"} <= required:
        score += 0.24
        factors.append("discount_profit_combination")
    elif {"discount", "profit"} <= all_candidate_fields:
        score += 0.12
        factors.append("optional_discount_profit_combination")

    if "profit" in required and {"category", "sub_category", "segment", "region", "product_name"} & all_candidate_fields:
        score += 0.12
        factors.append("profit_dimension_combination")

    if "order_id" in required and ({"quantity", "customer_id"} & all_candidate_fields or candidate.get("chart_id") == "basket_size_distribution"):
        score += 0.2
        factors.append("basket_structure")
    elif "order_id" in required:
        score += 0.12
        factors.append("order_structure")

    if dominant_story and dominant_story in best_when:
        score += 0.16
        factors.append(f"dominant_story:{dominant_story}")
    if shape_type and shape_type in best_when:
        score += 0.1
        factors.append(f"shape_type:{shape_type}")

    if best_when & {"productline_mix", "country_market", "discount_loss", "profit_quality", "order_basket"}:
        score += 0.06
        factors.append("business_story_candidate")

    return round(max(0.0, min(1.0, score)), 4), factors


def _distribution(values: list[str]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for value in values:
        distribution[value] = distribution.get(value, 0) + 1
    return dict(sorted(distribution.items()))


def _score_bucket(value: float | int | None, ranges: tuple[float, float]) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    number = float(value)
    low, high = ranges
    if number < low:
        return "low"
    if number < high:
        return "medium"
    return "high"


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _normalized_rejection_reason(reason: str) -> str:
    reason = str(reason or "").strip()
    if reason.startswith("missing required fields"):
        return "missing required fields"
    if reason.startswith("limited by data availability"):
        return "limited by data availability"
    if reason.startswith("too many categories") or reason.startswith("too many category combinations"):
        return "too many categories"
    return reason or "unknown"


def _is_hard_rejection(reason: str) -> bool:
    return any(str(reason or "").startswith(prefix) for prefix in HARD_REJECTION_PREFIXES)


def _is_soft_rejection(reason: str) -> bool:
    return _normalized_rejection_reason(reason) in SOFT_REJECTION_REASONS


def _is_available_for_strategy(candidate: dict[str, Any]) -> bool:
    reason = str(candidate.get("rejection_reason") or "")
    return not _is_hard_rejection(reason)


def _why_dataset_specific_chart_matters(candidate: dict[str, Any]) -> str:
    fields = set(candidate.get("required_fields") or [])
    factors = set(candidate.get("dataset_specificity_factors") or [])
    if "productline" in fields:
        return "Uses PRODUCTLINE structure, which can distinguish this dataset from generic product rankings."
    if "deal_size" in fields:
        return "Uses DEALSIZE structure, exposing order-size mix rather than only sales totals."
    if "order_status" in fields:
        return "Uses STATUS structure, exposing fulfillment or confirmation mix."
    if "country" in fields:
        return "Uses country market structure, making geographic differences visible."
    if "customer_id" in fields:
        return "Uses customer identity, exposing customer-base structure."
    if "discount_profit_combination" in factors:
        return "Combines discount and profit, which can reveal margin erosion rather than generic sales volume."
    if "basket_structure" in factors:
        return "Uses basket/order structure, which can reveal purchase-depth behavior."
    return "Has high dataset specificity and may make this dataset's chart strategy more distinctive."


def _selected_reason_breakdown(selected: list[dict[str, Any]]) -> dict[str, Any]:
    ranking_factors: list[str] = []
    selection_sources: list[str] = []
    score_buckets: list[str] = []
    specificity_buckets: list[str] = []
    for chart in selected:
        ranking_factors.extend(str(item) for item in chart.get("ranking_factors", []) or [])
        selection_sources.append(str(chart.get("selection_source") or "unknown"))
        score_buckets.append(_score_bucket(chart.get("score"), (1.0, 1.25)))
        specificity_buckets.append(
            _score_bucket(chart.get("dataset_specificity_score"), (0.35, DATASET_SPECIFIC_THRESHOLD))
        )
    return {
        "ranking_factor_distribution": _distribution(ranking_factors),
        "selection_source_distribution": _distribution(selection_sources),
        "score_bucket_distribution": _distribution(score_buckets),
        "dataset_specificity_bucket_distribution": _distribution(specificity_buckets),
    }


def _section_budget_pressure(candidate_charts: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return _section_budget_pressure_for_selection(candidate_charts, None)


def _section_budget_pressure_for_selection(
    candidate_charts: list[dict[str, Any]],
    selected_chart_ids: set[str] | None,
) -> dict[str, dict[str, int]]:
    sections = sorted({str(candidate.get("section_id")) for candidate in candidate_charts})
    pressure: dict[str, dict[str, int]] = {}
    for section_id in sections:
        section_candidates = [
            candidate for candidate in candidate_charts if str(candidate.get("section_id")) == section_id
        ]
        role = str(section_candidates[0].get("section_role") or "skipped") if section_candidates else "skipped"
        available = [candidate for candidate in section_candidates if _is_available_for_strategy(candidate)]
        pressure[section_id] = {
            "available_candidate_count": len(available),
            "selected_count": sum(
                1
                for candidate in section_candidates
                if (
                    str(candidate.get("chart_id")) in selected_chart_ids
                    if selected_chart_ids is not None
                    else candidate.get("availability_status") == "selected"
                )
            ),
            "rejected_by_budget_count": sum(
                1
                for candidate in section_candidates
                if _normalized_rejection_reason(str(candidate.get("rejection_reason") or ""))
                in {"budget exceeded", "support budget exceeded", "core budget exceeded"}
            ),
            "rejected_by_redundant_insight_count": sum(
                1 for candidate in section_candidates if candidate.get("rejection_reason") == "redundant insight type"
            ),
            "budget_limit": _budget_for_role(role),
        }
    return pressure


def _specificity_distribution(candidates: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"low": 0, "medium": 0, "high": 0}
    for candidate in candidates:
        if not _is_available_for_strategy(candidate):
            continue
        bucket = _score_bucket(candidate.get("dataset_specificity_score"), (0.35, DATASET_SPECIFIC_THRESHOLD))
        if bucket in buckets:
            buckets[bucket] += 1
    return buckets


def _specificity_gap(candidate_charts: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, float]:
    available_scores = [
        float(candidate.get("dataset_specificity_score") or 0)
        for candidate in candidate_charts
        if _is_available_for_strategy(candidate)
    ]
    selected_scores = [float(chart.get("dataset_specificity_score") or 0) for chart in selected]
    avg_available = _average(available_scores)
    avg_selected = _average(selected_scores)
    return {
        "max_available_specificity": round(max(available_scores), 4) if available_scores else 0.0,
        "avg_available_specificity": avg_available,
        "max_selected_specificity": round(max(selected_scores), 4) if selected_scores else 0.0,
        "avg_selected_specificity": avg_selected,
        "gap": round(avg_available - avg_selected, 4),
    }


def _top_unselected_dataset_specific_charts(
    candidate_charts: list[dict[str, Any]],
    selected_chart_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected_chart_ids = selected_chart_ids or set()
    unselected = [
        candidate
        for candidate in candidate_charts
        if str(candidate.get("chart_id")) not in selected_chart_ids
        and candidate.get("availability_status") != "selected"
        and float(candidate.get("dataset_specificity_score") or 0) >= DATASET_SPECIFIC_THRESHOLD
        and _is_soft_rejection(str(candidate.get("rejection_reason") or ""))
    ]
    unselected.sort(
        key=lambda candidate: (
            -float(candidate.get("dataset_specificity_score") or 0),
            -float(candidate.get("score") or 0),
            str(candidate.get("chart_id") or ""),
        )
    )
    output: list[dict[str, Any]] = []
    for candidate in unselected[:10]:
        item = {
            "chart_id": candidate.get("chart_id"),
            "section_id": candidate.get("section_id"),
            "chart_kind": candidate.get("chart_kind"),
            "title": candidate.get("title"),
            "dataset_specificity_score": candidate.get("dataset_specificity_score"),
            "availability_status": candidate.get("availability_status"),
            "rejection_reason": candidate.get("rejection_reason"),
            "score": candidate.get("score"),
            "ranking_factors": candidate.get("ranking_factors", []),
            "why_it_matters": _why_dataset_specific_chart_matters(candidate),
        }
        output.append(item)
    return output


def _build_diagnostics(
    selected: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    candidate_charts: list[dict[str, Any]],
    fields: set[str],
    evidence_pack: dict[str, Any] | None,
) -> dict[str, Any]:
    selected_chart_ids = [str(chart.get("chart_id")) for chart in selected]
    selected_chart_kinds = [str(chart.get("chart_kind")) for chart in selected]
    selected_sections = [str(chart.get("section_id")) for chart in selected]
    generic_count = sum(
        1
        for chart in selected
        if float(chart.get("dataset_specificity_score") or 0) < DATASET_SPECIFIC_THRESHOLD
    )
    selected_count = len(selected)
    dataset_specific_count = selected_count - generic_count
    signature = _dataset_signature(evidence_pack)
    selected_chart_id_set = set(selected_chart_ids)

    return {
        "selected_chart_ids": selected_chart_ids,
        "selected_chart_kinds": selected_chart_kinds,
        "selected_sections": selected_sections,
        "chart_kind_distribution": _distribution(selected_chart_kinds),
        "section_distribution": _distribution(selected_sections),
        "generic_chart_count": generic_count,
        "dataset_specific_chart_count": dataset_specific_count,
        "generic_chart_ratio": round(generic_count / selected_count, 4) if selected_count else 0.0,
        "dataset_specific_chart_ratio": round(dataset_specific_count / selected_count, 4) if selected_count else 0.0,
        "dataset_signature": signature,
        "dominant_story": str(signature.get("dominant_story") or ""),
        "available_field_groups": _available_field_groups(fields),
        "selected_reason_breakdown": _selected_reason_breakdown(selected),
        "rejected_reason_distribution": _distribution(
            [_normalized_rejection_reason(str(item.get("reason") or "")) for item in rejected]
        ),
        "hard_rejection_distribution": _distribution(
            [
                _normalized_rejection_reason(str(item.get("reason") or ""))
                for item in rejected
                if _is_hard_rejection(str(item.get("reason") or ""))
            ]
        ),
        "soft_rejection_distribution": _distribution(
            [
                _normalized_rejection_reason(str(item.get("reason") or ""))
                for item in rejected
                if _is_soft_rejection(str(item.get("reason") or ""))
            ]
        ),
        "section_budget_pressure": _section_budget_pressure_for_selection(candidate_charts, selected_chart_id_set),
        "candidate_specificity_distribution": _specificity_distribution(candidate_charts),
        "selected_vs_available_specificity_gap": _specificity_gap(candidate_charts, selected),
        "top_unselected_dataset_specific_charts": _top_unselected_dataset_specific_charts(candidate_charts, selected_chart_id_set),
    }


def _candidate_score(
    candidate: dict[str, Any],
    all_focuses: set[str],
    primary_focuses: set[str],
    evidence_pack: dict[str, Any] | None,
) -> tuple[float, list[str], list[str], str]:
    score = float(candidate.get("priority") or 0)
    factors: list[str] = [f"base_priority={score:.2f}"]
    supported = set(candidate.get("supported_focuses") or [])
    focus_hits = supported & all_focuses
    if focus_hits:
        score += 0.25
        factors.append("focus_match:" + ",".join(sorted(focus_hits)))
    if supported & primary_focuses:
        score += 0.15
        factors.append("selected_focus_match")

    evidence = _evidence_by_id(evidence_pack)
    evidence_ids = [
        evidence_id
        for evidence_id in candidate.get("answers_evidence_ids", []) or []
        if evidence_id in evidence
    ]
    if evidence_ids:
        strength = sum(float(evidence[item].get("strength") or 0) for item in evidence_ids)
        score += min(0.45, strength * 0.2)
        factors.append("evidence_match:" + ",".join(evidence_ids))

    tokens = _signature_tokens(evidence_pack)
    best_hits = [token for token in candidate.get("best_when", []) or [] if token in tokens or token in evidence]
    if best_hits:
        score += 0.2
        factors.append("best_when:" + ",".join(best_hits))
    avoid_hits = [token for token in candidate.get("avoid_when", []) or [] if token in tokens or token in _limitation_fields(evidence_pack)]
    if avoid_hits:
        score -= 0.4
        factors.append("avoid_when:" + ",".join(avoid_hits))

    summary_parts: list[str] = []
    for evidence_id in evidence_ids[:2]:
        item = evidence[evidence_id]
        meaning = str(item.get("business_meaning") or item.get("text") or evidence_id)
        formatted = str(item.get("formatted") or "")
        if formatted and formatted not in meaning:
            meaning = f"{meaning}（{formatted}）"
        summary_parts.append(meaning)
    if not summary_parts:
        summary_parts.append(str(candidate.get("business_question", "当前图表用于解释核心业务问题。")))

    return round(score, 4), factors, evidence_ids, "；".join(summary_parts)


def build_chart_selection_plan(
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    section_priority: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None = None,
    llm_client=None,
    llm_profile: str | None = None,
    llm_chart_selection_enabled: bool | None = None,
) -> dict[str, Any]:
    fields = _mapped_fields(schema_mapping)
    profile = dataset_profile or {}
    focus = analysis_focus or {}
    priority = section_priority or {}
    all_focuses = _focuses(focus)
    primary_focuses = _selected_focuses(focus)
    candidates = iter_chart_candidates()

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    selected_count_by_section: dict[str, int] = {}
    selected_insight_types_by_section: dict[str, set[str]] = {}
    scored_candidates: list[tuple[float, dict[str, Any], list[str], list[str], str, str]] = []
    candidate_charts: list[dict[str, Any]] = []
    candidate_charts_by_id: dict[str, dict[str, Any]] = {}

    for candidate in candidates:
        section_id = str(candidate["section_id"])
        chart_id = str(candidate["chart_id"])
        role = _section_role(section_id, priority)
        required = set(candidate.get("required_fields") or [])
        supported = set(candidate.get("supported_focuses") or [])
        specificity_score, specificity_factors = _dataset_specificity_score(candidate, fields, evidence_pack)
        candidate_entry = {
            "chart_id": chart_id,
            "section_id": section_id,
            "chart_kind": candidate["chart_kind"],
            "title": candidate["title"],
            "business_question": candidate["business_question"],
            "required_fields": sorted(required),
            "optional_fields": list(candidate.get("optional_fields") or []),
            "supported_focuses": list(candidate.get("supported_focuses") or []),
            "insight_type": candidate.get("insight_type") or "structure",
            "section_role": role,
            "dataset_specificity_score": specificity_score,
            "dataset_specificity_factors": specificity_factors,
            "availability_status": "available",
        }
        candidate_charts.append(candidate_entry)
        candidate_charts_by_id[chart_id] = candidate_entry

        if role == "skipped":
            candidate_entry["availability_status"] = "rejected"
            candidate_entry["rejection_reason"] = "skipped section"
            rejected.append(
                {"chart_id": chart_id, "section_id": section_id, "reason": "skipped section"}
            )
            continue
        if not required <= fields:
            reason = "missing required fields: " + ", ".join(sorted(required - fields))
            candidate_entry["availability_status"] = "rejected"
            candidate_entry["rejection_reason"] = reason
            rejected.append(
                {
                    "chart_id": chart_id,
                    "section_id": section_id,
                    "reason": reason,
                }
            )
            continue
        if section_id == "country_market" and not _is_country_slice_useful(profile):
            candidate_entry["availability_status"] = "rejected"
            candidate_entry["rejection_reason"] = "country field has no useful market slice"
            rejected.append(
                {
                    "chart_id": chart_id,
                    "section_id": section_id,
                    "reason": "country field has no useful market slice",
                }
            )
            continue
        shape_rejection = _reject_for_chart_shape(candidate, profile)
        if shape_rejection:
            candidate_entry["availability_status"] = "rejected"
            candidate_entry["rejection_reason"] = shape_rejection
            rejected.append(
                {
                    "chart_id": chart_id,
                    "section_id": section_id,
                    "reason": shape_rejection,
                }
            )
            continue
        if supported and not supported.intersection(all_focuses):
            candidate_entry["availability_status"] = "rejected"
            candidate_entry["rejection_reason"] = "lower evidence score"
            rejected.append(
                {"chart_id": chart_id, "section_id": section_id, "reason": "lower evidence score"}
            )
            continue
        limited_fields = required & _limitation_fields(evidence_pack)
        if limited_fields:
            reason = "limited by data availability: " + ", ".join(sorted(limited_fields))
            candidate_entry["availability_status"] = "rejected"
            candidate_entry["rejection_reason"] = reason
            rejected.append(
                {
                    "chart_id": chart_id,
                    "section_id": section_id,
                    "reason": reason,
                }
            )
            continue

        score, factors, evidence_ids, evidence_summary = _candidate_score(
            candidate,
            all_focuses,
            primary_focuses,
            evidence_pack,
        )
        candidate_entry["score"] = score
        candidate_entry["ranking_factors"] = factors
        scored_candidates.append((score, candidate, factors, evidence_ids, evidence_summary, role))

    for score, candidate, factors, evidence_ids, evidence_summary, role in sorted(
        scored_candidates,
        key=lambda item: (
            item[1].get("section_id"),
            -item[0],
            item[1].get("chart_id"),
        ),
    ):
        section_id = str(candidate["section_id"])
        chart_id = str(candidate["chart_id"])
        insight_type = str(candidate.get("insight_type") or "structure")
        budget = _budget_for_role(role)
        current_count = selected_count_by_section.get(section_id, 0)
        if current_count >= budget:
            reason = "support budget exceeded" if role == "support" else "core budget exceeded"
            candidate_charts_by_id[chart_id]["availability_status"] = "rejected"
            candidate_charts_by_id[chart_id]["rejection_reason"] = reason
            rejected.append(
                {
                    "chart_id": chart_id,
                    "section_id": section_id,
                    "reason": reason,
                }
            )
            continue
        section_insights = selected_insight_types_by_section.setdefault(section_id, set())
        if insight_type in section_insights:
            candidate_charts_by_id[chart_id]["availability_status"] = "rejected"
            candidate_charts_by_id[chart_id]["rejection_reason"] = "redundant insight type"
            rejected.append(
                {
                    "chart_id": chart_id,
                    "section_id": section_id,
                    "reason": "redundant insight type",
                }
            )
            continue

        supported = set(candidate.get("supported_focuses") or [])
        selected_count_by_section[section_id] = current_count + 1
        section_insights.add(insight_type)
        candidate_charts_by_id[chart_id]["availability_status"] = "selected"
        selected.append(
            {
                "chart_id": chart_id,
                "section_id": section_id,
                "chart_kind": candidate["chart_kind"],
                "title": candidate["title"],
                "reason": (
                    f"{role} section selected by focus "
                    f"{', '.join(sorted(supported.intersection(all_focuses)) or sorted(supported))}"
                ),
                "business_question": candidate["business_question"],
                "selection_source": "evidence_ranker",
                "evidence_ids": evidence_ids,
                "evidence_summary": evidence_summary,
                "score": score,
                "ranking_factors": factors,
                "insight_type": insight_type,
                "dataset_specificity_score": candidate_charts_by_id[chart_id]["dataset_specificity_score"],
            }
        )

    deterministic_selected = [dict(chart) for chart in selected]
    plan = {
        "candidate_charts": candidate_charts,
        "selected_charts": selected,
        "rejected_charts": rejected,
        "chart_budget": dict(CHART_BUDGET),
        "diagnostics": _build_diagnostics(selected, rejected, candidate_charts, fields, evidence_pack),
        "selection_mode": "deterministic",
        "deterministic_selected_charts": deterministic_selected,
        "llm_selected_chart_ids": [],
        "llm_sanitized_selected_chart_ids": [],
        "llm_selection_rationale": {},
        "llm_selection_trace": {
            "attempted": False,
            "applied": False,
            "status": "deterministic",
            "reason": "LLM chart selection was not attempted.",
            "invalid_chart_ids": [],
            "hard_rejected_chart_ids": [],
            "budget_rejected_chart_ids": [],
            "chart_kind_rejected_chart_ids": [],
            "fallback_reason": None,
        },
    }
    if llm_client is not None:
        from app.services.llm_chart_selector import build_llm_chart_selection_plan

        plan = build_llm_chart_selection_plan(
            schema_mapping=schema_mapping,
            dataset_profile=profile,
            analysis_focus=focus,
            section_priority=priority,
            evidence_pack=evidence_pack,
            deterministic_chart_selection_plan=plan,
            diagnostics=plan["diagnostics"],
            candidate_charts=candidate_charts,
            llm_client=llm_client,
            llm_profile=llm_profile,
            llm_chart_selection_enabled=llm_chart_selection_enabled,
            rebuild_diagnostics=lambda final_selected: _build_diagnostics(
                final_selected, rejected, candidate_charts, fields, evidence_pack
            ),
        )
    return plan


def selected_chart_ids(chart_selection_plan: dict[str, Any] | None) -> set[str]:
    if not isinstance(chart_selection_plan, dict):
        return set()
    return {
        str(item.get("chart_id"))
        for item in chart_selection_plan.get("selected_charts", [])
        if isinstance(item, dict) and item.get("chart_id")
    }


def selected_chart_metadata_by_title(chart_selection_plan: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(chart_selection_plan, dict):
        return {}
    return {
        str(item.get("title")): dict(item)
        for item in chart_selection_plan.get("selected_charts", [])
        if isinstance(item, dict) and item.get("title")
    }
