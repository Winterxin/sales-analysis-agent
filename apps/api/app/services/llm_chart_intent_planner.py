from __future__ import annotations

from typing import Any

from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_client import complete_json_for_stage


SYSTEM_PROMPT = """You plan chart intents for a sales-analysis dataset in shadow mode.
Return JSON only with keys: overall_chart_strategy, chart_intents, generic_fallback_reason.
Rules:
1. Do not choose from candidate chart IDs. First explain what this dataset is best suited to visualize.
2. Recommend field combinations and chart kinds that fit the data structure and business questions.
3. Do not force treemap, donut, funnel, heatmap, boxplot, or other distinctive charts for novelty.
4. Chart-kind diversity is secondary. Field fit, business fit, clarity, stability, and interpretability matter most.
5. Sparse datasets may fall back to generic trend or ranking charts.
6. Do not generate Python code, chart code, notebook cells, or prose for the final notebook.
7. If the existing candidate pool seems insufficient, describe a chart intent and fallback_if_unsupported.
"""

SUPPORTED_TEMPLATE_KINDS = {
    "bar",
    "line",
    "heatmap",
    "scatter",
    "histogram",
    "boxplot",
    "pareto",
    "stacked_bar",
    "combo",
    "donut",
}

CATEGORY_PROFILE_KEYS = {
    "category": "category_count",
    "sub_category": "sub_category_count",
    "product_name": "product_count",
    "productline": "productline_count",
    "segment": "segment_count",
    "region": "region_count",
    "country": "country_count",
    "state": "state_count",
    "city": "city_count",
    "deal_size": "distinct_deal_size_count",
    "order_status": "distinct_order_status_count",
    "customer_id": "customer_count",
}

TOP_N_TERMS = (
    "top",
    "top n",
    "top 10",
    "ranking",
    "rank",
    "head",
    "head items",
    "top products",
    "头部",
    "排行",
    "排名",
)

TOP_N_CANDIDATE_IDS = {
    "top_product_profit_gap_bar",
    "product_category_bar",
    "high_discount_product_bar",
    "product_category_pareto",
    "category_concentration_bar",
}

EXACT_INTENT_CANDIDATES = {
    "productline_monthly_trend": {"productline_monthly_trend"},
    "country_market_sales_trend": {"country_monthly_trend"},
    "productline_deal_size_cross_analysis": {"productline_deal_size_stacked_bar"},
    "order_status_deal_size_structure": {"status_deal_size_stacked_bar"},
    "discount_profit_relationship": {"discount_vs_profit_scatter"},
    "top_products_profit_gap": {"top_product_profit_gap_bar"},
    "quantity_distribution_overview": {"quantity_distribution"},
    "quantity_distribution": {"quantity_distribution"},
}

STRONG_INTENT_CANDIDATES = {
    "category_profit_sales_comparison": {"category_profit_margin_bar"},
    "category_profit_margin": {"category_profit_margin_bar"},
    "region_profit_performance": {"region_sales_profit_bar"},
    "region_profit_comparison": {"region_sales_profit_bar"},
    "segment_profit_quality": {"segment_sales_profit_bar", "segment_region_margin_heatmap"},
    "profit_distribution_overview": {"profit_distribution_if_available"},
    "profit_distribution": {"profit_distribution_if_available"},
}

MEANINGFUL_MATCH_TOKENS = {
    "category",
    "country",
    "deal",
    "discount",
    "margin",
    "monthly",
    "order",
    "product",
    "productline",
    "profit",
    "region",
    "sales",
    "segment",
    "status",
    "trend",
    "volatility",
}

SAFE_DIAGNOSTIC_KEYS = {
    "dataset_signature",
    "dominant_story",
    "available_field_groups",
    "generic_chart_ratio",
    "dataset_specific_chart_ratio",
    "top_unselected_dataset_specific_charts",
}


def _empty_plan(status: str, reason: str, *, llm_client: Any = None) -> dict[str, Any]:
    return {
        "mode": "shadow",
        "attempted": status not in {"skipped"},
        "applied_to_selected_charts": False,
        "overall_chart_strategy": "",
        "chart_intents": [],
        "validated_intents": [],
        "matched_existing_candidates": [],
        "unsupported_intents": [],
        "invalid_intents": [],
        "new_template_opportunities": [],
        "generic_fallback_reason": "",
        "trace": {
            "attempted": status not in {"skipped"},
            "status": status,
            "fallback_reason": reason if status == "fallback" else None,
            "prompt_chars": None,
            "response_chars": None,
            "model": getattr(llm_client, "configured_model", None) if llm_client is not None else None,
            "source": getattr(llm_client, "source", "disabled" if llm_client is None else "unknown"),
        },
    }


def _enabled(llm_client: Any, llm_profile: str | None, enabled: bool | None) -> tuple[bool, str]:
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return False, "LLM client is disabled."
    if enabled is False:
        return False, "LLM chart intent planning is disabled by settings or profile policy."
    profile = str(llm_profile or "full").strip().lower()
    if enabled is True or profile == "full":
        return True, "LLM chart intent planning enabled."
    return False, f"LLM chart intent planning is disabled for llm_profile={profile}."


def _safe_candidate_summary(chart_selection_plan: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = chart_selection_plan.get("candidate_charts")
    if not isinstance(candidates, list):
        return []
    output: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "chart_id": item.get("chart_id"),
                "section_id": item.get("section_id"),
                "chart_kind": item.get("chart_kind"),
                "business_question": item.get("business_question"),
                "required_fields": item.get("required_fields", []),
                "availability_status": item.get("availability_status"),
                "rejection_reason": item.get("rejection_reason"),
                "dataset_specificity_score": item.get("dataset_specificity_score"),
            }
        )
    return output


def _safe_diagnostics(chart_selection_plan: dict[str, Any]) -> dict[str, Any]:
    diagnostics = chart_selection_plan.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {}
    return {key: diagnostics.get(key) for key in SAFE_DIAGNOSTIC_KEYS if key in diagnostics}


def _payload(
    *,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any],
    section_priority: dict[str, Any],
    evidence_pack: dict[str, Any] | None,
    chart_selection_plan: dict[str, Any],
) -> dict[str, Any]:
    signature = (evidence_pack or {}).get("dataset_signature") or {}
    return {
        "task": "Plan chart intents for this dataset in shadow mode.",
        "dataset_signature": signature,
        "schema_fields": sorted(set(schema_mapping.field_mapping.values())),
        "dataset_profile_summary": {
            key: value
            for key, value in (dataset_profile or {}).items()
            if key.endswith("_count")
            or key
            in {
                "row_count",
                "column_count",
                "negative_profit_rate",
                "profit_margin_spread",
                "top_product_sales_share",
                "top_category_sales_share",
                "top_country_sales_share",
                "line_per_order_avg",
            }
        },
        "selected_focuses": list(analysis_focus.get("selected_focuses", []) or []),
        "support_focuses": list(analysis_focus.get("support_focuses", []) or []),
        "section_priority": {
            "core_sections": list(section_priority.get("core_sections", []) or []),
            "support_sections": list(section_priority.get("support_sections", []) or []),
            "skipped_sections": list(section_priority.get("skipped_sections", []) or []),
        },
        "candidate_chart_summary": _safe_candidate_summary(chart_selection_plan),
        "diagnostics": _safe_diagnostics(chart_selection_plan),
        "output_contract": {
            "overall_chart_strategy": "how this dataset should be visualized",
            "chart_intents": [
                {
                    "intent_id": "stable_snake_case_id",
                    "business_question": "question answered by the chart",
                    "suggested_chart_kind": "bar|line|heatmap|boxplot|scatter|...",
                    "required_fields": ["canonical_field"],
                    "optional_fields": ["canonical_field"],
                    "recommended_dimensions": {"x": "field_or_derived_bucket", "y": "metric"},
                    "why_this_chart": "why this fits the data",
                    "why_not_generic_chart": "why a generic chart is weaker",
                    "priority": "high|medium|low",
                    "fallback_if_unsupported": "candidate_chart_id_or_short_description",
                }
            ],
            "generic_fallback_reason": "why generic charts are acceptable if fields are sparse",
        },
    }


def _call_llm(llm_client: Any, payload: dict[str, Any]) -> dict[str, Any]:
    method = getattr(llm_client, "suggest_llm_chart_intent_plan", None)
    if method is not None:
        result = method(payload)
    else:
        result = complete_json_for_stage(
            llm_client,
            system_prompt=SYSTEM_PROMPT,
            user_payload=payload,
            cache_stage="llm_chart_intent_planning",
        )
    return result if isinstance(result, dict) else {}


def _intent_dicts(result: dict[str, Any]) -> list[dict[str, Any]] | None:
    intents = result.get("chart_intents")
    if not isinstance(intents, list):
        return None
    return [dict(item) for item in intents if isinstance(item, dict)]


def _category_count(field: str, dataset_profile: dict[str, Any] | None) -> int | None:
    key = CATEGORY_PROFILE_KEYS.get(field)
    if not key:
        return None
    value = (dataset_profile or {}).get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _string_blob(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_string_blob(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_string_blob(item) for item in value)
    return str(value or "")


def _intent_text(intent: dict[str, Any]) -> str:
    return " ".join(
        [
            str(intent.get("intent_id") or ""),
            str(intent.get("business_question") or ""),
            str(intent.get("why_this_chart") or ""),
            str(intent.get("why_not_generic_chart") or ""),
            str(intent.get("fallback_if_unsupported") or ""),
            _string_blob(intent.get("recommended_dimensions")),
        ]
    ).lower()


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        [
            str(candidate.get("chart_id") or ""),
            str(candidate.get("title") or ""),
            str(candidate.get("business_question") or ""),
            str(candidate.get("insight_type") or ""),
        ]
    ).lower()


def _tokens(text: str) -> set[str]:
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {token for token in normalized.split() if token in MEANINGFUL_MATCH_TOKENS}


def _business_question_match(intent: dict[str, Any], candidate: dict[str, Any]) -> bool:
    intent_tokens = _tokens(_intent_text(intent))
    candidate_tokens = _tokens(_candidate_text(candidate))
    return len(intent_tokens & candidate_tokens) >= 2


def _has_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _quality_rank(quality: str) -> int:
    return {
        "exact_match": 0,
        "strong_match": 1,
        "weak_match": 2,
        "false_friend_match": 3,
    }.get(quality, 9)


def score_intent_candidate_match(intent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any] | None:
    chart_id = str(candidate.get("chart_id") or "")
    if not chart_id:
        return None

    intent_required = {str(field) for field in intent.get("required_fields", []) or [] if str(field).strip()}
    candidate_required = {
        str(field) for field in candidate.get("required_fields", []) or [] if str(field).strip()
    }
    fields_overlap = sorted(intent_required & candidate_required)
    if not fields_overlap:
        return None

    missing_intent_fields = sorted(intent_required - candidate_required)
    extra_candidate_fields = sorted(candidate_required - intent_required)
    intent_kind = str(intent.get("suggested_chart_kind") or "").strip().lower()
    candidate_kind = str(candidate.get("chart_kind") or "").strip().lower()
    chart_kind_match = bool(intent_kind and intent_kind == candidate_kind)
    business_question_match = _business_question_match(intent, candidate)
    intent_id = str(intent.get("intent_id") or "").strip()
    intent_text = _intent_text(intent)
    candidate_text = _candidate_text(candidate)
    has_trend_intent = _has_any(intent_text, {"trend", "monthly", "month", "月", "趋势"})
    mentions_volatility = _has_any(intent_text, {"volatility", "波动"})

    quality: str | None = None
    score = 0.0
    reason = ""

    if has_trend_intent and "rolling_volatility" in chart_id and not mentions_volatility:
        quality = "false_friend_match"
        score = 0.18
        reason = "Candidate is a volatility diagnostic, not the requested monthly trend breakdown."
    elif (
        "category" in intent_required
        and "profit" in intent_required
        and chart_id in {"segment_region_sales_bar", "segment_region_low_margin_table_or_bar"}
        and "segment" not in intent_text
        and "region" not in intent_text
    ):
        quality = "false_friend_match"
        score = 0.22
        reason = "Candidate is segment/region focused while the intent asks for category profit analysis."
    elif (
        "region" in intent_required
        and "profit" in intent_required
        and chart_id == "segment_region_low_margin_table_or_bar"
        and "segment" not in intent_text
        and "low margin" not in intent_text
        and "低毛利" not in intent_text
    ):
        quality = "false_friend_match"
        score = 0.24
        reason = "Candidate emphasizes segment low-margin diagnostics, not the requested region profit view."
    elif chart_id in EXACT_INTENT_CANDIDATES.get(intent_id, set()):
        quality = "exact_match" if chart_kind_match else "strong_match"
        score = 0.96 if chart_kind_match else 0.86
        reason = "Candidate chart id directly implements this named intent."
    elif chart_id in STRONG_INTENT_CANDIDATES.get(intent_id, set()):
        quality = "strong_match"
        score = 0.84
        reason = "Candidate uses the same core fields and business framing as the intent."
    elif has_trend_intent and chart_id == "sales_trends_monthly_line" and (
        "productline" in intent_required or "country" in intent_required
    ):
        quality = "weak_match"
        score = 0.46
        reason = "Generic monthly trend is only a fallback because it omits the requested breakdown dimension."
    elif (
        {"productline", "deal_size"} <= intent_required
        and {"productline", "deal_size"} <= candidate_required
        and candidate_kind in {"stacked_bar", "heatmap", "bar"}
    ):
        quality = "strong_match"
        score = 0.86 if chart_kind_match else 0.8
        reason = "Candidate captures the productline and deal-size cross-structure."
    elif (
        {"order_status", "deal_size"} <= intent_required
        and {"order_status", "deal_size"} <= candidate_required
        and candidate_kind in {"stacked_bar", "heatmap", "bar"}
    ):
        quality = "strong_match"
        score = 0.86 if chart_kind_match else 0.8
        reason = "Candidate captures the order-status and deal-size structure."
    elif {"discount", "profit"} <= intent_required and {"discount", "profit"} <= candidate_required:
        quality = "strong_match" if chart_kind_match or business_question_match else "weak_match"
        score = 0.82 if quality == "strong_match" else 0.54
        reason = "Candidate covers the discount/profit relationship fields."
    elif {"category", "profit", "sales_amount"} <= intent_required and {
        "category",
        "profit",
        "sales_amount",
    } <= candidate_required:
        quality = "strong_match"
        score = 0.84
        reason = "Candidate covers category sales and profit quality."
    elif {"region", "profit", "sales_amount"} <= intent_required and {
        "region",
        "profit",
        "sales_amount",
    } <= candidate_required:
        quality = "strong_match"
        score = 0.84
        reason = "Candidate covers region sales and profit quality."
    elif {"segment", "profit", "sales_amount"} <= intent_required and {
        "segment",
        "profit",
        "sales_amount",
    } <= candidate_required:
        quality = "strong_match"
        score = 0.84
        reason = "Candidate covers segment sales and profit quality."
    elif intent_required == {"quantity"} and chart_id == "quantity_distribution" and candidate_kind == "histogram":
        quality = "strong_match" if chart_kind_match else "weak_match"
        score = 0.84 if chart_kind_match else 0.52
        reason = "Candidate directly visualizes the order-line quantity distribution."
    elif intent_required <= candidate_required and chart_kind_match and business_question_match:
        quality = "strong_match"
        score = 0.78
        reason = "Candidate covers required fields with matching chart kind and business framing."
    elif len(fields_overlap) >= 2 and (chart_kind_match or business_question_match):
        quality = "weak_match"
        score = 0.5 if chart_kind_match else 0.42
        reason = "Candidate shares some fields but does not express the full intent."
    elif len(fields_overlap) >= 2 and "sales_amount" in fields_overlap:
        quality = "weak_match"
        score = 0.36
        reason = "Candidate overlaps on generic sales fields only, so it is at most a fallback."

    if quality is None:
        return None
    if quality == "strong_match" and len(missing_intent_fields) > 1:
        quality = "weak_match"
        score = min(score, 0.55)
        reason = "Candidate misses multiple intent fields, so it is only a fallback."
    if quality == "weak_match" and "rolling_volatility" in candidate_text and not mentions_volatility:
        quality = "false_friend_match"
        score = 0.18
        reason = "Candidate is volatility-oriented and should not support a non-volatility intent."

    return {
        "chart_id": chart_id,
        "match_quality": quality,
        "match_score": round(score, 3),
        "match_reason": reason,
        "fields_overlap": fields_overlap,
        "missing_intent_fields": missing_intent_fields,
        "extra_candidate_fields": extra_candidate_fields,
        "chart_kind_match": chart_kind_match,
        "business_question_match": business_question_match,
    }


def _candidate_matches(intent: dict[str, Any], chart_selection_plan: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for candidate in chart_selection_plan.get("candidate_charts", []) or []:
        if not isinstance(candidate, dict):
            continue
        match = score_intent_candidate_match(intent, candidate)
        if match is not None:
            matches.append(match)
    return sorted(matches, key=lambda item: (_quality_rank(str(item["match_quality"])), -float(item["match_score"])))


def _is_top_n_intent(intent: dict[str, Any], matched: list[str]) -> bool:
    text = _intent_text(intent)
    normalized = "".join(char.lower() if char.isalnum() else " " for char in text)
    tokens = set(normalized.split())
    if any(term in text for term in {"top n", "top 10", "top products", "head items", "头部", "排行", "排名"}):
        return True
    if tokens & {"top", "ranking", "rank", "head"}:
        return True
    fallback = str(intent.get("fallback_if_unsupported") or "")
    return fallback in TOP_N_CANDIDATE_IDS or any(chart_id in TOP_N_CANDIDATE_IDS for chart_id in matched)


def _score(status: str, matched: list[str], supported_kind: bool) -> tuple[float, float, float]:
    if status.startswith("invalid"):
        return 0.25, 0.25, 0.0
    business = 0.82 if status == "valid_supported" else 0.72
    data_fit = 0.85
    readiness = 0.9 if matched else (0.45 if supported_kind else 0.2)
    return business, data_fit, readiness


def _validate_intents(
    intents: list[dict[str, Any]],
    *,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    chart_selection_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    fields = set(schema_mapping.field_mapping.values())
    validated: list[dict[str, Any]] = []
    for index, intent in enumerate(intents, start=1):
        intent_id = str(intent.get("intent_id") or f"intent_{index}")
        required = [str(field) for field in intent.get("required_fields", []) or [] if str(field).strip()]
        kind = str(intent.get("suggested_chart_kind") or "").strip().lower()
        missing = [field for field in required if field not in fields]
        candidate_matches = _candidate_matches(intent, chart_selection_plan) if not missing else []
        primary_matches = [
            str(item["chart_id"])
            for item in candidate_matches
            if item.get("match_quality") in {"exact_match", "strong_match"}
        ]
        fallback_matches = [
            str(item["chart_id"]) for item in candidate_matches if item.get("match_quality") == "weak_match"
        ]
        false_friend_matches = [
            str(item["chart_id"])
            for item in candidate_matches
            if item.get("match_quality") == "false_friend_match"
        ]
        supported_kind = kind in SUPPORTED_TEMPLATE_KINDS
        high_category_fields = [
            field for field in required if (_category_count(field, dataset_profile) or 0) > 50
        ]
        top_n_intent = _is_top_n_intent(intent, primary_matches)
        category_handling = "top_n" if high_category_fields and top_n_intent else (
            "full_category" if high_category_fields else "standard"
        )
        category_handling_reason = (
            "High-cardinality product field is acceptable because the intent explicitly uses Top-N/ranking truncation."
            if category_handling == "top_n"
            else (
                "High-cardinality category field requires explicit Top-N/ranking truncation."
                if category_handling == "full_category"
                else "No high-cardinality category handling was needed."
            )
        )

        status = "valid_supported"
        unsupported_reason = None
        if missing:
            status = "invalid_missing_fields"
            unsupported_reason = "Required fields are not mapped in schema_mapping: " + ", ".join(missing)
        elif high_category_fields and not top_n_intent:
            status = "invalid_too_many_categories"
            unsupported_reason = "One or more required category fields have too many categories for a stable chart."
        elif int((dataset_profile or {}).get("row_count") or 1) < 5:
            status = "invalid_low_data_volume"
            unsupported_reason = "Dataset has too few rows for this chart intent."
        elif not str(intent.get("business_question") or "").strip():
            status = "invalid_unclear_business_value"
            unsupported_reason = "Missing clear business question."
        elif not supported_kind:
            status = "valid_unsupported"
            unsupported_reason = "Suggested chart kind is not in the safe template registry."
        elif primary_matches:
            status = "valid_supported"
        elif fallback_matches:
            status = "valid_unsupported"
            unsupported_reason = (
                "Only weak fallback candidates were found; no strong executable candidate matches this intent."
            )
        else:
            status = "valid_unsupported"
            unsupported_reason = "No exact or strong existing candidate chart matches this intent."

        business, data_fit, readiness = _score(status, primary_matches, supported_kind)
        validated.append(
            {
                **intent,
                "intent_id": intent_id,
                "suggested_chart_kind": kind,
                "required_fields": required,
                "validation_status": status,
                "candidate_matches": candidate_matches,
                "primary_matched_candidate_chart_ids": primary_matches,
                "fallback_matched_candidate_chart_ids": fallback_matches,
                "false_friend_candidate_chart_ids": false_friend_matches,
                "matched_candidate_chart_ids": primary_matches,
                "matched_template_type": kind if supported_kind else None,
                "unsupported_reason": unsupported_reason,
                "category_handling": category_handling,
                "category_handling_reason": category_handling_reason,
                "business_value_score": business,
                "data_fit_score": data_fit,
                "execution_readiness_score": readiness,
            }
        )
    return validated


def build_llm_chart_intent_plan(
    *,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any],
    section_priority: dict[str, Any],
    evidence_pack: dict[str, Any] | None,
    chart_selection_plan: dict[str, Any],
    llm_client: Any,
    llm_profile: str | None,
    llm_chart_intent_planning_enabled: bool | None,
) -> dict[str, Any]:
    can_attempt, reason = _enabled(llm_client, llm_profile, llm_chart_intent_planning_enabled)
    if not can_attempt:
        return _empty_plan("skipped", reason, llm_client=llm_client)

    payload = _payload(
        schema_mapping=schema_mapping,
        dataset_profile=dataset_profile,
        analysis_focus=analysis_focus,
        section_priority=section_priority,
        evidence_pack=evidence_pack,
        chart_selection_plan=chart_selection_plan,
    )
    try:
        result = _call_llm(llm_client, payload)
    except Exception as exc:
        plan = _empty_plan("fallback", f"llm_error: {exc}", llm_client=llm_client)
        plan["trace"]["attempted"] = True
        return plan

    intents = _intent_dicts(result)
    if intents is None:
        plan = _empty_plan("fallback", "invalid_payload", llm_client=llm_client)
        plan["trace"]["attempted"] = True
        return plan

    validated = _validate_intents(
        intents,
        schema_mapping=schema_mapping,
        dataset_profile=dataset_profile,
        chart_selection_plan=chart_selection_plan,
    )
    unsupported = [item for item in validated if item["validation_status"] == "valid_unsupported"]
    invalid = [item for item in validated if str(item["validation_status"]).startswith("invalid")]
    matched = [
        {"intent_id": item["intent_id"], "matched_candidate_chart_ids": item["matched_candidate_chart_ids"]}
        for item in validated
        if item.get("matched_candidate_chart_ids")
    ]
    metrics = getattr(llm_client, "last_completion_metrics", {}) or {}
    return {
        "mode": "shadow",
        "attempted": True,
        "applied_to_selected_charts": False,
        "overall_chart_strategy": str(result.get("overall_chart_strategy") or ""),
        "chart_intents": intents,
        "validated_intents": validated,
        "matched_existing_candidates": matched,
        "unsupported_intents": unsupported,
        "invalid_intents": invalid,
        "new_template_opportunities": [
            {
                "intent_id": item["intent_id"],
                "suggested_chart_kind": item["suggested_chart_kind"],
                "required_fields": item["required_fields"],
                "business_question": item.get("business_question"),
                "fallback_if_unsupported": item.get("fallback_if_unsupported"),
                "unsupported_reason": item.get("unsupported_reason"),
            }
            for item in unsupported
        ],
        "generic_fallback_reason": str(result.get("generic_fallback_reason") or ""),
        "trace": {
            "attempted": True,
            "status": "success",
            "fallback_reason": None,
            "prompt_chars": metrics.get("prompt_chars"),
            "response_chars": metrics.get("response_chars"),
            "model": getattr(llm_client, "configured_model", None),
            "source": getattr(llm_client, "source", "unknown"),
        },
    }
