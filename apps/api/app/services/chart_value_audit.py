from __future__ import annotations

from typing import Any

from app.schemas.schema_mapping import SchemaMapping


SPECIFIC_FIELDS = {
    "profit",
    "discount",
    "deal_size",
    "order_status",
    "productline",
    "country",
    "customer_id",
    "quantity",
}
HIGH_VALUE_KINDS = {"scatter", "boxplot", "heatmap", "histogram"}


def _chart_id(chart: dict[str, Any]) -> str:
    return str(chart.get("chart_id") or "")


def _fields(chart: dict[str, Any]) -> set[str]:
    return {str(field) for field in chart.get("required_fields", []) or []}


def _text(chart: dict[str, Any]) -> str:
    parts = [
        chart.get("chart_id"),
        chart.get("title"),
        chart.get("business_question"),
        chart.get("insight_type"),
    ]
    return " ".join(str(part or "") for part in parts).lower()


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _dominant_story(evidence_pack: dict[str, Any] | None) -> str:
    if not isinstance(evidence_pack, dict):
        return ""
    signature = evidence_pack.get("dataset_signature")
    if isinstance(signature, dict):
        return str(signature.get("dominant_story") or "").lower()
    return str(evidence_pack.get("dominant_story") or "").lower()


def _main_story_fit(chart: dict[str, Any], dominant_story: str) -> float:
    score = 0.42
    text = _text(chart)
    quality = str(chart.get("best_intent_match_quality") or "")
    if quality == "exact_match":
        score += 0.28
    elif quality == "strong_match":
        score += 0.2
    if chart.get("matched_intent_ids"):
        score += 0.12
    if dominant_story and any(token in text for token in dominant_story.replace("_", " ").split() if len(token) > 3):
        score += 0.14
    if chart.get("evidence_ids"):
        score += 0.06
    return _clamp(score)


def _information_gain(chart: dict[str, Any]) -> float:
    fields = _fields(chart)
    score = 0.28
    if len(fields) >= 2:
        score += 0.12
    if len(fields) >= 3:
        score += 0.1
    special = fields & SPECIFIC_FIELDS
    score += min(0.24, 0.08 * len(special))
    if {"sales_amount", "profit"} <= fields:
        score += 0.14
    if {"discount", "profit"} <= fields:
        score += 0.16
    if {"productline", "order_datetime"} <= fields or {"country", "order_datetime"} <= fields:
        score += 0.14
    if fields == {"sales_amount"} or fields == {"product_name", "sales_amount"}:
        score -= 0.18
    return _clamp(score)


def _field_specificity(chart: dict[str, Any]) -> float:
    if chart.get("dataset_specificity_score") is not None:
        return _clamp(float(chart.get("dataset_specificity_score") or 0))
    fields = _fields(chart)
    return _clamp(0.3 + 0.12 * len(fields & SPECIFIC_FIELDS))


def _section_stats(candidate_charts: list[dict[str, Any]]) -> tuple[dict[str, int], dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    section_counts: dict[str, int] = {}
    section_kind_counts: dict[tuple[str, str], int] = {}
    section_insight_counts: dict[tuple[str, str], int] = {}
    for chart in candidate_charts:
        section = str(chart.get("section_id") or "")
        kind = str(chart.get("chart_kind") or "")
        insight = str(chart.get("insight_type") or "")
        if section:
            section_counts[section] = section_counts.get(section, 0) + 1
        if section and kind:
            key = (section, kind)
            section_kind_counts[key] = section_kind_counts.get(key, 0) + 1
        if section and insight:
            key = (section, insight)
            section_insight_counts[key] = section_insight_counts.get(key, 0) + 1
    return section_counts, section_kind_counts, section_insight_counts


def build_chart_value_audit(
    *,
    candidate_charts: list[dict[str, Any]],
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del schema_mapping, dataset_profile
    section_counts, section_kind_counts, section_insight_counts = _section_stats(candidate_charts)
    dominant_story = _dominant_story(evidence_pack)
    chart_value_by_id: dict[str, dict[str, Any]] = {}
    low_visual: list[str] = []
    low_info: list[str] = []
    high_value: list[str] = []

    for chart in candidate_charts:
        chart_id = _chart_id(chart)
        if not chart_id:
            continue
        section = str(chart.get("section_id") or "")
        kind = str(chart.get("chart_kind") or "")
        insight = str(chart.get("insight_type") or "")
        fields = _fields(chart)
        info = _information_gain(chart)
        field_specificity = _field_specificity(chart)
        story = _main_story_fit(chart, dominant_story)
        same_kind_count = section_kind_counts.get((section, kind), 0)
        same_insight_count = section_insight_counts.get((section, insight), 0)
        redundancy = 0.12
        if same_kind_count >= 2:
            redundancy += 0.28
        if same_kind_count >= 3:
            redundancy += 0.18
        if same_insight_count >= 2:
            redundancy += 0.18
        if section_counts.get(section, 0) >= 3:
            redundancy += 0.12
        if fields == {"sales_amount"} or fields == {"product_name", "sales_amount"}:
            redundancy += 0.12
        redundancy = _clamp(redundancy)

        visual = 0.48
        if kind == "bar":
            visual -= 0.1
            if same_kind_count >= 2:
                visual -= 0.14
            if info < 0.45:
                visual -= 0.12
        elif kind in HIGH_VALUE_KINDS:
            visual += 0.12 if info >= 0.45 else 0.02
        if "margin" in _text(chart) and kind == "bar" and same_kind_count >= 2:
            visual -= 0.08
        visual = _clamp(visual)

        flags: list[str] = []
        if same_kind_count >= 2 and kind == "bar":
            flags.append("same_section_bar_overlap")
        if info < 0.38:
            flags.append("low_information_gain")
            low_info.append(chart_id)
        if kind == "bar" and (info < 0.45 or (same_kind_count >= 2 and visual <= 0.36)):
            flags.append("low_visual_gain_risk")
            low_visual.append(chart_id)
        if redundancy >= 0.65:
            flags.append("high_redundancy_risk")

        overall = _clamp(
            0.35 * story
            + 0.30 * info
            + 0.20 * visual
            + 0.15 * field_specificity
            - 0.25 * redundancy
        )
        if overall >= 0.58 and str(chart.get("availability_status") or "") != "selected":
            high_value.append(chart_id)
        chart_value_by_id[chart_id] = {
            "main_story_fit_score": story,
            "information_gain_score": info,
            "visual_gain_score": visual,
            "redundancy_risk_score": redundancy,
            "field_specificity_score": field_specificity,
            "overall_chart_value_score": overall,
            "value_flags": flags,
            "value_reason": _value_reason(chart, flags, info, visual, redundancy),
        }

    return {
        "chart_value_by_id": chart_value_by_id,
        "low_visual_gain_risk_chart_ids": sorted(set(low_visual)),
        "low_information_gain_chart_ids": sorted(set(low_info)),
        "high_value_challenger_ids": sorted(set(high_value)),
        "summary": {
            "candidate_count": len(chart_value_by_id),
            "low_visual_gain_risk_count": len(set(low_visual)),
            "low_information_gain_count": len(set(low_info)),
            "high_value_challenger_count": len(set(high_value)),
        },
    }


def _value_reason(chart: dict[str, Any], flags: list[str], info: float, visual: float, redundancy: float) -> str:
    if flags:
        return (
            f"{chart.get('chart_id')} has information_gain={info}, visual_gain={visual}, "
            f"redundancy_risk={redundancy}; flags: {', '.join(flags)}."
        )
    return (
        f"{chart.get('chart_id')} has usable incremental value with information_gain={info}, "
        f"visual_gain={visual}, and redundancy_risk={redundancy}."
    )
