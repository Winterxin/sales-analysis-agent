from __future__ import annotations

from math import ceil
from typing import Any

VALID_LLM_PROFILES = {"quick", "full"}


def normalize_llm_profile(value: str | None) -> str:
    profile = str(value or "full").strip().lower()
    return profile if profile in VALID_LLM_PROFILES else "full"


def build_llm_profile_policy(profile: str | None) -> dict[str, Any]:
    normalized = normalize_llm_profile(profile)
    if normalized == "quick":
        return {
            "profile": "quick",
            "notebook_revision_enabled": False,
            "max_revision_decisions": 0,
            "postrun_reflection_enabled": True,
            "postrun_reflection_ratio": 0.5,
            "min_llm_postrun_reflections": 3,
            "max_llm_postrun_reflections": 6,
            "allow_postrun_fallback_reflections": False,
            "notebook_content_llm_sections": "core",
            "final_synthesis_enabled": True,
            "modeling_outcome_interpretation_enabled": True,
            "client_report_enabled": True,
            "report_summary_enabled": True,
            "notebook_narrative_enabled": True,
            "modeling_interpretation_enabled": True,
            "modeling_opportunity_decision_enabled": True,
            "chart_selection_enabled": True,
            "chart_intent_planning_enabled": False,
            "intent_guided_chart_selection_enabled": False,
            "description": (
                "Quick profile keeps core LLM analysis and client/report/modeling stages, "
                "allows lightweight single-pass LLM chart selection with deterministic "
                "fallback, disables chart intent planning and notebook revision, limits "
                "chart reflections dynamically, and leaves unreflected charts blank "
                "instead of filling them with deterministic template prose."
            ),
        }
    return {
        "profile": "full",
        "notebook_revision_enabled": True,
        "max_revision_decisions": 2,
        "postrun_reflection_enabled": True,
        "postrun_reflection_ratio": 1.0,
        "min_llm_postrun_reflections": 6,
        "max_llm_postrun_reflections": 8,
        "allow_postrun_fallback_reflections": False,
        "notebook_content_llm_sections": "all",
        "final_synthesis_enabled": True,
        "modeling_outcome_interpretation_enabled": True,
        "client_report_enabled": True,
        "report_summary_enabled": True,
        "notebook_narrative_enabled": True,
        "modeling_interpretation_enabled": True,
        "modeling_opportunity_decision_enabled": True,
        "chart_selection_enabled": True,
        "chart_intent_planning_enabled": True,
        "intent_guided_chart_selection_enabled": True,
        "description": (
            "Full profile preserves the complete high-quality agent path with revision "
            "decisions, broad section enrichment, and broader LLM chart reflection "
            "coverage while leaving failed or unreflected charts blank instead of "
            "filling them with deterministic template prose."
        ),
    }


def resolve_max_postrun_reflections(profile_policy: dict[str, Any], chart_count: int) -> int:
    chart_count = max(0, int(chart_count or 0))
    if chart_count == 0:
        return 0
    ratio = float(profile_policy.get("postrun_reflection_ratio") or 1.0)
    minimum = int(profile_policy.get("min_llm_postrun_reflections") or 0)
    maximum = int(profile_policy.get("max_llm_postrun_reflections") or chart_count)
    return min(maximum, max(minimum, ceil(chart_count * ratio), 1))
