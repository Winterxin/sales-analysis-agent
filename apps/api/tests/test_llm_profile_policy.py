from __future__ import annotations

from app.services.llm_profile_policy import (
    build_llm_profile_policy,
    normalize_llm_profile,
    resolve_max_postrun_reflections,
)


def test_llm_profile_defaults_to_full_and_normalizes_known_values() -> None:
    assert normalize_llm_profile(None) == "full"
    assert normalize_llm_profile("") == "full"
    assert normalize_llm_profile("QUICK") == "quick"
    assert normalize_llm_profile("full") == "full"
    assert normalize_llm_profile("invalid") == "full"


def test_quick_and_full_policy_fields() -> None:
    quick = build_llm_profile_policy("quick")
    full = build_llm_profile_policy("full")
    core_stages = {
        "report_summary_enabled",
        "notebook_narrative_enabled",
        "final_synthesis_enabled",
        "client_report_enabled",
        "modeling_interpretation_enabled",
        "modeling_opportunity_decision_enabled",
        "modeling_outcome_interpretation_enabled",
    }
    enhanced_stages = {
        "chart_selection_enabled",
        "chart_intent_planning_enabled",
        "intent_guided_chart_selection_enabled",
        "notebook_revision_enabled",
        "postrun_reflection_enabled",
    }

    assert quick["profile"] == "quick"
    assert core_stages <= set(quick)
    assert enhanced_stages <= set(quick)
    assert all(quick[key] is True for key in core_stages)
    assert quick["max_revision_decisions"] == 0
    assert quick["notebook_revision_enabled"] is False
    assert quick["postrun_reflection_enabled"] is True
    assert quick["allow_postrun_fallback_reflections"] is False
    assert quick["postrun_reflection_ratio"] == 0.5
    assert quick["min_llm_postrun_reflections"] == 3
    assert quick["max_llm_postrun_reflections"] == 6
    assert quick["modeling_interpretation_enabled"] is True
    assert quick["chart_selection_enabled"] is True
    assert quick["chart_intent_planning_enabled"] is False
    assert quick["intent_guided_chart_selection_enabled"] is False

    assert full["profile"] == "full"
    assert core_stages <= set(full)
    assert enhanced_stages <= set(full)
    assert all(full[key] is True for key in core_stages)
    assert full["max_revision_decisions"] == 2
    assert full["notebook_revision_enabled"] is True
    assert full["postrun_reflection_enabled"] is True
    assert full["allow_postrun_fallback_reflections"] is False
    assert full["postrun_reflection_ratio"] == 1.0
    assert full["max_llm_postrun_reflections"] >= quick["max_llm_postrun_reflections"]
    assert full["modeling_interpretation_enabled"] is True
    assert full["chart_selection_enabled"] is True
    assert full["chart_intent_planning_enabled"] is True
    assert full["intent_guided_chart_selection_enabled"] is True


def test_resolve_max_postrun_reflections_for_quick_is_dynamic() -> None:
    quick = build_llm_profile_policy("quick")

    assert resolve_max_postrun_reflections(quick, 4) == 3
    assert resolve_max_postrun_reflections(quick, 6) == 3
    assert resolve_max_postrun_reflections(quick, 8) == 4
    assert resolve_max_postrun_reflections(quick, 10) == 5
    assert resolve_max_postrun_reflections(quick, 12) == 6
