from __future__ import annotations

from types import SimpleNamespace

from app.application.analysis_run_service import (
    _chart_intent_planning_enabled,
    _chart_selection_enabled,
    _intent_guided_chart_selection_enabled,
    _stage_enabled_by_policy,
)
from app.services.llm_profile_policy import build_llm_profile_policy


def _ctx(profile: str, **settings_overrides: bool) -> SimpleNamespace:
    return SimpleNamespace(
        resolved_llm_profile=profile,
        llm_profile_policy=build_llm_profile_policy(profile),
        settings=SimpleNamespace(
            llm_chart_selection_enabled=settings_overrides.get(
                "llm_chart_selection_enabled", False
            ),
            llm_chart_intent_planning_enabled=settings_overrides.get(
                "llm_chart_intent_planning_enabled", False
            ),
            llm_intent_guided_chart_selection_enabled=settings_overrides.get(
                "llm_intent_guided_chart_selection_enabled", False
            ),
        ),
    )


def test_quick_chart_policy_keeps_single_pass_selection_only() -> None:
    ctx = _ctx(
        "quick",
        llm_chart_intent_planning_enabled=True,
        llm_intent_guided_chart_selection_enabled=True,
    )

    assert _chart_selection_enabled(ctx) is True
    assert _chart_intent_planning_enabled(ctx) is False
    assert _intent_guided_chart_selection_enabled(ctx) is False


def test_full_chart_enhancement_policy_enables_deep_stages() -> None:
    ctx = _ctx("full")

    assert _chart_selection_enabled(ctx) is True
    assert _chart_intent_planning_enabled(ctx) is True
    assert _intent_guided_chart_selection_enabled(ctx) is True


def test_core_stage_policy_keeps_quick_complete() -> None:
    ctx = _ctx("quick")

    for stage_key in (
        "report_summary_enabled",
        "notebook_narrative_enabled",
        "final_synthesis_enabled",
        "client_report_enabled",
        "modeling_interpretation_enabled",
        "modeling_opportunity_decision_enabled",
        "modeling_outcome_interpretation_enabled",
    ):
        assert _stage_enabled_by_policy(ctx, stage_key) is True
