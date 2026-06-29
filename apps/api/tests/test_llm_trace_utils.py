from __future__ import annotations

from app.services.llm_trace_utils import (
    build_aggregated_llm_stage_trace,
    build_llm_stage_trace,
    build_llm_trace_summary,
)


class TraceLLM:
    enabled = True
    source = "test"
    configured_model = "gpt-5.4-mini"
    last_completion_metrics = {}


class TraceLLMWithLastMetrics:
    enabled = True
    source = "test"
    configured_model = "gpt-5.4-mini"
    last_completion_metrics = {
        "elapsed_ms": 123,
        "prompt_chars": 456,
        "response_chars": 78,
        "attempt_count": 1,
        "cache_enabled": True,
        "cache_status": "hit",
        "cache_key": "previous-cache-key",
        "remote_elapsed_ms": 111,
        "saved_ms": 222,
        "error_type": "PreviousError",
    }


def test_unattempted_skipped_stage_does_not_inherit_previous_completion_metrics() -> None:
    expectations = [
        ("skipped_by_profile", "profile"),
        ("skipped_by_budget", "budget"),
        ("skipped_by_demo_safe", "demo_safe"),
        ("skipped", "skipped"),
        ("disabled", "disabled"),
    ]

    for status, fallback_type in expectations:
        trace = build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=TraceLLMWithLastMetrics(),
            status=status,
            reason="stage did not call the LLM",
            attempted=False,
            applied=False,
        )

        assert trace.elapsed_ms is None
        assert trace.prompt_chars is None
        assert trace.response_chars is None
        assert trace.attempt_count is None
        assert trace.cache_key is None
        assert trace.remote_elapsed_ms is None
        assert trace.saved_ms == 0.0
        assert trace.error_type is None
        assert trace.cache_status != "hit"
        assert trace.fallback_type == fallback_type
        assert trace.source == "test"
        assert trace.model == "gpt-5.4-mini"


def test_attempted_fallback_stage_keeps_completion_metrics() -> None:
    for status, fallback_type in [
        ("fallback_on_error", "error"),
        ("fallback_invalid_payload", "invalid_payload"),
    ]:
        trace = build_llm_stage_trace(
            stage="notebook_content",
            llm_client=TraceLLMWithLastMetrics(),
            status=status,
            reason="stage attempted the LLM and fell back",
            attempted=True,
            applied=False,
        )

        assert trace.prompt_chars == 456
        assert trace.response_chars == 78
        assert trace.cache_key == "previous-cache-key"
        assert trace.remote_elapsed_ms == 111
        assert trace.cache_status == "hit"
        assert trace.fallback_type == fallback_type


def test_build_aggregated_llm_stage_trace_sums_multiple_subcalls() -> None:
    trace = build_aggregated_llm_stage_trace(
        stage="notebook_content",
        llm_client=TraceLLM(),
        status="llm_applied",
        reason="sections generated",
        attempted=True,
        applied=True,
        subcalls=[
            {
                "stage": "notebook_content",
                "cache_status": "miss",
                "prompt_chars": 100,
                "response_chars": 20,
                "remote_elapsed_ms": 11.5,
                "saved_ms": 0,
                "cache_key": "abc123",
            },
            {
                "stage": "notebook_content",
                "cache_status": "hit",
                "prompt_chars": 200,
                "response_chars": 30,
                "remote_elapsed_ms": 0,
                "saved_ms": 9.5,
                "cache_key": "def456",
            },
            {
                "stage": "notebook_content",
                "cache_status": "bypassed",
                "prompt_chars": 50,
                "response_chars": 10,
                "remote_elapsed_ms": 4.25,
                "saved_ms": 0,
                "error_type": None,
            },
        ],
    )

    assert trace.cache_status == "mixed"
    assert trace.cache_key is None
    assert trace.prompt_chars == 350
    assert trace.response_chars == 60
    assert trace.remote_elapsed_ms == 15.75
    assert trace.saved_ms == 9.5
    assert trace.subcall_count == 3
    assert trace.aggregate_prompt_chars == 350
    assert trace.aggregate_response_chars == 60
    assert trace.aggregate_remote_elapsed_ms == 15.75
    assert trace.aggregate_saved_ms == 9.5
    assert trace.aggregate_cache_hit_count == 1
    assert trace.aggregate_cache_miss_count == 1
    assert trace.aggregate_cache_disabled_count == 0
    assert trace.aggregate_cache_bypassed_remote_count == 1
    assert trace.subcalls is not None
    assert set(trace.subcalls[0]) == {
        "stage",
        "cache_status",
        "prompt_chars",
        "response_chars",
        "remote_elapsed_ms",
        "saved_ms",
        "cache_key",
        "error_type",
    }


def test_llm_trace_summary_prefers_aggregate_fields_and_subcall_counts() -> None:
    summary = build_llm_trace_summary(
        {
            "notebook_content": {
                "stage": "notebook_content",
                "prompt_chars": 1,
                "remote_elapsed_ms": 1,
                "saved_ms": 1,
                "aggregate_prompt_chars": 350,
                "aggregate_remote_elapsed_ms": 15.75,
                "aggregate_saved_ms": 9.5,
                "aggregate_cache_hit_count": 1,
                "aggregate_cache_miss_count": 1,
                "aggregate_cache_disabled_count": 0,
                "aggregate_cache_bypassed_remote_count": 1,
                "subcalls": [
                    {"cache_status": "miss", "prompt_chars": 100, "remote_elapsed_ms": 11.5},
                    {"cache_status": "hit", "prompt_chars": 200, "remote_elapsed_ms": 0},
                    {"cache_status": "bypassed", "prompt_chars": 50, "remote_elapsed_ms": 4.25},
                ],
            },
            "client_report": {
                "stage": "client_report",
                "prompt_chars": 80,
                "remote_elapsed_ms": 3,
                "saved_ms": 0,
                "cache_status": "disabled",
            },
        }
    )

    assert summary["total_prompt_chars"] == 430
    assert summary["remote_prompt_chars"] == 230
    assert summary["remote_call_count"] == 3
    assert summary["cache_hit_count"] == 1
    assert summary["cache_miss_count"] == 1
    assert summary["cache_disabled_count"] == 1
    assert summary["cache_bypassed_remote_count"] == 1
    assert summary["total_llm_remote_elapsed_ms"] == 18.75
    assert summary["total_llm_saved_ms_by_cache"] == 9.5
    assert summary["evidence_pack_stage_count"] == 2


def test_llm_trace_summary_records_profile_policy_and_profile_limits() -> None:
    summary = build_llm_trace_summary(
        {
            "notebook_revision_decision": {
                "stage": "notebook_revision_decision",
                "status": "skipped_by_profile",
                "remote_elapsed_ms": 0,
            },
            "postrun_chart_reflection": {
                "stage": "postrun_chart_reflection",
                "status": "llm_partial",
                "remote_elapsed_ms": 12,
            },
        },
        llm_profile="quick",
        llm_profile_policy={"profile": "quick", "max_revision_decisions": 0},
        profile_limited_stage_count=2,
        max_llm_postrun_reflections_resolved=4,
        allow_postrun_fallback_reflections=False,
    )

    assert summary["llm_profile"] == "quick"
    assert summary["llm_profile_policy"]["profile"] == "quick"
    assert summary["skipped_by_profile_count"] == 1
    assert summary["profile_limited_stage_count"] == 2
    assert summary["max_llm_postrun_reflections_resolved"] == 4
    assert summary["allow_postrun_fallback_reflections"] is False
