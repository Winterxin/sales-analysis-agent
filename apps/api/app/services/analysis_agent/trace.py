from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field


class AgentTraceEvent(BaseModel):
    sequence: int
    event_type: str
    node: str
    round: int = 0
    requested_tools: list[str] = Field(default_factory=list)
    accepted_tools: list[str] = Field(default_factory=list)
    rejected_tools: list[str] = Field(default_factory=list)
    guard_errors: list[str] = Field(default_factory=list)
    decision: str | None = None
    decision_reason: str | None = None
    missing_questions: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    executed_tools: list[str] = Field(default_factory=list)
    tool_errors: dict[str, str] = Field(default_factory=dict)
    evidence_before_count: int | None = None
    evidence_after_count: int | None = None
    evidence_delta_count: int | None = None
    evidence_delta_keys: list[str] = Field(default_factory=list)
    no_progress: bool | None = None
    budget_remaining_ms: float | None = None
    llm_call_count: int = 0
    llm_metrics: list[dict[str, object]] = Field(default_factory=list)
    termination_reason: str | None = None


class AnalysisAgentTrace(BaseModel):
    task_id: str
    user_goal: str
    max_rounds: int
    max_tools_per_round: int
    events: list[AgentTraceEvent] = Field(default_factory=list)
    summary: dict[str, object] = Field(default_factory=dict)


def append_trace_event(
    events: list[AgentTraceEvent],
    *,
    event_type: str,
    node: str,
    round: int,
    **values: object,
) -> list[AgentTraceEvent]:
    event = AgentTraceEvent(
        sequence=len(events) + 1,
        event_type=event_type,
        node=node,
        round=round,
        **values,
    )
    return [*events, event]


def _metric_number(metrics: list[dict[str, object]], key: str) -> float:
    return round(
        sum(float(item.get(key) or 0.0) for item in metrics),
        3,
    )


def build_analysis_agent_trace_summary(
    events: list[AgentTraceEvent],
    *,
    rounds: int,
    termination_reason: str | None,
) -> dict[str, object]:
    all_metrics = [metric for event in events for metric in event.llm_metrics]
    planner_events = [
        event for event in events if event.node in {"plan", "plan_correction", "replan"}
    ]
    validation_events = [event for event in events if event.node == "validate_plan"]
    execute_events = [
        event for event in events if event.node in {"execute_tools", "fallback"}
    ]
    return {
        "rounds": rounds,
        "planner_calls": sum(event.llm_call_count for event in planner_events),
        "inspector_calls": sum(
            event.llm_call_count for event in events if event.node == "inspect"
        ),
        "requested_tool_calls": sum(len(event.requested_tools) for event in planner_events),
        "accepted_tool_calls": sum(len(event.accepted_tools) for event in validation_events),
        "rejected_tool_calls": sum(len(event.rejected_tools) for event in validation_events),
        "successful_tool_calls": sum(len(event.executed_tools) for event in execute_events),
        "failed_tool_calls": sum(len(event.tool_errors) for event in execute_events),
        "replans": sum(event.node == "replan" for event in events),
        "no_progress_triggered": any(event.no_progress is True for event in events),
        "fallback_used": any(event.node == "fallback" for event in events),
        "llm_calls": sum(event.llm_call_count for event in events),
        "total_llm_elapsed_ms": _metric_number(all_metrics, "elapsed_ms"),
        "total_prompt_chars": int(_metric_number(all_metrics, "prompt_chars")),
        "total_response_chars": int(_metric_number(all_metrics, "response_chars")),
        "cache_hit_count": sum(
            str(metric.get("cache_status") or "") == "hit" for metric in all_metrics
        ),
        "termination_reason": termination_reason,
    }


def build_analysis_agent_trace(
    *,
    task_id: str,
    user_goal: str,
    max_rounds: int,
    max_tools_per_round: int,
    events: list[AgentTraceEvent],
    rounds: int,
    termination_reason: str | None,
) -> AnalysisAgentTrace:
    return AnalysisAgentTrace(
        task_id=task_id,
        user_goal=user_goal,
        max_rounds=max_rounds,
        max_tools_per_round=max_tools_per_round,
        events=events,
        summary=build_analysis_agent_trace_summary(
            events,
            rounds=rounds,
            termination_reason=termination_reason,
        ),
    )


def snapshot_llm_metrics(llm_client: Any) -> int | None:
    snapshot = getattr(llm_client, "snapshot_completion_metrics", None)
    if not callable(snapshot):
        return None
    return int(snapshot())


def collect_llm_metrics(llm_client: Any, snapshot: int | None) -> list[dict[str, object]]:
    collect = getattr(llm_client, "collect_completion_metrics_since", None)
    if snapshot is not None and callable(collect):
        return [dict(item) for item in collect(snapshot)]
    last = getattr(llm_client, "last_completion_metrics", {})
    return [dict(last)] if isinstance(last, dict) and last else []


T = TypeVar("T")


def call_with_llm_metrics(
    llm_client: Any,
    call: Callable[[], T],
) -> tuple[T | None, list[dict[str, object]], Exception | None]:
    snapshot = snapshot_llm_metrics(llm_client)
    try:
        result = call()
    except Exception as exc:
        return None, collect_llm_metrics(llm_client, snapshot), exc
    return result, collect_llm_metrics(llm_client, snapshot), None
