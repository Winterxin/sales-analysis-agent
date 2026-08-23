from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.analysis_agent.state import SalesAnalysisAgentState
from app.services.analysis_agent.tool_calls import AnalysisToolCall


class PlannerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_calls: list[AnalysisToolCall]
    reasoning_summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def adapt_legacy_selected_tools(cls, value: object) -> object:
        if isinstance(value, dict) and "tool_calls" not in value and "selected_tools" in value:
            adapted = dict(value)
            adapted["tool_calls"] = [
                {"tool_name": name, "arguments": {}}
                for name in adapted.pop("selected_tools")
            ]
            return adapted
        return value


class InspectionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    reason: str = ""
    missing_questions: list[str] = Field(default_factory=list)
    next_tools: list[str] = Field(default_factory=list)


def _available_tool_payload(state: SalesAnalysisAgentState) -> list[dict[str, object]]:
    return [tool.model_dump() for tool in state.available_tools]


def plan_with_llm(
    state: SalesAnalysisAgentState,
    llm_client: Any,
    *,
    replan: bool,
    correction_errors: list[str] | None = None,
) -> PlannerDecision:
    mode = (
        "plan_correction"
        if correction_errors
        else "replan" if replan else "initial_plan"
    )
    payload = {
        "task": "Select deterministic analysis tools for the user goal.",
        "user_goal": state.user_goal,
        "dataset_profile": state.dataset_profile,
        "available_tools": _available_tool_payload(state),
        "already_executed_tools": [item.tool_name for item in state.executed_tools],
        "current_evidence": [item.model_dump() for item in state.evidence],
        "missing_questions": state.missing_questions,
        "inspect_suggested_tools": state.suggested_tools,
        "guard_suggested_tool_calls": [
            call.model_dump() for call in state.suggested_tool_calls
        ],
        "round": state.round + 1,
        "max_tools_per_round": state.max_tools_per_round,
        "mode": mode,
        "harness_rejection_reasons": correction_errors or [],
        "constraints": [
            "Choose only names from available_tools.",
            "Choose 2-4 tools when useful and never return Python code.",
            "On replan, choose only tools that can add evidence not already collected.",
            "On plan correction, fix every harness rejection and choose only available tools.",
            "Return JSON with tool_calls (tool_name + arguments) and reasoning_summary.",
        ],
    }
    raw = llm_client.plan_analysis_tools(payload)
    return PlannerDecision.model_validate(raw)


def inspect_with_llm(
    state: SalesAnalysisAgentState,
    llm_client: Any,
) -> InspectionDecision:
    payload = {
        "task": "Decide whether current deterministic evidence answers the user goal.",
        "user_goal": state.user_goal,
        "executed_tools": [item.model_dump() for item in state.executed_tools],
        "evidence": [item.model_dump() for item in state.evidence],
        "tool_errors": state.errors,
        "round": state.round,
        "remaining_rounds": max(0, state.max_rounds - state.round),
        "available_tools": _available_tool_payload(state),
        "constraints": [
            "Return status as enough or need_more.",
            "If need_more, provide missing_questions and candidate next_tools.",
            "Never invent evidence or tool names.",
        ],
    }
    raw = llm_client.inspect_analysis_evidence(payload)
    decision = InspectionDecision.model_validate(raw)
    if decision.status not in {"enough", "need_more"}:
        raise ValueError(f"Invalid inspection status: {decision.status}")
    return decision
