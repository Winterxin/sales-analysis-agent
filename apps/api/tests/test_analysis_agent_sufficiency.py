from __future__ import annotations

from app.services.analysis_agent.state import (
    AgentEvidence,
    SalesAnalysisAgentState,
    ToolExecutionRecord,
)
from app.services.analysis_agent.sufficiency import evaluate_sufficiency


def _state(goal: str, tool_name: str | None = "sales_trend_analysis") -> SalesAnalysisAgentState:
    executions = (
        [ToolExecutionRecord(tool_name=tool_name, round=1, success=True)]
        if tool_name
        else []
    )
    evidence = (
        [
            AgentEvidence(
                evidence_id="e1",
                tool_name=tool_name,
                round=1,
                summary_metrics={"value": 1},
            )
        ]
        if tool_name
        else []
    )
    return SalesAnalysisAgentState(
        task_id="task", user_goal=goal, executed_tools=executions, evidence=evidence
    )


def test_sufficiency_passes_when_goal_capability_is_satisfied() -> None:
    result = evaluate_sufficiency(
        _state("销售趋势"), available_fields={"order_datetime", "sales_amount"}
    )

    assert result.passed is True
    assert result.satisfied_capabilities == ["trend"]


def test_sufficiency_rejects_empty_execution_and_evidence() -> None:
    result = evaluate_sufficiency(_state("趋势", None), available_fields=set())

    assert result.passed is False
    assert "No successful tool" in result.reason


def test_sufficiency_suggests_profit_tool_when_profit_is_available() -> None:
    result = evaluate_sufficiency(
        _state("分析利润"), available_fields={"sales_amount", "profit"}
    )

    assert result.missing_capabilities == ["profit"]
    assert result.suggested_tool_calls[0].tool_name == "discount_profit_analysis"


def test_sufficiency_marks_profit_unavailable_without_field() -> None:
    result = evaluate_sufficiency(
        _state("分析利润"), available_fields={"sales_amount"}
    )

    assert result.unavailable_capabilities == ["profit"]
    assert result.suggested_tool_calls == []


def test_unknown_goal_uses_generic_evidence_floor() -> None:
    result = evaluate_sufficiency(
        _state("做一次综合经营复盘"), available_fields={"sales_amount"}
    )

    assert result.passed is True
    assert result.recognized_goal_capabilities == []
