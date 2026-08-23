from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.services.analysis_agent.facts import evidence_fact_set
from app.services.analysis_agent.state import SalesAnalysisAgentState
from app.services.analysis_agent.tool_calls import AnalysisToolCall


class SufficiencyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    recognized_goal_capabilities: list[str] = Field(default_factory=list)
    satisfied_capabilities: list[str] = Field(default_factory=list)
    missing_capabilities: list[str] = Field(default_factory=list)
    unavailable_capabilities: list[str] = Field(default_factory=list)
    reason: str
    suggested_tool_calls: list[AnalysisToolCall] = Field(default_factory=list)


_KEYWORDS = {
    "profit": ("利润", "profit"),
    "discount": ("折扣", "discount"),
    "trend": ("趋势", "时间", "trend", "time"),
    "product": ("商品", "产品", "product", "sku"),
    "region": ("区域", "地区", "region"),
    "customer": ("客户", "customer"),
}


def recognize_goal_capabilities(user_goal: str) -> list[str]:
    lowered = user_goal.casefold()
    return [
        capability
        for capability, keywords in _KEYWORDS.items()
        if any(keyword.casefold() in lowered for keyword in keywords)
    ]


def _capability_available(capability: str, fields: set[str]) -> bool:
    if capability == "profit":
        return "profit" in fields
    if capability == "discount":
        return "discount" in fields and "profit" in fields
    if capability == "trend":
        return {"order_datetime", "sales_amount"} <= fields
    if capability == "product":
        return "sales_amount" in fields and bool(
            {"product_name", "sku", "category", "productline"} & fields
        )
    if capability == "region":
        return "sales_amount" in fields and bool(
            {"region", "country", "state", "city"} & fields
        )
    if capability == "customer":
        return (
            {"segment", "sales_amount"} <= fields
            or {"order_id", "customer_id"} <= fields
        )
    return True


def _record_satisfies(capability: str, tool_name: str, arguments: dict[str, object]) -> bool:
    if capability in {"profit", "discount"}:
        if capability == "discount":
            return tool_name == "discount_profit_analysis"
        if tool_name == "dimension_breakdown_analysis":
            return arguments.get("metric") == "profit"
        return tool_name in {
            "discount_profit_analysis",
            "product_contribution_analysis",
            "country_market_analysis",
            "loss_risk_modeling",
        }
    if capability == "trend":
        return tool_name == "sales_trend_analysis"
    if capability == "product":
        return tool_name == "product_contribution_analysis"
    if capability == "region":
        return tool_name == "country_market_analysis" or (
            tool_name == "dimension_breakdown_analysis"
            and arguments.get("dimension") in {"region", "country", "state", "city"}
        )
    if capability == "customer":
        return tool_name == "order_structure_analysis" or (
            tool_name == "dimension_breakdown_analysis"
            and arguments.get("dimension") == "segment"
        )
    return False


def _suggestion(capability: str, fields: set[str]) -> AnalysisToolCall | None:
    if capability in {"profit", "discount"}:
        return AnalysisToolCall(tool_name="discount_profit_analysis")
    if capability == "trend":
        return AnalysisToolCall(
            tool_name="sales_trend_analysis", arguments={"granularity": "month"}
        )
    if capability == "product":
        return AnalysisToolCall(
            tool_name="product_contribution_analysis", arguments={"top_n": 10}
        )
    if capability == "region":
        dimension = next(
            (name for name in ("region", "country", "state", "city") if name in fields),
            None,
        )
        if dimension:
            return AnalysisToolCall(
                tool_name="dimension_breakdown_analysis",
                arguments={"dimension": dimension, "metric": "sales_amount", "top_n": 10},
            )
    if capability == "customer" and "segment" in fields:
        return AnalysisToolCall(
            tool_name="dimension_breakdown_analysis",
            arguments={"dimension": "segment", "metric": "sales_amount", "top_n": 10},
        )
    if capability == "customer" and {"order_id", "customer_id"} <= fields:
        return AnalysisToolCall(tool_name="order_structure_analysis")
    return None


def evaluate_sufficiency(
    state: SalesAnalysisAgentState,
    *,
    available_fields: set[str],
) -> SufficiencyResult:
    successful = [record for record in state.executed_tools if record.success]
    has_valid_evidence = bool(evidence_fact_set(state.evidence))
    if not successful or not has_valid_evidence:
        return SufficiencyResult(
            passed=False,
            reason="No successful tool execution with content-based evidence.",
        )

    recognized = recognize_goal_capabilities(state.user_goal)
    satisfied = [
        capability
        for capability in recognized
        if any(
            _record_satisfies(capability, record.tool_name, record.arguments)
            for record in successful
        )
    ]
    unavailable = [
        capability
        for capability in recognized
        if capability not in satisfied
        and not _capability_available(capability, available_fields)
    ]
    missing = [
        capability
        for capability in recognized
        if capability not in satisfied and capability not in unavailable
    ]
    suggestions = [
        suggestion
        for capability in missing
        if (suggestion := _suggestion(capability, available_fields)) is not None
    ]
    return SufficiencyResult(
        passed=not missing and not unavailable,
        recognized_goal_capabilities=recognized,
        satisfied_capabilities=satisfied,
        missing_capabilities=missing,
        unavailable_capabilities=unavailable,
        reason=(
            "Generic evidence floor passed."
            if not recognized
            else "All recognized goal capabilities are satisfied."
            if not missing and not unavailable
            else "One or more recognized goal capabilities remain unresolved."
        ),
        suggested_tool_calls=suggestions,
    )
