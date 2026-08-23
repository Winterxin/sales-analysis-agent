from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.report import ModuleReport
from app.services.analysis_agent.trace import AgentTraceEvent


DEFAULT_USER_GOAL = (
    "对当前销售数据进行综合经营分析，识别最值得关注的趋势、结构、异常和风险。"
)


class AvailableAnalysisTool(BaseModel):
    name: str
    description: str
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)


class ToolExecutionRecord(BaseModel):
    tool_name: str
    round: int
    success: bool
    result_summary: str | None = None
    error: str | None = None


class AgentEvidence(BaseModel):
    evidence_id: str
    tool_name: str
    round: int
    summary_metrics: dict[str, object] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    warning_count: int = 0
    result_signals: list[str] = Field(default_factory=list)


class SalesAnalysisAgentState(BaseModel):
    task_id: str
    user_goal: str = DEFAULT_USER_GOAL
    dataset_profile: dict[str, object] = Field(default_factory=dict)
    available_tools: list[AvailableAnalysisTool] = Field(default_factory=list)
    selected_tools: list[str] = Field(default_factory=list)
    requested_tools: list[str] = Field(default_factory=list)
    executed_tools: list[ToolExecutionRecord] = Field(default_factory=list)
    evidence: list[AgentEvidence] = Field(default_factory=list)
    round: int = 0
    decision: Literal["plan", "need_more", "enough", "fallback", "finalize"] = "plan"
    decision_reason: str = ""
    missing_questions: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    termination_reason: str | None = None
    consecutive_no_progress_rounds: int = 0
    evidence_fingerprint: str = ""
    max_rounds: int = 2
    max_tools_per_round: int = 4
    max_plan_corrections: int = 1
    plan_corrections: int = 0
    validation_action: str = "execute"
    last_guard_errors: list[str] = Field(default_factory=list)
    node_trace: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    report_modules: list[ModuleReport] = Field(default_factory=list, exclude=True)
    current_round_modules: list[ModuleReport] = Field(default_factory=list, exclude=True)
    trace_events: list[AgentTraceEvent] = Field(default_factory=list, exclude=True)

    def public_payload(self) -> dict[str, object]:
        return self.model_dump(
            exclude={"report_modules", "current_round_modules", "trace_events"}
        )
