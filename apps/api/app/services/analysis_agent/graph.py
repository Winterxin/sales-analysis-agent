from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.analysis.runner import run_analysis
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_agent.executor import execute_analysis_tools
from app.services.analysis_agent.guards import validate_selected_tools
from app.services.analysis_agent.planner import inspect_with_llm, plan_with_llm
from app.services.analysis_agent.state import (
    AgentEvidence,
    AvailableAnalysisTool,
    DEFAULT_USER_GOAL,
    SalesAnalysisAgentState,
    ToolExecutionRecord,
)
from app.services.analysis_tools import AnalysisToolRegistry, get_analysis_tool_registry
from app.services.run_budget import RunBudget


@dataclass(frozen=True)
class AnalysisAgentResult:
    report: AnalysisReport
    analysis_plan: AnalysisPlan
    state: SalesAnalysisAgentState


class _AgentRuntime(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    csv_path: Path
    schema_mapping: SchemaMapping
    fallback_plan: AnalysisPlan
    llm_client: Any
    run_budget: RunBudget
    registry: AnalysisToolRegistry


def _trace(state: SalesAnalysisAgentState, node: str) -> list[str]:
    return [*state.node_trace, node]


def _available_fields(schema_mapping: SchemaMapping) -> set[str]:
    return set(schema_mapping.field_mapping.values())


def _available_tool_state(
    registry: AnalysisToolRegistry,
    available_fields: set[str],
    dataset_profile: dict[str, object],
) -> list[AvailableAnalysisTool]:
    return [
        AvailableAnalysisTool(
            name=spec.name,
            description=spec.description,
            required_fields=sorted(spec.required_fields),
            optional_fields=sorted(spec.optional_fields),
        )
        for spec in registry.available_specs(available_fields, dataset_profile)
    ]


def _report_from_modules(
    task_id: str,
    dataset_type: str,
    modules: list[ModuleReport],
) -> AnalysisReport:
    return AnalysisReport(
        task_id=task_id,
        dataset_type=dataset_type,
        module_count=len(modules),
        summary=[finding for module in modules for finding in module.findings[:1]],
        modules=modules,
    )


def _fingerprint(evidence: list[AgentEvidence]) -> str:
    payload = [item.model_dump(mode="json") for item in evidence]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fallback_result(
    *,
    task_id: str,
    csv_path: Path,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, object],
    user_goal: str,
    fallback_plan: AnalysisPlan,
    reason: str,
    registry: AnalysisToolRegistry,
) -> AnalysisAgentResult:
    report = run_analysis(task_id, csv_path, schema_mapping, fallback_plan)
    available_fields = _available_fields(schema_mapping)
    executions = [
        ToolExecutionRecord(
            tool_name=module.module_id,
            round=0,
            success=True,
            result_summary=(module.findings[0] if module.findings else module.title),
        )
        for module in report.modules
    ]
    evidence = [
        AgentEvidence(
            evidence_id=f"fallback:{module.module_id}",
            tool_name=module.module_id,
            round=0,
            summary_metrics=module.summary_metrics,
            findings=module.findings[:3],
            warning_count=len(module.warnings),
        )
        for module in report.modules
    ]
    state = SalesAnalysisAgentState(
        task_id=task_id,
        user_goal=user_goal,
        dataset_profile=dataset_profile,
        available_tools=_available_tool_state(registry, available_fields, dataset_profile),
        executed_tools=executions,
        evidence=evidence,
        decision="fallback",
        decision_reason="Deterministic analysis plan executed.",
        termination_reason=reason,
        fallback_reason=reason,
        evidence_fingerprint=_fingerprint(evidence),
        node_trace=["fallback", "finalize"],
        report_modules=report.modules,
    )
    return AnalysisAgentResult(report=report, analysis_plan=fallback_plan, state=state)


def _build_graph(runtime: _AgentRuntime):
    available_fields = _available_fields(runtime.schema_mapping)

    def plan_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        try:
            if not runtime.run_budget.has_budget_for_stage("analysis_agent_plan"):
                raise RuntimeError("analysis_agent_plan budget exhausted")
            decision = plan_with_llm(state, runtime.llm_client, replan=False)
            return {
                "selected_tools": decision.selected_tools,
                "decision": "plan",
                "decision_reason": decision.reasoning_summary,
                "round": state.round + 1,
                "node_trace": _trace(state, "plan"),
            }
        except Exception as exc:
            return {
                "fallback_reason": "planner_failure_fallback",
                "errors": [*state.errors, f"planner_error:{type(exc).__name__}:{exc}"],
                "node_trace": _trace(state, "plan"),
            }

    def replan_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        try:
            if not runtime.run_budget.has_budget_for_stage("analysis_agent_plan"):
                raise RuntimeError("analysis_agent_plan budget exhausted")
            decision = plan_with_llm(state, runtime.llm_client, replan=True)
            return {
                "selected_tools": decision.selected_tools,
                "decision": "plan",
                "decision_reason": decision.reasoning_summary,
                "round": state.round + 1,
                "node_trace": _trace(state, "replan"),
            }
        except Exception as exc:
            return {
                "fallback_reason": "planner_failure_fallback",
                "errors": [*state.errors, f"replanner_error:{type(exc).__name__}:{exc}"],
                "node_trace": _trace(state, "replan"),
            }

    def validate_plan_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        successful = {item.tool_name for item in state.executed_tools if item.success}
        result = validate_selected_tools(
            state.selected_tools,
            registry=runtime.registry,
            available_fields=available_fields,
            dataset_profile=state.dataset_profile,
            executed_successfully=successful,
            max_tools_per_round=state.max_tools_per_round,
            has_budget=runtime.run_budget.has_budget_for_stage("analysis_agent_plan"),
        )
        return {
            "selected_tools": result.accepted_tools,
            "errors": [*state.errors, *result.errors],
            "node_trace": _trace(state, "validate_plan"),
        }

    def execute_tools_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        reports, execution_errors = execute_analysis_tools(
            state.selected_tools,
            csv_path=runtime.csv_path,
            schema_mapping=runtime.schema_mapping,
            registry=runtime.registry,
        )
        report_by_name = {report.module_id: report for report in reports}
        records = list(state.executed_tools)
        for tool_name in state.selected_tools:
            report = report_by_name.get(tool_name)
            error = execution_errors.get(tool_name)
            records.append(
                ToolExecutionRecord(
                    tool_name=tool_name,
                    round=state.round,
                    success=report is not None,
                    result_summary=(
                        report.findings[0]
                        if report is not None and report.findings
                        else report.title if report is not None else None
                    ),
                    error=error,
                )
            )
        errors = [
            *state.errors,
            *(f"tool_execution_error:{name}:{error}" for name, error in execution_errors.items()),
        ]
        return {
            "executed_tools": records,
            "report_modules": [*state.report_modules, *reports],
            "current_round_modules": reports,
            "errors": errors,
            "node_trace": _trace(state, "execute_tools"),
        }

    def build_evidence_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        evidence = list(state.evidence)
        existing_ids = {item.evidence_id for item in evidence}
        for module in state.current_round_modules:
            if not (module.summary_metrics or module.findings or any(module.tables.values())):
                continue
            evidence_id = f"round:{state.round}:tool:{module.module_id}"
            if evidence_id in existing_ids:
                continue
            evidence.append(
                AgentEvidence(
                    evidence_id=evidence_id,
                    tool_name=module.module_id,
                    round=state.round,
                    summary_metrics=module.summary_metrics,
                    findings=module.findings[:3],
                    warning_count=len(module.warnings),
                )
            )
        fingerprint = _fingerprint(evidence)
        no_progress = fingerprint == state.evidence_fingerprint
        return {
            "evidence": evidence,
            "evidence_fingerprint": fingerprint,
            "consecutive_no_progress_rounds": (
                state.consecutive_no_progress_rounds + 1 if no_progress else 0
            ),
            "current_round_modules": [],
            "node_trace": _trace(state, "build_evidence"),
        }

    def inspect_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        trace = _trace(state, "inspect")
        if state.consecutive_no_progress_rounds >= 1:
            return {
                "decision": "finalize",
                "decision_reason": "No new valid evidence was produced in the latest round.",
                "termination_reason": "no_progress",
                "node_trace": trace,
            }
        try:
            if not runtime.run_budget.has_budget_for_stage("analysis_agent_inspect"):
                raise RuntimeError("analysis_agent_inspect budget exhausted")
            decision = inspect_with_llm(state, runtime.llm_client)
        except Exception as exc:
            return {
                "fallback_reason": "inspect_failure_fallback",
                "errors": [*state.errors, f"inspect_error:{type(exc).__name__}:{exc}"],
                "node_trace": trace,
            }
        if decision.status == "enough":
            return {
                "decision": "enough",
                "decision_reason": decision.reason,
                "termination_reason": "evidence_sufficient",
                "missing_questions": [],
                "suggested_tools": [],
                "node_trace": trace,
            }
        if state.round >= state.max_rounds:
            return {
                "decision": "finalize",
                "decision_reason": decision.reason,
                "termination_reason": "max_rounds",
                "missing_questions": decision.missing_questions,
                "suggested_tools": decision.next_tools,
                "node_trace": trace,
            }
        return {
            "decision": "need_more",
            "decision_reason": decision.reason,
            "missing_questions": decision.missing_questions,
            "suggested_tools": decision.next_tools,
            "node_trace": trace,
        }

    def fallback_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        report = run_analysis(
            state.task_id,
            runtime.csv_path,
            runtime.schema_mapping,
            runtime.fallback_plan,
        )
        records = [
            ToolExecutionRecord(
                tool_name=module.module_id,
                round=0,
                success=True,
                result_summary=(module.findings[0] if module.findings else module.title),
            )
            for module in report.modules
        ]
        evidence = [
            AgentEvidence(
                evidence_id=f"fallback:{module.module_id}",
                tool_name=module.module_id,
                round=0,
                summary_metrics=module.summary_metrics,
                findings=module.findings[:3],
                warning_count=len(module.warnings),
            )
            for module in report.modules
        ]
        return {
            "decision": "fallback",
            "decision_reason": "Deterministic analysis plan executed after Agent failure.",
            "termination_reason": state.fallback_reason or "agent_failure_fallback",
            "selected_tools": [],
            "executed_tools": records,
            "evidence": evidence,
            "evidence_fingerprint": _fingerprint(evidence),
            "report_modules": report.modules,
            "node_trace": _trace(state, "fallback"),
        }

    def finalize_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        decision = "fallback" if state.fallback_reason else "finalize"
        return {"decision": decision, "node_trace": _trace(state, "finalize")}

    def route_after_plan(state: SalesAnalysisAgentState) -> str:
        return "fallback" if state.fallback_reason else "validate_plan"

    def route_after_inspect(state: SalesAnalysisAgentState) -> str:
        if state.fallback_reason:
            return "fallback"
        if state.decision == "need_more":
            return "replan"
        return "finalize"

    graph = StateGraph(SalesAnalysisAgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node("execute_tools", execute_tools_node)
    graph.add_node("build_evidence", build_evidence_node)
    graph.add_node("inspect", inspect_node)
    graph.add_node("replan", replan_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", route_after_plan, {"validate_plan": "validate_plan", "fallback": "fallback"}
    )
    graph.add_edge("validate_plan", "execute_tools")
    graph.add_edge("execute_tools", "build_evidence")
    graph.add_edge("build_evidence", "inspect")
    graph.add_conditional_edges(
        "inspect",
        route_after_inspect,
        {"replan": "replan", "fallback": "fallback", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "replan", route_after_plan, {"validate_plan": "validate_plan", "fallback": "fallback"}
    )
    graph.add_edge("fallback", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_analysis_agent(
    *,
    task_id: str,
    csv_path: Path,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, object],
    user_goal: str | None,
    fallback_plan: AnalysisPlan,
    llm_client: Any,
    run_budget: RunBudget,
    max_rounds: int = 2,
    max_tools_per_round: int = 4,
    registry: AnalysisToolRegistry | None = None,
) -> AnalysisAgentResult:
    registry = registry or get_analysis_tool_registry()
    resolved_goal = (user_goal or "").strip() or DEFAULT_USER_GOAL
    if not bool(getattr(llm_client, "enabled", False)):
        return _fallback_result(
            task_id=task_id,
            csv_path=csv_path,
            schema_mapping=schema_mapping,
            dataset_profile=dataset_profile,
            user_goal=resolved_goal,
            fallback_plan=fallback_plan,
            reason="llm_unavailable_fallback",
            registry=registry,
        )
    if not run_budget.has_budget_for_stage("analysis_agent_plan"):
        return _fallback_result(
            task_id=task_id,
            csv_path=csv_path,
            schema_mapping=schema_mapping,
            dataset_profile=dataset_profile,
            user_goal=resolved_goal,
            fallback_plan=fallback_plan,
            reason="budget_exhausted_fallback",
            registry=registry,
        )

    runtime = _AgentRuntime(
        csv_path=csv_path,
        schema_mapping=schema_mapping,
        fallback_plan=fallback_plan,
        llm_client=llm_client,
        run_budget=run_budget,
        registry=registry,
    )
    initial_state = SalesAnalysisAgentState(
        task_id=task_id,
        user_goal=resolved_goal,
        dataset_profile=dataset_profile,
        available_tools=_available_tool_state(
            registry, _available_fields(schema_mapping), dataset_profile
        ),
        evidence_fingerprint=_fingerprint([]),
        max_rounds=max(1, max_rounds),
        max_tools_per_round=max(1, max_tools_per_round),
    )
    output = _build_graph(runtime).invoke(initial_state, {"recursion_limit": 32})
    state = SalesAnalysisAgentState.model_validate(output)
    report = _report_from_modules(task_id, schema_mapping.dataset_type, state.report_modules)
    executed_plan = [
        record.tool_name
        for record in state.executed_tools
        if record.success and record.tool_name in {module.module_id for module in report.modules}
    ]
    plan = (
        fallback_plan
        if state.fallback_reason
        else AnalysisPlan(
            analysis_plan=list(dict.fromkeys(executed_plan)),
            chart_preferences={
                name: chart
                for name, chart in fallback_plan.chart_preferences.items()
                if name in executed_plan
            },
            reasoning_summary=[state.decision_reason],
        )
    )
    return AnalysisAgentResult(report=report, analysis_plan=plan, state=state)
