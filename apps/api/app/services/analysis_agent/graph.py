from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.analysis.runner import run_analysis
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_agent.executor import execute_analysis_tools
from app.services.analysis_agent.facts import evidence_fact_fingerprint, evidence_fact_set
from app.services.analysis_agent.guards import validate_selected_tools
from app.services.analysis_agent.planner import inspect_with_llm, plan_with_llm
from app.services.analysis_agent.state import (
    AgentEvidence,
    AvailableAnalysisTool,
    DEFAULT_USER_GOAL,
    SalesAnalysisAgentState,
    ToolExecutionRecord,
)
from app.services.analysis_agent.trace import (
    AnalysisAgentTrace,
    append_trace_event,
    build_analysis_agent_trace,
    call_with_llm_metrics,
)
from app.services.analysis_tools import AnalysisToolRegistry, get_analysis_tool_registry
from app.services.run_budget import RunBudget


@dataclass(frozen=True)
class AnalysisAgentResult:
    report: AnalysisReport
    analysis_plan: AnalysisPlan
    state: SalesAnalysisAgentState
    trace: AnalysisAgentTrace


class _AgentRuntime(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    csv_path: Path
    schema_mapping: SchemaMapping
    fallback_plan: AnalysisPlan
    llm_client: Any
    run_budget: RunBudget
    registry: AnalysisToolRegistry


def _node_path(state: SalesAnalysisAgentState, node: str) -> list[str]:
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


def _module_evidence(module: ModuleReport, *, round_number: int) -> AgentEvidence | None:
    signals = [
        f"table:{name}:rows:{len(rows)}"
        for name, rows in sorted(module.tables.items())
        if rows
    ]
    signals.extend(f"warning:{warning}" for warning in module.warnings[:3])
    if not (module.summary_metrics or module.findings or signals):
        return None
    return AgentEvidence(
        evidence_id=f"round:{round_number}:tool:{module.module_id}",
        tool_name=module.module_id,
        round=round_number,
        summary_metrics=module.summary_metrics,
        findings=module.findings[:3],
        warning_count=len(module.warnings),
        result_signals=signals,
    )


def _fallback_state(
    *,
    task_id: str,
    csv_path: Path,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, object],
    user_goal: str,
    fallback_plan: AnalysisPlan,
    reason: str,
    registry: AnalysisToolRegistry,
    max_rounds: int,
    max_tools_per_round: int,
) -> SalesAnalysisAgentState:
    report = run_analysis(task_id, csv_path, schema_mapping, fallback_plan)
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
        item
        for module in report.modules
        if (item := _module_evidence(module, round_number=0)) is not None
    ]
    events = append_trace_event(
        [],
        event_type="deterministic_fallback",
        node="fallback",
        round=0,
        executed_tools=[module.module_id for module in report.modules],
        decision="fallback",
        decision_reason="Deterministic analysis plan executed.",
        termination_reason=reason,
    )
    events = append_trace_event(
        events,
        event_type="finalize",
        node="finalize",
        round=0,
        executed_tools=[item.tool_name for item in executions if item.success],
        decision="fallback",
        termination_reason=reason,
    )
    return SalesAnalysisAgentState(
        task_id=task_id,
        user_goal=user_goal,
        dataset_profile=dataset_profile,
        available_tools=_available_tool_state(
            registry, _available_fields(schema_mapping), dataset_profile
        ),
        executed_tools=executions,
        evidence=evidence,
        decision="fallback",
        decision_reason="Deterministic analysis plan executed.",
        termination_reason=reason,
        fallback_reason=reason,
        evidence_fingerprint=evidence_fact_fingerprint(evidence_fact_set(evidence)),
        max_rounds=max_rounds,
        max_tools_per_round=max_tools_per_round,
        node_trace=["fallback", "finalize"],
        trace_events=events,
        report_modules=report.modules,
    )


def _result_from_state(
    state: SalesAnalysisAgentState,
    *,
    schema_mapping: SchemaMapping,
    fallback_plan: AnalysisPlan,
) -> AnalysisAgentResult:
    report = _report_from_modules(
        state.task_id, schema_mapping.dataset_type, state.report_modules
    )
    report_ids = {module.module_id for module in report.modules}
    executed_plan = [
        record.tool_name
        for record in state.executed_tools
        if record.success and record.tool_name in report_ids
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
    trace = build_analysis_agent_trace(
        task_id=state.task_id,
        user_goal=state.user_goal,
        max_rounds=state.max_rounds,
        max_tools_per_round=state.max_tools_per_round,
        events=state.trace_events,
        rounds=state.round,
        termination_reason=state.termination_reason,
    )
    return AnalysisAgentResult(report=report, analysis_plan=plan, state=state, trace=trace)


def _build_graph(runtime: _AgentRuntime):
    available_fields = _available_fields(runtime.schema_mapping)

    def _planner_node(
        state: SalesAnalysisAgentState,
        *,
        node: str,
        replan: bool,
        correction_errors: list[str] | None = None,
    ) -> dict[str, object]:
        if not runtime.run_budget.has_budget_for_stage("analysis_agent_plan"):
            llm_attempted = False
            decision = None
            metrics: list[dict[str, object]] = []
            error: Exception | None = RuntimeError("analysis_agent_plan budget exhausted")
        else:
            llm_attempted = True
            decision, metrics, error = call_with_llm_metrics(
                runtime.llm_client,
                lambda: plan_with_llm(
                    state,
                    runtime.llm_client,
                    replan=replan,
                    correction_errors=correction_errors,
                ),
            )
        requested = list(decision.selected_tools) if decision is not None else []
        reason = decision.reasoning_summary if decision is not None else None
        event_round = state.round if node == "plan_correction" else state.round + 1
        events = append_trace_event(
            state.trace_events,
            event_type=node,
            node=node,
            round=event_round,
            requested_tools=requested,
            guard_errors=correction_errors or [],
            decision="plan" if decision is not None else "fallback",
            decision_reason=reason or (str(error) if error else ""),
            budget_remaining_ms=runtime.run_budget.remaining_ms(),
            llm_call_count=int(llm_attempted),
            llm_metrics=metrics,
        )
        if error is not None:
            label = "replanner" if replan else "planner"
            return {
                "fallback_reason": "planner_failure_fallback",
                "errors": [*state.errors, f"{label}_error:{type(error).__name__}:{error}"],
                "node_trace": _node_path(state, node),
                "trace_events": events,
            }
        return {
            "requested_tools": requested,
            "selected_tools": requested,
            "decision": "plan",
            "decision_reason": reason or "",
            "round": event_round,
            "plan_corrections": (
                state.plan_corrections + 1
                if node == "plan_correction"
                else state.plan_corrections
            ),
            "node_trace": _node_path(state, node),
            "trace_events": events,
        }

    def plan_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        return _planner_node(state, node="plan", replan=False)

    def plan_correction_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        return _planner_node(
            state,
            node="plan_correction",
            replan=False,
            correction_errors=state.last_guard_errors,
        )

    def replan_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        return _planner_node(state, node="replan", replan=True)

    def validate_plan_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        successful = {item.tool_name for item in state.executed_tools if item.success}
        result = validate_selected_tools(
            state.requested_tools,
            registry=runtime.registry,
            available_fields=available_fields,
            dataset_profile=state.dataset_profile,
            executed_successfully=successful,
            max_tools_per_round=state.max_tools_per_round,
            has_budget=runtime.run_budget.has_budget_for_stage("analysis_agent_plan"),
        )
        if result.accepted_tools:
            action = "execute"
            fallback_reason = state.fallback_reason
            termination_reason = state.termination_reason
        elif state.round == 1 and not state.evidence:
            if not state.requested_tools:
                action = "fallback"
                fallback_reason = "initial_plan_empty_fallback"
            elif state.plan_corrections < state.max_plan_corrections:
                action = "correct"
                fallback_reason = state.fallback_reason
            else:
                action = "fallback"
                fallback_reason = "initial_plan_invalid_fallback"
            termination_reason = state.termination_reason
        elif state.evidence:
            action = "finalize"
            fallback_reason = state.fallback_reason
            termination_reason = "replan_no_valid_tools"
        else:
            action = "fallback"
            fallback_reason = "initial_plan_invalid_fallback"
            termination_reason = state.termination_reason
        events = append_trace_event(
            state.trace_events,
            event_type="plan_validation",
            node="validate_plan",
            round=state.round,
            requested_tools=state.requested_tools,
            accepted_tools=result.accepted_tools,
            rejected_tools=result.rejected_tools,
            guard_errors=result.errors,
            decision=action,
            budget_remaining_ms=runtime.run_budget.remaining_ms(),
            termination_reason=termination_reason,
        )
        return {
            "selected_tools": result.accepted_tools,
            "errors": [*state.errors, *result.errors],
            "last_guard_errors": result.errors,
            "validation_action": action,
            "fallback_reason": fallback_reason,
            "termination_reason": termination_reason,
            "decision": "finalize" if action == "finalize" else state.decision,
            "decision_reason": (
                "No additional valid tools remained after replan."
                if action == "finalize"
                else state.decision_reason
            ),
            "node_trace": _node_path(state, "validate_plan"),
            "trace_events": events,
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
            *(
                f"tool_execution_error:{name}:{error}"
                for name, error in execution_errors.items()
            ),
        ]
        events = append_trace_event(
            state.trace_events,
            event_type="tool_execution",
            node="execute_tools",
            round=state.round,
            accepted_tools=state.selected_tools,
            executed_tools=[report.module_id for report in reports],
            tool_errors=execution_errors,
            budget_remaining_ms=runtime.run_budget.remaining_ms(),
        )
        return {
            "executed_tools": records,
            "report_modules": [*state.report_modules, *reports],
            "current_round_modules": reports,
            "errors": errors,
            "node_trace": _node_path(state, "execute_tools"),
            "trace_events": events,
        }

    def build_evidence_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        evidence_before = list(state.evidence)
        evidence = list(evidence_before)
        existing_ids = {item.evidence_id for item in evidence}
        for module in state.current_round_modules:
            item = _module_evidence(module, round_number=state.round)
            if item is not None and item.evidence_id not in existing_ids:
                evidence.append(item)
        facts_before = evidence_fact_set(evidence_before)
        facts_after = evidence_fact_set(evidence)
        delta = sorted(facts_after - facts_before)
        no_progress = not delta
        events = append_trace_event(
            state.trace_events,
            event_type="evidence_update",
            node="build_evidence",
            round=state.round,
            evidence_before_count=len(facts_before),
            evidence_after_count=len(facts_after),
            evidence_delta_count=len(delta),
            evidence_delta_keys=delta,
            no_progress=no_progress,
            budget_remaining_ms=runtime.run_budget.remaining_ms(),
        )
        return {
            "evidence": evidence,
            "evidence_fingerprint": evidence_fact_fingerprint(facts_after),
            "consecutive_no_progress_rounds": (
                state.consecutive_no_progress_rounds + 1 if no_progress else 0
            ),
            "current_round_modules": [],
            "node_trace": _node_path(state, "build_evidence"),
            "trace_events": events,
        }

    def inspect_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        if state.consecutive_no_progress_rounds >= 1:
            has_facts = bool(evidence_fact_set(state.evidence))
            termination = "no_progress" if has_facts else "initial_execution_no_evidence_fallback"
            fallback_reason = state.fallback_reason if has_facts else termination
            events = append_trace_event(
                state.trace_events,
                event_type="inspection_skipped_no_progress",
                node="inspect",
                round=state.round,
                decision="finalize" if has_facts else "fallback",
                decision_reason="No new content-based evidence facts were produced.",
                no_progress=True,
                termination_reason=termination,
                budget_remaining_ms=runtime.run_budget.remaining_ms(),
            )
            return {
                "decision": "finalize" if has_facts else "fallback",
                "decision_reason": "No new content-based evidence facts were produced.",
                "termination_reason": termination if has_facts else state.termination_reason,
                "fallback_reason": fallback_reason,
                "node_trace": _node_path(state, "inspect"),
                "trace_events": events,
            }
        if not runtime.run_budget.has_budget_for_stage("analysis_agent_inspect"):
            llm_attempted = False
            decision = None
            metrics: list[dict[str, object]] = []
            error: Exception | None = RuntimeError("analysis_agent_inspect budget exhausted")
        else:
            llm_attempted = True
            decision, metrics, error = call_with_llm_metrics(
                runtime.llm_client,
                lambda: inspect_with_llm(state, runtime.llm_client),
            )
        if error is not None:
            events = append_trace_event(
                state.trace_events,
                event_type="inspection_error",
                node="inspect",
                round=state.round,
                decision="fallback",
                decision_reason=str(error),
                llm_call_count=int(llm_attempted),
                llm_metrics=metrics,
                budget_remaining_ms=runtime.run_budget.remaining_ms(),
            )
            return {
                "fallback_reason": "inspect_failure_fallback",
                "errors": [
                    *state.errors,
                    f"inspect_error:{type(error).__name__}:{error}",
                ],
                "node_trace": _node_path(state, "inspect"),
                "trace_events": events,
            }
        assert decision is not None
        if decision.status == "enough":
            next_decision = "enough"
            termination_reason = "evidence_sufficient"
        elif state.round >= state.max_rounds:
            next_decision = "finalize"
            termination_reason = "max_rounds"
        else:
            next_decision = "need_more"
            termination_reason = None
        events = append_trace_event(
            state.trace_events,
            event_type="inspection_decision",
            node="inspect",
            round=state.round,
            decision=next_decision,
            decision_reason=decision.reason,
            missing_questions=decision.missing_questions,
            suggested_tools=decision.next_tools,
            llm_call_count=int(llm_attempted),
            llm_metrics=metrics,
            budget_remaining_ms=runtime.run_budget.remaining_ms(),
            termination_reason=termination_reason,
        )
        return {
            "decision": next_decision,
            "decision_reason": decision.reason,
            "termination_reason": termination_reason,
            "missing_questions": decision.missing_questions if next_decision != "enough" else [],
            "suggested_tools": decision.next_tools if next_decision != "enough" else [],
            "node_trace": _node_path(state, "inspect"),
            "trace_events": events,
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
            item
            for module in report.modules
            if (item := _module_evidence(module, round_number=0)) is not None
        ]
        reason = state.fallback_reason or "agent_failure_fallback"
        events = append_trace_event(
            state.trace_events,
            event_type="deterministic_fallback",
            node="fallback",
            round=state.round,
            executed_tools=[module.module_id for module in report.modules],
            decision="fallback",
            decision_reason="Deterministic analysis plan executed after Agent failure.",
            termination_reason=reason,
            budget_remaining_ms=runtime.run_budget.remaining_ms(),
        )
        return {
            "decision": "fallback",
            "decision_reason": "Deterministic analysis plan executed after Agent failure.",
            "termination_reason": reason,
            "selected_tools": [],
            "executed_tools": records,
            "evidence": evidence,
            "evidence_fingerprint": evidence_fact_fingerprint(evidence_fact_set(evidence)),
            "report_modules": report.modules,
            "node_trace": _node_path(state, "fallback"),
            "trace_events": events,
        }

    def finalize_node(state: SalesAnalysisAgentState) -> dict[str, object]:
        decision = "fallback" if state.fallback_reason else "finalize"
        events = append_trace_event(
            state.trace_events,
            event_type="finalize",
            node="finalize",
            round=state.round,
            executed_tools=[item.tool_name for item in state.executed_tools if item.success],
            decision=decision,
            decision_reason=state.decision_reason,
            termination_reason=state.termination_reason,
            budget_remaining_ms=runtime.run_budget.remaining_ms(),
        )
        return {
            "decision": decision,
            "node_trace": _node_path(state, "finalize"),
            "trace_events": events,
        }

    def route_after_plan(state: SalesAnalysisAgentState) -> str:
        return "fallback" if state.fallback_reason else "validate_plan"

    def route_after_validate(state: SalesAnalysisAgentState) -> str:
        return state.validation_action

    def route_after_inspect(state: SalesAnalysisAgentState) -> str:
        if state.fallback_reason:
            return "fallback"
        if state.decision == "need_more":
            return "replan"
        return "finalize"

    graph = StateGraph(SalesAnalysisAgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("plan_correction", plan_correction_node)
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
    graph.add_conditional_edges(
        "plan_correction",
        route_after_plan,
        {"validate_plan": "validate_plan", "fallback": "fallback"},
    )
    graph.add_conditional_edges(
        "replan", route_after_plan, {"validate_plan": "validate_plan", "fallback": "fallback"}
    )
    graph.add_conditional_edges(
        "validate_plan",
        route_after_validate,
        {
            "execute": "execute_tools",
            "correct": "plan_correction",
            "fallback": "fallback",
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_tools", "build_evidence")
    graph.add_edge("build_evidence", "inspect")
    graph.add_conditional_edges(
        "inspect",
        route_after_inspect,
        {"replan": "replan", "fallback": "fallback", "finalize": "finalize"},
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
    max_plan_corrections: int = 1,
    registry: AnalysisToolRegistry | None = None,
) -> AnalysisAgentResult:
    registry = registry or get_analysis_tool_registry()
    resolved_goal = (user_goal or "").strip() or DEFAULT_USER_GOAL
    resolved_rounds = max(1, max_rounds)
    resolved_tool_limit = max(1, max_tools_per_round)
    direct_fallback_reason = None
    if not bool(getattr(llm_client, "enabled", False)):
        direct_fallback_reason = "llm_unavailable_fallback"
    elif not run_budget.has_budget_for_stage("analysis_agent_plan"):
        direct_fallback_reason = "budget_exhausted_fallback"
    if direct_fallback_reason is not None:
        state = _fallback_state(
            task_id=task_id,
            csv_path=csv_path,
            schema_mapping=schema_mapping,
            dataset_profile=dataset_profile,
            user_goal=resolved_goal,
            fallback_plan=fallback_plan,
            reason=direct_fallback_reason,
            registry=registry,
            max_rounds=resolved_rounds,
            max_tools_per_round=resolved_tool_limit,
        )
        return _result_from_state(
            state, schema_mapping=schema_mapping, fallback_plan=fallback_plan
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
        evidence_fingerprint=evidence_fact_fingerprint(set()),
        max_rounds=resolved_rounds,
        max_tools_per_round=resolved_tool_limit,
        max_plan_corrections=max(0, max_plan_corrections),
    )
    output = _build_graph(runtime).invoke(initial_state, {"recursion_limit": 40})
    state = SalesAnalysisAgentState.model_validate(output)
    return _result_from_state(
        state, schema_mapping=schema_mapping, fallback_plan=fallback_plan
    )
