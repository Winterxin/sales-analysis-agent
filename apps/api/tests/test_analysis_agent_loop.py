from __future__ import annotations

from pathlib import Path

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_agent.graph import run_analysis_agent
from app.services.analysis_tools import AnalysisToolRegistry, get_analysis_tool_registry
from app.services.run_budget import RunBudget


class FakeAgentLLM:
    enabled = True
    source = "fake"
    configured_model = "fake-agent"

    def __init__(self, plans: list[object], inspections: list[object]) -> None:
        self.plans = list(plans)
        self.inspections = list(inspections)
        self.metrics: list[dict[str, object]] = []
        self.plan_payloads: list[dict[str, object]] = []

    @property
    def last_completion_metrics(self) -> dict[str, object]:
        return dict(self.metrics[-1]) if self.metrics else {}

    def snapshot_completion_metrics(self) -> int:
        return len(self.metrics)

    def collect_completion_metrics_since(self, snapshot: int):
        return [dict(item) for item in self.metrics[snapshot:]]

    def _record(self, stage: str) -> None:
        self.metrics.append(
            {
                "stage": stage,
                "elapsed_ms": 3.0,
                "prompt_chars": 100,
                "response_chars": 30,
                "attempt_count": 1,
                "cache_status": "miss",
                "remote_elapsed_ms": 2.0,
                "saved_ms": 0.0,
                "model": self.configured_model,
                "source": self.source,
            }
        )

    def plan_analysis_tools(self, payload):
        self.plan_payloads.append(payload)
        self._record("analysis_agent_plan")
        value = self.plans.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def inspect_analysis_evidence(self, payload):
        self._record("analysis_agent_inspect")
        value = self.inspections.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class DisabledAgentLLM:
    enabled = False
    source = "disabled"
    configured_model = None


def _schema() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Product": "product_name",
        },
    )


def _fallback_plan() -> AnalysisPlan:
    return AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "discount_profit_analysis",
        ]
    )


def _budget() -> RunBudget:
    return RunBudget(demo_safe=False, total_seconds=None, min_stage_seconds=0)


def test_unknown_and_duplicate_tools_are_rejected(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[
            {
                "selected_tools": [
                    "missing_tool",
                    "sales_trend_analysis",
                    "sales_trend_analysis",
                ],
                "reasoning_summary": "trend",
            }
        ],
        inspections=[{"status": "enough", "reason": "done"}],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="analyze trend",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert [item.tool_name for item in result.state.executed_tools] == [
        "sales_trend_analysis"
    ]
    assert any("unknown_tool:missing_tool" in error for error in result.state.errors)
    assert any("duplicate_in_plan:sales_trend_analysis" in error for error in result.state.errors)


def test_harness_rejects_profit_tool_when_profit_is_not_mapped(
    sample_csv_path: Path,
) -> None:
    schema = _schema().model_copy(deep=True)
    schema.field_mapping = {
        original: canonical
        for original, canonical in schema.field_mapping.items()
        if canonical != "profit"
    }
    llm = FakeAgentLLM(
        plans=[
            {"selected_tools": ["discount_profit_analysis"], "reasoning_summary": "profit"},
            {"selected_tools": ["discount_profit_analysis"], "reasoning_summary": "retry"},
        ],
        inspections=[],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=schema,
        dataset_profile={"row_count": 10},
        user_goal="analyze profit",
        fallback_plan=AnalysisPlan(analysis_plan=[]),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.executed_tools == []
    assert any(
        "missing_required_fields:discount_profit_analysis:profit" in error
        for error in result.state.errors
    )
    assert result.state.termination_reason == "initial_plan_invalid_fallback"
    assert result.state.plan_corrections == 1


def test_max_rounds_terminates_normally(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[{"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "trend"}],
        inspections=[
            {
                "status": "need_more",
                "reason": "need discount",
                "missing_questions": ["discount"],
                "next_tools": ["discount_profit_analysis"],
            }
        ],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="analyze profit",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
        max_rounds=1,
    )

    assert result.state.termination_reason == "max_rounds"
    assert result.state.decision == "finalize"


def test_empty_initial_plan_uses_deterministic_fallback(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[{"selected_tools": [], "reasoning_summary": "none"}],
        inspections=[],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="analyze",
        fallback_plan=AnalysisPlan(analysis_plan=[]),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.termination_reason == "initial_plan_empty_fallback"
    assert result.report.module_count == 0


def test_llm_disabled_uses_deterministic_fallback(sample_csv_path: Path) -> None:
    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal=None,
        fallback_plan=_fallback_plan(),
        llm_client=DisabledAgentLLM(),
        run_budget=_budget(),
    )

    assert result.state.termination_reason == "llm_unavailable_fallback"
    assert result.report.module_count == 3


def test_planner_exception_uses_deterministic_fallback(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(plans=[RuntimeError("planner down")], inspections=[])

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="analyze",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.termination_reason == "planner_failure_fallback"
    assert result.report.module_count == 3


def test_planner_invalid_payload_uses_deterministic_fallback(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(plans=[{"unexpected": ["sales_trend_analysis"]}], inspections=[])

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="analyze",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.termination_reason == "planner_failure_fallback"
    assert any("ValidationError" in error for error in result.state.errors)


def test_inspector_exception_uses_deterministic_fallback(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[{"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "trend"}],
        inspections=[RuntimeError("inspector down")],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="analyze",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.termination_reason == "inspect_failure_fallback"
    assert result.report.module_count == 3


def test_agent_plan_execute_inspect_finalize(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[
            {
                "selected_tools": ["sales_trend_analysis", "discount_profit_analysis"],
                "reasoning_summary": "goal matched",
            }
        ],
        inspections=[{"status": "enough", "reason": "evidence is sufficient"}],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="why profit changed",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.node_trace == [
        "plan",
        "validate_plan",
        "execute_tools",
        "build_evidence",
        "inspect",
        "finalize",
    ]
    assert result.report.module_count == 2
    assert result.state.termination_reason == "evidence_sufficient"
    assert result.trace.summary["planner_calls"] == 1
    assert result.trace.summary["inspector_calls"] == 1
    assert result.trace.summary["llm_calls"] == 2
    assert result.trace.summary["total_prompt_chars"] == 200
    assert [event.sequence for event in result.trace.events] == list(
        range(1, len(result.trace.events) + 1)
    )
    validation_event = next(
        event for event in result.trace.events if event.node == "validate_plan"
    )
    assert validation_event.requested_tools == [
        "sales_trend_analysis",
        "discount_profit_analysis",
    ]
    assert validation_event.accepted_tools == validation_event.requested_tools


def test_agent_replans_for_incremental_evidence(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[
            {"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "trend"},
            {"selected_tools": ["product_contribution_analysis"], "reasoning_summary": "product"},
        ],
        inspections=[
            {
                "status": "need_more",
                "reason": "need profit evidence",
                "missing_questions": ["profit impact"],
                "next_tools": ["product_contribution_analysis"],
            },
            {"status": "enough", "reason": "complete"},
        ],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="why profit changed",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
        max_rounds=2,
    )

    assert result.state.round == 2
    assert [item.tool_name for item in result.state.executed_tools] == [
        "sales_trend_analysis",
        "product_contribution_analysis",
    ]
    assert result.state.node_trace.count("replan") == 1
    assert result.state.termination_reason == "evidence_sufficient"


def test_initial_all_rejected_plan_gets_one_successful_correction(
    sample_csv_path: Path,
) -> None:
    llm = FakeAgentLLM(
        plans=[
            {"selected_tools": ["invented_tool"], "reasoning_summary": "invalid"},
            {"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "corrected"},
        ],
        inspections=[{"status": "enough", "reason": "done"}],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="trend",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.plan_corrections == 1
    assert result.state.node_trace[:4] == [
        "plan",
        "validate_plan",
        "plan_correction",
        "validate_plan",
    ]
    assert result.state.termination_reason == "evidence_sufficient"
    assert result.trace.summary["fallback_used"] is False
    assert "unknown_tool:invented_tool" in llm.plan_payloads[1][
        "harness_rejection_reasons"
    ]


def test_replan_all_rejected_finalizes_existing_evidence(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[
            {"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "trend"},
            {"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "repeat"},
        ],
        inspections=[
            {
                "status": "need_more",
                "reason": "try again",
                "next_tools": ["sales_trend_analysis"],
            }
        ],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="trend",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.termination_reason == "replan_no_valid_tools"
    assert result.report.module_count == 1
    assert result.report.modules[0].module_id == "sales_trend_analysis"
    assert result.trace.summary["fallback_used"] is False
    assert any("duplicate_executed_tool" in error for error in result.state.errors)


def test_inspector_invalid_status_uses_fallback(sample_csv_path: Path) -> None:
    llm = FakeAgentLLM(
        plans=[{"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "trend"}],
        inspections=[{"status": "unknown", "reason": "bad"}],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="trend",
        fallback_plan=_fallback_plan(),
        llm_client=llm,
        run_budget=_budget(),
    )

    assert result.state.termination_reason == "inspect_failure_fallback"
    assert any("Invalid inspection status" in error for error in result.state.errors)


def test_budget_exhausted_before_plan_uses_fallback(sample_csv_path: Path) -> None:
    budget = RunBudget(demo_safe=False, total_seconds=0.001, min_stage_seconds=10)

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="trend",
        fallback_plan=_fallback_plan(),
        llm_client=FakeAgentLLM(plans=[], inspections=[]),
        run_budget=budget,
    )

    assert result.state.termination_reason == "budget_exhausted_fallback"
    assert result.trace.summary["llm_calls"] == 0


def test_tool_executor_exception_is_traced_and_falls_back(sample_csv_path: Path) -> None:
    base = get_analysis_tool_registry()
    specs = [base.get(name).model_copy(deep=True) for name in base.names()]

    def fail_executor(frame, canonical_columns):
        raise RuntimeError("injected tool failure")

    specs = [
        spec.model_copy(update={"executor": fail_executor})
        if spec.name == "sales_trend_analysis"
        else spec
        for spec in specs
    ]
    registry = AnalysisToolRegistry(specs)
    llm = FakeAgentLLM(
        plans=[{"selected_tools": ["sales_trend_analysis"], "reasoning_summary": "trend"}],
        inspections=[],
    )

    result = run_analysis_agent(
        task_id="task",
        csv_path=sample_csv_path,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10},
        user_goal="trend",
        fallback_plan=AnalysisPlan(analysis_plan=["data_quality_check"]),
        llm_client=llm,
        run_budget=_budget(),
        registry=registry,
    )

    assert result.state.termination_reason == "initial_execution_no_evidence_fallback"
    execution_event = next(
        event for event in result.trace.events if event.node == "execute_tools"
    )
    assert "injected tool failure" in execution_event.tool_errors["sales_trend_analysis"]
