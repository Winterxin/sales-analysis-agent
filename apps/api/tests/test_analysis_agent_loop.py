from __future__ import annotations

from pathlib import Path

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_agent.graph import run_analysis_agent
from app.services.run_budget import RunBudget


class FakeAgentLLM:
    enabled = True
    source = "fake"
    configured_model = "fake-agent"

    def __init__(self, plans: list[object], inspections: list[object]) -> None:
        self.plans = list(plans)
        self.inspections = list(inspections)

    def plan_analysis_tools(self, payload):
        value = self.plans.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def inspect_analysis_evidence(self, payload):
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
        plans=[{"selected_tools": ["discount_profit_analysis"], "reasoning_summary": "profit"}],
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
    assert result.state.termination_reason == "no_progress"


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


def test_no_evidence_progress_terminates(sample_csv_path: Path) -> None:
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

    assert result.state.termination_reason == "no_progress"
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
