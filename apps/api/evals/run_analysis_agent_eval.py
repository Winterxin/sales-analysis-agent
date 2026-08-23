from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from app.analysis.contracts import ModuleResult
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.schema_mapping import SchemaMapping
from app.services.analysis_agent.graph import run_analysis_agent
from app.services.analysis_tools import AnalysisToolRegistry, get_analysis_tool_registry
from app.services.run_budget import RunBudget


DEFAULT_CASES_PATH = Path(__file__).with_name("analysis_agent_golden_cases.json")


class ScriptedAgentLLM:
    enabled = True
    source = "golden_eval"
    configured_model = "scripted"

    def __init__(self, plans: list[object], inspections: list[object]) -> None:
        self.plans = list(plans)
        self.inspections = list(inspections)
        self.metrics: list[dict[str, object]] = []

    @property
    def last_completion_metrics(self) -> dict[str, object]:
        return dict(self.metrics[-1]) if self.metrics else {}

    def snapshot_completion_metrics(self) -> int:
        return len(self.metrics)

    def collect_completion_metrics_since(self, snapshot: int) -> list[dict[str, object]]:
        return [dict(item) for item in self.metrics[snapshot:]]

    def _next(self, values: list[object], stage: str) -> object:
        self.metrics.append(
            {
                "stage": stage,
                "elapsed_ms": 1.0,
                "prompt_chars": 100,
                "response_chars": 30,
                "attempt_count": 1,
                "cache_status": "bypassed",
                "remote_elapsed_ms": 1.0,
                "saved_ms": 0.0,
                "model": self.configured_model,
                "source": self.source,
            }
        )
        value = values.pop(0)
        if isinstance(value, dict) and "$exception" in value:
            raise RuntimeError(str(value["$exception"]))
        return value

    def plan_analysis_tools(self, payload: dict[str, object]) -> object:
        return self._next(self.plans, "analysis_agent_plan")

    def inspect_analysis_evidence(self, payload: dict[str, object]) -> object:
        return self._next(self.inspections, "analysis_agent_inspect")


class DisabledAgentLLM:
    enabled = False
    source = "disabled"
    configured_model = None


def _synthetic_frame(fields: list[str]) -> pd.DataFrame:
    rows = 40
    values: dict[str, object] = {
        "order_datetime": pd.date_range("2025-01-01", periods=rows, freq="D"),
        "sales_amount": [100.0 + index * 3 for index in range(rows)],
        "profit": [20.0 - (index % 5) * 8 for index in range(rows)],
        "discount": [(index % 4) * 0.1 for index in range(rows)],
        "product_name": [f"Product {index % 5}" for index in range(rows)],
        "quantity": [(index % 4) + 1 for index in range(rows)],
        "order_id": [f"order-{index}" for index in range(rows)],
        "customer_id": [f"customer-{index % 10}" for index in range(rows)],
        "country": ["CN" if index % 2 else "US" for index in range(rows)],
        "region": ["East" if index % 2 else "West" for index in range(rows)],
    }
    return pd.DataFrame({field: values[field] for field in fields})


def _registry_for_scenario(scenario: str | None) -> AnalysisToolRegistry | None:
    if scenario not in {"no_progress", "tool_failure", "partial_failure"}:
        return None
    base = get_analysis_tool_registry()
    specs = [base.get(name).model_copy(deep=True) for name in base.names()]

    def fail(frame, canonical_columns):
        raise RuntimeError("golden injected tool failure")

    def same_fact(module_id: str):
        def execute(frame, canonical_columns):
            return ModuleResult(
                module_id=module_id,
                title="Synthetic evidence",
                chart_type="bar",
                summary_metrics={"shared_metric": 1},
                findings=["The same normalized fact."],
            )

        return execute

    replacements = {}
    if scenario == "no_progress":
        replacements = {
            "data_quality_check": same_fact("data_quality_check"),
            "metric_distribution_analysis": same_fact("metric_distribution_analysis"),
        }
    elif scenario == "tool_failure":
        replacements = {"sales_trend_analysis": fail}
    elif scenario == "partial_failure":
        replacements = {"data_quality_check": fail}
    return AnalysisToolRegistry(
        [
            spec.model_copy(update={"executor": replacements[spec.name]})
            if spec.name in replacements
            else spec
            for spec in specs
        ]
    )


def _run_case(case: dict[str, Any], csv_path: Path) -> dict[str, object]:
    scenario = str(case.get("scenario") or "")
    llm_client = (
        DisabledAgentLLM()
        if scenario == "llm_disabled"
        else ScriptedAgentLLM(case.get("plans", []), case.get("inspections", []))
    )
    budget = (
        RunBudget(demo_safe=False, total_seconds=0.001, min_stage_seconds=10)
        if scenario == "budget_exhausted"
        else RunBudget(demo_safe=False, total_seconds=None, min_stage_seconds=0)
    )
    fields = list(case["schema_fields"])
    result = run_analysis_agent(
        task_id=str(case["case_id"]),
        csv_path=csv_path,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={field: field for field in fields},
        ),
        dataset_profile={"row_count": 40},
        user_goal=str(case["user_goal"]),
        fallback_plan=AnalysisPlan(analysis_plan=["data_quality_check"]),
        llm_client=llm_client,
        run_budget=budget,
        max_rounds=int(case.get("max_rounds", 2)),
        registry=_registry_for_scenario(scenario),
    )
    expected = case["expected"]
    actual_tools = [item.tool_name for item in result.state.executed_tools if item.success]
    forbidden = set(expected["forbidden_tools"])
    tool_calls = sum(
        len(event.accepted_tools)
        for event in result.trace.events
        if event.node == "execute_tools"
    ) + sum(
        len(event.executed_tools)
        for event in result.trace.events
        if event.node == "fallback"
    )
    checks = {
        "tools": sorted(actual_tools) == sorted(expected["expected_tools"]),
        "forbidden": not (forbidden & set(actual_tools)),
        "rounds": result.state.round == int(expected["expected_rounds"]),
        "replan": ("replan" in result.state.node_trace)
        is bool(expected["expected_replan"]),
        "fallback": bool(result.trace.summary["fallback_used"])
        is bool(expected["expected_fallback"]),
        "termination": result.state.termination_reason
        == expected["expected_termination_reason"],
        "tool_call_limit": tool_calls <= int(expected["max_tool_calls"]),
    }
    return {
        "case_id": case["case_id"],
        "passed": all(checks.values()),
        "checks": checks,
        "actual_tools": actual_tools,
        "termination_reason": result.state.termination_reason,
        "rounds": result.state.round,
        "replan": "replan" in result.state.node_trace,
        "fallback": result.trace.summary["fallback_used"],
        "tool_calls": tool_calls,
        "guard_errors": sum(
            len(event.guard_errors)
            for event in result.trace.events
            if event.node == "validate_plan"
        ),
    }


def run_golden_eval(cases_path: Path = DEFAULT_CASES_PATH) -> dict[str, object]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    with TemporaryDirectory(prefix="sales-agent-eval-") as temp_dir:
        for case in cases:
            csv_path = Path(temp_dir) / f"{case['case_id']}.csv"
            _synthetic_frame(list(case["schema_fields"])).to_csv(csv_path, index=False)
            results.append(_run_case(case, csv_path))
    count = len(results)
    return {
        "case_count": count,
        "case_pass_rate": round(
            sum(bool(item["passed"]) for item in results) / count if count else 0.0,
            4,
        ),
        "expected_tools_match": sum(
            bool(item["checks"]["tools"]) for item in results
        ),
        "forbidden_tool_violation_count": sum(
            not bool(item["checks"]["forbidden"]) for item in results
        ),
        "fallback_correctness": sum(
            bool(item["checks"]["fallback"]) for item in results
        ),
        "termination_correctness": sum(
            bool(item["checks"]["termination"]) for item in results
        ),
        "replan_correctness": sum(
            bool(item["checks"]["replan"]) for item in results
        ),
        "guard_violation_count": sum(int(item["guard_errors"]) for item in results),
        "average_tool_calls": round(
            sum(int(item["tool_calls"]) for item in results) / count if count else 0.0,
            3,
        ),
        "average_rounds": round(
            sum(int(item["rounds"]) for item in results) / count if count else 0.0,
            3,
        ),
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic analysis Agent golden evals.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()
    report = run_golden_eval(args.cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["case_pass_rate"] == 1.0 else 1)


if __name__ == "__main__":
    main()
