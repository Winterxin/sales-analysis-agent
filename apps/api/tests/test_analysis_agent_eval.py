from __future__ import annotations

from evals.run_analysis_agent_eval import run_golden_eval


def test_analysis_agent_golden_eval_passes_all_cases() -> None:
    report = run_golden_eval()

    assert report["case_count"] == 18
    assert report["case_pass_rate"] == 1.0, [
        case for case in report["cases"] if not case["passed"]
    ]
    assert report["forbidden_tool_violation_count"] == 0
    assert report["fallback_correctness"] == 18
    assert report["termination_correctness"] == 18
    assert report["replan_correctness"] == 18
