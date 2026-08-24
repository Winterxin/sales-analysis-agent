from __future__ import annotations

from evals.run_analysis_agent_eval import run_golden_eval


def test_analysis_agent_golden_eval_passes_all_cases() -> None:
    report = run_golden_eval()

    assert report["case_count"] == 33
    assert report["case_pass_rate"] == 1.0, [
        case for case in report["cases"] if not case["passed"]
    ]
    assert report["forbidden_tool_violation_count"] == 0
    assert report["fallback_correctness"] == 33
    assert report["termination_correctness"] == 33
    assert report["replan_correctness"] == 33
    assert report["argument_validation_correctness"] == 8
    assert report["duplicate_signature_correctness"] == 3
    assert report["sufficiency_guard_correctness"] == 7
    assert report["capability_unavailable_correctness"] == 1
