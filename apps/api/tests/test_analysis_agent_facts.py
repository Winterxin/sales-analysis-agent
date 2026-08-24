from __future__ import annotations

from app.services.analysis_agent.facts import evidence_fact_set
from app.services.analysis_agent.state import AgentEvidence


def test_evidence_facts_ignore_round_and_evidence_id() -> None:
    first = AgentEvidence(
        evidence_id="round:1:tool:sales_trend_analysis",
        tool_name="sales_trend_analysis",
        round=1,
        summary_metrics={"growth": 0.125},
        findings=[" Sales grew 12.5%  month over month. "],
    )
    repeated = first.model_copy(
        update={"evidence_id": "round:2:tool:sales_trend_analysis", "round": 2}
    )

    assert evidence_fact_set([first]) == evidence_fact_set([repeated])


def test_evidence_facts_detect_new_metric_value() -> None:
    first = AgentEvidence(
        evidence_id="one",
        tool_name="sales_trend_analysis",
        round=1,
        summary_metrics={"growth": 0.125},
    )
    changed = first.model_copy(
        update={"evidence_id": "two", "round": 2, "summary_metrics": {"growth": 0.2}}
    )

    assert evidence_fact_set([changed]) - evidence_fact_set([first])
