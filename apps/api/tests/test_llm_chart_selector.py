from __future__ import annotations

from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_chart_selector import build_llm_chart_selection_plan


class FakeLLM:
    enabled = True
    source = "fake"
    configured_model = "fake-model"

    def __init__(
        self,
        selected_chart_ids: list[str],
        *,
        remove_as_redundant: dict[str, str] | None = None,
        replace: list[dict[str, str]] | None = None,
        base_chart_audit: dict[str, dict[str, str]] | None = None,
        remove_only: dict[str, str] | None = None,
    ):
        self.selected_chart_ids = selected_chart_ids
        self.payloads: list[dict[str, object]] = []
        self.remove_as_redundant = remove_as_redundant or {}
        self.replace = replace or []
        self.base_chart_audit = base_chart_audit or {}
        self.remove_only = remove_only or {}

    def suggest_llm_chart_selection(self, payload: dict[str, object]) -> dict[str, object]:
        self.payloads.append(payload)
        return {
            "selected_chart_ids": self.selected_chart_ids,
            "selection_rationale": {
                chart_id: f"LLM reason for {chart_id}" for chart_id in self.selected_chart_ids
            },
            "kept_for_unique_value": {
                chart_id: f"Unique value for {chart_id}" for chart_id in self.selected_chart_ids
            },
            "base_chart_audit": self.base_chart_audit,
            "remove_as_redundant": self.remove_as_redundant,
            "replace": self.replace,
            "remove_only": self.remove_only,
            "redundancy_summary": "Compressed redundant base charts.",
            "overall_strategy": "Use the strongest safe chart set.",
            "rejected_high_specific_notes": {},
        }


def _candidate(
    chart_id: str,
    section_id: str = "sales_trends",
    *,
    availability_status: str = "selected",
    chart_kind: str | None = None,
    intent_guided_candidate: bool = False,
    dataset_specificity_score: float = 0.4,
) -> dict[str, object]:
    return {
        "chart_id": chart_id,
        "section_id": section_id,
        "chart_kind": chart_kind or ("bar" if section_id != "sales_trends" else "line"),
        "title": chart_id.replace("_", " ").title(),
        "business_question": f"What is the business value of {chart_id}?",
        "required_fields": ["sales_amount"],
        "section_role": "core",
        "availability_status": availability_status,
        "score": 0.8,
        "ranking_factors": ["evidence"],
        "insight_type": chart_id,
        "dataset_specificity_score": dataset_specificity_score,
        "intent_guided_candidate": intent_guided_candidate,
        "best_intent_match_quality": "strong_match" if intent_guided_candidate else None,
        "intent_support_score": 0.25 if intent_guided_candidate else 0,
    }


def _plan() -> tuple[SchemaMapping, dict[str, object], list[dict[str, object]]]:
    candidates = [
        _candidate("trend_line", "sales_trends"),
        _candidate("product_bar", "product_and_category"),
        _candidate("country_bar", "country_market"),
        _candidate("quantity_distribution", "metric_distributions"),
        _candidate("deal_size_sales_bar", "order_structure"),
        _candidate("deal_size_avg_order_value_bar", "order_structure"),
        _candidate("region_sales_profit_bar", "segment_and_region"),
        _candidate("profit_distribution_if_available", "metric_distributions"),
        _candidate(
            "profit_scatter_challenger",
            "product_and_category",
            availability_status="available",
            chart_kind="scatter",
            intent_guided_candidate=True,
            dataset_specificity_score=0.72,
        ),
        {
            **_candidate(
                "missing_field_challenger",
                "product_and_category",
                availability_status="available",
                chart_kind="scatter",
                intent_guided_candidate=True,
                dataset_specificity_score=0.78,
            ),
            "required_fields": ["missing_field"],
        },
    ]
    selected = [
        {
            **candidate,
            "reason": f"Base reason for {candidate['chart_id']}",
            "selection_source": "evidence_ranker",
            "evidence_ids": [f"evidence_{candidate['chart_id']}"],
            "evidence_summary": f"Evidence summary for {candidate['chart_id']}",
        }
        for candidate in candidates
        if candidate["availability_status"] == "selected"
    ]
    plan = {
        "candidate_charts": candidates,
        "selected_charts": selected,
        "deterministic_selected_charts": selected,
        "rejected_charts": [],
        "chart_budget": {
            "core_max_per_section": 2,
            "support_max_per_section": 2,
            "skipped_max_per_section": 0,
        },
        "diagnostics": {},
        "selection_mode": "deterministic",
        "llm_selection_trace": {},
    }
    mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount"},
        confidence=1.0,
    )
    return mapping, plan, candidates


def test_llm_sanitized_selection_preserves_base_evidence_fields() -> None:
    mapping, plan, candidates = _plan()

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=FakeLLM(["trend_line", "product_bar", "country_bar", "quantity_distribution"]),
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    by_id = {chart["chart_id"]: chart for chart in result["selected_charts"]}
    assert by_id["trend_line"]["evidence_ids"] == ["evidence_trend_line"]
    assert by_id["trend_line"]["evidence_summary"] == "Evidence summary for trend_line"
    assert by_id["trend_line"]["llm_selection_reason"] == "LLM reason for trend_line"


def test_llm_sanitized_selection_does_not_restore_base_charts_when_guard_disabled() -> None:
    mapping, plan, candidates = _plan()

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=FakeLLM(["trend_line", "product_bar"]),
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    trace = result["llm_selection_trace"]
    assert trace["minimum_chart_count"] == 0
    assert trace["chart_count_guard_applied"] is False
    assert trace["chart_count_before_guard"] == 2
    assert trace["chart_count_after_guard"] == 2
    assert trace["restored_base_chart_ids"] == []
    assert trace["chart_count_guard_reason"] == "disabled_no_mechanical_restoration"
    assert [chart["chart_id"] for chart in result["selected_charts"]] == ["trend_line", "product_bar"]


def test_llm_selector_payload_includes_redundancy_review_context() -> None:
    mapping, plan, candidates = _plan()
    llm = FakeLLM(["trend_line", "profit_scatter_challenger"])

    build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=llm,
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    payload = llm.payloads[0]
    assert "chart_value_audit" in payload
    assert "base_chart_value_review" in payload
    assert "profit_scatter_challenger" in [item["chart_id"] for item in payload["high_value_challengers"]]
    assert [item["chart_id"] for item in payload["base_selected_charts"]] == [
        chart["chart_id"] for chart in plan["selected_charts"]
    ]
    assert [item["chart_id"] for item in payload["intent_supported_challengers"]] == [
        "profit_scatter_challenger"
    ]
    assert "profit_scatter_challenger" in [
        item["chart_id"] for item in payload["high_specificity_unselected_candidates"]
    ]
    assert "profit_scatter_challenger" in [item["chart_id"] for item in payload["non_bar_challengers"]]
    assert "same_section_same_kind" in payload["redundancy_groups"]
    assert payload["redundancy_pressure"]["bar_chart_count"] >= 1


def test_llm_selector_traces_chart_value_audit_decisions() -> None:
    mapping, plan, candidates = _plan()

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=FakeLLM(
            ["trend_line", "quantity_distribution"],
            remove_as_redundant={"product_bar": "Low visual gain and redundant product section bar."},
        ),
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    trace = result["llm_selection_trace"]
    assert "chart_value_audit_summary" in trace
    assert "product_bar" in trace["low_visual_gain_risk_chart_ids"]
    assert "product_bar" in trace["removed_low_visual_gain_chart_ids"]
    assert trace["average_value_score_before"] >= 0
    assert trace["average_value_score_after"] >= 0
    assert "quantity_distribution" in trace["selected_chart_value_scores"]


def test_llm_selector_records_redundant_removal_and_challenger_replacement() -> None:
    mapping, plan, candidates = _plan()
    llm = FakeLLM(
        ["trend_line", "profit_scatter_challenger"],
        remove_as_redundant={"product_bar": "Product bar overlaps with stronger scatter challenger."},
        replace=[
            {
                "remove": "product_bar",
                "add": "profit_scatter_challenger",
                "reason": "Scatter explains profit relationship with more incremental value.",
            }
        ],
    )

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=llm,
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    final_ids = [chart["chart_id"] for chart in result["selected_charts"]]
    trace = result["llm_selection_trace"]
    assert "product_bar" not in final_ids
    assert "profit_scatter_challenger" in final_ids
    assert trace["remove_as_redundant"] == {
        "product_bar": "Product bar overlaps with stronger scatter challenger."
    }
    assert trace["replacement_decisions"] == [
        {
            "remove": "product_bar",
            "add": "profit_scatter_challenger",
            "reason": "Scatter explains profit relationship with more incremental value.",
        }
    ]
    assert "product_bar" in trace["base_removed_chart_ids"]
    assert trace["newly_selected_challenger_ids"] == ["profit_scatter_challenger"]
    assert trace["base_retention_ratio"] < 1
    assert trace["final_bar_chart_ratio"] < trace["redundancy_pressure_before"]["bar_chart_ratio"]


def test_llm_selector_removes_standalone_challenger_without_valid_replacement() -> None:
    mapping, plan, candidates = _plan()

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=FakeLLM(["trend_line", "product_bar", "profit_scatter_challenger"]),
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    trace = result["llm_selection_trace"]
    assert [chart["chart_id"] for chart in result["selected_charts"]] == ["trend_line", "product_bar"]
    assert trace["standalone_added_candidate_ids"] == ["profit_scatter_challenger"]
    assert trace["selected_non_base_without_valid_replace"] == ["profit_scatter_challenger"]
    assert trace["replacement_consistency_status"] == "fixed"
    assert trace["final_chart_count_le_base"] is True


def test_llm_selector_records_true_replacement_only_when_add_is_final() -> None:
    mapping, plan, candidates = _plan()

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=FakeLLM(
            ["trend_line", "profit_scatter_challenger"],
            replace=[{"remove": "product_bar", "add": "profit_scatter_challenger", "reason": "Better fit."}],
        ),
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    trace = result["llm_selection_trace"]
    assert trace["true_replacements"] == [
        {"remove": "product_bar", "add": "profit_scatter_challenger", "reason": "Better fit."}
    ]
    assert trace["suggested_replacements_not_applied"] == []
    assert trace["replacement_consistency_status"] == "ok"


def test_llm_selector_suggested_replacement_not_applied_is_remove_only() -> None:
    mapping, plan, candidates = _plan()

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=FakeLLM(
            ["trend_line"],
            replace=[{"remove": "product_bar", "add": "profit_scatter_challenger", "reason": "Suggested but omitted."}],
        ),
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    trace = result["llm_selection_trace"]
    assert trace["true_replacements"] == []
    assert trace["suggested_replacements_not_applied"] == [
        {"remove": "product_bar", "add": "profit_scatter_challenger", "reason": "Suggested but omitted."}
    ]
    assert "product_bar" in trace["remove_only_chart_ids"]


def test_llm_selector_records_invalid_replacement_for_missing_fields() -> None:
    mapping, plan, candidates = _plan()

    result = build_llm_chart_selection_plan(
        schema_mapping=mapping,
        dataset_profile={},
        analysis_focus={"selected_focuses": []},
        section_priority={"core_sections": [], "support_sections": [], "skipped_sections": []},
        evidence_pack={"dataset_signature": {}},
        deterministic_chart_selection_plan=plan,
        diagnostics={},
        candidate_charts=candidates,
        llm_client=FakeLLM(
            ["trend_line"],
            replace=[{"remove": "product_bar", "add": "missing_field_challenger", "reason": "Unsafe add."}],
        ),
        llm_profile="full",
        llm_chart_selection_enabled=True,
        rebuild_diagnostics=lambda selected: {"selected_chart_ids": [item["chart_id"] for item in selected]},
    )

    trace = result["llm_selection_trace"]
    assert trace["invalid_replacements"] == [
        {
            "remove": "product_bar",
            "add": "missing_field_challenger",
            "reason": "Unsafe add.",
            "invalid_reason": "unsafe_add_candidate",
        }
    ]
    assert trace["true_replacements"] == []
