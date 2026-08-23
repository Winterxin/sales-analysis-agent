from __future__ import annotations

from app.services.analysis_agent.guards import validate_tool_calls
from app.services.analysis_agent.tool_calls import AnalysisToolCall, normalized_call_signature
from app.services.analysis_tools import get_analysis_tool_registry


FIELDS = {
    "order_datetime",
    "sales_amount",
    "profit",
    "product_name",
    "region",
}


def _validate(calls, *, fields=FIELDS, executed=frozenset()):
    return validate_tool_calls(
        calls,
        registry=get_analysis_tool_registry(),
        available_fields=set(fields),
        dataset_profile={"row_count": 100},
        executed_successfully=set(executed),
        max_tools_per_round=4,
        has_budget=True,
    )


def test_dimension_argument_is_accepted_when_dataset_supports_it() -> None:
    result = _validate(
        [
            AnalysisToolCall(
                tool_name="dimension_breakdown_analysis",
                arguments={"dimension": "region", "metric": "profit", "top_n": 10},
            )
        ]
    )

    assert result.errors == []
    assert result.accepted_tool_calls[0].arguments["dimension"] == "region"


def test_dimension_argument_is_rejected_when_field_is_unmapped() -> None:
    result = _validate(
        [
            AnalysisToolCall(
                tool_name="dimension_breakdown_analysis",
                arguments={"dimension": "region"},
            )
        ],
        fields=(FIELDS - {"region"}) | {"country"},
    )

    assert "unsupported_dataset_argument:dimension_breakdown_analysis:dimension:region" in result.errors


def test_profit_metric_is_rejected_when_profit_is_unmapped() -> None:
    result = _validate(
        [
            AnalysisToolCall(
                tool_name="dimension_breakdown_analysis",
                arguments={"dimension": "region", "metric": "profit"},
            )
        ],
        fields=FIELDS - {"profit"},
    )

    assert "unsupported_dataset_argument:dimension_breakdown_analysis:metric:profit" in result.errors


def test_out_of_range_and_unknown_arguments_are_rejected() -> None:
    too_large = _validate(
        [AnalysisToolCall(tool_name="product_contribution_analysis", arguments={"top_n": 5000})]
    )
    unknown = _validate(
        [AnalysisToolCall(tool_name="sales_trend_analysis", arguments={"timezone": "UTC"})]
    )

    assert any("less_than_equal" in error for error in too_large.errors)
    assert any("extra_forbidden" in error for error in unknown.errors)


def test_same_signature_is_duplicate_but_distinct_arguments_are_allowed() -> None:
    first = AnalysisToolCall(
        tool_name="dimension_breakdown_analysis",
        arguments={"dimension": "region", "metric": "profit", "top_n": 10},
    )
    signature = normalized_call_signature(first.tool_name, first.arguments)

    duplicate = _validate([first], executed={signature})
    distinct = _validate(
        [
            AnalysisToolCall(
                tool_name="dimension_breakdown_analysis",
                arguments={"dimension": "region", "metric": "sales_amount", "top_n": 10},
            )
        ],
        executed={signature},
    )

    assert any("duplicate_executed_tool_call" in error for error in duplicate.errors)
    assert len(distinct.accepted_tool_calls) == 1


def test_signature_is_independent_of_argument_key_order() -> None:
    left = normalized_call_signature(
        "dimension_breakdown_analysis",
        {"dimension": "region", "metric": "profit", "top_n": 10},
    )
    right = normalized_call_signature(
        "dimension_breakdown_analysis",
        {"top_n": 10, "metric": "profit", "dimension": "region"},
    )

    assert left == right
