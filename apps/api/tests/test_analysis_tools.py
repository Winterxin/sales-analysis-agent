from __future__ import annotations

from app.services.analysis_tools import get_analysis_tool_registry


def test_discount_profit_tool_available_with_profit_and_discount() -> None:
    registry = get_analysis_tool_registry()

    available, missing = registry.is_available(
        "discount_profit_analysis", {"sales_amount", "profit", "discount"}
    )

    assert available is True
    assert missing == []


def test_profit_dependent_tool_rejected_without_profit() -> None:
    registry = get_analysis_tool_registry()

    available, missing = registry.is_available(
        "discount_profit_analysis", {"sales_amount", "discount"}
    )

    assert available is False
    assert missing == ["profit"]
