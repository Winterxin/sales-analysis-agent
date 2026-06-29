from __future__ import annotations

import pandas as pd

from app.analysis.modules import dimension_breakdown


def test_dimension_breakdown_omits_missing_state_from_city_slice_label() -> None:
    frame = pd.DataFrame(
        [
            {"State": pd.NA, "City": "Madrid", "Sales": 1000.0, "Profit": 120.0},
        ]
    )

    result = dimension_breakdown.run(
        frame,
        {
            "state": "State",
            "city": "City",
            "sales_amount": "Sales",
            "profit": "Profit",
        },
    )

    label = result.summary_metrics["top_slice_label"]
    assert label == "Madrid"
    assert "nan" not in label.lower()
    assert "none" not in label.lower()
    assert "<na>" not in label.lower()


def test_dimension_breakdown_keeps_state_and_city_slice_label() -> None:
    frame = pd.DataFrame(
        [
            {"State": "CA", "City": "Los Angeles", "Sales": 1000.0, "Profit": 120.0},
        ]
    )

    result = dimension_breakdown.run(
        frame,
        {
            "state": "State",
            "city": "City",
            "sales_amount": "Sales",
            "profit": "Profit",
        },
    )

    assert result.summary_metrics["top_slice_label"] == "CA / Los Angeles"
