from __future__ import annotations

import pandas as pd

from app.analysis.modules import dimension_breakdown


def test_dimension_breakdown_outputs_profit_quality_and_weak_cuts() -> None:
    frame = pd.DataFrame(
        {
            "Segment": [
                "Consumer",
                "Consumer",
                "Corporate",
                "Corporate",
                "Home Office",
                "Home Office",
            ],
            "Region": ["East", "West", "East", "West", "East", "West"],
            "Sales": [500, 600, 700, 650, 800, 750],
            "Profit": [80, 90, 40, -60, 20, -120],
            "Quantity": [5, 6, 7, 6, 8, 7],
        }
    )

    result = dimension_breakdown.run(
        frame,
        {
            "segment": "Segment",
            "region": "Region",
            "sales_amount": "Sales",
            "profit": "Profit",
            "quantity": "Quantity",
        },
    )

    assert result.module_id == "dimension_breakdown_analysis"
    assert result.summary_metrics["primary_dimension"] == "segment"
    assert result.summary_metrics["secondary_dimension"] == "region"
    assert result.summary_metrics["top_slice_label"] == "Home Office / East"
    assert result.summary_metrics["high_sales_low_profit_cut_count"] >= 1
    assert result.tables["segment_region_matrix"][0]["profit_margin"] == 0.025
    assert result.tables["primary_profit_quality"]
    assert result.tables["weak_performance_cuts"]
    assert result.tables["top_performance_cuts"]
