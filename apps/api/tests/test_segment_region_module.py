from __future__ import annotations

import pandas as pd

from app.analysis.modules import dimension_breakdown


def test_dimension_breakdown_builds_segment_region_matrix_and_weak_dimensions() -> None:
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
            "Sales": [200, 120, 180, 90, 80, 60],
            "Profit": [40, 5, 35, -10, 8, -12],
        }
    )

    result = dimension_breakdown.run(
        frame,
        {
            "segment": "Segment",
            "region": "Region",
            "sales_amount": "Sales",
            "profit": "Profit",
        },
    )

    assert result.module_id == "dimension_breakdown_analysis"
    assert result.summary_metrics["primary_dimension"] == "segment"
    assert result.summary_metrics["secondary_dimension"] == "region"
    assert result.tables["segment_region_matrix"]
    assert result.tables["weak_performance_cuts"]
    assert any("客群" in finding or "segment" in finding.lower() for finding in result.findings)
