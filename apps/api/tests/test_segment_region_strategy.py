from __future__ import annotations

from app.schemas.report import ModuleReport
from app.services.segment_region_strategy import build_segment_region_strategy


def test_segment_region_strategy_selects_profit_quality_views() -> None:
    module = ModuleReport(
        module_id="dimension_breakdown_analysis",
        title="客群与区域分析",
        chart_type="stacked_bar",
        summary_metrics={
            "primary_dimension": "segment",
            "secondary_dimension": "region",
            "high_sales_low_profit_cut_count": 2,
        },
        tables={
            "segment_region_matrix": [
                {
                    "Segment": "Home Office",
                    "Region": "West",
                    "Sales": 750.0,
                    "Profit": -120.0,
                    "profit_margin": -0.16,
                }
            ],
            "primary_profit_quality": [
                {
                    "Segment": "Home Office",
                    "Sales": 1550.0,
                    "Profit": -100.0,
                    "profit_margin": -0.0645,
                }
            ],
            "weak_performance_cuts": [
                {
                    "Segment": "Home Office",
                    "Region": "West",
                    "Sales": 750.0,
                    "Profit": -120.0,
                    "profit_margin": -0.16,
                }
            ],
        },
    )

    strategy = build_segment_region_strategy(module)
    charts = [view["chart"] for view in strategy["views"]]

    assert strategy["source"] == "deterministic_fallback"
    assert len(charts) <= 4
    assert "segment_region_sales_heatmap" in charts
    assert "segment_region_profit_margin_heatmap" in charts
    assert "segment_region_bubble" in charts
    assert "weak_segment_region_bar" in charts


class FocusedSegmentRegionStrategyLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_segment_region_strategy(self, payload):
        return {
            "views": [
                {
                    "chart": "segment_region_profit_margin_heatmap",
                    "reason": "Profit margin heatmap shows weak combinations directly.",
                },
                {
                    "chart": "segment_region_bubble",
                    "reason": "Bubble view bridges scale and profit quality.",
                },
                {
                    "chart": "weak_segment_region_bar",
                    "reason": "Weak cut bar turns the diagnosis into action.",
                },
            ]
        }


def test_llm_segment_region_strategy_does_not_append_unselected_fallback_views() -> None:
    module = ModuleReport(
        module_id="dimension_breakdown_analysis",
        title="客群与区域分析",
        chart_type="stacked_bar",
        summary_metrics={"primary_dimension": "segment", "secondary_dimension": "region"},
        tables={
            "dimension_totals": [{"Segment": "Consumer", "Sales": 1000.0, "Profit": 50.0}],
            "segment_region_matrix": [
                {
                    "Segment": "Consumer",
                    "Region": "Central",
                    "Sales": 800.0,
                    "Profit": -20.0,
                    "profit_margin": -0.025,
                }
            ],
            "primary_profit_quality": [
                {"Segment": "Consumer", "Sales": 1000.0, "Profit": 50.0, "profit_margin": 0.05}
            ],
            "weak_performance_cuts": [
                {
                    "Segment": "Consumer",
                    "Region": "Central",
                    "Sales": 800.0,
                    "Profit": -20.0,
                    "profit_margin": -0.025,
                }
            ],
        },
    )

    strategy = build_segment_region_strategy(
        module,
        llm_client=FocusedSegmentRegionStrategyLLM(),
    )

    assert strategy["source"] == "llm_sanitized"
    assert [view["chart"] for view in strategy["views"]] == [
        "segment_region_profit_margin_heatmap",
        "segment_region_bubble",
        "weak_segment_region_bar",
    ]
