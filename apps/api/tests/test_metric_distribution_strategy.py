from __future__ import annotations

from app.schemas.report import ModuleReport
from app.services.metric_distribution_strategy import build_metric_distribution_strategy


def _metric_module() -> ModuleReport:
    return ModuleReport(
        module_id="metric_distribution_analysis",
        title="Metric Distribution",
        chart_type="histogram",
        findings=[],
        summary_metrics={"metric_count": 2},
        tables={
            "metric_quantiles": [
                {
                    "metric": "sales_amount",
                    "column": "Sales",
                    "label": "销售额",
                    "count": 9994,
                    "mean": 229.86,
                    "median": 54.49,
                    "p90": 572.71,
                    "p95": 956.98,
                    "p99": 2481.69,
                    "min": 0.44,
                    "max": 22638.48,
                    "skew": 12.97,
                    "mean_median_ratio": 4.22,
                    "p99_median_ratio": 45.55,
                    "non_negative": True,
                },
                {
                    "metric": "profit",
                    "column": "Profit",
                    "label": "利润",
                    "count": 9994,
                    "mean": 28.65,
                    "median": 8.67,
                    "p90": 89.28,
                    "p95": 168.47,
                    "p99": 580.66,
                    "min": -6599.98,
                    "max": 8399.98,
                    "skew": 7.56,
                    "mean_median_ratio": 3.3,
                    "p99_median_ratio": 66.97,
                    "non_negative": False,
                },
            ],
            "metric_outliers": [
                {
                    "metric": "sales_amount",
                    "column": "Sales",
                    "label": "销售额",
                    "outlier_ratio": 0.1167,
                },
                {
                    "metric": "profit",
                    "column": "Profit",
                    "label": "利润",
                    "outlier_ratio": 0.1882,
                },
            ],
            "relationship_candidates": [
                {
                    "x_metric": "discount",
                    "x_column": "Discount",
                    "x_label": "折扣",
                    "y_metric": "profit",
                    "y_column": "Profit",
                    "y_label": "利润",
                    "correlation": -0.62,
                    "abs_correlation": 0.62,
                    "business_priority": 1.0,
                    "observation_count": 9994,
                    "color_metric": "sales_amount",
                    "color_column": "Sales",
                    "color_label": "销售额",
                },
                {
                    "x_metric": "sales_amount",
                    "x_column": "Sales",
                    "x_label": "销售额",
                    "y_metric": "profit",
                    "y_column": "Profit",
                    "y_label": "利润",
                    "correlation": 0.41,
                    "abs_correlation": 0.41,
                    "business_priority": 0.8,
                    "observation_count": 9994,
                    "color_metric": "discount",
                    "color_column": "Discount",
                    "color_label": "折扣",
                },
            ],
        },
    )


def test_fallback_strategy_recommends_log_and_p99_for_long_tailed_sales() -> None:
    strategy = build_metric_distribution_strategy(_metric_module(), llm_client=None)

    sales_decision = next(item for item in strategy["decisions"] if item["metric"] == "sales_amount")

    assert sales_decision["decision"] == "use_log_and_p99_clipped"
    assert "log_histogram" in sales_decision["charts"]
    assert "p99_clipped_histogram" in sales_decision["charts"]
    assert "quantile_markers" in sales_decision["charts"]
    assert "raw_histogram" not in sales_decision["charts"]
    assert "right skew" not in sales_decision["reason"].lower()
    assert any(
        view["chart"] == "correlation_heatmap" for view in strategy["relationship_views"]
    )
    scatter_view = next(
        view for view in strategy["relationship_views"] if view["chart"] == "scatter_relationships"
    )
    assert len(scatter_view["pairs"]) == 2
    assert scatter_view["pairs"][0]["x_metric"] == "discount"
    assert scatter_view["pairs"][0]["y_metric"] == "profit"


class UnsafeMetricStrategyLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.last_payload = None

    def suggest_metric_distribution_strategy(self, payload):
        self.last_payload = payload
        return {
            "decisions": [
                {
                    "metric": "sales_amount",
                    "decision": "invented_decision",
                    "charts": ["log_histogram", "delete_files", "p99_clipped_histogram"],
                    "reason": "Sales is long-tailed, so use a safer transformed view.",
                }
            ],
            "relationship_views": [
                {
                    "chart": "scatter_relationships",
                    "pairs": [
                        {
                            "x_metric": "discount",
                            "y_metric": "profit",
                            "color_metric": "sales_amount",
                        },
                        {
                            "x_metric": "delete_files",
                            "y_metric": "profit",
                        },
                    ],
                    "reason": "Discount and profit deserve a relationship view.",
                },
                {
                    "chart": "shell_exec",
                    "pairs": [],
                    "reason": "unsafe",
                },
            ],
        }


def test_llm_strategy_is_sanitized_to_safe_chart_candidates() -> None:
    llm = UnsafeMetricStrategyLLM()

    strategy = build_metric_distribution_strategy(_metric_module(), llm_client=llm)

    sales_decision = next(item for item in strategy["decisions"] if item["metric"] == "sales_amount")
    assert "delete_files" not in sales_decision["charts"]
    assert sales_decision["decision"] == "use_log_and_p99_clipped"
    assert any(
        view["chart"] == "correlation_heatmap" for view in strategy["relationship_views"]
    )
    scatter_view = next(
        view for view in strategy["relationship_views"] if view["chart"] == "scatter_relationships"
    )
    assert len(scatter_view["pairs"]) == 1
    assert scatter_view["pairs"][0]["x_metric"] == "discount"
    assert scatter_view["pairs"][0]["y_metric"] == "profit"
    assert llm.last_payload["allowed_charts"] == [
        "raw_histogram",
        "log_histogram",
        "p99_clipped_histogram",
        "boxplot",
        "quantile_markers",
        "correlation_heatmap",
        "scatter_relationships",
    ]
