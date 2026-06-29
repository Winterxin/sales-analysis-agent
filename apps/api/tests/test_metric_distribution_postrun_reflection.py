from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.services.notebook_postrun_reflection import (
    build_postrun_chart_reflections,
    extract_postrun_chart_contexts,
)


def test_extract_postrun_chart_contexts_collects_metric_distribution_charts(
    tmp_path: Path,
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 指标分布与异常值分析"),
            new_code_cell(
                "metric_summary = metric_profile.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).transpose()\n"
                "metric_summary",
                outputs=[
                    new_output(
                        "execute_result",
                        data={
                            "text/plain": (
                                "          count    mean     50%      90%       99%\n"
                                "Sales    9994.0  229.85  54.49  572.706  2481.6946\n"
                                "Profit   9994.0   28.65   8.67   89.281   580.6578"
                            )
                        },
                        execution_count=5,
                    )
                ],
                execution_count=5,
            ),
            new_code_cell(
                "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
                "for ax, col in zip(axes.flatten(), available_metric_columns):\n"
                "    sns.histplot(metric_profile[col].dropna(), kde=True, ax=ax)\n"
                "plt.suptitle('核心数值指标分布画像', y=1.02)\n"
                "plt.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "image/png": "abc123",
                            "text/plain": "<Figure size 1200x800 with 4 Axes>",
                        },
                    )
                ],
                execution_count=6,
            ),
            new_code_cell(
                "metric_long = metric_profile.rename(columns=metric_columns).melt(var_name='指标', value_name='数值').dropna()\n"
                "plt.figure(figsize=(10, 5))\n"
                "sns.boxplot(data=metric_long, x='指标', y='数值')\n"
                "plt.title('核心数值指标箱线图')\n"
                "plt.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "image/png": "def456",
                            "text/plain": "<Figure size 1000x500 with 1 Axes>",
                        },
                    )
                ],
                execution_count=7,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="metric_distributions",
                title="指标分布与异常值分析",
                purpose="Profile numeric metrics.",
            )
        ],
    )

    contexts = extract_postrun_chart_contexts(notebook_path, outline)

    assert len(contexts) == 2
    assert contexts[0]["section_id"] == "metric_distributions"
    assert contexts[0]["chart_title"] == "核心数值指标分布画像"
    assert contexts[0]["chart_kind"] == "histogram"
    assert "Sales" in contexts[0]["table_preview"]
    assert contexts[1]["chart_title"] == "核心数值指标箱线图"
    assert contexts[1]["chart_kind"] == "boxplot"


class MetricDistributionContextLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.last_context = None

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        self.last_context = chart_context
        return {
            "reflection_markdown": (
                "核心数值指标分布画像显示销售额和利润都存在明显长尾，其中 Sales 的 P99 达到 2481.69，"
                "远高于中位数 54.49。这个信号说明后续趋势和商品贡献不能只看均值，需要把异常大单与常规订单分开复盘，"
                "否则少数极端订单会掩盖真实经营结构。"
            )
        }


def test_build_postrun_chart_reflections_sends_metric_distribution_context_to_llm() -> None:
    chart_contexts = [
        {
            "section_id": "metric_distributions",
            "section_title": "指标分布与异常值分析",
            "chart_title": "核心数值指标分布画像",
            "table_preview": "Sales 9994.0 229.85 54.49 572.706 2481.6946",
            "code_source": "sns.histplot(metric_profile[col].dropna(), kde=True, ax=ax)",
            "cell_index": 6,
            "output_formats": ["image/png"],
            "chart_kind": "histogram",
            "chart_summary": "核心数值指标分布画像用于观察分布形态、偏态和长尾。",
        }
    ]
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="metric_distribution_analysis",
                title="指标分布分析",
                chart_type="histogram",
                findings=["已完成 4 个核心数值指标的分布画像。"],
                summary_metrics={
                    "metric_count": 4,
                    "highest_tail_metric": "sales_amount",
                },
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
                            "p99": 2481.6946,
                            "min": 0.44,
                            "max": 22638.48,
                            "skew": 12.97,
                            "mean_median_ratio": 4.22,
                            "p99_median_ratio": 45.55,
                            "non_negative": True,
                        }
                    ],
                    "metric_outliers": [
                        {
                            "metric": "profit",
                            "label": "利润",
                            "outlier_ratio": 0.1882,
                        }
                    ],
                },
            )
        ],
    )
    llm = MetricDistributionContextLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        llm_client=llm,
    )

    assert trace.status == "llm_applied"
    assert "P99" in reflections[6]
    assert llm.last_context["module_summary_metrics"]["highest_tail_metric"] == "sales_amount"
    assert "metric_quantiles" in llm.last_context["module_tables_preview"]
    assert "metric_outliers" in llm.last_context["module_tables_preview"]
    strategy = llm.last_context["metric_distribution_strategy"]
    sales_decision = next(item for item in strategy["decisions"] if item["metric"] == "sales_amount")
    assert sales_decision["decision"] == "use_log_and_p99_clipped"
    assert "log_histogram" in sales_decision["charts"]
    assert "p99_clipped_histogram" in sales_decision["charts"]
    assert any(
        view["chart"] == "correlation_heatmap" for view in strategy["relationship_views"]
    )


def test_build_postrun_chart_reflections_uses_metric_distribution_fallback_when_llm_disabled() -> None:
    chart_contexts = [
        {
            "section_id": "metric_distributions",
            "section_title": "指标分布与异常值分析",
            "chart_title": "核心数值指标分布画像",
            "table_preview": "Sales 9994.0 229.85 54.49 572.706 2481.6946",
            "code_source": "sns.histplot(metric_profile[col].dropna(), kde=True, ax=ax)",
            "cell_index": 6,
            "output_formats": ["image/png"],
            "chart_kind": "histogram",
            "chart_summary": "核心数值指标分布画像用于观察分布形态、偏态和长尾。",
        }
    ]
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="metric_distribution_analysis",
                title="指标分布分析",
                chart_type="histogram",
                findings=["异常值占比最高的指标是利润，占比约 0.1882。"],
                summary_metrics={
                    "metric_count": 4,
                    "highest_tail_metric": "sales_amount",
                    "highest_outlier_metric": "profit",
                },
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
                            "p99": 2481.6946,
                            "min": 0.44,
                            "max": 22638.48,
                            "skew": 12.97,
                            "mean_median_ratio": 4.22,
                            "p99_median_ratio": 45.55,
                            "non_negative": True,
                        }
                    ],
                    "metric_outliers": [
                        {
                            "metric": "profit",
                            "label": "利润",
                            "outlier_ratio": 0.1882,
                        }
                    ],
                },
            )
        ],
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        llm_client=None,
    )

    assert trace.status == "fallback"
    assert 6 in reflections
    assert "P99" in reflections[6]
    assert "18.82%" in reflections[6]


def test_metric_distribution_fallback_uses_strategy_context_when_rich() -> None:
    chart_contexts = [
        {
            "section_id": "metric_distributions",
            "section_title": "指标分布与异常值分析",
            "chart_title": "核心数值指标分布画像：按策略选择展示尺度",
            "table_preview": "Sales 9994.0 229.85 54.49 572.706 2481.6946\nProfit 9994.0 28.65 8.67 89.28 580.66",
            "code_source": "np.log1p(series.clip(lower=0)); series.clip(upper=clip_upper); sns.boxplot(data=metric_long)",
            "cell_index": 16,
            "output_formats": ["image/png"],
            "chart_kind": "histogram",
            "chart_summary": "核心数值指标分布画像用于观察分布形态、偏态和长尾。",
        }
    ]
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="metric_distribution_analysis",
                title="指标分布分析",
                chart_type="histogram",
                findings=[],
                summary_metrics={
                    "metric_count": 2,
                    "highest_tail_metric": "profit",
                    "highest_outlier_metric": "profit",
                },
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
                            "mean": 28.66,
                            "median": 8.67,
                            "p90": 89.28,
                            "p95": 168.47,
                            "p99": 580.66,
                            "min": -6599.98,
                            "max": 8399.98,
                            "skew": 7.56,
                            "mean_median_ratio": 3.31,
                            "p99_median_ratio": 66.97,
                            "non_negative": False,
                        },
                    ],
                    "metric_outliers": [
                        {
                            "metric": "sales_amount",
                            "label": "销售额",
                            "outlier_ratio": 0.1167,
                        },
                        {
                            "metric": "profit",
                            "label": "利润",
                            "outlier_ratio": 0.1882,
                        },
                    ],
                },
            )
        ],
    )

    reflections, _ = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        llm_client=None,
    )

    assert 16 in reflections
    assert "log1p" in reflections[16] or "P99" in reflections[16]
    assert "580.66" in reflections[16]
