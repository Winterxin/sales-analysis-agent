from __future__ import annotations

from app.schemas.report import AnalysisReport, ModuleReport
from app.services.business_review_builder import build_business_review


def test_build_business_review_contains_three_business_sections() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["整体销售额为 1000。", "Top 商品为 A。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1000, "total_quantity": 20},
                findings=["销售趋势整体平稳。"],
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                summary_metrics={"top_product": "A"},
                findings=["Top 商品为 A。"],
            ),
        ],
    )

    review = build_business_review(report, output_language="zh-CN")

    assert "# 销售经营复盘报告" in review
    assert "## 1. 总体表现" in review
    assert "## 2. 关键发现" in review
    assert "## 6. 后续行动建议" in review
    assert "整体销售额为 1000。" in review


def test_english_business_review_missing_dimension_uses_natural_fallback() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        summary=[],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1000, "total_quantity": 20},
                findings=[],
            ),
            ModuleReport(
                module_id="data_quality_check",
                title="数据质量检查",
                chart_type="table",
                summary_metrics={"missing_order_datetime": 2},
                findings=[],
            ),
        ],
    )

    review = build_business_review(report, output_language="en")

    assert "# Business Review" in review
    assert "暂无" not in review
    assert "No suitable business dimension is available" in review
    assert "No product-level ranking is available" in review
    assert "No reliable forecast baseline was produced" in review
    assert "profit margin" not in review.lower()
    assert "high-discount" not in review.lower()


def test_english_business_review_uses_primary_and_secondary_dimensions() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        summary=[],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1000, "total_quantity": 20},
                findings=[],
            ),
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="客群与区域分析",
                chart_type="heatmap",
                summary_metrics={"primary_dimension": "segment", "secondary_dimension": "region"},
                tables={
                    "segment_region_matrix": [
                        {"segment": "Consumer", "region": "West", "sales_amount": 500.0}
                    ]
                },
                findings=["Consumer x West has the largest sales slice."],
            ),
        ],
    )

    review = build_business_review(report, output_language="en")

    assert "The analysis supports a `Segment × Region` sales breakdown." in review
    assert "No suitable business dimension is available" not in review


def test_chinese_business_review_without_discount_uses_profit_cost_inventory_wording() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["整体销售额为 1000。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1000, "total_quantity": 20},
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                summary_metrics={"top_product": "A"},
                findings=["已识别高销售低利润商品，需要进一步核查毛利、定价、成本和补货策略。"],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="利润质量分析",
                chart_type="table",
                summary_metrics={"total_profit_amount": 120, "negative_profit_rate": 0.08},
                tables={},
            ),
        ],
    )

    review = build_business_review(report, output_language="zh-CN")

    assert "利润、成本和库存" in review
    assert "对低利润商品和类目做专项分析，判断成本、定价或履约因素是否持续压低利润。" in review
    assert "利润、折扣、库存" not in review
    assert "对高折扣订单做专项分析" not in review


def test_chinese_business_review_with_discount_keeps_existing_discount_wording() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["整体销售额为 1000。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1000, "total_quantity": 20},
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                summary_metrics={"top_product": "A"},
                findings=["已识别高销售低利润商品，需要进一步核查毛利、折扣和补货策略。"],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="table",
                tables={"discount_threshold_candidates": [{"discount_bucket": "30%+"}]},
            ),
        ],
    )

    review = build_business_review(report, output_language="zh-CN")

    assert "利润、折扣、库存" in review
    assert "对高折扣订单做专项分析，判断折扣是否真正带来销售增长。" in review


def test_english_business_review_discount_behavior_is_unchanged() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        summary=[],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend Analysis",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1000, "total_quantity": 20},
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="Discount Profit Analysis",
                chart_type="table",
                summary_metrics={"total_profit_amount": 120},
                tables={"discount_threshold_candidates": [{"discount_bucket": "30%+"}]},
            ),
        ],
    )

    review = build_business_review(report, output_language="en")

    assert "Review high-discount orders only where discount and profit evidence are both available." in review
