from __future__ import annotations

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.ingestion import IngestionSummary
from app.schemas.schema_mapping import SchemaMapping


def build_analysis_plan(
    schema_mapping: SchemaMapping, ingestion: IngestionSummary
) -> AnalysisPlan:
    modules = ["data_quality_check"]
    chart_preferences = {
        "data_quality_check": "bar",
    }
    reasoning = [
        "默认执行数据质量检查，确保后续分析可信。",
        "检测到时间和销售金额字段，启用销售趋势分析。",
    ]

    mapped_fields = set(schema_mapping.field_mapping.values())
    core_numeric_fields = {"sales_amount", "quantity", "discount", "profit"} & mapped_fields
    if len(core_numeric_fields) >= 2:
        modules.append("metric_distribution_analysis")
        chart_preferences["metric_distribution_analysis"] = "histogram"
        reasoning.append(
            "检测到多个核心数值字段，先启用指标分布画像，帮助识别长尾、异常值和后续分析重点。"
        )

    modules.append("sales_trend_analysis")
    chart_preferences["sales_trend_analysis"] = "line"
    if {"product_name", "sku"} & mapped_fields:
        modules.append("product_contribution_analysis")
        chart_preferences["product_contribution_analysis"] = "bar"
        reasoning.append("检测到商品字段，启用商品贡献分析。")

    if {"segment", "region", "state", "city", "country"} & mapped_fields:
        modules.append("dimension_breakdown_analysis")
        chart_preferences["dimension_breakdown_analysis"] = "stacked_bar"
        reasoning.append("检测到客群或地理维度，启用维度拆解分析。")

    if "country" in mapped_fields:
        modules.append("country_market_analysis")
        chart_preferences["country_market_analysis"] = "bar"
        reasoning.append("检测到国家字段，启用国家市场专题分析。")

    if {"order_id", "customer_id", "deal_size", "order_status", "unit_price"} & mapped_fields:
        modules.append("order_structure_analysis")
        chart_preferences["order_structure_analysis"] = "histogram"
        reasoning.append("检测到订单、客户、交易规模、状态或单价字段，启用订单结构专题分析。")

    if "profit" in mapped_fields:
        modules.append("discount_profit_analysis")
        chart_preferences["discount_profit_analysis"] = (
            "scatter" if "discount" in mapped_fields else "bar"
        )
        if "discount" in mapped_fields:
            reasoning.append("检测到折扣和利润字段，启用折扣与利润专题分析。")
        else:
            reasoning.append("检测到利润字段，启用利润概览降级分析。")

        modules.append("loss_risk_modeling")
        chart_preferences["loss_risk_modeling"] = "modeling"
        reasoning.append(
            "检测到利润字段，尝试进行亏损风险建模；若样本量、类别分布或特征条件不足，将在建模模块中优雅跳过。"
        )

    if ingestion.row_count >= 30 and "order_datetime" in mapped_fields:
        modules.append("forecast_analysis")
        chart_preferences["forecast_analysis"] = "line"
        reasoning.append("时间跨度具备基本预测条件，启用基础预测。")

    return AnalysisPlan(
        analysis_plan=modules,
        chart_preferences=chart_preferences,
        reasoning_summary=reasoning,
    )
