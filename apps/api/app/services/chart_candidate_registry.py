from __future__ import annotations

from typing import Any


ChartCandidate = dict[str, Any]


def _candidate(
    chart_id: str,
    section_id: str,
    chart_kind: str,
    title: str,
    required_fields: list[str],
    supported_focuses: list[str],
    business_question: str,
    priority: float,
    *,
    optional_fields: list[str] | None = None,
    answers_evidence_ids: list[str] | None = None,
    best_when: list[str] | None = None,
    avoid_when: list[str] | None = None,
    insight_type: str = "structure",
) -> ChartCandidate:
    return {
        "chart_id": chart_id,
        "section_id": section_id,
        "chart_kind": chart_kind,
        "title": title,
        "required_fields": required_fields,
        "optional_fields": optional_fields or [],
        "supported_focuses": supported_focuses,
        "business_question": business_question,
        "priority": priority,
        "answers_evidence_ids": answers_evidence_ids or [],
        "best_when": best_when or [],
        "avoid_when": avoid_when or [],
        "insight_type": insight_type,
    }


CHART_CANDIDATES: list[ChartCandidate] = [
    _candidate("sales_trends_monthly_line", "sales_trends", "line", "销售趋势与滚动均线", ["order_datetime", "sales_amount"], ["trend_volatility_focus"], "销售额是否存在阶段性增长、下滑或波动拐点？", 0.95, optional_fields=["quantity"], answers_evidence_ids=["monthly_volatility", "recent_growth_rate"], best_when=["trend_volatility"], insight_type="trend"),
    _candidate("sales_trends_month_heatmap", "sales_trends", "heatmap", "年月销售热力图", ["order_datetime", "sales_amount"], ["trend_volatility_focus"], "销售高峰是否集中在特定月份或年份组合？", 0.7, answers_evidence_ids=["monthly_volatility"], best_when=["trend_volatility"], insight_type="trend"),
    _candidate("recent_growth_bar", "sales_trends", "bar", "近期增长率对比", ["order_datetime", "sales_amount"], ["trend_volatility_focus"], "近期增长是改善、放缓还是回落？", 0.76, answers_evidence_ids=["recent_growth_rate"], best_when=["recent_growth_rate"], insight_type="trend"),
    _candidate("rolling_volatility_line", "sales_trends", "line", "滚动波动率趋势", ["order_datetime", "sales_amount"], ["trend_volatility_focus"], "销售波动是否在近期扩大？", 0.72, answers_evidence_ids=["monthly_volatility"], best_when=["monthly_volatility"], insight_type="risk"),

    _candidate("product_category_bar", "product_and_category", "bar", "头部商品销售额对比", ["product_name", "sales_amount"], ["product_concentration_focus"], "销售额是否过度集中在少数商品？", 0.88, optional_fields=["profit"], answers_evidence_ids=["product_count", "top_product_sales_share"], best_when=["product_concentration"], insight_type="concentration"),
    _candidate("product_category_pareto", "product_and_category", "pareto", "头部商品累计贡献 Pareto 分布", ["product_name", "sales_amount"], ["product_concentration_focus"], "头部商品累计贡献是否形成明显集中度风险？", 0.82, answers_evidence_ids=["top_product_sales_share"], best_when=["product_concentration"], insight_type="concentration"),
    _candidate("product_profit_bridge_scatter", "product_and_category", "scatter", "商品销售额与利润桥接", ["product_name", "sales_amount", "profit"], ["product_concentration_focus", "profit_quality_focus"], "高销售商品是否同步贡献利润，还是存在规模幻觉？", 0.74, answers_evidence_ids=["profit_margin_spread", "top_product_sales_share"], best_when=["profit_quality"], insight_type="quality"),
    _candidate("category_profit_margin_bar", "product_and_category", "bar", "类目利润率质量对比", ["category", "sales_amount", "profit"], ["profit_quality_focus", "product_concentration_focus"], "类目贡献和利润率是否同步？", 0.79, answers_evidence_ids=["profit_margin_spread", "top_category_sales_share"], best_when=["profit_quality"], insight_type="quality"),
    _candidate("subcategory_sales_profit_matrix", "product_and_category", "scatter", "子类目销售利润矩阵", ["sub_category", "sales_amount", "profit"], ["profit_quality_focus", "product_concentration_focus"], "子类目是否存在高销售低利润组合？", 0.73, answers_evidence_ids=["profit_margin_spread"], best_when=["profit_quality"], insight_type="quality"),
    _candidate("top_product_profit_gap_bar", "product_and_category", "bar", "头部商品利润缺口", ["product_name", "sales_amount", "profit"], ["profit_quality_focus", "product_concentration_focus"], "头部商品中哪些贡献规模但利润不足？", 0.8, answers_evidence_ids=["profit_margin_spread", "top_product_sales_share"], best_when=["profit_quality"], insight_type="risk"),
    _candidate("category_concentration_bar", "product_and_category", "bar", "类目集中度对比", ["category", "sales_amount"], ["product_concentration_focus"], "销售额是否集中在少数类目？", 0.77, answers_evidence_ids=["top_category_sales_share"], best_when=["product_concentration"], insight_type="concentration"),
    _candidate("productline_sales_bar", "product_and_category", "bar", "PRODUCTLINE 销售额贡献", ["productline", "sales_amount"], ["productline_performance_focus"], "哪个 PRODUCTLINE 是销售贡献主线，资源是否需要重新分配？", 0.98, optional_fields=["quantity", "deal_size"], answers_evidence_ids=["productline_count"], best_when=["productline_mix"], insight_type="mix"),
    _candidate("productline_sales_donut", "product_and_category", "donut", "PRODUCTLINE 销售额占比", ["productline", "sales_amount"], ["productline_performance_focus"], "产品线销售贡献是否集中在少数 PRODUCTLINE？", 0.68, answers_evidence_ids=["productline_count"], best_when=["productline_mix"], insight_type="mix"),
    _candidate("productline_monthly_trend", "product_and_category", "line", "PRODUCTLINE 月度销售趋势", ["productline", "order_datetime", "sales_amount"], ["productline_performance_focus", "trend_volatility_focus"], "不同产品线的月度销售趋势是否同步？", 0.84, answers_evidence_ids=["productline_count", "monthly_volatility"], best_when=["productline_mix"], insight_type="trend"),
    _candidate("productline_quantity_sales_combo", "product_and_category", "combo", "PRODUCTLINE 销售额与销量对比", ["productline", "sales_amount", "quantity"], ["productline_performance_focus"], "产品线的销售额贡献和销量贡献是否匹配？", 0.8, answers_evidence_ids=["productline_count"], best_when=["productline_mix"], insight_type="mix"),
    _candidate("productline_quantity_bar", "product_and_category", "bar", "PRODUCTLINE 数量贡献", ["productline", "quantity"], ["productline_performance_focus"], "产品线的销量贡献和销售额贡献是否一致？", 0.86, optional_fields=["sales_amount"], answers_evidence_ids=["productline_count"], best_when=["productline_mix"], insight_type="mix"),
    _candidate("productline_deal_size_stacked_bar", "product_and_category", "stacked_bar", "PRODUCTLINE x 交易规模销售结构", ["productline", "deal_size", "sales_amount"], ["productline_performance_focus", "deal_size_focus"], "产品线销售是否由特定交易规模驱动？", 0.76, answers_evidence_ids=["productline_count", "distinct_deal_size_count"], best_when=["productline_mix"], insight_type="mix"),
    _candidate("productline_deal_size_heatmap", "product_and_category", "heatmap", "PRODUCTLINE x 交易规模销售热力图", ["productline", "deal_size", "sales_amount"], ["productline_performance_focus", "deal_size_focus"], "产品线和交易规模的交叉贡献是否存在明显集中？", 0.74, answers_evidence_ids=["productline_count", "distinct_deal_size_count"], best_when=["productline_mix"], insight_type="mix"),

    _candidate("segment_region_sales_bar", "segment_and_region", "bar", "客群/区域销售额对比", ["sales_amount"], ["segment_region_focus"], "哪些客群、区域或类目贡献主要销售额？", 0.9, optional_fields=["segment", "region", "category", "profit"], answers_evidence_ids=["region_count", "segment_count", "category_count"], best_when=["superstore_like"], insight_type="mix"),
    _candidate("segment_region_margin_heatmap", "segment_and_region", "heatmap", "客群 x 区域利润率热力图", ["sales_amount", "profit"], ["segment_region_focus", "profit_quality_focus"], "利润率压力是否集中在特定客群和区域组合？", 0.84, optional_fields=["segment", "region", "category"], answers_evidence_ids=["profit_margin_spread"], best_when=["profit_quality"], insight_type="quality"),
    _candidate("segment_sales_profit_bar", "segment_and_region", "bar", "客群销售与利润对比", ["segment", "sales_amount", "profit"], ["segment_region_focus", "profit_quality_focus"], "客群销售贡献是否同步转化为利润？", 0.82, answers_evidence_ids=["segment_count", "profit_margin_spread"], best_when=["profit_quality"], insight_type="quality"),
    _candidate("region_sales_profit_bar", "segment_and_region", "bar", "区域销售与利润对比", ["region", "sales_amount", "profit"], ["segment_region_focus", "profit_quality_focus"], "区域规模和利润质量是否匹配？", 0.81, answers_evidence_ids=["region_count", "profit_margin_spread"], best_when=["profit_quality"], insight_type="quality"),
    _candidate("segment_region_low_margin_table_or_bar", "segment_and_region", "bar", "低利润客群区域组合", ["sales_amount", "profit"], ["segment_region_focus", "profit_quality_focus"], "低利润风险集中在哪些切片？", 0.78, optional_fields=["segment", "region"], answers_evidence_ids=["profit_margin_spread", "negative_profit_rate"], best_when=["discount_loss"], insight_type="risk"),

    _candidate("country_sales_bar", "country_market", "bar", "国家销售额对比", ["country", "sales_amount"], ["country_market_focus"], "销售额主要由哪些国家市场贡献？", 0.96, optional_fields=["order_id", "customer_id"], answers_evidence_ids=["country_count", "top_country_sales_share"], best_when=["country_market"], insight_type="mix"),
    _candidate("country_sales_donut", "country_market", "donut", "国家销售额占比", ["country", "sales_amount"], ["country_market_focus"], "国家市场销售贡献是否集中在少数市场？", 0.67, answers_evidence_ids=["country_count", "top_country_sales_share"], best_when=["country_market"], insight_type="mix"),
    _candidate("country_avg_order_value_bar", "country_market", "bar", "国家平均订单金额对比", ["country", "sales_amount", "order_id"], ["country_market_focus"], "不同国家的客单价是否存在经营差异？", 0.78, answers_evidence_ids=["country_count"], best_when=["country_market"], insight_type="quality"),
    _candidate("country_monthly_trend", "country_market", "line", "头部国家月度销售趋势", ["country", "sales_amount", "order_datetime"], ["country_market_focus", "trend_volatility_focus"], "头部国家市场的趋势是否同步，还是存在局部波动？", 0.74, answers_evidence_ids=["top_country_sales_share", "monthly_volatility"], best_when=["country_market"], insight_type="trend"),
    _candidate("country_productline_heatmap", "country_market", "heatmap", "国家 x PRODUCTLINE 销售热力图", ["country", "productline", "sales_amount"], ["country_market_focus", "productline_performance_focus"], "不同国家的产品线销售结构是否存在差异？", 0.73, answers_evidence_ids=["country_count", "productline_count"], best_when=["country_market", "productline_mix"], insight_type="mix"),
    _candidate("country_customer_count_bar", "country_market", "bar", "国家客户数对比", ["country", "customer_id"], ["country_market_focus"], "客户基础是否集中在少数国家？", 0.73, answers_evidence_ids=["country_count", "customer_count"], best_when=["country_market"], insight_type="structure"),
    _candidate("country_order_count_bar", "country_market", "bar", "国家订单数对比", ["country", "order_id"], ["country_market_focus"], "订单量是否和销售额国家结构一致？", 0.74, answers_evidence_ids=["country_count", "order_count"], best_when=["country_market"], insight_type="structure"),
    _candidate("country_concentration_pareto", "country_market", "pareto", "国家销售集中度 Pareto", ["country", "sales_amount"], ["country_market_focus"], "国家市场是否过度集中？", 0.82, answers_evidence_ids=["top_country_sales_share"], best_when=["country_market"], insight_type="concentration"),

    _candidate("basket_size_distribution", "order_structure", "histogram", "订单篮子大小分布", ["order_id"], ["customer_order_structure_focus"], "订单是否由多行商品组成，篮子深度是否值得运营？", 0.9, optional_fields=["quantity"], answers_evidence_ids=["line_per_order_avg"], best_when=["order_basket"], insight_type="structure"),
    _candidate("invoice_value_distribution", "order_structure", "histogram", "发票金额分布", ["order_id", "sales_amount"], ["customer_order_structure_focus"], "订单金额是否存在长尾或大单依赖？", 0.84, answers_evidence_ids=["order_count"], best_when=["order_basket"], insight_type="structure"),
    _candidate("deal_size_sales_bar", "order_structure", "bar", "DEALSIZE 销售额对比", ["deal_size", "sales_amount"], ["deal_size_focus"], "销售额主要来自大单、中单还是小单？", 0.98, answers_evidence_ids=["distinct_deal_size_count"], best_when=["productline_mix"], insight_type="mix"),
    _candidate("deal_size_sales_donut", "order_structure", "donut", "DEALSIZE 销售额占比", ["deal_size", "sales_amount"], ["deal_size_focus"], "不同交易规模对销售额的贡献占比如何？", 0.7, answers_evidence_ids=["distinct_deal_size_count"], best_when=["productline_mix"], insight_type="mix"),
    _candidate("deal_size_order_count_bar", "order_structure", "bar", "DEALSIZE 订单数对比", ["deal_size", "order_id"], ["deal_size_focus"], "不同交易规模的订单数量是否和销售贡献匹配？", 0.78, answers_evidence_ids=["distinct_deal_size_count", "order_count"], best_when=["productline_mix"], insight_type="structure"),
    _candidate("deal_size_avg_order_value_bar", "order_structure", "bar", "DEALSIZE 平均订单金额对比", ["deal_size", "sales_amount", "order_id"], ["deal_size_focus"], "不同交易规模的平均订单金额是否形成清晰分层？", 0.82, answers_evidence_ids=["distinct_deal_size_count"], best_when=["productline_mix"], insight_type="quality"),
    _candidate("order_status_breakdown", "order_structure", "bar", "STATUS 销售额结构", ["order_status", "sales_amount"], ["order_status_focus"], "订单状态是否影响可确认销售额和履约质量？", 0.96, answers_evidence_ids=["distinct_order_status_count"], best_when=["productline_mix"], insight_type="structure"),
    _candidate("status_sales_donut", "order_structure", "donut", "STATUS 销售额占比", ["order_status", "sales_amount"], ["order_status_focus"], "不同订单状态的销售额占比是否健康？", 0.69, answers_evidence_ids=["distinct_order_status_count"], best_when=["productline_mix"], insight_type="structure"),
    _candidate("status_order_count_bar", "order_structure", "bar", "STATUS 订单数对比", ["order_status", "order_id"], ["order_status_focus"], "不同订单状态的订单数是否显示异常履约压力？", 0.8, answers_evidence_ids=["distinct_order_status_count", "order_count"], best_when=["productline_mix"], insight_type="structure"),
    _candidate("status_deal_size_stacked_bar", "order_structure", "stacked_bar", "STATUS x 交易规模销售结构", ["order_status", "deal_size", "sales_amount"], ["order_status_focus", "deal_size_focus"], "异常订单状态是否集中在特定交易规模？", 0.75, answers_evidence_ids=["distinct_order_status_count", "distinct_deal_size_count"], best_when=["productline_mix"], insight_type="structure"),
    _candidate("price_quantity_scatter", "order_structure", "scatter", "单价与数量关系", ["unit_price", "quantity", "sales_amount"], ["customer_order_structure_focus"], "单价和购买数量之间是否存在结构性关系？", 0.62, answers_evidence_ids=["line_per_order_avg"], best_when=["order_basket"], insight_type="structure"),
    _candidate("order_line_count_by_status", "order_structure", "bar", "STATUS 订单行数对比", ["order_status", "order_id"], ["order_status_focus"], "不同订单状态的订单行数结构是否不同？", 0.72, answers_evidence_ids=["distinct_order_status_count", "line_per_order_avg"], best_when=["productline_mix"], insight_type="structure"),
    _candidate("order_value_by_deal_size", "order_structure", "boxplot", "DEALSIZE 订单金额分布", ["deal_size", "sales_amount"], ["deal_size_focus"], "不同 DEALSIZE 的订单金额分布是否分层？", 0.75, answers_evidence_ids=["distinct_deal_size_count"], best_when=["productline_mix"], insight_type="quality"),

    _candidate("discount_profit_quality_bar", "discount_and_profit", "bar", "各折扣区间利润质量：平均利润与亏损率", ["discount", "profit", "sales_amount"], ["discount_erosion_focus", "profit_quality_focus"], "高折扣是否显著侵蚀利润质量？", 0.96, answers_evidence_ids=["negative_profit_rate", "profit_margin_spread"], best_when=["discount_loss"], insight_type="risk"),
    _candidate("discount_category_heatmap", "discount_and_profit", "heatmap", "类目 x 折扣区间亏损率热力图", ["discount", "profit", "sales_amount", "category"], ["discount_erosion_focus", "segment_region_focus"], "亏损是否集中在特定类目和折扣区间？", 0.82, answers_evidence_ids=["negative_profit_rate", "top_category_sales_share"], best_when=["discount_loss"], insight_type="risk"),
    _candidate("discount_segment_heatmap", "discount_and_profit", "heatmap", "客群 x 折扣区间亏损率热力图", ["discount", "profit", "sales_amount", "segment"], ["discount_erosion_focus", "segment_region_focus"], "折扣风险是否集中在特定客群？", 0.81, answers_evidence_ids=["negative_profit_rate", "segment_count"], best_when=["discount_loss"], insight_type="risk"),
    _candidate("discount_region_heatmap", "discount_and_profit", "heatmap", "区域 x 折扣区间亏损率热力图", ["discount", "profit", "sales_amount", "region"], ["discount_erosion_focus", "segment_region_focus"], "折扣风险是否集中在特定区域？", 0.8, answers_evidence_ids=["negative_profit_rate", "region_count"], best_when=["discount_loss"], insight_type="risk"),
    _candidate("high_discount_product_bar", "discount_and_profit", "bar", "高折扣低利润商品", ["discount", "profit", "sales_amount", "product_name"], ["discount_erosion_focus", "profit_quality_focus"], "哪些商品在高折扣下拖累利润？", 0.86, answers_evidence_ids=["negative_profit_rate", "top_product_sales_share"], best_when=["discount_loss"], insight_type="risk"),
    _candidate("profit_margin_by_discount_box", "discount_and_profit", "boxplot", "折扣区间利润率分布", ["discount", "profit", "sales_amount"], ["discount_erosion_focus", "profit_quality_focus"], "折扣越深，利润率分布是否整体下移？", 0.83, answers_evidence_ids=["profit_margin_spread"], best_when=["profit_quality"], insight_type="quality"),
    _candidate("discount_vs_profit_scatter", "discount_and_profit", "scatter", "折扣与利润散点关系", ["discount", "profit", "sales_amount"], ["discount_erosion_focus", "profit_quality_focus"], "折扣和利润之间是否存在方向性关系？", 0.76, answers_evidence_ids=["negative_profit_rate", "profit_margin_spread"], best_when=["profit_quality"], insight_type="quality"),

    _candidate("metric_sales_distribution", "metric_distributions", "histogram", "核心数值指标分布画像：按策略选择展示尺度", ["sales_amount"], ["discount_erosion_focus", "profit_quality_focus", "product_concentration_focus", "customer_order_structure_focus"], "销售额等核心指标是否存在长尾、异常值或尺度问题？", 0.66, optional_fields=["quantity", "discount", "profit"], answers_evidence_ids=["top_product_sales_share"], insight_type="structure"),
    _candidate("sales_amount_log_distribution", "metric_distributions", "histogram", "销售额对数分布", ["sales_amount"], ["product_concentration_focus", "customer_order_structure_focus"], "销售额长尾是否需要用对数尺度观察？", 0.64, answers_evidence_ids=["top_product_sales_share"], insight_type="structure"),
    _candidate(
        "quantity_distribution",
        "metric_distributions",
        "histogram",
        "订单行数量分布",
        ["quantity"],
        ["customer_order_structure_focus", "deal_size_focus", "order_status_focus", "productline_performance_focus"],
        "What is the distribution of quantities per order line?",
        0.65,
        answers_evidence_ids=["line_per_order_avg"],
        best_when=["order_basket", "productline_mix", "quantity"],
        avoid_when=[
            "quantity missing",
            "quantity has little variation",
            "quantity is non-numeric or cannot be converted to numeric",
        ],
        insight_type="quantity_distribution",
    ),
    _candidate("profit_distribution_if_available", "metric_distributions", "histogram", "利润分布", ["profit"], ["profit_quality_focus"], "利润是否存在负值、长尾或极端点？", 0.68, answers_evidence_ids=["negative_profit_rate", "profit_margin_spread"], insight_type="quality"),
]


def iter_chart_candidates() -> list[ChartCandidate]:
    return [dict(candidate) for candidate in CHART_CANDIDATES]
