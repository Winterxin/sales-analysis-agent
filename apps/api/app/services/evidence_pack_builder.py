from __future__ import annotations

from typing import Any

from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping


PROFILE_METRICS = [
    "country_count",
    "order_count",
    "customer_count",
    "product_count",
    "productline_count",
    "distinct_deal_size_count",
    "distinct_order_status_count",
    "top_country_sales_share",
    "top_category_sales_share",
    "top_product_sales_share",
    "negative_profit_rate",
    "profit_margin_spread",
    "monthly_volatility",
    "recent_growth_rate",
    "repeat_customer_rate",
    "line_per_order_avg",
]

RATIO_METRICS = {
    "top_country_sales_share",
    "top_category_sales_share",
    "top_product_sales_share",
    "negative_profit_rate",
    "profit_margin_spread",
    "monthly_volatility",
    "recent_growth_rate",
    "repeat_customer_rate",
}

FOCUS_METRIC_MAP = {
    "discount_erosion_focus": ["negative_profit_rate", "profit_margin_spread"],
    "profit_quality_focus": ["negative_profit_rate", "profit_margin_spread"],
    "product_concentration_focus": [
        "product_count",
        "top_product_sales_share",
        "top_category_sales_share",
    ],
    "segment_region_focus": ["region_count", "segment_count", "category_count", "profit_margin_spread"],
    "country_market_focus": ["country_count", "top_country_sales_share"],
    "customer_order_structure_focus": [
        "order_count",
        "customer_count",
        "repeat_customer_rate",
        "line_per_order_avg",
    ],
    "productline_performance_focus": ["productline_count"],
    "deal_size_focus": ["distinct_deal_size_count"],
    "order_status_focus": ["distinct_order_status_count"],
    "trend_volatility_focus": ["monthly_volatility", "recent_growth_rate"],
}

METRIC_LABELS = {
    "country_count": "国家数",
    "order_count": "订单数",
    "customer_count": "客户数",
    "product_count": "商品数",
    "productline_count": "PRODUCTLINE 数",
    "distinct_deal_size_count": "DEALSIZE 类型数",
    "distinct_order_status_count": "STATUS 类型数",
    "top_country_sales_share": "头部国家销售占比",
    "top_category_sales_share": "头部类目销售占比",
    "top_product_sales_share": "头部商品销售占比",
    "negative_profit_rate": "负利润记录占比",
    "profit_margin_spread": "利润率跨度",
    "monthly_volatility": "月度波动率",
    "recent_growth_rate": "近期增长率",
    "repeat_customer_rate": "复购客户占比",
    "line_per_order_avg": "平均订单行数",
}


def _format_value(metric: str, value: Any) -> str:
    if value is None:
        return "数据限制说明：缺少可验证字段"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        number = float(value)
        if metric in RATIO_METRICS:
            if abs(number) < 0.00005:
                return "0%"
            return f"{number * 100:.2f}%"
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"
    text = str(value).strip()
    return text if text else "数据限制说明：缺少可验证字段"


def _strength(metric: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        return 0.35
    number = abs(float(value))
    if metric in {"negative_profit_rate", "top_country_sales_share", "top_category_sales_share", "top_product_sales_share"}:
        return min(1.0, 0.35 + number)
    if metric in {"profit_margin_spread", "monthly_volatility", "repeat_customer_rate"}:
        return min(1.0, 0.35 + number)
    if metric in {"country_count", "productline_count", "distinct_deal_size_count", "distinct_order_status_count"}:
        return min(1.0, 0.3 + number / 20)
    if metric in {"order_count", "customer_count", "product_count"}:
        return min(1.0, 0.35 + number / 10000)
    return min(1.0, 0.4 + number / 10)


def _shape_type(fields: set[str], profile: dict[str, Any]) -> str:
    if {"productline", "deal_size", "order_status"} & fields:
        return "productline_order_like"
    if "country" in fields and "customer_id" in fields and "profit" not in fields:
        return "retail_like"
    if {"discount", "profit", "segment", "region", "category"} & fields and {"discount", "profit"} <= fields:
        return "superstore_like"
    if profile.get("productline_count") is not None:
        return "productline_order_like"
    return "generic_sales"


def _dominant_story(fields: set[str], profile: dict[str, Any], selected_focuses: list[str]) -> str:
    if "productline_performance_focus" in selected_focuses:
        return "productline_mix"
    if "country_market_focus" in selected_focuses and profile.get("country_count"):
        return "country_market"
    negative_rate = profile.get("negative_profit_rate")
    if "discount_erosion_focus" in selected_focuses and isinstance(negative_rate, (int, float)) and negative_rate > 0.12:
        return "discount_loss"
    if "profit_quality_focus" in selected_focuses and "profit" in fields:
        return "profit_quality"
    if "customer_order_structure_focus" in selected_focuses:
        return "order_basket"
    if "product_concentration_focus" in selected_focuses:
        return "product_concentration"
    if "trend_volatility_focus" in selected_focuses:
        return "trend_volatility"
    return "product_concentration" if "product_name" in fields else "generic_sales"


def _business_meaning(metric: str, formatted: str) -> str:
    if metric == "negative_profit_rate" and formatted == "0%":
        return "负利润记录占比为 0%，当前不应把亏损排查作为主线，应转向利润质量和结构差异。"
    meanings = {
        "negative_profit_rate": f"负利润记录占比为 {formatted}，应优先检查折扣与利润质量。",
        "profit_margin_spread": f"利润率跨度为 {formatted}，说明利润质量存在切片差异。",
        "top_country_sales_share": f"头部国家销售占比为 {formatted}，需要判断市场是否过度集中。",
        "top_category_sales_share": f"头部类目销售占比为 {formatted}，需要判断类目结构是否集中。",
        "top_product_sales_share": f"头部商品销售占比为 {formatted}，需要识别单品依赖风险。",
        "monthly_volatility": f"月度波动率为 {formatted}，趋势分析应关注经营节奏是否稳定。",
        "recent_growth_rate": f"近期增长率为 {formatted}，应结合趋势判断增长是否延续。",
        "repeat_customer_rate": f"复购客户占比为 {formatted}，订单结构应关注复购和客户运营。",
        "line_per_order_avg": f"平均订单行数为 {formatted}，适合分析篮子深度和订单结构。",
    }
    return meanings.get(metric, f"{METRIC_LABELS.get(metric, metric)}为 {formatted}，可作为当前分析主线的证据。")


def _evidence_item(metric: str, value: Any, source: str = "dataset_profile") -> dict[str, Any]:
    formatted = _format_value(metric, value)
    return {
        "evidence_id": metric,
        "metric": metric,
        "value": value,
        "formatted": formatted,
        "strength": round(_strength(metric, value), 4),
        "business_meaning": _business_meaning(metric, formatted),
        "source": source,
    }


def _limitations(fields: set[str]) -> list[dict[str, str]]:
    limitations: list[dict[str, str]] = []
    required = {
        "profit": ("不能判断利润质量", "改用销售额、订单结构或商品集中度分析"),
        "discount": ("不能判断折扣侵蚀", "改用利润质量、类目结构或趋势波动分析"),
        "country": ("不能判断国家市场差异", "改用区域、客群、商品或订单结构分析"),
        "customer_id": ("不能判断客户复购", "改用订单行数、发票金额或商品结构分析"),
        "productline": ("不能判断 PRODUCTLINE 组合", "改用商品、类目或交易结构分析"),
        "deal_size": ("不能判断 DEALSIZE 差异", "改用订单金额分布或商品结构分析"),
        "order_status": ("不能判断 STATUS 履约结构", "改用订单金额、国家市场或趋势分析"),
    }
    for field, (impact, alternative) in required.items():
        if field not in fields:
            limitations.append(
                {
                    "field": field,
                    "impact": f"数据限制说明：{impact}",
                    "alternative": alternative,
                }
            )
    return limitations


def build_evidence_pack(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    section_priority: dict[str, Any] | None,
) -> dict[str, Any]:
    profile = dataset_profile or {}
    fields = set(schema_mapping.field_mapping.values())
    selected_focuses = [
        str(item)
        for item in (analysis_focus or {}).get("selected_focuses", [])
        if str(item).strip()
    ]
    missing_fields = sorted(
        field
        for field in ["profit", "discount", "country", "customer_id", "productline", "deal_size", "order_status"]
        if field not in fields
    )
    signature = {
        "shape_type": _shape_type(fields, profile),
        "dominant_story": _dominant_story(fields, profile, selected_focuses),
        "available_fields": sorted(fields),
        "missing_fields": missing_fields,
        "core_sections": list((section_priority or {}).get("core_sections", [])),
    }

    focus_evidence: dict[str, list[dict[str, Any]]] = {}
    for focus in selected_focuses + [
        str(item)
        for item in (analysis_focus or {}).get("support_focuses", [])
        if str(item).strip()
    ]:
        for metric in FOCUS_METRIC_MAP.get(focus, []):
            if profile.get(metric) is None:
                continue
            if metric == "negative_profit_rate" and isinstance(profile.get(metric), (int, float)) and float(profile.get(metric)) <= 0:
                continue
            focus_evidence.setdefault(focus, []).append(_evidence_item(metric, profile.get(metric)))

    distinctive_facts: list[dict[str, Any]] = []
    for metric in PROFILE_METRICS:
        if profile.get(metric) is None:
            continue
        item = _evidence_item(metric, profile.get(metric))
        distinctive_facts.append(
            {
                "fact_id": metric,
                "text": item["business_meaning"],
                "strength": item["strength"],
                "source": item["source"],
            }
        )

    for module in report.modules:
        for index, finding in enumerate(module.findings[:2]):
            text = str(finding).strip()
            if text:
                distinctive_facts.append(
                    {
                        "fact_id": f"{module.module_id}_{index}",
                        "text": text,
                        "strength": 0.45,
                        "source": "module_report",
                    }
                )

    distinctive_facts.sort(key=lambda item: float(item.get("strength") or 0), reverse=True)
    return {
        "dataset_signature": signature,
        "focus_evidence": focus_evidence,
        "distinctive_facts": distinctive_facts[:12],
        "limitations": _limitations(fields),
    }
