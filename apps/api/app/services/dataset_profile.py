from __future__ import annotations

from typing import Any

import pandas as pd

from app.schemas.schema_mapping import SchemaMapping
from app.services.canonical_fields import (
    canonical_columns_for_frame,
    canonical_field_source_decisions,
    clean_numeric_series,
    drop_empty_unnamed_columns,
    numeric_coercion_summary,
)


def _canonical_columns(schema_mapping: SchemaMapping) -> dict[str, str]:
    return {
        canonical: original for original, canonical in schema_mapping.field_mapping.items()
    }


def _series(frame: pd.DataFrame, columns: dict[str, str], canonical: str):
    column = columns.get(canonical)
    if column and column in frame.columns:
        return frame[column]
    return None


def _numeric(frame: pd.DataFrame, columns: dict[str, str], canonical: str):
    series = _series(frame, columns, canonical)
    if series is None:
        return None
    return clean_numeric_series(series)


def _share(frame: pd.DataFrame, columns: dict[str, str], dimension: str) -> float | None:
    dim_col = columns.get(dimension)
    sales_col = columns.get("sales_amount")
    if not dim_col or not sales_col or dim_col not in frame.columns or sales_col not in frame.columns:
        return None
    grouped_source = frame[[dim_col]].copy()
    grouped_source["_sales_amount_numeric"] = clean_numeric_series(frame[sales_col])
    grouped = grouped_source.groupby(dim_col, dropna=True)["_sales_amount_numeric"].sum()
    total = float(grouped.sum())
    if total <= 0 or grouped.empty:
        return None
    return round(float(grouped.max() / total), 4)


def _nunique(frame: pd.DataFrame, columns: dict[str, str], canonical: str) -> int | None:
    series = _series(frame, columns, canonical)
    if series is None:
        return None
    return int(series.dropna().nunique())


def _line_per_order_avg(frame: pd.DataFrame, columns: dict[str, str]) -> float | None:
    order_col = columns.get("order_id")
    if not order_col or order_col not in frame.columns:
        return None
    counts = frame.dropna(subset=[order_col]).groupby(order_col).size()
    return round(float(counts.mean()), 4) if len(counts) else None


def _monthly_profile(frame: pd.DataFrame, columns: dict[str, str]) -> tuple[float | None, float | None]:
    date_col = columns.get("order_datetime")
    sales_col = columns.get("sales_amount")
    if not date_col or not sales_col or date_col not in frame.columns or sales_col not in frame.columns:
        return None, None
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    sales = clean_numeric_series(frame[sales_col])
    monthly = sales.groupby(dates.dt.to_period("M")).sum().dropna()
    monthly = monthly[monthly.index.notna()]
    if len(monthly) < 2:
        return None, None
    mean = float(monthly.mean())
    volatility = round(float(monthly.std(ddof=0) / mean), 4) if mean else None
    recent_growth = None
    if float(monthly.iloc[-2]) != 0:
        recent_growth = round(float((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]), 4)
    return volatility, recent_growth


def build_dataset_profile(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> dict[str, Any]:
    frame = drop_empty_unnamed_columns(frame)
    source_decisions = canonical_field_source_decisions(frame, schema_mapping)
    columns = canonical_columns_for_frame(frame, schema_mapping)
    fields = set(columns)
    product_dimension = "productline" if "productline" in fields else "category"

    profit = _numeric(frame, columns, "profit")
    sales = _numeric(frame, columns, "sales_amount")
    negative_profit_rate = None
    profit_margin_spread = None
    if profit is not None:
        valid_profit = profit.dropna()
        negative_profit_rate = round(float((valid_profit < 0).mean()), 4) if len(valid_profit) else None
        if sales is not None:
            margin = profit / sales.replace(0, pd.NA)
            margin = pd.to_numeric(margin, errors="coerce").dropna()
            if len(margin):
                profit_margin_spread = round(float(margin.quantile(0.9) - margin.quantile(0.1)), 4)

    invoice = _series(frame, columns, "order_id")
    customer = _series(frame, columns, "customer_id")
    repeat_customer_rate = None
    avg_basket_size = None
    line_per_order_avg = _line_per_order_avg(frame, columns)
    if customer is not None and invoice is not None:
        customer_orders = frame[[columns["customer_id"], columns["order_id"]]].dropna().drop_duplicates()
        counts = customer_orders.groupby(columns["customer_id"])[columns["order_id"]].nunique()
        repeat_customer_rate = round(float((counts > 1).mean()), 4) if len(counts) else None
        avg_basket_size = line_per_order_avg

    monthly_volatility, recent_growth_rate = _monthly_profile(frame, columns)
    missing_capabilities: list[str] = []
    if not {"discount", "profit"} <= fields:
        missing_capabilities.append("discount_erosion_focus")
    if not {"profit", "sales_amount"} <= fields:
        missing_capabilities.append("profit_quality_focus")
    if not {"customer_id", "order_id"} <= fields:
        missing_capabilities.append("customer_order_structure_focus")

    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "has_date": "order_datetime" in fields,
        "has_profit": "profit" in fields,
        "has_discount": "discount" in fields,
        "has_customer_id": "customer_id" in fields,
        "has_invoice_id": "order_id" in fields,
        "has_country": "country" in fields,
        "has_region": "region" in fields,
        "has_segment": "segment" in fields,
        "has_category": "category" in fields,
        "has_product": "product_name" in fields,
        "has_productline": "productline" in fields,
        "has_deal_size": "deal_size" in fields,
        "has_order_status": "order_status" in fields,
        "has_unit_price": "unit_price" in fields,
        "has_quantity": "quantity" in fields,
        "top_product_sales_share": _share(frame, columns, "product_name"),
        "top_category_sales_share": _share(frame, columns, product_dimension),
        "top_country_sales_share": _share(frame, columns, "country"),
        "country_count": _nunique(frame, columns, "country"),
        "region_count": _nunique(frame, columns, "region"),
        "segment_count": _nunique(frame, columns, "segment"),
        "category_count": _nunique(frame, columns, "category"),
        "productline_count": _nunique(frame, columns, "productline"),
        "product_count": _nunique(frame, columns, "product_name"),
        "order_count": _nunique(frame, columns, "order_id"),
        "customer_count": _nunique(frame, columns, "customer_id"),
        "line_per_order_avg": line_per_order_avg,
        "distinct_order_status_count": _nunique(frame, columns, "order_status"),
        "distinct_deal_size_count": _nunique(frame, columns, "deal_size"),
        "negative_profit_rate": negative_profit_rate,
        "profit_margin_spread": profit_margin_spread,
        "monthly_volatility": monthly_volatility,
        "recent_growth_rate": recent_growth_rate,
        "repeat_customer_rate": repeat_customer_rate,
        "avg_basket_size": avg_basket_size,
        "missing_capabilities": missing_capabilities,
        "numeric_coercion_summary": numeric_coercion_summary(frame, columns),
        "canonical_field_source_decision": {
            canonical: decision.as_dict()
            for canonical, decision in source_decisions.items()
            if decision.rejected_duplicate_columns or canonical in {"sales_amount", "profit", "cost", "unit_price", "quantity", "discount"}
        },
    }


def select_analysis_focuses(profile: dict[str, Any]) -> dict[str, Any]:
    focus_reasons: dict[str, str] = {}
    skipped: list[str] = []
    support: list[str] = []

    def skip(focus: str, reason: str) -> None:
        if focus not in skipped:
            skipped.append(focus)
        focus_reasons[focus] = reason

    def supports(focus: str, reason: str) -> None:
        if focus not in support:
            support.append(focus)
        focus_reasons[focus] = reason

    has_discount_profit = bool(profile.get("has_discount") and profile.get("has_profit"))
    has_slice = bool(profile.get("has_segment") or profile.get("has_region") or profile.get("has_category"))
    has_product = bool(profile.get("product_count") or profile.get("top_product_sales_share") or profile.get("has_category"))
    country_count = profile.get("country_count")
    top_country_share = profile.get("top_country_sales_share")
    country_ok = (
        bool(profile.get("has_country"))
        and isinstance(country_count, int)
        and country_count >= 2
        and (top_country_share is None or float(top_country_share) < 0.95)
    )
    if not country_ok:
        skip(
            "country_market_focus",
            "country 字段无有效市场切片价值：需要至少 2 个国家且头部国家销售占比低于 95%。",
        )
    else:
        focus_reasons["country_market_focus"] = "存在多个国家市场，且头部国家没有完全垄断销售额。"

    order_conditions = [
        bool(profile.get("has_customer_id")),
        bool(profile.get("has_quantity")),
        bool(profile.get("has_unit_price")),
        profile.get("line_per_order_avg") is not None and float(profile.get("line_per_order_avg") or 0) > 1.2,
        profile.get("repeat_customer_rate") is not None,
    ]
    order_ok = bool(profile.get("has_invoice_id")) and sum(order_conditions) >= 2
    if not order_ok:
        skip("customer_order_structure_focus", "订单结构需要发票/订单字段，且至少两个客户、数量、单价、篮子或复购条件。")
    else:
        focus_reasons["customer_order_structure_focus"] = "订单字段和订单结构画像足够支持复购、篮子或客单分析。"

    if not has_discount_profit:
        skip("discount_erosion_focus", "缺少折扣或利润字段，不能把折扣侵蚀作为主线。")
    if not profile.get("has_profit"):
        skip("profit_quality_focus", "缺少利润字段，不能把利润质量作为主线。")
    if not has_slice:
        skip("segment_region_focus", "缺少客群、区域或类目字段，不能做有效切片主线。")
    if not has_product:
        skip("product_concentration_focus", "缺少商品或类目结构字段，不能做商品集中度主线。")
    if not profile.get("has_productline"):
        skip("productline_performance_focus", "缺少 PRODUCTLINE 字段。")
    if not profile.get("has_deal_size"):
        skip("deal_size_focus", "缺少 DEALSIZE 字段。")
    if not profile.get("has_order_status"):
        skip("order_status_focus", "缺少 STATUS 字段。")
    if not (profile.get("has_date") and profile.get("monthly_volatility") is not None):
        skip("trend_volatility_focus", "缺少日期或月度波动不足，趋势波动只能作为限制说明。")

    is_superstore = has_discount_profit and has_slice
    is_online_retail = (
        not profile.get("has_profit")
        and not profile.get("has_discount")
        and country_ok
        and order_ok
        and bool(profile.get("has_unit_price"))
    )
    is_sample_sales = bool(profile.get("has_productline") and profile.get("has_deal_size") and profile.get("has_order_status"))

    if is_sample_sales:
        ordered = [
            "productline_performance_focus",
            "deal_size_focus",
            "order_status_focus",
            "country_market_focus",
            "trend_volatility_focus",
        ]
    elif is_online_retail:
        ordered = [
            "country_market_focus",
            "customer_order_structure_focus",
            "product_concentration_focus",
            "trend_volatility_focus",
        ]
    elif is_superstore:
        ordered = [
            "discount_erosion_focus",
            "profit_quality_focus",
            "segment_region_focus",
            "product_concentration_focus",
            "trend_volatility_focus",
            "customer_order_structure_focus",
        ]
        if order_ok:
            supports("customer_order_structure_focus", "折扣、利润和切片字段更适合作为主线，订单结构降为支撑分析。")
    else:
        ordered = [
            "discount_erosion_focus",
            "profit_quality_focus",
            "segment_region_focus",
            "product_concentration_focus",
            "country_market_focus",
            "customer_order_structure_focus",
            "productline_performance_focus",
            "deal_size_focus",
            "order_status_focus",
            "trend_volatility_focus",
        ]

    enabled = {
        "discount_erosion_focus": has_discount_profit,
        "profit_quality_focus": bool(profile.get("has_profit")),
        "segment_region_focus": has_slice,
        "product_concentration_focus": has_product,
        "country_market_focus": country_ok,
        "customer_order_structure_focus": order_ok,
        "productline_performance_focus": bool(profile.get("has_productline")),
        "deal_size_focus": bool(profile.get("has_deal_size")),
        "order_status_focus": bool(profile.get("has_order_status")),
        "trend_volatility_focus": bool(profile.get("has_date") and profile.get("monthly_volatility") is not None),
    }

    selected: list[str] = []
    for focus in ordered:
        if enabled.get(focus) and focus not in support and focus not in selected:
            selected.append(focus)
    if is_sample_sales:
        selected = selected[:4]
    elif is_online_retail:
        selected = selected[:3]
    elif is_superstore:
        selected = selected[:4]
    else:
        selected = selected[:3]
    if not selected:
        selected = ["data_limitation_focus"]
        focus_reasons["data_limitation_focus"] = "字段不足，无法形成稳定业务主线。"

    for focus, enabled_flag in enabled.items():
        if enabled_flag and focus not in selected and focus not in support:
            supports(focus, "字段可支持该分析，但优先级低于本数据集的核心主线。")
    skipped = [focus for focus in skipped if focus not in selected and focus not in support]
    return {
        "selected_focuses": selected,
        "support_focuses": support,
        "skipped_focuses": skipped,
        "focus_reasons": focus_reasons,
    }
