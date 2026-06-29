from __future__ import annotations

import pandas as pd

from app.analysis.contracts import ModuleResult


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    country_col = canonical_columns.get("country")
    sales_col = canonical_columns.get("sales_amount")
    order_col = canonical_columns.get("order_id")
    customer_col = canonical_columns.get("customer_id")
    date_col = canonical_columns.get("order_datetime")
    if not country_col or not sales_col:
        return ModuleResult(
            module_id="country_market_analysis",
            title="国家市场分析",
            chart_type="bar",
            warnings=["缺少国家或销售额字段，无法开展国家市场分析。"],
        )

    metric_map = {sales_col: "sum"}
    grouped = frame.groupby(country_col, as_index=False, dropna=False).agg(metric_map)
    grouped["order_count"] = (
        frame.groupby(country_col, dropna=False)[order_col].nunique().values
        if order_col
        else frame.groupby(country_col, dropna=False).size().values
    )
    if customer_col:
        grouped["customer_count"] = frame.groupby(country_col, dropna=False)[customer_col].nunique().values
    grouped["avg_order_value"] = grouped[sales_col] / grouped["order_count"].replace(0, pd.NA)
    grouped = grouped.sort_values(sales_col, ascending=False)
    total_grouped_sales = float(grouped[sales_col].sum()) if not grouped.empty else 0.0
    grouped["sales_share"] = grouped[sales_col] / total_grouped_sales if total_grouped_sales else 0.0

    table_grouped = grouped.rename(columns={country_col: "country", sales_col: "sales_amount"})
    tables = {"country_market_totals": table_grouped.head(20).to_dict(orient="records")}
    if date_col:
        monthly = frame.assign(_month=pd.to_datetime(frame[date_col], errors="coerce").dt.to_period("M").astype(str))
        monthly = (
            monthly.groupby([country_col, "_month"], as_index=False, dropna=False)[sales_col]
            .sum()
            .sort_values([country_col, "_month"])
        )
        tables["country_monthly_trend"] = monthly.head(120).to_dict(orient="records")

    top = grouped.iloc[0] if not grouped.empty else None
    total_sales = float(grouped[sales_col].sum()) if not grouped.empty else 0.0
    top_share = float(top[sales_col] / total_sales) if top is not None and total_sales else 0.0
    return ModuleResult(
        module_id="country_market_analysis",
        title="国家市场分析",
        chart_type="bar",
        summary_metrics={
            "top_country": str(top[country_col]) if top is not None else "",
            "top_country_sales_share": round(top_share, 4),
            "country_count": int(grouped[country_col].nunique()) if not grouped.empty else 0,
        },
        tables=tables,
        chart_payload={
            "x": grouped[country_col].head(10).tolist(),
            "y": grouped[sales_col].head(10).tolist(),
            "series_name": "country_sales",
        },
        findings=[
            f"销售额最高国家是 {top[country_col]}，销售占比约 {top_share:.1%}。"
            if top is not None
            else "未识别到有效国家市场。"
        ],
    )
