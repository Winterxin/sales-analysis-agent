from __future__ import annotations

import pandas as pd

from app.analysis.contracts import ModuleResult


def _safe_round(value: object, digits: int = 4) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(numeric):
        return 0.0
    return round(numeric, digits)


def _profit_margin(profit: object, sales: object) -> float:
    try:
        profit_value = float(profit)  # type: ignore[arg-type]
        sales_value = float(sales)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if sales_value == 0 or pd.isna(profit_value) or pd.isna(sales_value):
        return 0.0
    return round(profit_value / sales_value, 4)


def _round_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: _safe_round(value))
    return rounded.to_dict(orient="records")


def _products_to_share(grouped: pd.DataFrame, share: float) -> int:
    if grouped.empty or "cumulative_sales_share" not in grouped.columns:
        return 0
    reached = grouped[grouped["cumulative_sales_share"] >= share]
    if reached.empty:
        return int(len(grouped))
    return int(reached.index[0]) + 1


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    product_col = canonical_columns.get("product_name") or canonical_columns.get("sku")
    sales_col = canonical_columns["sales_amount"]
    quantity_col = canonical_columns.get("quantity")
    profit_col = canonical_columns.get("profit")
    discount_col = canonical_columns.get("discount")
    category_col = canonical_columns.get("category")
    sub_category_col = canonical_columns.get("sub_category")

    if product_col is None:
        return ModuleResult(
            module_id="product_contribution_analysis",
            title="商品贡献分析",
            chart_type="bar",
            warnings=["缺少商品字段，无法展开商品贡献分析。"],
        )

    working = frame.copy()
    working[sales_col] = pd.to_numeric(working[sales_col], errors="coerce")
    if quantity_col:
        working[quantity_col] = pd.to_numeric(working[quantity_col], errors="coerce")
    if profit_col:
        working[profit_col] = pd.to_numeric(working[profit_col], errors="coerce")

    aggregations: dict[str, str] = {sales_col: "sum"}
    if quantity_col:
        aggregations[quantity_col] = "sum"
    if profit_col:
        aggregations[profit_col] = "sum"

    grouped = (
        working.dropna(subset=[product_col, sales_col])
        .groupby(product_col, as_index=False)
        .agg(aggregations)
        .sort_values(sales_col, ascending=False)
        .reset_index(drop=True)
    )
    total_sales = float(grouped[sales_col].sum()) if not grouped.empty else 0.0
    grouped["sales_share"] = grouped[sales_col] / total_sales if total_sales else 0.0
    grouped["cumulative_sales_share"] = grouped["sales_share"].cumsum()
    if profit_col:
        grouped["profit_margin"] = grouped.apply(
            lambda row: _profit_margin(row[profit_col], row[sales_col]),
            axis=1,
        )

    top_rows = _round_records(grouped.head(15))
    tables: dict[str, list[dict[str, object]]] = {"top_products": top_rows}

    if category_col:
        category_keys = [category_col]
        if sub_category_col:
            category_keys.append(sub_category_col)
        category_aggregations = {sales_col: "sum"}
        if quantity_col:
            category_aggregations[quantity_col] = "sum"
        if profit_col:
            category_aggregations[profit_col] = "sum"
        category_sales = (
            working.dropna(subset=category_keys + [sales_col])
            .groupby(category_keys, as_index=False)
            .agg(category_aggregations)
            .sort_values(sales_col, ascending=False)
            .reset_index(drop=True)
        )
        category_sales["sales_share"] = (
            category_sales[sales_col] / float(category_sales[sales_col].sum())
            if not category_sales.empty and float(category_sales[sales_col].sum()) != 0
            else 0.0
        )
        if profit_col:
            category_sales["profit_margin"] = category_sales.apply(
                lambda row: _profit_margin(row[profit_col], row[sales_col]),
                axis=1,
            )
        tables["category_sales"] = _round_records(category_sales.head(30))

        if profit_col:
            category_profit_quality = (
                working.dropna(subset=[category_col, sales_col])
                .groupby(category_col, as_index=False)[[sales_col, profit_col]]
                .sum()
                .sort_values(profit_col)
                .reset_index(drop=True)
            )
            category_profit_quality["profit_margin"] = category_profit_quality.apply(
                lambda row: _profit_margin(row[profit_col], row[sales_col]),
                axis=1,
            )
            tables["category_profit_quality"] = _round_records(category_profit_quality)

    if profit_col and not grouped.empty:
        sales_threshold = float(grouped[sales_col].quantile(0.75))
        high_sales_low_profit = grouped[
            (grouped[sales_col] >= sales_threshold)
            & (grouped["profit_margin"] <= 0.08)
        ].copy()
        if high_sales_low_profit.empty:
            high_sales_low_profit = grouped.sort_values(
                by=["profit_margin", sales_col],
                ascending=[True, False],
            ).head(10)
        else:
            high_sales_low_profit = high_sales_low_profit.sort_values(
                by=["profit_margin", sales_col],
                ascending=[True, False],
            ).head(10)
        tables["high_sales_low_profit_products"] = _round_records(high_sales_low_profit)

    summary_metrics: dict[str, object] = {
        "distinct_products": int(grouped[product_col].nunique()) if not grouped.empty else 0,
        "top_product": top_rows[0][product_col] if top_rows else None,
        "top_1_sales_share": _safe_round(grouped["sales_share"].head(1).sum())
        if not grouped.empty
        else 0.0,
        "top_5_sales_share": _safe_round(grouped["sales_share"].head(5).sum())
        if not grouped.empty
        else 0.0,
        "top_10_sales_share": _safe_round(grouped["sales_share"].head(10).sum())
        if not grouped.empty
        else 0.0,
        "products_to_80pct_sales": _products_to_share(grouped, 0.8),
    }
    if profit_col and tables.get("high_sales_low_profit_products"):
        summary_metrics["high_sales_low_profit_count"] = len(
            tables["high_sales_low_profit_products"]
        )

    findings = [
        f"Top 商品为 {top_rows[0][product_col]}。" if top_rows else "暂无商品数据。"
    ]
    if top_rows:
        findings.append(
            f"头部商品销售集中度：Top 10 销售额占比约 {summary_metrics['top_10_sales_share']}，达到 80% 销售额需要约 {summary_metrics['products_to_80pct_sales']} 个商品。"
        )
    if profit_col and tables.get("high_sales_low_profit_products"):
        if discount_col:
            findings.append("已识别高销售低利润商品，需要进一步核查毛利、折扣和补货策略。")
        else:
            findings.append("已识别高销售低利润商品，需要进一步核查毛利、定价、成本和补货策略。")

    return ModuleResult(
        module_id="product_contribution_analysis",
        title="商品贡献分析",
        chart_type="bar",
        summary_metrics=summary_metrics,
        tables=tables,
        chart_payload={
            "x": [row[product_col] for row in top_rows],
            "y": [row[sales_col] for row in top_rows],
            "series_name": "sales_amount",
        },
        findings=findings,
    )
