from __future__ import annotations

import pandas as pd

from app.analysis.contracts import ModuleResult


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    order_col = canonical_columns.get("order_id")
    customer_col = canonical_columns.get("customer_id")
    sales_col = canonical_columns.get("sales_amount")
    quantity_col = canonical_columns.get("quantity")
    productline_col = canonical_columns.get("productline")
    deal_size_col = canonical_columns.get("deal_size")
    status_col = canonical_columns.get("order_status")
    unit_price_col = canonical_columns.get("unit_price")
    if not sales_col:
        return ModuleResult(
            module_id="order_structure_analysis",
            title="订单结构分析",
            chart_type="histogram",
            warnings=["缺少销售额字段，无法开展订单结构分析。"],
        )

    tables: dict[str, list[dict[str, object]]] = {}
    findings: list[str] = []
    summary: dict[str, object] = {}
    if order_col:
        invoice_values = frame.groupby(order_col, as_index=False)[sales_col].sum()
        invoice_values = invoice_values.rename(columns={sales_col: "invoice_value"})
        tables["invoice_values"] = invoice_values.head(50).to_dict(orient="records")
        if quantity_col:
            basket = frame.groupby(order_col, as_index=False).agg(
                basket_size=(quantity_col, "sum"),
                line_count=(quantity_col, "size"),
            )
            tables["basket_sizes"] = basket.head(50).to_dict(orient="records")
            summary["avg_basket_size"] = float(basket["line_count"].mean()) if not basket.empty else 0.0
        findings.append(f"共识别 {len(invoice_values)} 个订单，可按发票金额分布拆解订单结构。")
    if customer_col and order_col:
        customer_orders = frame.groupby(customer_col, as_index=False)[order_col].nunique()
        customer_orders = customer_orders.rename(columns={order_col: "order_count"})
        repeat_rate = float((customer_orders["order_count"] > 1).mean()) if not customer_orders.empty else 0.0
        tables["customer_order_counts"] = customer_orders.head(50).to_dict(orient="records")
        summary["repeat_customer_rate"] = round(repeat_rate, 4)
        findings.append(f"复购客户占比约 {repeat_rate:.1%}。")
    total_sales = float(frame[sales_col].sum()) if sales_col in frame.columns else 0.0
    if productline_col:
        productline = (
            frame.groupby(productline_col, as_index=False, dropna=False)
            .agg(sales_amount=(sales_col, "sum"))
            .sort_values("sales_amount", ascending=False)
        )
        if quantity_col:
            quantities = frame.groupby(productline_col, as_index=False, dropna=False).agg(quantity=(quantity_col, "sum"))
            productline = productline.merge(quantities, on=productline_col, how="left")
        if order_col:
            order_counts = frame.groupby(productline_col, dropna=False)[order_col].nunique().reset_index(name="order_count")
        else:
            order_counts = frame.groupby(productline_col, dropna=False).size().reset_index(name="order_count")
        productline = productline.merge(order_counts, on=productline_col, how="left")
        productline = productline.rename(columns={productline_col: "productline"})
        productline["sales_share"] = productline["sales_amount"] / total_sales if total_sales else 0.0
        tables["productline_sales"] = productline.to_dict(orient="records")
        findings.append("已按 PRODUCTLINE 拆解产品线销售贡献。")
    if deal_size_col:
        deal_size = (
            frame.groupby(deal_size_col, as_index=False, dropna=False)
            .agg(sales_amount=(sales_col, "sum"))
            .sort_values("sales_amount", ascending=False)
        )
        if order_col:
            deal_orders = frame.groupby(deal_size_col, dropna=False)[order_col].nunique().reset_index(name="order_count")
        else:
            deal_orders = frame.groupby(deal_size_col, dropna=False).size().reset_index(name="order_count")
        deal_size = deal_size.merge(deal_orders, on=deal_size_col, how="left")
        deal_size = deal_size.rename(columns={deal_size_col: "deal_size"})
        deal_size["sales_share"] = deal_size["sales_amount"] / total_sales if total_sales else 0.0
        deal_size["avg_order_value"] = deal_size["sales_amount"] / deal_size["order_count"].replace(0, pd.NA)
        tables["deal_size_sales"] = deal_size.to_dict(orient="records")
        findings.append("已按 DEALSIZE 拆解不同交易规模的销售贡献。")
    if status_col:
        status = (
            frame.groupby(status_col, as_index=False, dropna=False)
            .agg(sales_amount=(sales_col, "sum"))
            .sort_values("sales_amount", ascending=False)
        )
        if order_col:
            status_orders = frame.groupby(status_col, dropna=False)[order_col].nunique().reset_index(name="order_count")
        else:
            status_orders = frame.groupby(status_col, dropna=False).size().reset_index(name="order_count")
        status_rows = frame.groupby(status_col, dropna=False).size().reset_index(name="order_rows")
        status = status.merge(status_orders, on=status_col, how="left").merge(status_rows, on=status_col, how="left")
        status = status.rename(columns={status_col: "order_status"})
        status["sales_share"] = status["sales_amount"] / total_sales if total_sales else 0.0
        tables["order_status_breakdown"] = status.to_dict(orient="records")
        findings.append("已按 STATUS 拆解订单状态结构。")
    if unit_price_col and quantity_col:
        tables["price_quantity_sample"] = frame[[unit_price_col, quantity_col, sales_col]].head(100).to_dict(orient="records")

    return ModuleResult(
        module_id="order_structure_analysis",
        title="订单结构分析",
        chart_type="histogram",
        summary_metrics=summary,
        tables=tables,
        chart_payload={},
        findings=findings or ["已生成订单结构分析底表。"],
    )
