from __future__ import annotations

import pandas as pd

from app.analysis.contracts import ModuleResult


DISCOUNT_BUCKET_ORDER = ["0%", "0-10%", "10-20%", "20-30%", "30%+"]


def _bucket_discount(value: float) -> str:
    if value <= 0:
        return "0%"
    if value <= 0.1:
        return "0-10%"
    if value <= 0.2:
        return "10-20%"
    if value <= 0.3:
        return "20-30%"
    return "30%+"


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


def _round_records(rows: pd.DataFrame) -> list[dict[str, object]]:
    rounded = rows.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: _safe_round(value))
    return rounded.to_dict(orient="records")


def _discount_bucket_rows(
    discount_profit: pd.DataFrame,
    *,
    discount_col: str,
    profit_col: str,
    sales_col: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bucket in DISCOUNT_BUCKET_ORDER:
        bucket_frame = discount_profit[discount_profit["discount_bucket"] == bucket]
        if bucket_frame.empty:
            continue

        profit_sum = float(bucket_frame[profit_col].sum())
        sales_sum = float(bucket_frame[sales_col].sum()) if sales_col else 0.0
        negative_count = int((bucket_frame[profit_col] < 0).sum())
        order_count = int(len(bucket_frame))
        row: dict[str, object] = {
            "discount_bucket": bucket,
            "order_count": order_count,
            "avg_discount": _safe_round(bucket_frame[discount_col].mean()),
            "avg_profit": _safe_round(bucket_frame[profit_col].mean()),
            "total_profit": _safe_round(profit_sum, 2),
            "negative_profit_count": negative_count,
            "negative_profit_rate": round(negative_count / order_count, 4)
            if order_count
            else 0.0,
        }
        if sales_col:
            row.update(
                {
                    "avg_sales": _safe_round(bucket_frame[sales_col].mean()),
                    "total_sales": _safe_round(sales_sum, 2),
                    "profit_margin": _profit_margin(profit_sum, sales_sum),
                }
            )
        rows.append(row)
    return rows


def _discount_threshold_candidate_rows(bucket_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in bucket_rows:
        avg_profit = float(row.get("avg_profit", 0) or 0)
        profit_margin = float(row.get("profit_margin", 0) or 0)
        negative_profit_rate = float(row.get("negative_profit_rate", 0) or 0)
        if negative_profit_rate >= 0.5 or profit_margin < 0 or avg_profit < 0:
            risk_level = "high"
            action_hint = "设为折扣审批红线，优先复盘商品、客户和授权口径。"
        elif negative_profit_rate >= 0.15 or profit_margin < 0.05:
            risk_level = "medium"
            action_hint = "纳入周度监控，观察亏损率和利润率是否继续恶化。"
        else:
            risk_level = "low"
            action_hint = "保留为对照区间，暂不作为折扣收紧重点。"

        order_count = int(row.get("order_count", 0) or 0)
        candidates.append(
            {
                "discount_bucket": row.get("discount_bucket", ""),
                "row_count": order_count,
                "order_count": order_count,
                "sales_amount": _safe_round(row.get("total_sales", 0), 2),
                "avg_profit": _safe_round(avg_profit),
                "profit_margin": _safe_round(profit_margin),
                "negative_profit_rate": _safe_round(negative_profit_rate),
                "risk_level": risk_level,
                "action_hint": action_hint,
            }
        )
    risk_rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        candidates,
        key=lambda item: (
            risk_rank.get(str(item.get("risk_level")), 3),
            -float(item.get("negative_profit_rate", 0) or 0),
            float(item.get("profit_margin", 0) or 0),
            float(item.get("avg_profit", 0) or 0),
        ),
    )


def _target_cap_for_bucket(bucket: object) -> float:
    label = str(bucket or "")
    normalized = label.replace("%", "").replace("+", "").strip()
    lower_bound = normalized.split("-", 1)[0].strip()
    try:
        return max(float(lower_bound) / 100, 0.0)
    except (TypeError, ValueError):
        pass
    return 0.2


def _highest_risk_threshold(threshold_rows: list[dict[str, object]]) -> dict[str, object] | None:
    high_rows = [row for row in threshold_rows if str(row.get("risk_level", "")).lower() == "high"]
    if not high_rows:
        return None
    return high_rows[0]


def _discount_cap_what_if_rows(
    discount_profit: pd.DataFrame,
    *,
    threshold_rows: list[dict[str, object]],
    discount_col: str,
    profit_col: str,
    sales_col: str | None,
) -> list[dict[str, object]]:
    if not sales_col:
        return []
    high_risk = _highest_risk_threshold(threshold_rows)
    if not high_risk:
        return []

    high_risk_bucket = str(high_risk.get("discount_bucket") or high_risk.get("threshold") or "")
    target_cap = _target_cap_for_bucket(high_risk_bucket)
    working = discount_profit[[discount_col, profit_col, sales_col, "discount_bucket"]].copy()
    working[discount_col] = pd.to_numeric(working[discount_col], errors="coerce").clip(lower=0, upper=1)
    working[profit_col] = pd.to_numeric(working[profit_col], errors="coerce")
    working[sales_col] = pd.to_numeric(working[sales_col], errors="coerce")
    working = working.dropna(subset=[discount_col, profit_col, sales_col])
    if working.empty:
        return []

    if high_risk_bucket:
        affected = working[working["discount_bucket"].astype(str) == high_risk_bucket].copy()
    else:
        affected = working[working[discount_col] > target_cap].copy()
    affected = affected[affected[discount_col] > target_cap].copy()
    if affected.empty:
        return []

    recovered_discount_amount = (affected[discount_col] - target_cap).clip(lower=0) * affected[sales_col]
    estimated_profit = affected[profit_col] + recovered_discount_amount
    current_profit = float(affected[profit_col].sum())
    estimated_profit_after_cap = float(estimated_profit.sum())
    affected_count = int(len(affected))
    return [
        {
            "scenario_name": "高风险折扣收紧情景估算",
            "current_threshold": high_risk_bucket,
            "high_risk_bucket": high_risk_bucket,
            "target_discount_cap": _safe_round(target_cap),
            "affected_row_count": affected_count,
            "affected_sales_amount": _safe_round(affected[sales_col].sum(), 2),
            "current_profit": _safe_round(current_profit, 2),
            "estimated_profit_after_cap": _safe_round(estimated_profit_after_cap, 2),
            "estimated_profit_delta": _safe_round(estimated_profit_after_cap - current_profit, 2),
            "current_negative_profit_rate": _safe_round(float((affected[profit_col] < 0).mean())),
            "estimated_negative_profit_rate_after_cap": _safe_round(float((estimated_profit < 0).mean())),
            "assumption_note": "这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。",
        }
    ]


def _product_risk_rows(
    discount_profit: pd.DataFrame,
    *,
    product_col: str,
    sales_col: str,
    profit_col: str,
    discount_col: str,
) -> list[dict[str, object]]:
    grouped = (
        discount_profit.groupby(product_col, as_index=False)
        .agg(
            **{
                sales_col: (sales_col, "sum"),
                profit_col: (profit_col, "sum"),
                "avg_discount": (discount_col, "mean"),
                "order_count": (profit_col, "size"),
                "negative_profit_count": (profit_col, lambda series: int((series < 0).sum())),
            }
        )
        .copy()
    )
    if grouped.empty:
        return []

    grouped["profit_margin"] = grouped.apply(
        lambda row: _profit_margin(row[profit_col], row[sales_col]),
        axis=1,
    )
    grouped["negative_profit_rate"] = (
        grouped["negative_profit_count"] / grouped["order_count"]
    ).round(4)
    sales_threshold = float(grouped[sales_col].quantile(0.75))
    candidates = grouped[
        (grouped[sales_col] >= sales_threshold) & (grouped["profit_margin"] <= 0.05)
    ]
    if candidates.empty:
        candidates = grouped

    candidates = candidates.sort_values(
        by=["profit_margin", sales_col],
        ascending=[True, False],
    ).head(10)
    return _round_records(candidates)


def _category_discount_risk_rows(
    discount_profit: pd.DataFrame,
    *,
    category_col: str,
    sales_col: str,
    profit_col: str,
    discount_col: str,
) -> list[dict[str, object]]:
    grouped = (
        discount_profit.groupby([category_col, "discount_bucket"], as_index=False)
        .agg(
            **{
                sales_col: (sales_col, "sum"),
                profit_col: (profit_col, "sum"),
                "avg_discount": (discount_col, "mean"),
                "order_count": (profit_col, "size"),
                "negative_profit_count": (profit_col, lambda series: int((series < 0).sum())),
            }
        )
        .copy()
    )
    if grouped.empty:
        return []

    grouped["profit_margin"] = grouped.apply(
        lambda row: _profit_margin(row[profit_col], row[sales_col]),
        axis=1,
    )
    grouped["negative_profit_rate"] = (
        grouped["negative_profit_count"] / grouped["order_count"]
    ).round(4)
    grouped = grouped.sort_values(
        by=["negative_profit_rate", "profit_margin", sales_col],
        ascending=[False, True, False],
    ).head(30)
    return _round_records(grouped)


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    sales_col = canonical_columns.get("sales_amount")
    profit_col = canonical_columns.get("profit")
    discount_col = canonical_columns.get("discount")
    category_col = canonical_columns.get("category")
    product_col = canonical_columns.get("product_name") or canonical_columns.get("sku")

    if profit_col is None:
        return ModuleResult(
            module_id="discount_profit_analysis",
            title="折扣与利润分析",
            chart_type="bar",
            warnings=["缺少 profit 字段，无法展开利润专题分析。"],
        )

    working = frame.copy()
    working[profit_col] = pd.to_numeric(working[profit_col], errors="coerce")
    if sales_col:
        working[sales_col] = pd.to_numeric(working[sales_col], errors="coerce")
    if discount_col:
        working[discount_col] = pd.to_numeric(working[discount_col], errors="coerce").fillna(0)

    profit_only = working.dropna(subset=[profit_col]).copy()
    negative_profit_orders = int((profit_only[profit_col] < 0).sum())
    summary_metrics: dict[str, object] = {
        "negative_profit_order_count": negative_profit_orders,
        "total_profit_amount": _safe_round(profit_only[profit_col].sum(), 2)
        if not profit_only.empty
        else 0.0,
    }
    tables: dict[str, list[dict[str, object]]] = {}
    findings: list[str] = []
    warnings: list[str] = []
    chart_payload: dict[str, object] = {}
    chart_type = "bar"

    if sales_col and category_col:
        profit_by_category = (
            profit_only.groupby(category_col, as_index=False)[[sales_col, profit_col]]
            .sum()
            .sort_values(profit_col)
        )
        tables["profit_by_category"] = _round_records(profit_by_category.head(10))

    if sales_col and product_col:
        loss_making_products = (
            profit_only.groupby(product_col, as_index=False)[[sales_col, profit_col]]
            .sum()
            .sort_values(profit_col)
        )
        tables["loss_making_products"] = _round_records(loss_making_products.head(10))

    if discount_col is None:
        warnings.append("缺少 discount 字段，已降级为利润概览分析。")
        findings.append(f"当前共有 {negative_profit_orders} 条利润为负的记录，需要优先排查亏损来源。")
        if tables.get("loss_making_products"):
            findings.append("已输出亏损商品列表，可继续排查高销售低利润商品。")
        chart_payload = {
            "x": [
                row.get(category_col, "unknown")
                for row in tables.get("profit_by_category", [])
            ],
            "y": [row.get(profit_col, 0) for row in tables.get("profit_by_category", [])],
            "series_name": "profit",
        }
        return ModuleResult(
            module_id="discount_profit_analysis",
            title="折扣与利润分析",
            chart_type=chart_type,
            summary_metrics=summary_metrics,
            tables=tables,
            chart_payload=chart_payload,
            findings=findings,
            warnings=warnings,
        )

    discount_profit = profit_only.dropna(subset=[discount_col]).copy()
    discount_profit["discount_bucket"] = discount_profit[discount_col].map(_bucket_discount)
    bucket_rows = _discount_bucket_rows(
        discount_profit,
        discount_col=discount_col,
        profit_col=profit_col,
        sales_col=sales_col,
    )
    tables["discount_buckets"] = bucket_rows
    tables["discount_profit_risk_buckets"] = bucket_rows
    tables["discount_threshold_candidates"] = _discount_threshold_candidate_rows(bucket_rows)
    tables["discount_cap_what_if"] = _discount_cap_what_if_rows(
        discount_profit,
        threshold_rows=tables["discount_threshold_candidates"],
        discount_col=discount_col,
        profit_col=profit_col,
        sales_col=sales_col,
    )
    if sales_col and not tables["discount_cap_what_if"]:
        warnings.append("未识别到高风险折扣区间，已跳过折扣收紧 what-if 情景估算。")
    elif not sales_col:
        warnings.append("缺少 sales_amount 字段，无法计算折扣收紧 what-if 情景估算。")

    high_discount_rows = discount_profit[discount_profit[discount_col] >= 0.3]
    summary_metrics.update(
        {
            "avg_discount": _safe_round(discount_profit[discount_col].mean())
            if not discount_profit.empty
            else 0.0,
            "high_discount_order_count": int((discount_profit[discount_col] >= 0.3).sum()),
            "negative_profit_rate": round(float((discount_profit[profit_col] < 0).mean()), 4)
            if not discount_profit.empty
            else 0.0,
        }
    )
    correlation_source = discount_profit[[discount_col, profit_col]].dropna()
    if len(correlation_source) >= 2:
        summary_metrics["discount_profit_correlation"] = _safe_round(
            correlation_source[discount_col].corr(correlation_source[profit_col])
        )

    if bucket_rows:
        worst_bucket = min(bucket_rows, key=lambda row: float(row.get("avg_profit", 0) or 0))
        summary_metrics["worst_discount_bucket"] = worst_bucket["discount_bucket"]
        summary_metrics["worst_bucket_negative_profit_rate"] = worst_bucket.get(
            "negative_profit_rate", 0.0
        )
    if tables["discount_threshold_candidates"]:
        highest_risk_threshold = tables["discount_threshold_candidates"][0]
        summary_metrics["highest_risk_discount_threshold"] = highest_risk_threshold.get(
            "discount_bucket", ""
        )
        summary_metrics["highest_risk_threshold_negative_profit_rate"] = highest_risk_threshold.get(
            "negative_profit_rate", 0.0
        )

    if sales_col and product_col:
        tables["high_sales_low_profit_items"] = _product_risk_rows(
            discount_profit,
            product_col=product_col,
            sales_col=sales_col,
            profit_col=profit_col,
            discount_col=discount_col,
        )
        summary_metrics["high_sales_low_profit_count"] = len(
            tables["high_sales_low_profit_items"]
        )

        if not high_discount_rows.empty:
            high_discount_losses = (
                high_discount_rows.groupby(product_col, as_index=False)[[sales_col, profit_col]]
                .sum()
                .sort_values(profit_col)
            )
            tables["loss_making_products"] = _round_records(high_discount_losses.head(10))

    if sales_col and category_col:
        tables["category_discount_risk"] = _category_discount_risk_rows(
            discount_profit,
            category_col=category_col,
            sales_col=sales_col,
            profit_col=profit_col,
            discount_col=discount_col,
        )
        summary_metrics["category_discount_risk_count"] = len(
            tables["category_discount_risk"]
        )

    chart_type = "scatter"
    chart_payload = {
        "x": discount_profit[discount_col].round(4).tolist(),
        "y": discount_profit[profit_col].round(2).tolist(),
        "series_name": "discount_vs_profit",
    }
    findings.append(
        f"当前共有 {summary_metrics['high_discount_order_count']} 条高折扣记录，整体负利润记录占比为 {summary_metrics['negative_profit_rate']}。"
    )
    if bucket_rows:
        worst_bucket = min(bucket_rows, key=lambda row: float(row.get("avg_profit", 0) or 0))
        findings.append(
            f"折扣区间 {worst_bucket['discount_bucket']} 的平均利润最低，亏损率为 {worst_bucket.get('negative_profit_rate', 0.0)}，值得重点复盘。"
        )
    if tables.get("high_sales_low_profit_items"):
        findings.append("已识别高销售低利润商品，可继续核查折扣审批、毛利和促销效率。")
    if tables.get("discount_cap_what_if"):
        findings.append("已输出折扣收紧 what-if 情景估算；这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。")

    return ModuleResult(
        module_id="discount_profit_analysis",
        title="折扣与利润分析",
        chart_type=chart_type,
        summary_metrics=summary_metrics,
        tables=tables,
        chart_payload=chart_payload,
        findings=findings,
        warnings=warnings,
    )
