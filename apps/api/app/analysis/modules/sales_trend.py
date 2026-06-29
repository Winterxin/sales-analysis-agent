from __future__ import annotations

import pandas as pd

from app.analysis.contracts import ModuleResult


WEEKDAY_LABELS = {
    0: "周一",
    1: "周二",
    2: "周三",
    3: "周四",
    4: "周五",
    5: "周六",
    6: "周日",
}


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    date_col = canonical_columns["order_datetime"]
    sales_col = canonical_columns["sales_amount"]
    quantity_col = canonical_columns.get("quantity")
    has_quantity = bool(quantity_col and quantity_col in frame.columns)

    working = frame.dropna(subset=[date_col]).copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    working = working.dropna(subset=[date_col]).sort_values(date_col)

    working["order_day"] = working[date_col].dt.floor("D")
    working["order_month"] = working[date_col].dt.to_period("M").astype(str)
    working["year"] = working[date_col].dt.year
    working["month"] = working[date_col].dt.month
    working["weekday"] = working[date_col].dt.weekday

    aggregations: dict[str, str] = {sales_col: "sum"}
    if has_quantity and quantity_col:
        aggregations[quantity_col] = "sum"

    daily = working.groupby("order_day", as_index=False).agg(aggregations).sort_values("order_day")
    daily["period"] = daily["order_day"].dt.date.astype(str)

    monthly = (
        working.groupby("order_month", as_index=False)
        .agg(aggregations)
        .sort_values("order_month")
    )

    weekday_aggregations: dict[str, tuple[str, str]] = {
        "avg_sales": (sales_col, "mean"),
        "day_count": ("order_day", "count"),
    }
    if has_quantity and quantity_col:
        weekday_aggregations["avg_quantity"] = (quantity_col, "mean")
    weekday_profile = (
        daily.assign(
            weekday=daily["order_day"].dt.weekday,
            weekday_cn=daily["order_day"].dt.weekday.map(WEEKDAY_LABELS),
        )
        .groupby(["weekday", "weekday_cn"], as_index=False)
        .agg(**weekday_aggregations)
        .sort_values("weekday")
    )

    year_month_totals = (
        working.groupby(["year", "month"], as_index=False)
        .agg(aggregations)
        .sort_values(["year", "month"])
    )
    year_month_totals["month_name"] = year_month_totals["month"].map(
        {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
    )

    row_columns = ["period", sales_col]
    if has_quantity and quantity_col:
        row_columns.append(quantity_col)
    rows = daily[row_columns].to_dict(orient="records")
    peak_row = max(rows, key=lambda row: float(row[sales_col])) if rows else {}
    trough_row = min(rows, key=lambda row: float(row[sales_col])) if rows else {}

    summary_metrics = {
        "total_sales_amount": round(float(daily[sales_col].sum()), 2) if not daily.empty else 0.0,
        "peak_period": peak_row.get("period"),
        "trough_period": trough_row.get("period"),
        "daily_grain_count": int(len(daily)),
        "monthly_grain_count": int(len(monthly)),
        "year_count": int(year_month_totals["year"].nunique()) if not year_month_totals.empty else 0,
        "has_multi_year": bool(year_month_totals["year"].nunique() >= 2) if not year_month_totals.empty else False,
    }
    warnings: list[str] = []
    if has_quantity and quantity_col:
        summary_metrics["total_quantity"] = int(daily[quantity_col].sum()) if not daily.empty else 0
    else:
        summary_metrics["skipped_quantity_dependent_analysis"] = True
        summary_metrics["skipped_quantity_dependent_analysis_reason"] = "missing quantity"
        warnings.append("skipped_quantity_dependent_analysis: missing quantity")

    return ModuleResult(
        module_id="sales_trend_analysis",
        title="销售趋势分析",
        chart_type="line",
        summary_metrics=summary_metrics,
        tables={
            "daily_totals": rows,
            "monthly_totals": monthly.to_dict(orient="records"),
            "weekday_profile": weekday_profile.to_dict(orient="records"),
            "year_month_totals": year_month_totals.to_dict(orient="records"),
        },
        chart_payload={
            "x": [row["period"] for row in rows],
            "y": [row[sales_col] for row in rows],
            "series_name": "sales_amount",
        },
        findings=[
            f"共覆盖 {len(rows)} 个日度时间粒度。",
            f"月度趋势共覆盖 {len(monthly)} 个时间点。",
            (
                f"峰值日期为 {peak_row.get('period')}，谷值日期为 {trough_row.get('period')}。"
                if rows
                else "当前没有可用于趋势分析的有效时间记录。"
            ),
        ],
        warnings=warnings,
    )
