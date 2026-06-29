from __future__ import annotations

import pandas as pd

from app.analysis.contracts import ModuleResult


def _pick_dimensions(canonical_columns: dict[str, str]) -> tuple[str | None, str | None]:
    priority = ("segment", "region", "state", "city", "country")
    available = [candidate for candidate in priority if candidate in canonical_columns]
    primary = available[0] if available else None
    secondary = available[1] if len(available) > 1 else None
    return primary, secondary


def _format_slice_label(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            continue
        parts.append(text)
    return " / ".join(parts)


def _build_grouped_view(
    frame: pd.DataFrame,
    group_cols: list[str],
    sales_col: str,
    profit_col: str | None,
    quantity_col: str | None,
) -> pd.DataFrame:
    metric_map = {sales_col: "sum"}
    if profit_col:
        metric_map[profit_col] = "sum"
    if quantity_col:
        metric_map[quantity_col] = "sum"

    grouped = (
        frame.groupby(group_cols, as_index=False, observed=False, dropna=False)
        .agg(metric_map)
        .sort_values(sales_col, ascending=False)
    )
    counts = (
        frame.groupby(group_cols, observed=False, dropna=False)
        .size()
        .reset_index(name="order_count")
    )
    grouped = grouped.merge(counts, on=group_cols, how="left")
    total_sales = float(grouped[sales_col].sum())
    grouped["sales_share"] = grouped[sales_col] / total_sales if total_sales else 0

    if profit_col:
        grouped["profit_margin"] = grouped[profit_col] / grouped[sales_col].replace(0, pd.NA)
        loss_rate = (
            frame.assign(_negative_profit=frame[profit_col] < 0)
            .groupby(group_cols, observed=False, dropna=False)["_negative_profit"]
            .mean()
            .reset_index(name="negative_profit_rate")
        )
        grouped = grouped.merge(loss_rate, on=group_cols, how="left")

    return grouped


def _weak_profit_cuts(
    grouped: pd.DataFrame,
    sales_col: str,
    profit_col: str,
) -> pd.DataFrame:
    if grouped.empty or "profit_margin" not in grouped:
        return grouped.head(0)

    sales_threshold = grouped[sales_col].quantile(0.5)
    weak = grouped[
        (grouped[sales_col] >= sales_threshold)
        & ((grouped[profit_col] < 0) | (grouped["profit_margin"] <= 0.08))
    ]
    weak = weak if not weak.empty else grouped
    return weak.sort_values(
        ["profit_margin", profit_col, sales_col],
        ascending=[True, True, False],
    ).head(10)


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    sales_col = canonical_columns["sales_amount"]
    profit_col = canonical_columns.get("profit")
    quantity_col = canonical_columns.get("quantity")
    primary_dimension, secondary_dimension = _pick_dimensions(canonical_columns)

    if primary_dimension is None:
        return ModuleResult(
            module_id="dimension_breakdown_analysis",
            title="客群与区域分析",
            chart_type="stacked_bar",
            warnings=["未找到可用于拆解的客群或地域维度。"],
        )

    primary_col = canonical_columns[primary_dimension]
    grouped_primary = _build_grouped_view(
        frame,
        [primary_col],
        sales_col=sales_col,
        profit_col=profit_col,
        quantity_col=quantity_col,
    )
    primary_rows = grouped_primary.to_dict(orient="records")

    tables: dict[str, list[dict[str, object]]] = {
        "dimension_totals": primary_rows,
        "primary_profit_quality": (
            grouped_primary.sort_values("profit_margin", ascending=True).to_dict(orient="records")
            if profit_col and "profit_margin" in grouped_primary
            else primary_rows
        ),
    }
    summary_metrics: dict[str, object] = {
        "primary_dimension": primary_dimension,
        "secondary_dimension": secondary_dimension,
        "group_count": len(primary_rows),
    }
    findings: list[str] = []
    chart_type = "bar"
    chart_payload = {
        "x": [row[primary_col] for row in primary_rows],
        "y": [row[sales_col] for row in primary_rows],
        "series_name": primary_dimension,
    }

    analysis_scope = grouped_primary
    scope_columns = [primary_col]

    if secondary_dimension is not None:
        secondary_col = canonical_columns[secondary_dimension]
        cross_breakdown = _build_grouped_view(
            frame,
            [primary_col, secondary_col],
            sales_col=sales_col,
            profit_col=profit_col,
            quantity_col=quantity_col,
        )
        tables["segment_region_matrix"] = cross_breakdown.to_dict(orient="records")
        analysis_scope = cross_breakdown
        scope_columns = [primary_col, secondary_col]
        chart_type = "stacked_bar"
        chart_payload = {
            "x": [row[primary_col] for row in tables["segment_region_matrix"]],
            "y": [row[sales_col] for row in tables["segment_region_matrix"]],
            "series_name": secondary_dimension,
        }
        findings.append(f"已按 {primary_dimension} x {secondary_dimension} 完成双维拆解。")
    else:
        findings.append(f"已按 {primary_dimension} 完成单维拆解。")

    top_scope = analysis_scope.sort_values(sales_col, ascending=False).head(10)
    tables["top_performance_cuts"] = top_scope.to_dict(orient="records")
    if not top_scope.empty:
        top_row = top_scope.iloc[0]
        summary_metrics["top_slice_label"] = _format_slice_label(top_row, scope_columns)
        summary_metrics["top_slice_sales"] = float(top_row[sales_col])
        if profit_col:
            summary_metrics["top_slice_profit"] = float(top_row[profit_col])
        findings.append(
            f"销售额最高切片是 {summary_metrics['top_slice_label']}，销售额约 {summary_metrics['top_slice_sales']:.2f}。"
        )

    if profit_col:
        weak = _weak_profit_cuts(analysis_scope, sales_col, profit_col)
        tables["weak_performance_cuts"] = weak.to_dict(orient="records")
        summary_metrics["negative_profit_group_count"] = int((analysis_scope[profit_col] < 0).sum())
        summary_metrics["high_sales_low_profit_cut_count"] = len(weak)
        if not weak.empty:
            weakest_row = weak.iloc[0]
            weakest_label = _format_slice_label(weakest_row, scope_columns)
            summary_metrics["weakest_slice_label"] = weakest_label
            summary_metrics["weakest_slice_profit"] = float(weakest_row[profit_col])
            summary_metrics["weakest_slice_margin"] = float(weakest_row.get("profit_margin", 0))
            findings.append(
                f"{weakest_label} 属于高销售低利润切片，利润率约 {summary_metrics['weakest_slice_margin']:.1%}，值得重点复盘。"
            )

    return ModuleResult(
        module_id="dimension_breakdown_analysis",
        title="客群与区域分析",
        chart_type=chart_type,
        summary_metrics=summary_metrics,
        tables=tables,
        chart_payload=chart_payload,
        findings=findings,
    )
