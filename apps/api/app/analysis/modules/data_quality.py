from __future__ import annotations

import pandas as pd

from app.analysis.contracts import ModuleResult


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    quantity_col = canonical_columns.get("quantity")
    sales_col = canonical_columns.get("sales_amount")
    date_col = canonical_columns.get("order_datetime")

    missing_dates = int(frame[date_col].isna().sum()) if date_col else 0
    negative_sales = int((frame[sales_col] < 0).sum()) if sales_col else 0
    negative_quantity = int((frame[quantity_col] < 0).sum()) if quantity_col else 0
    duplicate_rows = int(frame.duplicated().sum())

    return ModuleResult(
        module_id="data_quality_check",
        title="数据质量检查",
        chart_type="bar",
        summary_metrics={
            "missing_order_datetime": missing_dates,
            "negative_sales_amount": negative_sales,
            "negative_quantity": negative_quantity,
            "duplicate_rows": duplicate_rows,
        },
        tables={
            "quality_metrics": [
                {"metric": "missing_order_datetime", "value": missing_dates},
                {"metric": "negative_sales_amount", "value": negative_sales},
                {"metric": "negative_quantity", "value": negative_quantity},
                {"metric": "duplicate_rows", "value": duplicate_rows},
            ]
        },
        chart_payload={
            "labels": [
                "missing_order_datetime",
                "negative_sales_amount",
                "negative_quantity",
                "duplicate_rows",
            ],
            "values": [missing_dates, negative_sales, negative_quantity, duplicate_rows],
        },
        findings=[
            f"发现 {missing_dates} 条时间缺失记录。",
            f"发现 {negative_sales} 条负销售额记录。",
            f"发现 {duplicate_rows} 条重复记录。",
        ],
    )
