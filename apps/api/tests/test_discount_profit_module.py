from __future__ import annotations

import pandas as pd

from app.analysis.modules import discount_profit


def test_discount_profit_module_identifies_margin_risk_and_discount_buckets() -> None:
    frame = pd.DataFrame(
        {
            "Discount": [0.0, 0.1, 0.2, 0.4, 0.4, 0.5],
            "Profit": [40, 35, 20, -10, -15, -20],
            "Sales": [400, 320, 280, 200, 150, 120],
            "Category": ["Tech", "Tech", "Office", "Office", "Furniture", "Furniture"],
            "Product Name": ["A", "B", "C", "D", "E", "F"],
        }
    )

    result = discount_profit.run(
        frame,
        {
            "discount": "Discount",
            "profit": "Profit",
            "sales_amount": "Sales",
            "category": "Category",
            "product_name": "Product Name",
        },
    )

    assert result.module_id == "discount_profit_analysis"
    assert result.chart_type == "scatter"
    assert result.summary_metrics["negative_profit_order_count"] == 3
    assert result.summary_metrics["high_discount_order_count"] == 3
    assert result.tables["discount_buckets"]
    assert result.tables["loss_making_products"]
    assert any("折扣" in finding for finding in result.findings)


def test_discount_profit_module_outputs_strategy_ready_risk_tables() -> None:
    frame = pd.DataFrame(
        {
            "Discount": [0.0, 0.05, 0.2, 0.35, 0.4, 0.45, 0.5],
            "Profit": [90, 70, 30, -30, -70, -90, 5],
            "Sales": [900, 700, 300, 600, 900, 1000, 1200],
            "Category": [
                "Tech",
                "Tech",
                "Office",
                "Office",
                "Furniture",
                "Furniture",
                "Furniture",
            ],
            "Product Name": ["A", "B", "C", "D", "E", "F", "G"],
        }
    )

    result = discount_profit.run(
        frame,
        {
            "discount": "Discount",
            "profit": "Profit",
            "sales_amount": "Sales",
            "category": "Category",
            "product_name": "Product Name",
        },
    )

    bucket_rows = result.tables["discount_profit_risk_buckets"]
    high_risk_bucket = next(row for row in bucket_rows if row["discount_bucket"] == "30%+")

    assert high_risk_bucket["negative_profit_rate"] == 0.75
    assert high_risk_bucket["order_count"] == 4
    assert "profit_margin" in high_risk_bucket
    assert result.tables["high_sales_low_profit_items"]
    assert result.tables["category_discount_risk"]
    assert result.summary_metrics["worst_discount_bucket"] == "30%+"
    assert result.summary_metrics["high_sales_low_profit_count"] >= 1


def test_discount_profit_module_outputs_discount_threshold_candidates() -> None:
    frame = pd.DataFrame(
        {
            "Discount": [0.0, 0.05, 0.12, 0.25, 0.35, 0.45],
            "Profit": [100, 60, 20, -10, -40, -70],
            "Sales": [1000, 600, 400, 300, 500, 700],
        }
    )

    result = discount_profit.run(
        frame,
        {
            "discount": "Discount",
            "profit": "Profit",
            "sales_amount": "Sales",
        },
    )

    threshold_rows = result.tables["discount_threshold_candidates"]
    high_risk_row = next(row for row in threshold_rows if row["discount_bucket"] == "30%+")

    assert threshold_rows
    assert {
        "discount_bucket",
        "row_count",
        "order_count",
        "sales_amount",
        "avg_profit",
        "profit_margin",
        "negative_profit_rate",
        "risk_level",
        "action_hint",
    } <= set(high_risk_row)
    assert high_risk_row["risk_level"] == "high"
    assert high_risk_row["negative_profit_rate"] == 1.0


def test_discount_profit_module_outputs_discount_cap_what_if_when_high_risk_exists() -> None:
    frame = pd.DataFrame(
        {
            "Discount": [0.0, 0.1, 0.2, 0.45, 0.5, 0.55],
            "Profit": [80.0, 40.0, 10.0, -40.0, -60.0, -80.0],
            "Sales": [800.0, 500.0, 400.0, 500.0, 600.0, 700.0],
        }
    )

    result = discount_profit.run(
        frame,
        {
            "discount": "Discount",
            "profit": "Profit",
            "sales_amount": "Sales",
        },
    )

    scenario_rows = result.tables["discount_cap_what_if"]
    scenario = scenario_rows[0]

    assert scenario_rows
    assert {
        "scenario_name",
        "current_threshold",
        "high_risk_bucket",
        "target_discount_cap",
        "affected_row_count",
        "affected_sales_amount",
        "current_profit",
        "estimated_profit_after_cap",
        "estimated_profit_delta",
        "current_negative_profit_rate",
        "estimated_negative_profit_rate_after_cap",
    } <= set(scenario)
    assert scenario["estimated_profit_delta"] > 0
    assert scenario["estimated_profit_after_cap"] > scenario["current_profit"]
    assert "情景估算" in result.findings[-1]


def test_discount_profit_module_skips_discount_cap_what_if_without_high_risk() -> None:
    frame = pd.DataFrame(
        {
            "Discount": [0.0, 0.05, 0.1],
            "Profit": [80.0, 40.0, 10.0],
            "Sales": [800.0, 500.0, 400.0],
        }
    )

    result = discount_profit.run(
        frame,
        {
            "discount": "Discount",
            "profit": "Profit",
            "sales_amount": "Sales",
        },
    )

    assert result.tables["discount_cap_what_if"] == []
    assert any("未识别到高风险折扣区间" in warning for warning in result.warnings)
