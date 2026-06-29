from __future__ import annotations

import pandas as pd

from app.analysis.modules import product_contribution


def test_product_contribution_module_outputs_concentration_and_profit_quality() -> None:
    frame = pd.DataFrame(
        {
            "Product Name": ["A", "A", "B", "C", "D", "E"],
            "Category": ["Tech", "Tech", "Tech", "Office", "Office", "Furniture"],
            "Sub-Category": ["Phones", "Phones", "Copiers", "Binders", "Storage", "Chairs"],
            "Sales": [700, 300, 400, 250, 200, 150],
            "Quantity": [7, 3, 4, 5, 4, 3],
            "Profit": [140, 60, -40, 25, 10, -15],
        }
    )

    result = product_contribution.run(
        frame,
        {
            "product_name": "Product Name",
            "category": "Category",
            "sub_category": "Sub-Category",
            "sales_amount": "Sales",
            "quantity": "Quantity",
            "profit": "Profit",
        },
    )

    top_product = result.tables["top_products"][0]

    assert result.module_id == "product_contribution_analysis"
    assert top_product["Product Name"] == "A"
    assert top_product["Sales"] == 1000
    assert top_product["sales_share"] == 0.5
    assert top_product["cumulative_sales_share"] == 0.5
    assert top_product["profit_margin"] == 0.2
    assert result.summary_metrics["top_1_sales_share"] == 0.5
    assert result.summary_metrics["products_to_80pct_sales"] == 3
    assert result.tables["category_sales"]
    assert result.tables["category_profit_quality"]
    assert result.tables["high_sales_low_profit_products"]
    assert "已识别高销售低利润商品，需要进一步核查毛利、定价、成本和补货策略。" in result.findings
    assert "已识别高销售低利润商品，需要进一步核查毛利、折扣和补货策略。" not in result.findings


def test_product_contribution_module_keeps_discount_wording_when_discount_is_mapped() -> None:
    frame = pd.DataFrame(
        {
            "Product Name": ["A", "A", "B", "C", "D", "E"],
            "Category": ["Tech", "Tech", "Tech", "Office", "Office", "Furniture"],
            "Sales": [700, 300, 400, 250, 200, 150],
            "Quantity": [7, 3, 4, 5, 4, 3],
            "Profit": [140, 60, -40, 25, 10, -15],
            "Discount": [0.1, 0.05, 0.3, 0.0, 0.0, 0.2],
        }
    )

    result = product_contribution.run(
        frame,
        {
            "product_name": "Product Name",
            "category": "Category",
            "sales_amount": "Sales",
            "quantity": "Quantity",
            "profit": "Profit",
            "discount": "Discount",
        },
    )

    assert "已识别高销售低利润商品，需要进一步核查毛利、折扣和补货策略。" in result.findings
