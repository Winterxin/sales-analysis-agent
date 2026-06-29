from __future__ import annotations

import pandas as pd

from app.schemas.schema_mapping import SchemaMapping
from app.services.dataset_profile import build_dataset_profile, select_analysis_focuses


def test_dataset_profile_derives_retail_order_structure_without_profit_focus() -> None:
    frame = pd.DataFrame(
        {
            "InvoiceNo": ["A", "A", "B", "C"],
            "InvoiceDate": ["2024-01-01", "2024-01-01", "2024-02-01", "2024-03-01"],
            "CustomerID": ["C1", "C1", "C1", "C2"],
            "Country": ["UK", "UK", "France", "UK"],
            "StockCode": ["S1", "S2", "S1", "S3"],
            "Description": ["A", "B", "A", "C"],
            "Quantity": [2, 3, 1, 4],
            "UnitPrice": [10.0, 20.0, 10.0, 15.0],
            "__sales_amount": [20.0, 60.0, 10.0, 60.0],
        }
    )
    mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "InvoiceNo": "order_id",
            "InvoiceDate": "order_datetime",
            "CustomerID": "customer_id",
            "Country": "country",
            "Description": "product_name",
            "Quantity": "quantity",
            "UnitPrice": "unit_price",
            "__sales_amount": "sales_amount",
        },
        confidence=1.0,
    )

    profile = build_dataset_profile(frame, mapping)
    focuses = select_analysis_focuses(profile)

    assert profile["row_count"] == 4
    assert profile["column_count"] == 9
    assert profile["has_invoice_id"]
    assert profile["has_unit_price"]
    assert profile["repeat_customer_rate"] == 0.5
    assert "discount_erosion_focus" not in focuses["selected_focuses"]
    assert "profit_quality_focus" not in focuses["selected_focuses"]
    assert "customer_order_structure_focus" in focuses["selected_focuses"]
    assert "country_market_focus" in focuses["selected_focuses"]


def test_country_count_one_skips_country_market_focus() -> None:
    frame = pd.DataFrame(
        {
            "Order ID": ["A", "B", "C"],
            "Country": ["United States", "United States", "United States"],
            "Sales": [100.0, 200.0, 300.0],
        }
    )
    mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Country": "country",
            "Sales": "sales_amount",
        },
        confidence=1.0,
    )

    profile = build_dataset_profile(frame, mapping)
    focuses = select_analysis_focuses(profile)

    assert profile["country_count"] == 1
    assert "country_market_focus" not in focuses["selected_focuses"]
    assert "country_market_focus" in focuses["skipped_focuses"]
    assert "无有效市场切片价值" in focuses["focus_reasons"]["country_market_focus"]


def test_superstore_focus_prioritizes_discount_profit_segment_product_over_orders() -> None:
    profile = {
        "has_profit": True,
        "has_discount": True,
        "has_segment": True,
        "has_region": True,
        "has_category": True,
        "has_productline": False,
        "has_product": True,
        "has_customer_id": True,
        "has_invoice_id": True,
        "has_quantity": True,
        "has_unit_price": False,
        "line_per_order_avg": 2.0,
        "repeat_customer_rate": 0.8,
        "top_product_sales_share": 0.04,
        "top_category_sales_share": 0.36,
        "country_count": 1,
        "has_country": True,
    }

    focuses = select_analysis_focuses(profile)

    assert focuses["selected_focuses"][:4] == [
        "discount_erosion_focus",
        "profit_quality_focus",
        "segment_region_focus",
        "product_concentration_focus",
    ]
    assert "customer_order_structure_focus" not in focuses["selected_focuses"][:3]
    assert "customer_order_structure_focus" in focuses["support_focuses"]


def test_online_retail_focus_has_no_profit_or_discount_mainline() -> None:
    profile = {
        "has_profit": False,
        "has_discount": False,
        "has_country": True,
        "country_count": 3,
        "top_country_sales_share": 0.84,
        "has_customer_id": True,
        "has_invoice_id": True,
        "has_quantity": True,
        "has_unit_price": True,
        "line_per_order_avg": 20.9,
        "repeat_customer_rate": 0.7,
        "top_product_sales_share": 0.021,
    }

    focuses = select_analysis_focuses(profile)

    assert focuses["selected_focuses"][:3] == [
        "country_market_focus",
        "customer_order_structure_focus",
        "product_concentration_focus",
    ]
    assert "discount_erosion_focus" not in focuses["selected_focuses"]
    assert "profit_quality_focus" not in focuses["selected_focuses"]


def test_focus_selector_uses_deal_size_and_status_for_sample_sales() -> None:
    frame = pd.DataFrame(
        {
            "ORDERNUMBER": [1, 2, 3],
            "ORDERDATE": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "PRODUCTLINE": ["Classic Cars", "Motorcycles", "Classic Cars"],
            "COUNTRY": ["USA", "France", "USA"],
            "DEALSIZE": ["Small", "Medium", "Large"],
            "STATUS": ["Shipped", "Cancelled", "Shipped"],
            "QUANTITYORDERED": [10, 5, 8],
            "PRICEEACH": [100.0, 80.0, 120.0],
            "SALES": [1000.0, 400.0, 960.0],
        }
    )
    mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "ORDERNUMBER": "order_id",
            "ORDERDATE": "order_datetime",
            "PRODUCTLINE": "productline",
            "COUNTRY": "country",
            "DEALSIZE": "deal_size",
            "STATUS": "order_status",
            "QUANTITYORDERED": "quantity",
            "PRICEEACH": "unit_price",
            "SALES": "sales_amount",
        },
        confidence=1.0,
    )

    profile = build_dataset_profile(frame, mapping)
    focuses = select_analysis_focuses(profile)

    assert profile["has_productline"]
    assert profile["has_deal_size"]
    assert profile["has_order_status"]
    assert "deal_size_focus" in focuses["selected_focuses"]
    assert "order_status_focus" in focuses["selected_focuses"]
    assert "productline_performance_focus" in focuses["selected_focuses"]
    assert "country_market_focus" in focuses["selected_focuses"]
    assert "discount_erosion_focus" not in focuses["selected_focuses"]


def test_dataset_profile_coerces_dirty_sales_amount_before_aggregation() -> None:
    frame = pd.DataFrame(
        {
            "Category": ["A", "A", "B", "B"],
            "grand_total": ["$1,200.50", " 300 ", None, "bad"],
            "Date": ["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-02"],
        }
    )
    mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Category": "category",
            "grand_total": "sales_amount",
            "Date": "order_datetime",
        },
        confidence=1.0,
    )

    profile = build_dataset_profile(frame, mapping)

    assert profile["top_category_sales_share"] == 1.0
    coercion = profile["numeric_coercion_summary"]["sales_amount"]
    assert coercion["selected_column"] == "grand_total"
    assert coercion["valid_count"] == 2
    assert coercion["numeric_conversion_rate"] == 0.5


def test_dataset_profile_selects_best_duplicate_sales_amount_source() -> None:
    frame = pd.DataFrame(
        {
            "grand_total": ["1,200", "2,500", "3,000", "4,000"],
            "MV": ["abc", "", "id-3", "N/A"],
            "Category": ["A", "A", "B", "B"],
        }
    )
    mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "grand_total": "sales_amount",
            "MV": "sales_amount",
            "Category": "category",
        },
        confidence=1.0,
    )

    profile = build_dataset_profile(frame, mapping)

    decision = profile["canonical_field_source_decision"]["sales_amount"]
    assert decision["selected_original_column"] == "grand_total"
    assert "MV" in decision["rejected_duplicate_columns"]
    assert decision["numeric_conversion_rate"] == 1.0
    assert profile["top_category_sales_share"] == 0.6542
