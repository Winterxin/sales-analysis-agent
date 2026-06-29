from app.services.schema_rules import map_schema_fields
from app.services.schema_mapper import map_schema


def test_schema_rules_map_common_sales_columns() -> None:
    mapping = map_schema_fields(
        [
            "Order ID",
            "Order Date",
            "Product",
            "Quantity Ordered",
            "Sales",
            "Channel",
        ]
    )

    assert mapping.dataset_type == "sales_transaction"
    assert mapping.field_mapping["Order ID"] == "order_id"
    assert mapping.field_mapping["Order Date"] == "order_datetime"
    assert mapping.field_mapping["Product"] == "product_name"
    assert mapping.field_mapping["Sales"] == "sales_amount"


def test_schema_rules_map_common_quantity_aliases() -> None:
    mapping = map_schema_fields(
        [
            "Order_Quantity",
            "qty_ordered",
            "Qty Ordered",
            "Units Ordered",
            "quantityordered",
            "quantity",
        ]
    )

    assert mapping.field_mapping["Order_Quantity"] == "quantity"
    assert mapping.field_mapping["qty_ordered"] == "quantity"
    assert mapping.field_mapping["Qty Ordered"] == "quantity"
    assert mapping.field_mapping["Units Ordered"] == "quantity"
    assert mapping.field_mapping["quantityordered"] == "quantity"
    assert mapping.field_mapping["quantity"] == "quantity"


def test_schema_rules_map_weekly_store_sales_without_quantity_as_sales_dataset() -> None:
    mapping = map_schema_fields(["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"])

    assert mapping.field_mapping["Store"] == "store"
    assert mapping.field_mapping["Dept"] == "category"
    assert mapping.field_mapping["Date"] == "order_datetime"
    assert mapping.field_mapping["Weekly_Sales"] == "sales_amount"
    assert "quantity" in mapping.missing_required_fields


def test_schema_rules_map_retail_total_amount_and_price_per_unit() -> None:
    mapping = map_schema_fields(
        ["Transaction ID", "Date", "Customer ID", "Product Category", "Quantity", "Price per Unit", "Total Amount"]
    )

    assert mapping.field_mapping["Transaction ID"] == "order_id"
    assert mapping.field_mapping["Product Category"] == "category"
    assert mapping.field_mapping["Price per Unit"] == "unit_price"
    assert mapping.field_mapping["Total Amount"] == "sales_amount"
    assert mapping.dataset_type == "sales_transaction"


class FakeSchemaLLMClient:
    enabled = True

    def suggest_schema_mapping(self, columns, rule_mapping):
        from app.schemas.schema_mapping import SchemaMapping

        assert columns == ["创建时间", "商品标题", "实收金额"]
        return SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "创建时间": "order_datetime",
                "商品标题": "product_name",
                "实收金额": "sales_amount",
            },
            confidence=0.95,
            missing_required_fields=["order_id", "quantity"],
            uncertain_fields=[],
        )


def test_schema_mapper_uses_llm_when_rule_confidence_is_low() -> None:
    result = map_schema(
        ["创建时间", "商品标题", "实收金额"],
        llm_client=FakeSchemaLLMClient(),
    )

    assert result.dataset_type == "sales_transaction"
    assert result.field_mapping["创建时间"] == "order_datetime"
    assert result.field_mapping["商品标题"] == "product_name"
