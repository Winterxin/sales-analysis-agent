from __future__ import annotations

import re

from app.schemas.schema_mapping import SchemaMapping

ALIASES: dict[str, set[str]] = {
    "order_id": {"orderid", "order id", "ordernumber", "invoiceno", "invoice no", "transactionid", "transaction id", "订单号", "交易号"},
    "order_datetime": {"orderdate", "order date", "invoicedate", "invoice date", "createdat", "date", "日期", "下单时间"},
    "ship_datetime": {"shipdate", "ship date", "发货时间", "发货日期"},
    "product_name": {"product", "productname", "product name", "description", "商品", "商品名称", "itemtitle"},
    "sku": {"sku", "skuid", "stockcode", "productid", "product id", "商品编号"},
    "quantity": {
        "quantityordered",
        "quantity",
        "orderquantity",
        "order quantity",
        "order_quantity",
        "qtyordered",
        "qty ordered",
        "qty_ordered",
        "qty",
        "unitsordered",
        "units ordered",
        "units_ordered",
        "数量",
        "销量",
    },
    "sales_amount": {
        "sales",
        "amount",
        "gmv",
        "salesamount",
        "weeklysales",
        "weekly sales",
        "weekly_sales",
        "grandtotal",
        "grand total",
        "grand_total",
        "total",
        "totalamount",
        "total amount",
        "total_amount",
        "revenue",
        "销售额",
        "实收金额",
    },
    "unit_price": {"unitprice", "unit price", "priceperunit", "price per unit", "price", "priceeach", "price each", "单价"},
    "category": {"category", "productcategory", "product category", "department", "dept", "类目"},
    "productline": {"productline", "product line", "产品线"},
    "sub_category": {"subcategory", "sub-category", "子类目"},
    "channel": {"channel", "渠道"},
    "store": {"store", "shop", "门店", "店铺"},
    "customer_id": {"customerid", "customer id", "客户id"},
    "country": {"country", "国家"},
    "city": {"city", "城市"},
    "state": {"state", "province", "州", "省"},
    "region": {"region", "地区", "区域"},
    "segment": {"segment", "客户分群", "客群"},
    "discount": {"discount", "折扣"},
    "profit": {"profit", "利润"},
    "ship_mode": {"shipmode", "ship mode", "配送方式", "物流方式"},
    "deal_size": {"dealsize", "deal size", "交易规模"},
    "order_status": {"status", "orderstatus", "order status", "订单状态"},
}

REQUIRED_FIELDS = {"order_id", "order_datetime", "quantity", "sales_amount"}


def _normalize(column_name: str) -> str:
    lowered = column_name.strip().lower()
    return re.sub(r"[\s_\-()/]+", "", lowered)


def map_schema_fields(columns: list[str]) -> SchemaMapping:
    normalized_aliases = {
        canonical_name: {_normalize(alias) for alias in aliases}
        for canonical_name, aliases in ALIASES.items()
    }
    field_mapping: dict[str, str] = {}
    uncertain_fields: list[str] = []

    for original in columns:
        normalized = _normalize(original)
        match = None
        for canonical_name, aliases in normalized_aliases.items():
            if normalized in aliases:
                match = canonical_name
                break
        if match:
            field_mapping[original] = match
        else:
            uncertain_fields.append(original)

    mapped_values = set(field_mapping.values())
    missing_required = sorted(REQUIRED_FIELDS - mapped_values)
    confidence = round(len(field_mapping) / max(len(columns), 1), 2)
    dataset_type = (
        "sales_transaction"
        if {"order_datetime", "sales_amount"} <= mapped_values
        else "unknown"
    )

    return SchemaMapping(
        dataset_type=dataset_type,
        field_mapping=field_mapping,
        confidence=confidence,
        missing_required_fields=missing_required,
        uncertain_fields=uncertain_fields,
    )
