from __future__ import annotations

from app.schemas.schema_mapping import SchemaMapping


FIELD_LABELS = {
    "order_datetime": "订单日期",
    "sales_amount": "销售额",
    "profit": "利润",
    "discount": "折扣",
    "product_name": "商品",
    "category": "类目",
    "segment": "客群",
    "region": "区域",
    "quantity": "数量",
    "country": "国家",
    "customer_id": "客户",
    "order_id": "订单",
    "unit_price": "单价",
    "deal_size": "交易规模",
    "order_status": "订单状态",
    "productline": "产品线",
}


def mapped_fields(schema_mapping: SchemaMapping) -> set[str]:
    return {
        str(value)
        for value in schema_mapping.field_mapping.values()
        if str(value).strip()
    }


def field_list(fields: set[str]) -> str:
    if not fields:
        return "未要求特定字段"
    return ", ".join(FIELD_LABELS.get(field, field) for field in sorted(fields))
