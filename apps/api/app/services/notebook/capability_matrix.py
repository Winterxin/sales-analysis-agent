from __future__ import annotations

from app.schemas.schema_mapping import SchemaMapping

from app.services.notebook.field_utils import field_list, mapped_fields


CAPABILITY_RULES = [
    {
        "name": "销售趋势分析",
        "required": {"order_datetime", "sales_amount"},
        "alternative": "改用总体销售额、订单数或类目销售结构做静态概览。",
    },
    {
        "name": "折扣侵蚀分析",
        "required": {"discount", "profit"},
        "alternative": "改用利润分布或低利润商品清单识别盈利风险。",
    },
    {
        "name": "切片分析",
        "required_any": {"segment", "region", "category"},
        "required": {"sales_amount"},
        "alternative": "改用商品或时间维度做聚合对比，不能输出客群、区域或类目结论。",
    },
    {
        "name": "高销售低利润分析",
        "required": {"product_name", "sales_amount", "profit"},
        "alternative": "改用头部销售商品排名，不能判断利润质量。",
    },
    {
        "name": "基线预测",
        "required": {"order_datetime", "sales_amount"},
        "alternative": "改用历史汇总和峰值月份描述，不能生成时间序列预测。",
    },
]


def capability_rows(schema_mapping: SchemaMapping) -> list[dict[str, str]]:
    mapped = mapped_fields(schema_mapping)
    rows: list[dict[str, str]] = []
    for rule in CAPABILITY_RULES:
        required = set(rule.get("required", set()))
        required_any = set(rule.get("required_any", set()))
        missing = sorted(required - mapped)
        has_any = True
        if required_any:
            has_any = bool(required_any & mapped)
            if not has_any:
                missing.extend(sorted(required_any))
        enabled = not missing and has_any
        rows.append(
            {
                "capability": str(rule["name"]),
                "status": "enabled" if enabled else "disabled",
                "required_fields": field_list(required | required_any),
                "data_limit": "字段满足，允许输出对应结论。"
                if enabled
                else "数据限制说明：缺少 "
                + field_list(set(missing))
                + "；替代分析："
                + str(rule["alternative"]),
            }
        )
    return rows


def build_capability_matrix_markdown(schema_mapping: SchemaMapping) -> str:
    rows = capability_rows(schema_mapping)
    lines = [
        "## Analysis Capability Matrix",
        "",
        "| capability | status | required_fields | data_limit_or_alternative |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {capability} | {status} | {required_fields} | {data_limit} |".format(
                **row
            )
        )
    return "\n".join(lines)


def build_compact_capability_matrix_markdown(schema_mapping: SchemaMapping) -> str:
    rows = capability_rows(schema_mapping)
    enabled_count = sum(1 for row in rows if row["status"] == "enabled")
    partial_count = sum(1 for row in rows if row["status"] == "partial")
    missing_rows = [row for row in rows if row["status"] == "disabled"]
    lines = [
        "## Analysis Capability Matrix",
        "",
        "### 简版 Capability Matrix",
        "",
        f"- 已支持能力数量：`{enabled_count}`",
        f"- 部分支持能力数量：`{partial_count}`",
        f"- 缺失能力数量：`{len(missing_rows)}`",
        "",
        "关键缺失能力 Top 5：",
    ]
    if missing_rows:
        lines.extend(f"- {row['capability']}：disabled；{row['data_limit']}" for row in missing_rows[:5])
    else:
        lines.append("- 暂无关键缺失能力。")
    lines.extend(
        [
            "",
            "完整能力矩阵可在 full/debug 模式查看。",
        ]
    )
    return "\n".join(lines)


_mapped_fields = mapped_fields
_field_list = field_list
_capability_rows = capability_rows
_capability_matrix_markdown = build_capability_matrix_markdown
_compact_capability_matrix_markdown = build_compact_capability_matrix_markdown
