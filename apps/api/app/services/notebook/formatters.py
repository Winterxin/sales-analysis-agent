from __future__ import annotations

import math
import re
from typing import Any

from app.services.notebook.markdown_sanitizer import clean_business_text


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "数据限制说明：缺少可验证字段"
        return f"{value:,.2f}"
    text = str(value).strip()
    if text.lower() in {"", "unknown", "none", "nan", "null"}:
        return "数据限制说明：缺少可验证字段"
    return clean_business_text(text)


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return "数据限制说明：缺少可验证字段"
        return f"{value * 100:.2f}%"
    if value is None:
        return "数据限制说明：缺少可验证字段"
    text = format_scalar(value)
    if text.endswith("%"):
        return re.sub(r"%{2,}$", "%", text)
    return text
