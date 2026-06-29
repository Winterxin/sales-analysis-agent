from __future__ import annotations

from app.schemas.report import AnalysisReport, ModuleReport
from app.services.output_language import english_safe_bullets, is_english_output


def _module(report: AnalysisReport, module_id: str) -> ModuleReport | None:
    return next((module for module in report.modules if module.module_id == module_id), None)


def _metric(module: ModuleReport | None, key: str, default: object = "暂无") -> object:
    if module is None:
        return default
    return module.summary_metrics.get(key, default)


def _english_metric(module: ModuleReport | None, key: str, fallback: object) -> object:
    value = _metric(module, key, None)
    if value in (None, "", "暂无"):
        return fallback
    return value


def _has_metric(module: ModuleReport | None, *keys: str) -> bool:
    if module is None:
        return False
    return any(module.summary_metrics.get(key) not in (None, "", "暂无") for key in keys)


def _has_table_rows(module: ModuleReport | None, *table_names: str) -> bool:
    if module is None:
        return False
    return any(bool(module.tables.get(name)) for name in table_names)


def _dimension_label_en(module: ModuleReport | None) -> str | None:
    if module is None:
        return None
    metrics = module.summary_metrics or {}
    primary = metrics.get("primary_dimension")
    secondary = metrics.get("secondary_dimension")
    dimension = metrics.get("dimension")
    if primary and secondary:
        return f"{str(primary).replace('_', ' ').title()} × {str(secondary).replace('_', ' ').title()}"
    if primary:
        return str(primary).replace("_", " ").title()
    if dimension and dimension not in ("暂无", "unknown"):
        return str(dimension).replace("_", " ").title()
    for table_name in ("segment_region_matrix", "dimension_totals", "weak_performance_cuts"):
        rows = module.tables.get(table_name, [])
        if rows and isinstance(rows[0], dict):
            candidate_keys = [
                key
                for key in rows[0]
                if str(key).lower() in {"segment", "region", "category", "country", "city", "market"}
            ]
            if candidate_keys:
                return " × ".join(str(key).replace("_", " ").title() for key in candidate_keys[:2])
    return None


def build_business_review(report: AnalysisReport, output_language: str | None = None) -> str:
    trend = _module(report, "sales_trend_analysis")
    product = _module(report, "product_contribution_analysis")
    quality = _module(report, "data_quality_check")
    dimension = _module(report, "dimension_breakdown_analysis")
    forecast = _module(report, "forecast_analysis")

    summary_items = "\n".join(f"- {item}" for item in report.summary) or "- 暂无自动摘要。"
    quality_findings = "\n".join(f"- {item}" for item in (quality.findings if quality else [])) or "- 暂无明显数据质量问题。"
    module_findings = "\n".join(
        f"- {finding}"
        for module in report.modules
        for finding in module.findings
    ) or "- 暂无模块发现。"

    total_sales = _metric(trend, "total_sales_amount")
    total_quantity = _metric(trend, "total_quantity")
    peak_period = _metric(trend, "peak_period")
    trough_period = _metric(trend, "trough_period")
    top_product = _metric(product, "top_product")
    dimension_name = _metric(dimension, "dimension")
    forecast_baseline = _metric(forecast, "baseline_sales_amount")
    discount = _module(report, "discount_profit_analysis")
    has_discount = _has_table_rows(
        discount,
        "discount_buckets",
        "discount_threshold_candidates",
        "discount_cap_what_if",
    )

    if is_english_output(output_language):
        has_profit = _has_metric(product, "total_profit_amount", "profit_margin") or _has_metric(
            discount,
            "total_profit_amount",
            "negative_profit_rate",
        )
        total_sales_en = _english_metric(trend, "total_sales_amount", "not available from the mapped sales field")
        total_quantity_en = _english_metric(trend, "total_quantity", "not available from the mapped quantity field")
        peak_period_en = _english_metric(
            trend,
            "peak_period",
            "Peak and trough periods are not available from the mapped time field.",
        )
        trough_period_en = _english_metric(
            trend,
            "trough_period",
            "Peak and trough periods are not available from the mapped time field.",
        )
        top_product_en = _english_metric(
            product,
            "top_product",
            "No product-level ranking is available from the mapped fields.",
        )
        dimension_en = _dimension_label_en(dimension)
        forecast_en = _english_metric(
            forecast,
            "baseline_sales_amount",
            "No reliable forecast baseline was produced for this run.",
        )
        english_summary_items = "\n".join(
            f"- {item}"
            for item in english_safe_bullets(
                report.summary,
                [
                    f"The run completed {report.module_count} analysis modules for a `{report.dataset_type}` dataset.",
                    f"Current total sales are `{total_sales}` and total quantity is `{total_quantity}`.",
                    "Use the Notebook and Client Report together for chart-level evidence and client-ready interpretation.",
                ],
                limit=5,
            )
        )
        english_quality_findings = "\n".join(
            f"- {item}"
            for item in english_safe_bullets(
                quality.findings if quality else [],
                ["No major data quality issue is available from the current checks."],
                limit=4,
            )
        )
        english_module_details = "\n".join(
            f"- {module.module_id.replace('_', ' ')}: {len(module.findings)} findings, {len(module.tables)} tables."
            for module in report.modules
        ) or "- No module-level details are available yet."
        product_sentence = (
            f"The current top product is `{top_product_en}`. Review sales concentration before treating it as a growth driver."
            if top_product_en != "No product-level ranking is available from the mapped fields."
            else str(top_product_en)
        )
        dimension_sentence = (
            f"The analysis supports a `{dimension_en}` sales breakdown."
            if dimension_en
            else "No suitable business dimension is available for a reliable sales breakdown in this dataset."
        )
        actions = [
            "- Review sales trend, order status, and volatility patterns before making operating changes.",
            "- Check product, SKU, or category concentration where those fields are available.",
        ]
        if has_profit:
            actions.append("- Compare high-sales slices with profit outcomes before using sales alone as the main decision signal.")
        else:
            actions.append("- Add profit fields before making profitability decisions.")
        if has_profit and has_discount:
            actions.append("- Review high-discount orders only where discount and profit evidence are both available.")
        elif not has_discount:
            actions.append("- Add discount fields before making pricing or scenario decisions.")
        actions.append(f"- Forecast follow-up: `{forecast_en}`")
        return "\n".join(
            [
                "# Business Review",
                "",
                "## 1. Overall Performance",
                "",
                f"The dataset was identified as `{report.dataset_type}`, and the system executed {report.module_count} analysis modules.",
                f"Current total sales are `{total_sales_en}`, with total quantity of `{total_quantity_en}`.",
                (
                    f"The sales peak period is `{peak_period_en}`, and the trough period is `{trough_period_en}`."
                    if peak_period_en == peak_period and trough_period_en == trough_period
                    else str(peak_period_en)
                ),
                "",
                "## 2. Key Findings",
                "",
                english_summary_items,
                "",
                "## 3. Product Performance",
                "",
                product_sentence,
                "",
                "## 4. Dimension Breakdown",
                "",
                dimension_sentence,
                "",
                "## 5. Risks and Exceptions",
                "",
                english_quality_findings,
                "",
                "## 6. Recommended Next Actions",
                "",
                *actions,
                "",
                "## 7. Module Details",
                "",
                english_module_details,
                "",
            ]
        )

    return "\n".join(
        [
            "# 销售经营复盘报告",
            "",
            "## 1. 总体表现",
            "",
            f"本次数据被识别为 `{report.dataset_type}`，系统共执行 {report.module_count} 个分析模块。",
            f"当前汇总销售额为 `{total_sales}`，累计销量为 `{total_quantity}`。",
            f"销售峰值日期为 `{peak_period}`，低谷日期为 `{trough_period}`。",
            "",
            "## 2. 关键发现",
            "",
            summary_items,
            "",
            "## 3. 商品表现",
            "",
            (
                f"当前 Top 商品为 `{top_product}`。建议结合利润、折扣、库存进一步判断它是“高销售贡献”还是“高利润贡献”。"
                if has_discount
                else f"当前 Top 商品为 `{top_product}`。建议结合利润、成本和库存进一步判断它是“高销售贡献”还是“高利润贡献”。"
            ),
            "",
            "## 4. 维度拆解",
            "",
            f"系统当前优先按 `{dimension_name}` 维度拆解销售额。后续可继续扩展到区域、客群、类目和子类目组合分析。",
            "",
            "## 5. 风险与异常",
            "",
            quality_findings,
            "",
            "## 6. 后续行动建议",
            "",
            "- 优先拆解高销售额商品的利润表现，避免只看销售额不看利润。",
            (
                "- 对高折扣订单做专项分析，判断折扣是否真正带来销售增长。"
                if has_discount
                else "- 对低利润商品和类目做专项分析，判断成本、定价或履约因素是否持续压低利润。"
            ),
            "- 按区域、客群、类目组合拆解销售额和利润，定位优势市场与弱势市场。",
            f"- 当前 28 天基础预测日均销售额约为 `{forecast_baseline}`，建议结合节假日和促销日历校正。",
            "",
            "## 7. 模块明细",
            "",
            module_findings,
            "",
        ]
    )
