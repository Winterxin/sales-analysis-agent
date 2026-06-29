from __future__ import annotations

import re

from app.schemas.report import AnalysisReport, ModuleReport
from app.services.report_builder import build_html_report


def _report_with_chinese_modules() -> AnalysisReport:
    return AnalysisReport(
        task_id="task-report-html",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["Total sales increased across the selected period."],
        modules=[
            ModuleReport(
                module_id="data_quality_check",
                title="数据质量检查",
                chart_type="table",
                findings=["字段完整度较高，但存在少量缺失值。"],
            ),
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["销售额在最近月份出现波动。"],
            ),
        ],
    )


def test_build_html_report_english_uses_safe_module_titles_without_chinese_findings() -> None:
    html = build_html_report(_report_with_chinese_modules(), output_language="en")

    assert "<html lang='en'>" in html
    assert "Analysis Modules" in html
    assert "Data Quality Check" in html
    assert "Sales Trend Analysis" in html
    assert not re.search(r"[\u4e00-\u9fff]", html)
    assert "数据质量检查" not in html
    assert "字段完整度较高" not in html


def test_build_html_report_english_filters_chinese_fallback_summary() -> None:
    report = _report_with_chinese_modules().model_copy(
        update={"summary": ["发现 0 条时间缺失记录。", "已完成 2 个核心数值指标的分布画像。"]}
    )

    html = build_html_report(report, output_language="en")

    assert "<html lang='en'>" in html
    assert "Summary is available in report.json." in html
    assert "发现 0 条时间缺失记录" not in html
    assert not re.search(r"[\u4e00-\u9fff]", html)


def test_build_html_report_chinese_keeps_existing_default_module_rendering() -> None:
    report = _report_with_chinese_modules()
    default_html = build_html_report(report)
    zh_html = build_html_report(report, output_language="zh-CN")

    assert zh_html == default_html
    assert "数据质量检查" in zh_html
    assert "字段完整度较高，但存在少量缺失值。" in zh_html
    assert "Analysis Modules" not in zh_html
