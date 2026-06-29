from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


MODULE_SECTION_IDS = {
    "metric_distribution_analysis": "metric_distributions",
    "sales_trend_analysis": "sales_trends",
    "product_contribution_analysis": "product_and_category",
    "dimension_breakdown_analysis": "segment_and_region",
    "country_market_analysis": "country_market",
    "order_structure_analysis": "order_structure",
    "discount_profit_analysis": "discount_and_profit",
    "loss_risk_modeling": "modeling",
    "forecast_analysis": "forecast",
    "data_quality_check": "data_quality",
}

SECTION_LABELS = {
    "metric_distributions": "指标分布与异常值分析",
    "sales_trends": "销售趋势分析",
    "product_and_category": "商品与类目分析",
    "segment_and_region": "客群与区域分析",
    "country_market": "国家市场分析",
    "order_structure": "订单结构分析",
    "discount_and_profit": "折扣与利润分析",
    "modeling": "建模分析",
    "forecast": "基础预测",
    "data_quality": "数据说明",
}

SECTION_KIND = {
    "metric_distributions": "EDA",
    "sales_trends": "EDA",
    "product_and_category": "贡献结构",
    "segment_and_region": "切片表现",
    "country_market": "市场结构",
    "order_structure": "订单结构",
    "discount_and_profit": "利润质量",
    "modeling": "建模",
    "forecast": "预测",
    "data_quality": "数据说明",
}

NOTEBOOK_TITLE_SECTION_IDS = {
    "指标分布与异常值分析": "metric_distributions",
    "销售趋势分析": "sales_trends",
    "商品与类目分析": "product_and_category",
    "客群与区域分析": "segment_and_region",
    "订单结构分析": "order_structure",
    "折扣与利润分析": "discount_and_profit",
    "建模分析": "modeling",
    "建模分析：亏损风险识别": "modeling",
    "Action Plan": "action_plan",
    "结论与行动建议": "conclusions",
}

KPI_SPECS = (
    ("total_sales_amount", "累计销售额", "number"),
    ("negative_profit_rate", "负利润订单占比", "percent"),
    ("monthly_volatility", "月度波动率", "percent"),
    ("recent_growth_rate", "近期增长率", "percent"),
    ("best_recall", "模型 Recall", "percent"),
)


def build_analysis_results_payload(
    task_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    files = manifest.get("files", {})
    if not isinstance(files, dict) or "report_json" not in files:
        raise FileNotFoundError("Analysis results are not available")

    report = _read_json_path(files["report_json"])
    profile = _read_optional_json(files.get("dataset_profile_json"))
    narrative = _read_optional_json(files.get("notebook_narrative_json"))
    notebook = _extract_notebook_reading(files)
    narrative_by_section = {
        str(section.get("section_id")): section
        for section in narrative.get("sections", [])
        if isinstance(section, dict) and section.get("section_id")
    }

    modules = [module for module in report.get("modules", []) if isinstance(module, dict)]
    sections = [
        _build_section(module, narrative_by_section, notebook)
        for module in modules
        if module.get("module_id") != "data_quality_check"
        and MODULE_SECTION_IDS.get(str(module.get("module_id") or ""), "") != "modeling"
    ]
    modeling = _build_modeling(modules, notebook)
    action_plan = notebook.get("action_plan") or {"columns": [], "rows": []}

    return {
        "task_id": task_id,
        "dataset": _build_dataset(manifest, profile, report, modules, len(sections)),
        "downloads": _build_downloads(task_id, files),
        "kpis": _build_kpis(report, profile, modeling),
        "summary": notebook.get("core_judgments") or _build_executive_summary(modules, report.get("summary", [])),
        "sections": sections,
        "modeling": modeling,
        "action_plan": action_plan,
        "data_note": _build_data_note(modules, profile),
    }


def _read_json_path(path_value: object) -> dict[str, Any]:
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError("Analysis results are not available")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path_value: object | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_notebook_reading(files: dict[str, Any]) -> dict[str, Any]:
    notebook_path = files.get("analysis_notebook_canonical") or files.get("analysis_notebook")
    if not notebook_path:
        return {"sections": {}, "core_judgments": [], "action_plan": {"columns": [], "rows": []}}
    path = Path(str(notebook_path))
    if not path.exists():
        return {"sections": {}, "core_judgments": [], "action_plan": {"columns": [], "rows": []}}
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown_cells = [
        _cell_source(cell)
        for cell in notebook.get("cells", [])
        if isinstance(cell, dict) and cell.get("cell_type") == "markdown"
    ]
    sections = _notebook_sections(markdown_cells)
    return {
        "sections": sections,
        "core_judgments": _extract_core_judgments(sections.get("conclusions", "")),
        "action_plan": _extract_action_plan(sections.get("action_plan", "")),
    }


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _notebook_sections(markdown_cells: list[str]) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_id: str | None = None
    for cell_text in markdown_cells:
        title = _markdown_h2_title(cell_text)
        if title:
            current_id = _section_id_from_title(title)
            if current_id:
                sections.setdefault(current_id, []).append(cell_text)
            continue
        if current_id:
            sections.setdefault(current_id, []).append(cell_text)
    return {
        section_id: "\n\n".join(_clean_markdown(part) for part in parts if part.strip()).strip()
        for section_id, parts in sections.items()
    }


def _markdown_h2_title(text: str) -> str | None:
    match = re.search(r"^##\s+(.+?)\s*(?:<a\s|$)", text, flags=re.MULTILINE)
    if not match:
        return None
    return re.sub(r"\s*<a\b.*$", "", match.group(1)).strip()


def _section_id_from_title(title: str) -> str | None:
    if title in NOTEBOOK_TITLE_SECTION_IDS:
        return NOTEBOOK_TITLE_SECTION_IDS[title]
    if title.startswith("建模分析"):
        return "modeling"
    return None


def _clean_markdown(text: str) -> str:
    text = re.sub(r"<a\s+id=\"[^\"]+\"></a>", "", text)
    text = re.sub(r"^##\s+.+$", "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_core_judgments(conclusion_text: str) -> list[str]:
    if not conclusion_text:
        return []
    match = re.search(r"###\s*核心判断(?P<body>.*?)(?:###\s|\Z)", conclusion_text, flags=re.S)
    body = match.group("body") if match else conclusion_text
    bullets = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")):
            bullets.append(line[2:].strip())
    return bullets[:4]


def _extract_action_plan(action_text: str) -> dict[str, Any]:
    if not action_text:
        return {"columns": [], "rows": []}
    table_lines = [line.strip() for line in action_text.splitlines() if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return {"columns": [], "rows": []}
    columns = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(columns):
            continue
        rows.append(dict(zip(columns, cells, strict=False)))
    return {"columns": columns, "rows": rows[:5]}


def _build_dataset(
    manifest: dict[str, Any],
    profile: dict[str, Any],
    report: dict[str, Any],
    modules: list[dict[str, Any]],
    section_count: int,
) -> dict[str, Any]:
    row_count = profile.get("row_count")
    column_count = profile.get("column_count")
    trend = next((item for item in modules if item.get("module_id") == "sales_trend_analysis"), {})
    trend_metrics = trend.get("summary_metrics", {}) if isinstance(trend, dict) else {}
    date_span_days = profile.get("date_span_days")
    year_count = trend_metrics.get("year_count") if isinstance(trend_metrics, dict) else None
    return {
        "filename": manifest.get("uploaded_filename") or "analysis.csv",
        "dataset_type": report.get("dataset_type"),
        "row_count": row_count,
        "column_count": column_count,
        "date_span_days": date_span_days,
        "time_span_label": f"{year_count} 年" if year_count else None,
        "section_count": section_count,
    }


def _build_downloads(task_id: str, files: dict[str, Any]) -> dict[str, str]:
    artifact_by_file = {
        "analysis_notebook": "notebook",
        "client_report_html": "client-report-html",
        "client_report_json": "client-report-json",
        "report_json": "report-json",
    }
    downloads: dict[str, str] = {}
    for file_key, artifact_name in artifact_by_file.items():
        if file_key in files:
            label = {
                "analysis_notebook": "notebook",
                "client_report_html": "client_report_html",
                "client_report_json": "client_report_json",
                "report_json": "report_json",
            }[file_key]
            downloads[label] = f"/api/v1/analysis/tasks/{task_id}/artifacts/{artifact_name}"
    return downloads


def _build_kpis(
    report: dict[str, Any],
    profile: dict[str, Any],
    modeling: dict[str, Any] | None,
) -> list[dict[str, str]]:
    source: dict[str, Any] = dict(profile)
    for module in report.get("modules", []):
        if not isinstance(module, dict):
            continue
        metrics = module.get("summary_metrics", {})
        if isinstance(metrics, dict):
            source.update(metrics)
    if modeling:
        for key, value in modeling.get("raw_metrics", {}).items():
            source[key] = value

    kpis: list[dict[str, str]] = []
    for key, label, value_type in KPI_SPECS:
        if key not in source or source[key] is None:
            continue
        kpis.append({"label": label, "value": _format_value(source[key], value_type)})
    return kpis[:5]


def _build_section(
    module: dict[str, Any],
    narrative_by_section: dict[str, dict[str, Any]],
    notebook: dict[str, Any],
) -> dict[str, Any]:
    module_id = str(module.get("module_id") or "")
    section_id = MODULE_SECTION_IDS.get(module_id, module_id)
    narrative = narrative_by_section.get(section_id, {})
    notebook_text = notebook.get("sections", {}).get(section_id, "")
    notebook_conclusion = _extract_section_conclusion(notebook_text)
    findings = [str(item) for item in module.get("findings", []) if item]
    full_text = notebook_text or "\n".join(findings)
    return {
        "section_id": section_id,
        "title": SECTION_LABELS.get(section_id) or str(module.get("title") or section_id),
        "kind": SECTION_KIND.get(section_id, "分析"),
        "takeaway": notebook_conclusion
        or narrative.get("business_takeaway")
        or _first_text(findings),
        "summary": _first_text(findings),
        "findings": _short_list(findings, limit=3),
        "full_text": full_text,
        "is_long_text": len(full_text) > 260,
        "chart": {
            "type": module.get("chart_type") or "table",
            "payload": module.get("chart_payload") or _chart_payload_from_tables(module),
        },
        "table": _first_table(module.get("tables", {})),
    }


def _build_executive_summary(modules: list[dict[str, Any]], fallback: object) -> list[str]:
    preferred_modules = [
        "discount_profit_analysis",
        "loss_risk_modeling",
        "sales_trend_analysis",
        "product_contribution_analysis",
        "dimension_breakdown_analysis",
        "country_market_analysis",
        "order_structure_analysis",
    ]
    summary: list[str] = []
    for module_id in preferred_modules:
        module = next((item for item in modules if item.get("module_id") == module_id), None)
        if not module:
            continue
        for finding in module.get("findings", []):
            text = str(finding)
            if text and text not in summary:
                summary.append(text)
                break
        if len(summary) >= 4:
            return summary
    return _short_list(fallback, limit=4)


def _extract_section_conclusion(section_text: str) -> str:
    if not section_text:
        return ""
    match = re.search(r"本节结论[:：]\s*(.+)", section_text, flags=re.S)
    if match:
        return _first_paragraph(match.group(1))
    if "### 建模分析小结" in section_text:
        return _first_paragraph(section_text.split("### 建模分析小结", 1)[1])
    return ""


def _first_paragraph(text: str) -> str:
    text = text.strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    return paragraphs[0] if paragraphs else text


def _build_modeling(modules: list[dict[str, Any]], notebook: dict[str, Any]) -> dict[str, Any] | None:
    module = next((item for item in modules if item.get("module_id") == "loss_risk_modeling"), None)
    if not module:
        return None
    metrics = module.get("summary_metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return None
    notebook_text = notebook.get("sections", {}).get("modeling", "")
    return {
        "title": "建模分析：亏损风险识别",
        "best_model": metrics.get("best_model"),
        "narrative": _extract_section_conclusion(notebook_text),
        "full_text": notebook_text,
        "is_long_text": len(notebook_text) > 500,
        "metrics": {
            "Recall": _format_value(metrics.get("best_recall"), "percent"),
            "F1": _format_value(metrics.get("best_f1"), "decimal"),
            "ROC AUC": _format_value(metrics.get("best_roc_auc"), "decimal"),
        },
        "raw_metrics": metrics,
        "findings": _short_list(module.get("findings", []), limit=3),
        "confusion_matrix": module.get("tables", {}).get("confusion_matrix", []),
    }


def _build_data_note(modules: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
    quality = next((item for item in modules if item.get("module_id") == "data_quality_check"), {})
    return {
        "profile": {
            "row_count": profile.get("row_count"),
            "column_count": profile.get("column_count"),
        },
        "findings": _short_list(quality.get("findings", []), limit=3),
    }


def _chart_payload_from_tables(module: dict[str, Any]) -> dict[str, Any]:
    table = _first_table(module.get("tables", {}))
    rows = table.get("rows", [])
    if not rows:
        return {}
    columns = table.get("columns", [])
    if len(columns) < 2:
        return {}
    x_key = columns[0]
    y_key = next((column for column in columns[1:] if _is_number(rows[0].get(column))), columns[1])
    return {
        "x": [row.get(x_key) for row in rows[:24]],
        "y": [row.get(y_key) for row in rows[:24]],
        "series_name": y_key,
    }


def _first_table(tables: object) -> dict[str, Any]:
    if not isinstance(tables, dict):
        return {"title": "", "columns": [], "rows": []}
    for name, rows in tables.items():
        if isinstance(rows, list) and rows:
            first_row = rows[0] if isinstance(rows[0], dict) else {}
            return {
                "title": str(name),
                "columns": list(first_row.keys())[:6],
                "rows": rows[:5],
            }
    return {"title": "", "columns": [], "rows": []}


def _short_list(items: object, *, limit: int) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(item) for item in items if item][:limit]


def _first_text(items: list[str]) -> str:
    return items[0] if items else ""


def _format_value(value: object, value_type: str) -> str:
    if value is None:
        return "-"
    if value_type == "percent" and isinstance(value, int | float):
        return f"{value * 100:.0f}%" if abs(value) <= 1 else f"{value:.0f}%"
    if value_type == "number" and isinstance(value, int | float):
        return f"{value:,.0f}"
    if value_type == "decimal" and isinstance(value, int | float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
