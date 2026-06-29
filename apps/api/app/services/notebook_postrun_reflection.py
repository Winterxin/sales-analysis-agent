from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import nbformat
from nbformat import NotebookNode

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.notebook_outline import NotebookOutline
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.evidence_guard import downgrade_strong_inference_claims
from app.services.final_synthesis_guard import guard_english_public_narrative
from app.services.llm_trace_utils import (
    build_aggregated_llm_stage_trace,
    build_llm_stage_trace,
    describe_llm_error,
)
from app.services.metric_distribution_strategy import build_metric_distribution_strategy
from app.services.chart_selection_planner import selected_chart_metadata_by_title
from app.services.notebook_toolset import insert_markdown_cell_after_index, load_notebook, save_notebook
from app.services.notebook.markdown_sanitizer import (
    remove_template_discourse_markers,
    sanitize_final_markdown_text,
)
from app.services.output_language import (
    contains_cjk,
    englishize_notebook_text,
    is_english_output,
    user_facing_language_instruction,
)
from app.services.notebook.output_policy import is_compact_notebook_mode

SECTION_TO_MODULE = {
    "metric_distributions": "metric_distribution_analysis",
    "sales_trends": "sales_trend_analysis",
    "product_and_category": "product_contribution_analysis",
    "segment_and_region": "dimension_breakdown_analysis",
    "discount_and_profit": "discount_profit_analysis",
    "forecast": "forecast_analysis",
}

MAX_LLM_POSTRUN_REFLECTIONS = 6


def _explicit_english_output(output_language: str | None) -> bool:
    return output_language is not None and is_english_output(output_language)


def _section_title_map(outline: NotebookOutline) -> dict[str, str]:
    return {section.title.strip(): section.section_id for section in outline.sections}


def _normalize_source(cell: NotebookNode) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def _markdown_heading_title(source: str) -> str | None:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            return stripped[3:].strip()
    return None


def _text_preview_from_output(output: NotebookNode) -> str | None:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return None
    text_plain = data.get("text/plain")
    if isinstance(text_plain, list):
        return "".join(str(part) for part in text_plain).strip()
    if isinstance(text_plain, str):
        return text_plain.strip()
    return None


def _table_preview_from_cells(section_cells: list[NotebookNode]) -> str:
    previews: list[str] = []
    for cell in section_cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            preview = _text_preview_from_output(output)
            if preview:
                previews.append(preview)
    return previews[-1] if previews else ""


def _candidate_table_names_from_chart_source(source: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r"\bpx\.\w+\(\s*([A-Za-z_]\w*)",
        r"\bsns\.\w+\(\s*data\s*=\s*([A-Za-z_]\w*)",
        r"\bdata\s*=\s*([A-Za-z_]\w*)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            candidate = match.group(1)
            if candidate not in candidates:
                candidates.append(candidate)

    known_tables = [
        "top_products",
        "category_sales",
        "segment_region",
        "segment_view",
        "discount_profit",
        "discount_buckets",
        "monthly_sales",
        "forecast_df",
        "pareto_df",
        "metric_profile",
        "metric_summary",
        "metric_long",
    ]
    lowered_source = source.lower()
    for table_name in known_tables:
        if table_name.lower() in lowered_source and table_name not in candidates:
            candidates.append(table_name)
    return candidates


def _table_preview_from_cell(cell: NotebookNode) -> str:
    previews: list[str] = []
    for output in cell.get("outputs", []):
        preview = _text_preview_from_output(output)
        if preview:
            previews.append(preview)
    return previews[-1] if previews else ""


def _table_preview_for_chart(section_cells: list[NotebookNode], chart_source: str) -> str:
    candidates = _candidate_table_names_from_chart_source(chart_source)
    if candidates:
        for candidate in candidates:
            candidate_pattern = re.compile(rf"\b{re.escape(candidate)}\b", flags=re.IGNORECASE)
            for cell in reversed(section_cells):
                if cell.get("cell_type") != "code":
                    continue
                source = _normalize_source(cell)
                if not candidate_pattern.search(source):
                    continue
                preview = _table_preview_from_cell(cell)
                if preview:
                    return preview
    return _table_preview_from_cells(section_cells)


def _chart_title_from_output(output: NotebookNode) -> str | None:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return None
    plotly_payload = data.get("application/vnd.plotly.v1+json")
    if not isinstance(plotly_payload, dict):
        return None
    layout = plotly_payload.get("layout", {})
    if not isinstance(layout, dict):
        return None
    title = layout.get("title")
    if isinstance(title, dict):
        text = title.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def _chart_title_from_code(source: str) -> str | None:
    patterns = [
        r"\bplt\.title\(\s*(['\"])(.*?)\1",
        r"\bplt\.suptitle\(\s*(['\"])(.*?)\1",
        r"(?:\b\w+|\[[^\]]+\])\.set_title\(\s*(['\"])(.*?)\1",
        r"\btitle\s*=\s*(['\"])(.*?)\1",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.DOTALL)
        if match:
            title = match.group(2).strip()
            if title:
                return title
    return None


def _plotly_chart_kind_from_output(output: NotebookNode) -> str | None:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return None
    plotly_payload = data.get("application/vnd.plotly.v1+json")
    if not isinstance(plotly_payload, dict):
        return None
    traces = plotly_payload.get("data", [])
    if not isinstance(traces, list) or not traces:
        return None
    trace = traces[0]
    if not isinstance(trace, dict):
        return None
    trace_type = str(trace.get("type", "")).strip().lower()
    trace_mode = str(trace.get("mode", "")).strip().lower()
    if trace_type == "scatter" and "lines" in {part.strip() for part in trace_mode.split("+")}:
        return "line"
    return trace_type or None


def _infer_chart_kind_from_code(source: str) -> str | None:
    lowered = source.lower()
    if (
        "make_subplots" in lowered
        and re.search(r"secondary_y\s*=\s*true", source, flags=re.IGNORECASE)
        and "go.bar" in lowered
        and "go.scatter" in lowered
    ):
        return "dual_axis_bar_line"
    if (
        "pareto" in lowered
        or "cumulative_share" in lowered
        or ("twinx" in lowered and ".bar(" in lowered and ".plot(" in lowered)
    ):
        return "pareto"
    if "boxplot" in lowered:
        return "boxplot"
    if "heatmap" in lowered:
        return "heatmap"
    if "histplot" in lowered or "histogram" in lowered:
        return "histogram"
    if "treemap" in lowered:
        return "treemap"
    if "px.line" in lowered or "sns.lineplot" in lowered or "plt.plot" in lowered or ".plot(" in lowered:
        return "line"
    if "scatter" in lowered:
        return "scatter"
    if "px.bar" in lowered or "sns.barplot" in lowered or "plt.bar" in lowered or ".bar(" in lowered:
        return "bar"
    return None


def _resolve_chart_kind(source: str, output: NotebookNode | None = None) -> str | None:
    code_kind = _infer_chart_kind_from_code(source)
    output_kind = _plotly_chart_kind_from_output(output) if output is not None else None
    if code_kind == "scatter" and output_kind == "line":
        return "line"
    if code_kind == "line" and output_kind == "scatter":
        return "line"
    return code_kind or output_kind


def _series_pairs_from_plotly_trace(trace: dict[str, Any]) -> list[tuple[str, float]]:
    xs = trace.get("x")
    ys = trace.get("y")
    if not isinstance(xs, list) or not isinstance(ys, list):
        return []
    pairs: list[tuple[str, float]] = []
    for x, y in zip(xs, ys):
        try:
            value = float(y)
        except (TypeError, ValueError):
            continue
        pairs.append((str(x), value))
    return pairs


def _plotly_chart_kind_and_summary(
    output: NotebookNode,
    source: str = "",
) -> tuple[str | None, str | None]:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return None, None
    plotly_payload = data.get("application/vnd.plotly.v1+json")
    if not isinstance(plotly_payload, dict):
        return None, None
    traces = plotly_payload.get("data", [])
    if not isinstance(traces, list) or not traces:
        return None, None

    trace = traces[0]
    if not isinstance(trace, dict):
        return None, None

    output_kind = str(trace.get("type", "")).strip().lower() or None
    chart_kind = _resolve_chart_kind(source, output) or output_kind
    if chart_kind == "bar":
        orientation = str(trace.get("orientation", "v")).lower()
        xs = trace.get("x")
        ys = trace.get("y")
        if isinstance(xs, list) and isinstance(ys, list) and xs and ys:
            if orientation == "h":
                label = str(ys[0])
                try:
                    value = float(xs[0])
                    return chart_kind, f"头部对象是 {label}，对应数值约为 {value:.3f}。"
                except (TypeError, ValueError):
                    return chart_kind, f"头部对象是 {label}。"
            label = str(xs[0])
            try:
                value = float(ys[0])
                return chart_kind, f"头部对象是 {label}，对应数值约为 {value:.3f}。"
            except (TypeError, ValueError):
                return chart_kind, f"头部对象是 {label}。"
    if chart_kind == "line":
        pairs = _series_pairs_from_plotly_trace(trace)
        if pairs:
            high_point = max(pairs, key=lambda item: item[1])
            low_point = min(pairs, key=lambda item: item[1])
            return (
                chart_kind,
                f"高点出现在 {high_point[0]}（约 {high_point[1]:.3f}），低点出现在 {low_point[0]}（约 {low_point[1]:.3f}）。",
            )
    if chart_kind == "dual_axis_bar_line":
        return chart_kind, "该图同时展示柱形指标和折线指标，适合比较平均利润与亏损率是否同步恶化。"
    return chart_kind, None


def _chart_summary_from_code_and_output(
    source: str,
    output: NotebookNode,
    chart_title: str | None = None,
    table_preview: str | None = None,
) -> str:
    plotly_kind, plotly_summary = _plotly_chart_kind_and_summary(output, source=source)
    if plotly_summary:
        return plotly_summary

    chart_kind = plotly_kind or _resolve_chart_kind(source, output)
    title = chart_title or "当前图表"
    table_preview = table_preview or ""

    if chart_kind == "pareto":
        return f"{title}用于同时观察头部贡献和累计占比，适合判断销售额是否被少数商品过度拉动。"
    if chart_kind == "bar":
        if table_preview:
            first_line = table_preview.splitlines()[-1]
            return f"{title}显示头部对象的相对贡献，表格预览中最醒目的对象是 {first_line}。"
        return f"{title}主要用于比较不同对象的规模差异，便于识别头部和尾部。"
    if chart_kind == "line":
        if table_preview:
            first_line = table_preview.splitlines()[-1]
            return f"{title}主要观察时间趋势和波动节奏，当前预览里可直接看到 {first_line}。"
        return f"{title}主要用于观察趋势变化、高低点和波动区间。"
    if chart_kind == "dual_axis_bar_line":
        return f"{title}是双轴柱线组合图，用柱形和折线同步观察平均利润与亏损率。"
    if chart_kind == "heatmap":
        return f"{title}强调不同切片之间的强弱差异，适合识别持续偏弱或偏强的组合。"
    if chart_kind == "boxplot":
        return f"{title}用于比较不同折扣区间下利润分布的中位数、波动范围和异常点。"
    if chart_kind == "scatter":
        return f"{title}用于观察变量之间是否存在方向性关系，以及高风险点是否集中。"
    if chart_kind == "treemap":
        return f"{title}用于观察类目结构和集中度，面积越大通常表示贡献越高。"
    if chart_kind == "histogram":
        return f"{title}用于观察分布形态、偏态和是否存在长尾。"
    return f"{title}用于观察当前分析对象的相对差异。"


def _output_formats(output: NotebookNode) -> list[str]:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return []
    return [str(key) for key in data.keys()]


def _image_data_uri_from_output(output: NotebookNode) -> list[str]:
    data = output.get("data", {})
    if not isinstance(data, dict):
        return []
    image_urls: list[str] = []
    png = data.get("image/png")
    if isinstance(png, str) and png.strip():
        image_urls.append(f"data:image/png;base64,{png.strip()}")
    jpeg = data.get("image/jpeg")
    if isinstance(jpeg, str) and jpeg.strip():
        image_urls.append(f"data:image/jpeg;base64,{jpeg.strip()}")
    return image_urls


def _module_for_section(section_id: str, report: AnalysisReport | None) -> ModuleReport | None:
    if report is None:
        return None
    target_module = SECTION_TO_MODULE.get(section_id)
    if target_module is None:
        return None
    for module in report.modules:
        if module.module_id == target_module:
            return module
    return None


def _format_metric_number(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{decimals}f}"


def _format_metric_percent(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) <= 1:
        number *= 100
    return f"{number:.{decimals}f}%"


def _trim_table_preview(table_preview: str, max_lines: int = 3, max_chars: int = 420) -> str:
    if not table_preview.strip():
        return ""
    lines = [line.rstrip() for line in table_preview.splitlines() if line.strip()]
    compact = "\n".join(lines[:max_lines])
    if len(compact) > max_chars:
        return compact[: max_chars - 3] + "..."
    return compact


def _module_tables_preview(module: ModuleReport, max_rows: int = 3, max_chars: int = 320) -> dict[str, str]:
    previews: dict[str, str] = {}
    for table_name, rows in module.tables.items():
        if not rows:
            continue
        snippet = "\n".join(str(row) for row in rows[:max_rows])
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3] + "..."
        previews[table_name] = snippet
    return previews


def _metric_distribution_strategy_for_module(module: ModuleReport | None) -> dict[str, Any] | None:
    if module is None or module.module_id != "metric_distribution_analysis":
        return None
    return build_metric_distribution_strategy(module, llm_client=None)


def _strategy_decision_by_metric(
    strategy: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if not strategy:
        return {}
    decisions = strategy.get("decisions", [])
    if not isinstance(decisions, list):
        return {}
    return {
        str(decision.get("metric")): decision
        for decision in decisions
        if isinstance(decision, dict) and decision.get("metric")
    }


def _metric_label_by_metric(module: ModuleReport | None) -> dict[str, str]:
    if module is None:
        return {}
    rows = module.tables.get("metric_quantiles", [])
    return {
        str(row.get("metric")): str(row.get("label") or row.get("metric"))
        for row in rows
        if isinstance(row, dict) and row.get("metric")
    }


def _metric_strategy_summary(strategy: dict[str, Any] | None, module: ModuleReport | None) -> str:
    if not strategy:
        return ""
    labels = _metric_label_by_metric(module)
    decisions = strategy.get("decisions", [])
    if not isinstance(decisions, list):
        return ""

    highlights: list[str] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        metric = str(decision.get("metric") or "")
        label = labels.get(metric, str(decision.get("label") or metric or "指标"))
        charts = set(decision.get("charts", []))
        if "log_histogram" in charts and "p99_clipped_histogram" in charts:
            highlights.append(
                f"**{label}** 用 **log1p 分布** 拉开常规订单，并用 **P99 截尾分布** 避免极端大单压扁主体区间"
            )
        elif "log_histogram" in charts:
            highlights.append(f"**{label}** 用 **log1p 分布** 恢复常规订单之间的差异")
        elif "p99_clipped_histogram" in charts:
            highlights.append(f"**{label}** 用 **P99 截尾分布** 排除极端值压缩")
        elif "boxplot" in charts:
            highlights.append(f"**{label}** 用 **箱线图** 确认异常值和波动范围")

    if not highlights:
        return ""
    return "策略层的处理不是只盯一个最高长尾指标，而是分指标选择视角：" + "；".join(highlights) + "。"


def _compact_section_chart_context(
    chart_context: dict[str, Any],
    current_cell_index: int,
) -> dict[str, Any]:
    cell_index = int(chart_context.get("cell_index", -1))
    return {
        "cell_index": cell_index,
        "is_current_chart": cell_index == current_cell_index,
        "chart_title": chart_context.get("chart_title"),
        "chart_kind": chart_context.get("chart_kind"),
        "chart_summary": chart_context.get("chart_summary"),
        "table_preview": _trim_table_preview(str(chart_context.get("table_preview", "")), max_lines=2),
    }


def _section_contexts_by_section(
    chart_contexts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    contexts_by_section: dict[str, list[dict[str, Any]]] = {}
    for chart_context in chart_contexts:
        section_id = str(chart_context.get("section_id", ""))
        contexts_by_section.setdefault(section_id, []).append(chart_context)
    return contexts_by_section


def _postrun_reflection_priority(chart_context: dict[str, Any]) -> int:
    section_id = str(chart_context.get("section_id", ""))
    chart_kind = str(chart_context.get("chart_kind", "generic"))
    title = str(chart_context.get("chart_title", "")).lower()
    code_source = str(chart_context.get("code_source", "")).lower()
    summary = str(chart_context.get("chart_summary", "")).lower()
    combined_text = " ".join([title, code_source, summary])

    section_weight = {
        "discount_and_profit": 35,
        "product_and_category": 32,
        "segment_and_region": 28,
        "metric_distributions": 22,
        "sales_trends": 18,
        "forecast": 10,
    }.get(section_id, 0)
    kind_weight = {
        "pareto": 60,
        "heatmap": 52,
        "dual_axis_bar_line": 52,
        "scatter": 50,
        "boxplot": 46,
        "treemap": 42,
        "line": 38,
        "histogram": 34,
        "bar": 32,
    }.get(chart_kind, 20)
    keyword_weight = 0
    for keyword in [
        "pareto",
        "帕累托",
        "profit",
        "利润",
        "margin",
        "毛利",
        "loss",
        "亏损",
        "discount",
        "折扣",
        "risk",
        "风险",
        "高销售低利润",
        "negative",
        "concentration",
        "集中",
    ]:
        if keyword in combined_text:
            keyword_weight += 6
    return section_weight + kind_weight + min(keyword_weight, 30)


def _prioritized_chart_contexts(
    chart_contexts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed_contexts = list(enumerate(chart_contexts))
    indexed_contexts.sort(
        key=lambda item: (-_postrun_reflection_priority(item[1]), item[0])
    )
    return [context for _, context in indexed_contexts]


def _context_for_llm(
    chart_context: dict[str, Any],
    report: AnalysisReport | None,
    section_chart_contexts: list[dict[str, Any]] | None = None,
    output_language: str | None = None,
) -> dict[str, Any]:
    chart_kind = chart_context.get("chart_kind")
    english = _explicit_english_output(output_language)
    model_diagnostic_type = _model_diagnostic_chart_type(chart_context) if english else None
    if english:
        kind_focus = {
            "line": "Prioritize trend changes, peaks, troughs, volatility, and plausible operating triggers supported by evidence.",
            "bar": "Prioritize head-tail concentration, category or product gaps, and potential over-reliance.",
            "treemap": "Prioritize share structure, top-category dependence, and long-tail drag.",
            "pareto": "Prioritize top contribution concentration, cumulative-share breakpoints, and single-item dependence.",
            "heatmap": "Prioritize strong and weak slices, persistent weak combinations, and structural gaps.",
            "boxplot": "Prioritize median differences, spread, outliers, and risk intervals across groups.",
            "dual_axis_bar_line": (
                "This is a dual-axis bar-line chart; go.Scatter represents a line here, not a scatter plot. "
                "Describe it as a dual-axis chart or bar-line combination."
            ),
            "scatter": "Prioritize variable relationships, clusters, and high-risk combinations.",
            "histogram": "Prioritize skew, long tails, and the risk that averages do not represent typical records.",
        }
        default_focus = "Prioritize the most important business finding and contrast visible in the executed chart."
    else:
        kind_focus = {
            "line": "请优先解释趋势变化、高低点、阶段性波动和可能的经营触发因素。",
            "bar": "请优先解释头部集中度、尾部差距和是否存在过度依赖。",
            "treemap": "请优先解释结构占比、头部类目依赖和长尾拖累。",
            "pareto": "请优先解释头部贡献集中度、累计占比拐点和单品依赖风险。",
            "heatmap": "请优先解释强弱切片、稳定偏弱区域和结构性差距。",
            "boxplot": "请优先解释不同区间的中位数、波动范围、异常值和风险区间。",
            "dual_axis_bar_line": (
                "这是双轴柱线组合图；go.Scatter 在这里表示折线，不是散点图。"
                "不要把该图称为散点图，应描述为双轴图、柱线组合图或平均利润与亏损率组合图。"
            ),
            "scatter": "请优先解释变量关系、聚集点和高风险组合。",
            "histogram": "请优先解释分布偏态、长尾和均值不代表典型水平的风险。",
        }
        default_focus = "请优先解释最重要的经营发现和差异。"
    if model_diagnostic_type:
        context: dict[str, Any] = {
            "chart_title": chart_context.get("chart_title"),
            "chart_kind": chart_kind,
            "model_diagnostic_chart_type": model_diagnostic_type,
            "chart_summary": chart_context.get("chart_summary"),
            "table_preview": _trim_table_preview(str(chart_context.get("table_preview", ""))),
            "analysis_focus": (
                "Use only the current model diagnostic chart and its model metrics. "
                "Do not import other business-dimension evidence, scenario outputs, or example tables."
            ),
            "section_review_instruction": (
                "This is a model diagnostic chart. Interpret only the current chart_context, "
                "threshold, recall, precision, false-positive, false-negative, review-load, "
                "or feature-importance evidence that appears in this chart. Keep the commentary "
                "inside the displayed model diagnostic evidence and avoid other business slices, "
                "scenario outputs, example records, or monetary improvement claims."
            ),
        }
        for optional_key in (
            "model_metrics",
            "model_metric_evidence",
            "axis_labels",
            "axis_metadata",
            "label_metadata",
        ):
            if optional_key in chart_context:
                context[optional_key] = chart_context.get(optional_key)
        return context
    cell_index = int(chart_context.get("cell_index", -1))
    section_chart_contexts = section_chart_contexts or [chart_context]
    section_chart_count = len(section_chart_contexts)
    section_chart_position = next(
        (
            index + 1
            for index, section_context in enumerate(section_chart_contexts)
            if int(section_context.get("cell_index", -1)) == cell_index
        ),
        1,
    )

    context: dict[str, Any] = {
        "section_id": chart_context.get("section_id"),
        "section_title": chart_context.get("section_title"),
        "chart_title": chart_context.get("chart_title"),
        "chart_kind": chart_kind,
        "chart_summary": chart_context.get("chart_summary"),
        "table_preview": _trim_table_preview(str(chart_context.get("table_preview", ""))),
        "output_formats": chart_context.get("output_formats", []),
        "code_source": str(chart_context.get("code_source", ""))[:600],
        "image_urls": chart_context.get("image_urls", []),
        "analysis_focus": kind_focus.get(str(chart_kind), default_focus),
        "section_chart_count": section_chart_count,
        "section_chart_position": section_chart_position,
        "section_peer_charts": [
            _compact_section_chart_context(section_context, cell_index)
            for section_context in section_chart_contexts
        ],
        "section_review_instruction": (
            "Treat the current chart as one piece of evidence in this notebook section. "
            "When peer charts exist, connect the current chart to at least one adjacent "
            "section-level signal instead of writing an isolated chart description. "
            "Prioritize the current section, its module findings, and peer charts; use "
            "global report_summary only when it is directly relevant to this section."
        ),
    }
    if chart_kind == "dual_axis_bar_line":
        if english:
            context["chart_kind_instruction"] = (
                "This is a dual-axis bar-line chart; go.Scatter represents a line in this chart, not a scatter plot. "
                "Do not call it a scatter plot in the final wording."
            )
        else:
            context["chart_kind_instruction"] = (
                "这是双轴柱线组合图；go.Scatter 在本图中表示折线，不是散点图。"
                "最终措辞不要称为散点图，应称为双轴图、柱线组合图或平均利润与亏损率组合图。"
            )

    module = _module_for_section(str(chart_context.get("section_id", "")), report)
    if module is not None:
        context["module_summary_metrics"] = module.summary_metrics
        if not model_diagnostic_type:
            context["module_findings"] = module.findings
            context["module_tables_preview"] = _module_tables_preview(module)
            metric_strategy = _metric_distribution_strategy_for_module(module)
            if metric_strategy is not None:
                context["metric_distribution_strategy"] = metric_strategy
                context["metric_strategy_instruction"] = (
                    "For metric distribution charts, explain why the selected scale or "
                    "diagnostic view matters. If log_histogram or p99_clipped_histogram "
                    "is selected, connect it to mean/median/P99 gaps and warn that raw "
                    "averages may be misleading."
                )

    if report is not None and not model_diagnostic_type:
        context["report_summary"] = report.summary

    return context


def _is_low_quality_llm_reflection(
    reflection_markdown: str,
    *,
    output_language: str | None = None,
) -> bool:
    stripped = reflection_markdown.strip()
    if _explicit_english_output(output_language) and contains_cjk(stripped):
        return True
    if len(stripped) < 80:
        return True
    if _sentence_count(stripped) < 3:
        return True
    if not _has_concrete_evidence(stripped):
        return True
    forbidden_starts = [
        "这张图对应的是",
        "这是一张用于分析",
        "当前展示的是",
        "这张图主要展示",
        "这张图主要用于",
        "当前图表用于观察",
        "如果图表显示",
    ]
    if any(stripped.startswith(prefix) for prefix in forbidden_starts):
        return True
    forbidden_markers = [
        "### 关键观察",
        "### 业务含义",
        "### 后续追问",
        "### 下一步",
        "- 关键观察",
        "- 业务含义",
        "如果图表显示",
        "这张图主要用于",
        "当前图表用于观察",
    ]
    return any(marker in stripped for marker in forbidden_markers)


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[。！？.!?]+", text) if part.strip()])


def _has_concrete_evidence(text: str) -> bool:
    if re.search(r"\d{4}[-/年]\d{1,2}|\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?%?", text):
        return True
    return bool(re.search(r"[A-Z][A-Za-z0-9][A-Za-z0-9 .&'()/+-]{2,}", text))


def _normalize_llm_reflection_markdown(reflection_markdown: str) -> str:
    normalized = reflection_markdown.strip()
    normalized = normalized.replace("\\r\\n", "\n")
    normalized = normalized.replace("\\n", "\n")
    normalized = normalized.replace("\\t", " ")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _is_chart_cell(cell: NotebookNode) -> bool:
    if cell.get("cell_type") != "code":
        return False
    source = _normalize_source(cell).lower()
    if "fig.show()" in source or "plt.show()" in source:
        return True
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if not isinstance(data, dict):
            continue
        if "application/vnd.plotly.v1+json" in data:
            return True
        if "image/png" in data or "image/jpeg" in data:
            return True
    return False


def extract_postrun_chart_contexts(
    notebook_path: Path,
    outline: NotebookOutline,
    chart_selection_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    notebook = nbformat.read(notebook_path, as_version=4)
    title_map = _section_title_map(outline)
    contexts: list[dict[str, Any]] = []
    selected_by_title = selected_chart_metadata_by_title(chart_selection_plan)
    selected_by_section_kind: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if isinstance(chart_selection_plan, dict):
        for item in chart_selection_plan.get("selected_charts", []):
            if not isinstance(item, dict):
                continue
            key = (str(item.get("section_id", "")), str(item.get("chart_kind", "")))
            selected_by_section_kind.setdefault(key, []).append(dict(item))
    current_section_id: str | None = None
    current_section_title: str | None = None
    current_section_cells: list[NotebookNode] = []

    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") == "markdown":
            heading = _markdown_heading_title(_normalize_source(cell))
            if heading and heading in title_map:
                current_section_title = heading
                current_section_id = title_map[heading]
                current_section_cells = [cell]
                continue

        if current_section_id is None:
            continue

        current_section_cells.append(cell)

        if not _is_chart_cell(cell):
            continue

        cell_source = _normalize_source(cell)
        table_preview = _table_preview_for_chart(current_section_cells[:-1], cell_source)
        outputs = cell.get("outputs", [])
        chart_title: str | None = _chart_title_from_code(cell_source)
        chart_kind: str | None = None
        chart_summary: str | None = None
        output_formats: list[str] = []
        image_urls: list[str] = []

        for output in outputs:
            if chart_title is None:
                chart_title = _chart_title_from_output(output)
            if chart_kind is None:
                chart_kind = _resolve_chart_kind(cell_source, output)
            if chart_summary is None:
                chart_summary = _chart_summary_from_code_and_output(
                    cell_source,
                    output,
                    chart_title=chart_title,
                    table_preview=table_preview,
                )
            output_formats.extend(_output_formats(output))
            image_urls.extend(_image_data_uri_from_output(output))

        chart_kind = chart_kind or _resolve_chart_kind(cell_source)
        if chart_summary is None:
            chart_summary = _chart_summary_from_code_and_output(
                cell_source,
                outputs[0] if outputs else NotebookNode(),
                chart_title=chart_title,
                table_preview=table_preview,
            )

        context = {
            "section_id": current_section_id,
            "section_title": current_section_title,
            "chart_title": chart_title or "当前图表",
            "chart_kind": chart_kind or "generic",
            "chart_summary": chart_summary,
            "table_preview": table_preview,
            "code_source": cell_source,
            "cell_index": index,
            "output_formats": sorted(set(output_formats)),
            "image_urls": image_urls,
        }
        title_text = str(context["chart_title"])
        metadata = selected_by_title.get(title_text)
        if metadata is None:
            metadata = next(
                (
                    item
                    for title, item in selected_by_title.items()
                    if title and (title in title_text or title_text in title)
                ),
                None,
            )
        if metadata is None:
            metadata = next(
                (
                    item
                    for title, item in selected_by_title.items()
                    if title and title in cell_source
                ),
                None,
            )
        if metadata is None:
            same_kind = selected_by_section_kind.get(
                (str(current_section_id), str(context["chart_kind"])),
                [],
            )
            if len(same_kind) == 1:
                metadata = same_kind[0]
        if metadata is None and current_section_id == "discount_and_profit":
            title_tokens = {"折扣区间", "利润", "亏损率"}
            if all(token in title_text for token in title_tokens):
                metadata = selected_by_title.get("各折扣区间利润质量：平均利润与亏损率")
        if metadata is not None:
            context.update(
                {
                    "chart_id": metadata.get("chart_id"),
                    "business_question": metadata.get("business_question"),
                    "selection_reason": metadata.get("reason"),
                    "selection_source": metadata.get("selection_source"),
                }
            )
        contexts.append(context)

    return contexts


def _sanitize_unqualified_reflection_speculation(text: str) -> str:
    sanitized = text
    replacements = {
        "审批流程存在漏洞": "需结合审批记录进一步验证折扣流程是否存在漏洞",
        "折扣审批存在漏洞": "需结合审批记录进一步验证折扣流程是否存在漏洞",
        "供应链成本过高": "需结合成本字段进一步判断是否存在成本压力",
        "成本失控": "需结合成本字段进一步判断是否存在成本压力",
        "履约成本过高": "需结合履约成本字段进一步验证履约压力",
        "促销 ROI 不足": "需结合活动成本和转化数据验证促销 ROI",
        "客户生命周期价值不足": "需结合 LTV 或复购价值字段进一步验证客户长期价值",
    }
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    sanitized = sanitized.replace("值得关注的是", "需要复核的是")
    sanitized = sanitized.replace("值得关注", "需要复核")
    return sanitized


def _sanitize_reflection_chart_kind_wording(text: str, chart_kind: str | None) -> str:
    if chart_kind != "dual_axis_bar_line":
        return text
    sanitized = text
    replacements = {
        "散点图清晰揭示了": "该双轴图清晰显示了",
        "散点图显示": "该双轴图显示",
        "散点图": "双轴图",
    }
    for source, target in replacements.items():
        sanitized = sanitized.replace(source, target)
    return sanitized


def _cjk_length(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _reflection_sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[。！？.!?\n]+", text) if part.strip()])


def _ensure_min_reflection_sentences(text: str, min_sentences: int = 3) -> str:
    if _reflection_sentence_count(text) >= min_sentences:
        return text
    parts = [part.strip() for part in re.split(r"[，；;]", text) if part.strip()]
    if len(parts) < min_sentences:
        return text
    head = parts[: min_sentences - 1]
    tail = "，".join(parts[min_sentences - 1 :]).strip()
    compact = "。".join([*head, tail]).rstrip("。")
    return compact + "。"


def compress_chart_reflection_text(text: str, max_cjk_chars: int = 220) -> str:
    normalized = _sanitize_unqualified_reflection_speculation(text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""
    if _cjk_length(normalized) <= max_cjk_chars:
        return _ensure_min_reflection_sentences(normalized)

    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？!?])\s*", normalized)
        if item.strip()
    ]
    if not sentences:
        return normalized[:max_cjk_chars].rstrip("，、；： ") + "。"

    selected: list[str] = []
    number_sentences = [sentence for sentence in sentences if re.search(r"-?\d+(?:\.\d+)?%?", sentence)]
    meaning_sentences = [
        sentence
        for sentence in sentences
        if any(term in sentence for term in ("说明", "意味着", "显示", "表明", "风险", "侵蚀", "分化", "不同步"))
    ]
    action_sentences = [
        sentence
        for sentence in sentences
        if any(term in sentence for term in ("后续", "下一步", "应", "需要", "优先", "验证", "复盘"))
    ]
    for bucket, limit in ((number_sentences, 2), (meaning_sentences, 1), (action_sentences, 1)):
        added = 0
        for sentence in bucket:
            if sentence in selected:
                continue
            selected.append(sentence)
            added += 1
            if added >= limit:
                break
    if not selected:
        selected = sentences[:3]

    compact: list[str] = []
    for sentence in selected:
        candidate = "".join(compact + [sentence])
        if _cjk_length(candidate) > max_cjk_chars:
            continue
        compact.append(sentence)
    result = "".join(compact) or selected[0]
    if _cjk_length(result) > max_cjk_chars:
        result = result[: max_cjk_chars + 40]
        while _cjk_length(result) > max_cjk_chars:
            result = result[:-1]
        result = result.rstrip("，、；： ")
        if not result.endswith(("。", "！", "？")):
            result += "。"
    return _ensure_min_reflection_sentences(result)


def apply_postrun_chart_reflections(
    notebook_path: Path,
    output_path: Path,
    reflections_by_chart_cell: dict[int, str],
    notebook_output_mode: str | None = None,
    output_language: str | None = None,
) -> None:
    notebook = load_notebook(notebook_path)
    inserted = 0
    heading = "### Chart Commentary" if _explicit_english_output(output_language) else "### 图表解读"
    for cell_index in sorted(reflections_by_chart_cell):
        reflection = reflections_by_chart_cell[cell_index]
        if not reflection:
            continue
        reflection = compress_chart_reflection_text(reflection)
        reflection = sanitize_final_markdown_text(reflection)
        reflection = remove_template_discourse_markers(reflection)
        if not reflection:
            continue
        reflection_source = (
            reflection
            if notebook_output_mode is not None and is_compact_notebook_mode(notebook_output_mode)
            else f"{heading}\n\n{reflection}"
        )
        insert_markdown_cell_after_index(
            notebook,
            cell_index + inserted,
            reflection_source,
        )
        inserted += 1
    save_notebook(notebook, output_path)


def _call_postrun_reflection_llm(
    llm_client: Any,
    chart_context: dict[str, Any],
    fallback_markdown: str,
    *,
    language_instruction: str | None,
) -> dict[str, Any]:
    method = llm_client.suggest_postrun_chart_reflection
    signature = inspect.signature(method)
    if (
        "language_instruction" in signature.parameters
        or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    ):
        return method(
            chart_context,
            fallback_markdown,
            language_instruction=language_instruction,
        )
    return method(chart_context, fallback_markdown)


def _fallback_postrun_reflection(
    chart_context: dict[str, Any],
    report: AnalysisReport | None = None,
) -> str:
    chart_kind = str(chart_context.get("chart_kind", "generic"))
    chart_title = str(chart_context.get("chart_title", "当前图表"))
    chart_summary = str(chart_context.get("chart_summary", "")).strip()
    table_preview = str(chart_context.get("table_preview", "")).strip()
    section_id = str(chart_context.get("section_id", ""))
    code_source = str(chart_context.get("code_source", ""))
    module = _module_for_section(section_id, report)
    metrics = module.summary_metrics if module is not None else {}
    tables = module.tables if module is not None else {}

    if section_id == "metric_distributions":
        quantile_rows = [
            row for row in tables.get("metric_quantiles", []) if isinstance(row, dict)
        ]
        outlier_rows = [
            row for row in tables.get("metric_outliers", []) if isinstance(row, dict)
        ]
        tail_metric = metrics.get("highest_tail_metric")
        tail_row = next(
            (row for row in quantile_rows if row.get("metric") == tail_metric),
            None,
        )
        if tail_row is None and quantile_rows:
            tail_row = max(
                quantile_rows,
                key=lambda row: float(row.get("p99") or 0)
                / max(float(row.get("median") or 0), 1e-9),
            )
        outlier_metric = metrics.get("highest_outlier_metric")
        outlier_row = next(
            (row for row in outlier_rows if row.get("metric") == outlier_metric),
            None,
        )
        if outlier_row is None and outlier_rows:
            outlier_row = max(outlier_rows, key=lambda row: float(row.get("outlier_ratio") or 0))

        tail_label = str((tail_row or {}).get("label") or (tail_row or {}).get("metric") or "核心指标")
        tail_median = _format_metric_number((tail_row or {}).get("median"))
        tail_p99 = _format_metric_number((tail_row or {}).get("p99"))
        tail_mean = _format_metric_number((tail_row or {}).get("mean"))
        tail_mean_median_ratio = _format_metric_number((tail_row or {}).get("mean_median_ratio"))
        tail_metric = str((tail_row or {}).get("metric") or "")
        outlier_label = str(
            (outlier_row or {}).get("label") or (outlier_row or {}).get("metric") or "核心指标"
        )
        outlier_ratio = _format_metric_percent((outlier_row or {}).get("outlier_ratio"))
        strategy = _metric_distribution_strategy_for_module(module)
        strategy_by_metric = _strategy_decision_by_metric(strategy)
        tail_decision = strategy_by_metric.get(tail_metric, {})
        charts = set(tail_decision.get("charts", [])) if isinstance(tail_decision, dict) else set()
        strategy_text = _metric_strategy_summary(strategy, module)
        if "log_histogram" in charts and "p99_clipped_histogram" in charts:
            tail_strategy_text = "最高长尾指标也触发了 **log1p 分布** 和 **P99 截尾分布** 的组合视角。"
        elif "log_histogram" in charts:
            tail_strategy_text = "最高长尾指标也触发了 **log1p 分布** 视角。"
        elif "p99_clipped_histogram" in charts:
            tail_strategy_text = "最高长尾指标也触发了 **P99 截尾分布** 视角。"
        elif "boxplot" in charts:
            tail_strategy_text = "最高长尾指标更适合用 **箱线图** 确认异常值和波动范围。"
        else:
            tail_strategy_text = ""
        chart_role = (
            "箱线图更适合确认异常值和波动范围是否集中在少数指标上"
            if chart_kind == "boxplot"
            else "分布图更适合先判断均值是否会被长尾订单拉偏"
        )
        return (
            f"{chart_title}的重点不是看图形是否平滑，而是先判断后续分析能不能直接相信平均值。"
            f"当前长尾最明显的指标是 **{tail_label}**，均值约 **{tail_mean}**，中位数约 **{tail_median}**，P99 约 **{tail_p99}**，"
            f"均值/中位数约 **{tail_mean_median_ratio}**；"
            f"异常值占比最高的是 **{outlier_label}**，约 **{outlier_ratio}**。{chart_role}，"
            f"{strategy_text}{tail_strategy_text}因此后面的趋势、商品和利润专题都应该把常规订单与极端订单分开看，避免少数大单掩盖真实经营结构。"
        )

    if section_id == "forecast" and chart_kind == "line":
        baseline = metrics.get("baseline_sales_amount")
        horizon = metrics.get("horizon_days")
        baseline_text = _format_metric_number(baseline) if baseline is not None else "未知"
        horizon_text = str(horizon) if horizon is not None else "未知"
        preview_text = _trim_table_preview(table_preview, max_lines=1)
        method_text = (
            "从代码看，这里更像是用最近几期均值外推出来的基线预测，"
            if "tail(" in code_source and ".mean()" in code_source
            else "当前这条预测线更像基线判断，"
        )
        return (
            f"{chart_title}真正有用的不是曲线长什么样，而是未来 {horizon_text} 天的经营基线大约落在 **{baseline_text}**。"
            f"{method_text}{preview_text + '。' if preview_text else ''}它适合回答不额外做动作时销售大概会维持在哪个水平，不适合直接拿来当精细预算。"
            "下一步应把季节性、促销和节假日因素补进去，再看哪些月份会被基线预测低估。"
        )

    if section_id == "discount_and_profit" and "亏损率" in chart_title:
        bucket_rows = (
            tables.get("discount_profit_risk_buckets")
            or tables.get("discount_buckets")
            or []
        )
        worst_bucket = None
        if bucket_rows:
            worst_bucket = min(
                [row for row in bucket_rows if isinstance(row, dict)],
                key=lambda row: float(row.get("avg_profit") or 0),
                default=None,
            )
        if worst_bucket is not None:
            bucket_label = str(worst_bucket.get("discount_bucket", "风险折扣区间"))
            avg_profit = _format_metric_number(worst_bucket.get("avg_profit"))
            loss_rate = _format_metric_percent(
                worst_bucket.get("negative_profit_rate") or worst_bucket.get("loss_rate")
            )
            order_count = _format_metric_number(worst_bucket.get("order_count"))
            return (
                f"{chart_title}同时展示平均利润和亏损率，最弱的折扣区间是 **{bucket_label}**，"
                f"平均利润约 **{avg_profit}**，亏损率约 **{loss_rate}**，订单数约 **{order_count}**。"
                "这说明折扣风险不能只看销售额或单个亏损点，而要同时看折扣区间的利润带和亏损订单密度。"
                "下一步应把该折扣区间拆到商品、客户和审批口径，优先处理高销售但低利润的订单。"
            )

    if chart_kind == "dual_axis_bar_line":
        summary = chart_summary or _trim_table_preview(table_preview, max_lines=1) or "当前双轴图缺少可直接引用的数值。"
        return (
            f"{chart_title}需要按双轴图理解：柱形看平均利润，折线看亏损率。{summary}。"
            "当平均利润下探而亏损率抬升时，折扣风险已经同时体现在利润水平和亏损订单密度上。"
            "后续应把最弱折扣区间拆到商品、客户和审批口径，优先处理高销售但低利润的订单。"
        )

    if section_id == "discount_and_profit" and chart_kind == "boxplot":
        high_discount_orders = metrics.get("high_discount_order_count")
        negative_profit_rate = metrics.get("negative_profit_rate")
        discount_buckets = tables.get("discount_buckets", [])
        worst_bucket = None
        if discount_buckets:
            worst_bucket = min(
                discount_buckets,
                key=lambda row: float(row.get("avg_profit", 0)),
            )
        if worst_bucket is not None:
            worst_bucket_label = str(worst_bucket.get("discount_bucket", "未知区间"))
            worst_bucket_profit = _format_metric_number(worst_bucket.get("avg_profit"))
            return (
                f"{chart_title}更值得看的不是离群点，而是折扣区间整体利润带有没有下移。"
                f"当前高折扣记录已有 **{high_discount_orders}** 条，负利润占比约 **{_format_metric_percent(negative_profit_rate)}**，"
                f"其中 **{worst_bucket_label}** 桶的平均利润约为 **{worst_bucket_profit}**，这说明高折扣已经不是个别异常，而是在系统性压缩利润。"
                "后续应优先复盘最深折扣区间里的大单和低毛利 SKU，确认是促销策略问题还是审批失控。"
            )

    if chart_kind == "line":
        summary = chart_summary or _trim_table_preview(table_preview, max_lines=1) or "当前趋势上下文缺少可直接引用的峰谷数值。"
        return (
            f"{chart_title}先看峰谷差距和阶段节奏。{summary}。"
            "高点与低点并列出现时，销售更可能受到活动、季节或大单影响，而不是平滑增长。"
            "后续应把峰值月份对应的商品、渠道和订单金额拆开，确认增长来自可复制需求还是一次性冲高。"
        )
    if chart_kind in {"bar", "treemap"}:
        focus = chart_summary or "当前图已经显示出对象之间的贡献差异。"
        preview = f"表格预览中还能直接看到 {table_preview.splitlines()[-1]}。" if table_preview else ""
        if section_id == "product_and_category":
            implication = "商品或类目的头尾差距越大，经营越需要同时管理爆款依赖和长尾利润压力。"
            next_step = "后续应把头部销售贡献、利润率和促销成本合并成一张商品复盘表。"
        elif section_id == "discount_and_profit":
            implication = "折扣区间之间的利润差距越大，越说明促销强度正在改变订单盈利质量。"
            next_step = "后续应把低平均利润和高亏损率区间拆到商品、客户和审批人维度。"
        elif section_id == "segment_and_region":
            implication = "组合切片之间的差距越大，越说明经营问题落在具体客群和区域交叉对象上。"
            next_step = "后续应把弱势组合切片与折扣、履约成本和商品结构放在一起复盘。"
        else:
            implication = "头部对象与尾部对象差距越大，经营越依赖少数对象贡献。"
            next_step = "后续应把头部贡献与利润率放在同一张复盘表里，优先处理销售高但利润不同步的对象。"
        return (
            f"{chart_title}显示的核心问题是结构集中度。{focus}{preview}"
            f"{implication}{next_step}"
        )
    if chart_kind == "heatmap":
        return (
            f"{chart_title}最有价值的是横向比较不同切片的强弱差异。{chart_summary or _trim_table_preview(table_preview, max_lines=1)}。"
            "强势格子和弱势格子同时存在，说明问题更可能来自客群、区域或品类组合，而不是总体需求不足。"
            "后续应锁定持续偏弱的组合，复盘折扣、产品结构和履约成本是否共同压低利润。"
        )
    if chart_kind == "boxplot":
        return (
            f"{chart_title}重点不是单个点，而是各区间的中位数、分布宽度和异常值。{chart_summary or _trim_table_preview(table_preview, max_lines=1)}。"
            "高风险区间与低风险区间的利润带拉开时，问题更像系统性定价或审批风险。"
            "后续应优先复盘低位区间的大单和低毛利 SKU，确认折扣是否超过可承受利润边界。"
        )
    if chart_kind == "scatter":
        return (
            f"{chart_title}更适合看变量之间是否存在方向关系，以及风险点是否集中。{chart_summary or _trim_table_preview(table_preview, max_lines=1)}。"
            "高销售与低利润同时出现时，规模贡献和盈利质量已经不同步。"
            "后续应筛出高销售、低利润或高折扣订单，逐笔检查价格授权、赠品和成本归集。"
        )
    if chart_kind == "histogram":
        return (
            f"{chart_title}直接指向分布偏态和长尾问题。{chart_summary or _trim_table_preview(table_preview, max_lines=1)}。"
            "均值与典型订单差距越大，头部大单或异常值对结论的影响越强。"
            "后续应同时报告中位数、P90 和 P99，并把极端订单单独拆出，避免均值误导经营判断。"
        )
    return (
        f"{chart_title}已经给出了当前切片的主要差异。{chart_summary or _trim_table_preview(table_preview, max_lines=1)}。"
        "头部与尾部、增长与下滑或销售与利润不同步的地方，是本图最需要保留的经营信号。"
        "后续应把这些差异回填到明细表，优先确认能否通过定价、促销或商品组合调整改善。"
    )


def _english_context_text(value: object, fallback: str) -> str:
    text = englishize_notebook_text(value).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u3002\uff0c\uff1a\uff1b]", " ", text).strip(" ,;:")
    if not text or contains_cjk(text):
        return fallback
    return text


_INTERNAL_PREVIEW_MARKERS = (
    "blocked_fields",
    "field_mapping",
    "dtype",
    "columns",
    "index(",
    "dataframe",
    "unnamed:",
    "shows (",
)


def _is_usable_english_chart_evidence(value: object) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or contains_cjk(text):
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _INTERNAL_PREVIEW_MARKERS):
        return False
    if re.fullmatch(r"(?:nan|none|null)", lowered):
        return False
    tokens = text.split()
    if not tokens:
        return False
    identifier_tokens = [
        token
        for token in tokens
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token.strip(" ,;:()[]{}"))
    ]
    if len(tokens) >= 3 and len(identifier_tokens) == len(tokens):
        return False
    snake_or_camel_tokens = [
        token
        for token in tokens
        if "_" in token or re.search(r"[a-z][A-Z]", token)
    ]
    if snake_or_camel_tokens and len(snake_or_camel_tokens) >= max(1, len(tokens) - 1):
        return False
    if re.search(r"\d", text):
        return True
    if re.search(r"\b(?:shows|compares|indicates|highlights|contains|includes|category|segment|region|product|discount|sales|profit|margin|trend|threshold)\b", lowered):
        return True
    return bool(re.search(r"\b[A-Z][A-Za-z&'()/+-]{2,}\b", text) and len(tokens) >= 2)


def _looks_header_only_preview(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return False
    if not _is_usable_english_chart_evidence(cleaned):
        return True
    if re.search(r"\d", cleaned):
        return False
    tokens = cleaned.split()
    if len(tokens) < 4:
        return False
    identifier_tokens = [
        token
        for token in tokens
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token)
        or token in {"Sales", "Quantity", "Profit", "Discount"}
    ]
    return len(identifier_tokens) == len(tokens)


def _model_diagnostic_chart_type(chart_context: dict[str, Any]) -> str | None:
    section_id = str(chart_context.get("section_id") or "").lower()
    identity_text = " ".join(
        str(chart_context.get(key) or "")
        for key in (
            "chart_title",
            "chart_id",
            "business_question",
            "chart_summary",
            "table_preview",
        )
    ).lower()
    code_text = str(chart_context.get("code_source") or "").lower()

    def _has(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    def _has_confusion_identity(text: str) -> bool:
        if "confusion" in text or "confusionmatrixdisplay" in text:
            return True
        return all(re.search(rf"\b{token}\b", text) for token in ("tp", "fp", "fn", "tn"))

    feature_terms = ("feature importance", "importance", "predictive signal", "shap")
    threshold_terms = ("threshold", "trade-off", "tradeoff", "review_load", "review load", "precision-recall")
    generic_model_terms = ("recall", "precision", "classification", "loss-risk", "loss risk", "model score", "risk score")
    model_scope_terms = ("model", "classifier", "classification", "loss-risk", "loss risk", "risk score", "recall", "precision")

    if _has_confusion_identity(identity_text):
        return "confusion_matrix"
    if _has(identity_text, feature_terms):
        return "feature_importance"
    if _has(identity_text, threshold_terms) and (
        section_id == "modeling" or _has(identity_text, model_scope_terms)
    ):
        return "threshold"
    if _has(identity_text, generic_model_terms):
        return "model_diagnostic"

    fallback_text = code_text
    if section_id != "modeling" and not _has(
        " ".join((identity_text, fallback_text)),
        ("confusion", "recall", "precision", "classification", "model score"),
    ):
        return None
    if _has_confusion_identity(fallback_text):
        return "confusion_matrix"
    if _has(fallback_text, feature_terms):
        return "feature_importance"
    if _has(fallback_text, threshold_terms):
        return "threshold"
    if _has(fallback_text, generic_model_terms):
        return "model_diagnostic"
    return None


def _model_chart_fallback(chart_context: dict[str, Any], diagnostic_type: str | None = None) -> str:
    diagnostic_type = diagnostic_type or _model_diagnostic_chart_type(chart_context)
    chart_title = _english_context_text(chart_context.get("chart_title"), "This model diagnostic chart")
    if diagnostic_type == "threshold":
        return " ".join(
            [
                f"{chart_title} summarizes model recall and review workload for the available review-threshold choices.",
                "Select a review threshold based on review capacity and the observed false-positive / false-negative trade-off.",
                "Use the model only for manual review prioritization.",
            ]
        )
    if diagnostic_type == "confusion_matrix":
        return " ".join(
            [
                f"{chart_title} compares correctly identified loss-risk orders with missed orders and extra review load.",
                "Review false positives and false negatives with manual review capacity before changing the review threshold.",
                "Use the model only as a manual-review priority signal.",
            ]
        )
    if diagnostic_type == "feature_importance":
        return " ".join(
            [
                f"{chart_title} ranks predictive signals used by the model.",
                "These signals are not causal explanations and should be checked against order-level evidence before changing business rules.",
                "Use the model only as a manual-review priority signal.",
            ]
        )
    return " ".join(
        [
            f"{chart_title} summarizes model diagnostic evidence.",
            "Interpret recall, precision, false positives, false negatives, and review load before changing review thresholds.",
            "Use the model only as a manual-review priority signal.",
        ]
    )


_MODEL_NUMERIC_PATTERN = re.compile(
    r"(?<![A-Za-z])[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*[-–]\s*[-+]?\d[\d,]*(?:\.\d+)?)?%?"
)


def _normalize_model_number_token(value: str) -> str:
    token = str(value or "").strip().lower()
    token = token.replace("–", "-").replace("—", "-")
    token = re.sub(r"\s*-\s*", "-", token)
    token = token.replace("$", "").replace(",", "")
    return token


def _model_number_tokens(text: str) -> set[str]:
    return {
        _normalize_model_number_token(match.group(0))
        for match in _MODEL_NUMERIC_PATTERN.finditer(str(text or ""))
        if _normalize_model_number_token(match.group(0))
    }


def _visible_model_number_tokens(chart_context: dict[str, Any]) -> set[str]:
    evidence_text = " ".join(
        str(chart_context.get(key) or "")
        for key in ("chart_summary", "table_preview", "model_metrics", "model_metric_evidence")
    )
    return _model_number_tokens(evidence_text)


def _model_chart_structured_evidence_text(chart_context: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "chart_summary",
        "table_preview",
        "model_metrics",
        "model_metric_evidence",
        "axis_labels",
        "axis_metadata",
        "label_metadata",
    ):
        value = chart_context.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            values.append(json.dumps(value, ensure_ascii=False, default=str))
        else:
            values.append(str(value))
    return " ".join(values)


def _has_model_metric_number(text: str, metric_terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for term in metric_terms:
        term_pattern = re.escape(term).replace(r"\ ", r"[\s_-]+")
        if re.search(rf"\b{term_pattern}\b[^0-9-]{{0,40}}[-+]?\d", lowered):
            return True
        if re.search(rf"[-+]?\d[\d,.%]*[^A-Za-z0-9]{{0,20}}\b{term_pattern}\b", lowered):
            return True
    return False


def _has_structured_model_chart_evidence(
    chart_context: dict[str, Any],
    diagnostic_type: str | None,
) -> bool:
    evidence_text = _model_chart_structured_evidence_text(chart_context)
    lowered = evidence_text.lower()
    numbers = _model_number_tokens(evidence_text)
    if diagnostic_type == "confusion_matrix":
        matrix_terms = ("tp", "fp", "fn", "tn")
        has_matrix_labels = all(re.search(rf"\b{term}\b", lowered) for term in matrix_terms)
        if has_matrix_labels and len(numbers) >= 4:
            return True
        metric_terms = (
            "false positive",
            "false-positive",
            "false positives",
            "false-positive",
            "false negative",
            "false-negative",
            "false negatives",
            "true positive",
            "true-positive",
            "true negative",
            "true-negative",
        )
        return _has_model_metric_number(evidence_text, metric_terms)
    if diagnostic_type == "threshold":
        if "threshold" not in lowered:
            return False
        return _has_model_metric_number(
            evidence_text,
            ("recall", "precision", "review load", "review-load", "review_load"),
        )
    if diagnostic_type == "feature_importance":
        if _has_model_metric_number(evidence_text, ("importance", "feature importance")):
            return True
        return bool(re.search(r"\bfeature\b.{0,80}\bimportance\b.{0,80}[-+]?\d", lowered))
    if diagnostic_type == "model_diagnostic":
        return _has_model_metric_number(
            evidence_text,
            ("recall", "precision", "roc auc", "auc", "f1", "accuracy", "review load"),
        )
    return True


def _split_english_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _model_chart_sentence_allowed(
    sentence: str,
    diagnostic_type: str | None,
    chart_context: dict[str, Any],
) -> bool:
    lowered = sentence.lower()
    if (
        re.search(r"\bhigh\b(?!-risk)", lowered)
        or re.search(r"\blow\b", lowered)
        or any(
            term in lowered
            for term in (
                "concerning",
                "notable",
                "notably",
                "overly aggressive",
                "underperforming",
                "significant review workload",
                "deployed",
                "deployment",
                "current operational state",
            )
        )
    ):
        return False
    sentence_numbers = _model_number_tokens(sentence)
    if sentence_numbers and not sentence_numbers.issubset(_visible_model_number_tokens(chart_context)):
        return False
    if diagnostic_type == "threshold":
        disallowed_terms = (
            "discount",
            "high-discount",
            "discount-profit",
            "profit",
            "negative profit",
            "margin",
            "sales",
            "revenue",
            "cost",
            "price",
            "shipping",
            "fulfillment",
            "product",
            "sku",
            "category",
            "sub-category",
            "segment",
            "region",
            "country",
            "customer",
            "campaign",
            "promotion",
            "inventory",
            "$",
            "what-if",
            "business goal",
            "directly contributes",
            "causes",
            "drives",
            "because",
            "conflicts with",
        )
        if any(term in lowered for term in disallowed_terms):
            return False
        allowed_terms = (
            "threshold",
            "recall",
            "precision",
            "review workload",
            "review load",
            "review capacity",
            "false positive",
            "false negative",
            "manual review",
        )
        return any(term in lowered for term in allowed_terms)
    if diagnostic_type == "confusion_matrix":
        disallowed_terms = (
            "discount",
            "high-discount",
            "discount-profit",
            "profit",
            "negative profit",
            "margin",
            "sales",
            "revenue",
            "cost",
            "price",
            "shipping",
            "fulfillment",
            "product",
            "sku",
            "category",
            "sub-category",
            "segment",
            "region",
            "country",
            "customer",
            "campaign",
            "promotion",
            "inventory",
            "$",
            "what-if",
            "business goal",
            "directly contributes",
            "causes",
            "drives",
            "because",
            "true scale",
            "adjust the threshold to reduce",
        )
        if any(term in lowered for term in disallowed_terms):
            return False
        allowed_terms = (
            "confusion matrix",
            "tp",
            "fp",
            "fn",
            "tn",
            "false positive",
            "false negative",
            "manual review",
            "review threshold",
            "loss-risk",
            "loss risk",
        )
        return any(term in lowered for term in allowed_terms)
    if diagnostic_type == "feature_importance":
        if any(term in lowered for term in ("causes", "proves", "directly determines", "directly contributes")):
            return False
        return any(term in lowered for term in ("feature", "signal", "model", "manual review", "order-level"))
    if any(
        term in lowered
        for term in (
            "discount",
            "profit",
            "margin",
            "product",
            "sku",
            "category",
            "segment",
            "region",
            "what-if",
            "$",
            "proves",
            "proof",
        )
    ):
        return False
    return any(term in lowered for term in ("model", "recall", "precision", "false positive", "false negative", "review"))


def _isolate_model_chart_reflection(
    reflection: str,
    chart_context: dict[str, Any],
) -> str:
    diagnostic_type = _model_diagnostic_chart_type(chart_context)
    if not diagnostic_type:
        return reflection
    if not _has_structured_model_chart_evidence(chart_context, diagnostic_type):
        return _model_chart_fallback(chart_context, diagnostic_type)
    safe_sentences = [
        sentence
        for sentence in _split_english_sentences(reflection)
        if _model_chart_sentence_allowed(sentence, diagnostic_type, chart_context)
    ]
    if len(safe_sentences) < 2:
        return _model_chart_fallback(chart_context, diagnostic_type)
    return " ".join(safe_sentences[:3])


def _chart_context_evidence_text(chart_context: dict[str, Any]) -> str:
    return " ".join(
        str(chart_context.get(key) or "")
        for key in ("chart_title", "chart_id", "business_question", "chart_summary", "table_preview", "section_id")
    )


def _chart_grain(chart_context: dict[str, Any]) -> str:
    text = _chart_context_evidence_text(chart_context).lower()
    if any(term in text for term in ("product name", "product-level", " product ", "products ", "sku", "item name")):
        return "product"
    if "sub-category" in text or "sub_category" in text:
        return "sub_category"
    if "category" in text or "categories" in text:
        return "category"
    if "segment" in text:
        return "segment"
    if "region" in text:
        return "region"
    if "country" in text or "market" in text:
        return "country"
    if any(term in text for term in ("month", "date", "period", "trend", "time")):
        return "time"
    if "discount" in text and any(term in text for term in ("tier", "bucket", "band")):
        return "discount_tier"
    return "aggregate"


def _walk_report_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _walk_report_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_report_values(item)


def _known_product_entities(report: AnalysisReport | None) -> set[str]:
    entities: set[str] = set()
    if report is None:
        return entities
    product_key_pattern = re.compile(r"\b(product|sku|item)\b", flags=re.IGNORECASE)
    for module in report.modules:
        for key, value in _walk_report_values(module.tables or {}):
            if product_key_pattern.search(str(key)) and isinstance(value, str):
                cleaned = re.sub(r"\s+", " ", value).strip()
                if len(cleaned) >= 4:
                    entities.add(cleaned)
    return entities


def _aggregate_chart_fallback(chart_context: dict[str, Any], grain: str) -> str:
    title = _english_context_text(chart_context.get("chart_title"), "This aggregate chart")
    grain_label = grain.replace("_", " ")
    metric_text = str(chart_context.get("chart_summary") or chart_context.get("business_question") or "").lower()
    if "margin" in metric_text or "profit" in metric_text:
        evidence = f"{title} compares profit-margin differences across visible {grain_label} slices."
    else:
        evidence = f"{title} compares the visible {grain_label} slices in the current chart."
    return " ".join(
        [
            evidence,
            f"Review weaker {grain_label} slices with the underlying records before assigning cost, discount, or operating causes.",
        ]
    )


def _aggregate_peer_cross_grain_takeover(sentence: str, chart_context: dict[str, Any]) -> bool:
    lowered = sentence.lower()
    peer_terms = (
        "peer chart",
        "peer charts",
        "concurrent chart",
        "concurrent top product",
        "adjacent chart",
        "adjacent charts",
        "other chart",
        "other charts",
        "top product",
        "top sku",
        "product gap",
    )
    if not any(term in lowered for term in peer_terms):
        return False
    cross_grain_terms = (
        "sku",
        "individual product",
        "specific item",
        "specific furniture item",
        "product-level",
        "product name",
        "sub-category",
        "compounds",
        "explains",
        "pinpointing",
        "drags down",
        "drag down",
        "drives",
        "causes",
        "likely worsens",
        "likely compounds",
    )
    if any(term in lowered for term in cross_grain_terms):
        return True
    visible_text = _chart_context_evidence_text(chart_context).lower()
    for granular_term in ("storage", "supplies", "furniture items"):
        if granular_term in lowered and granular_term not in visible_text:
            return True
    return False


def _scatter_sentence_has_ungrounded_strength(sentence: str, chart_context: dict[str, Any]) -> bool:
    if str(chart_context.get("chart_kind") or "").lower() != "scatter":
        return False
    lowered = sentence.lower()
    if not ("correlation" in lowered or "association" in lowered):
        return False
    if not any(
        term in lowered
        for term in (
            "clear",
            "strong",
            "strongly",
            "highly",
            "almost perfect",
            "near-perfect",
            "near perfect",
        )
    ):
        return False
    visible_text = _chart_context_evidence_text(chart_context).lower()
    has_visible_strength = bool(
        re.search(r"\br\s*=\s*[-+]?\d", visible_text)
        or re.search(r"\bcorrelation coefficient\b", visible_text)
        or re.search(r"\b(?:clear|strong|strongly|highly|near-perfect|near perfect)\s+(?:negative\s+|positive\s+)?(?:correlation|association)\b", visible_text)
    )
    return not has_visible_strength


def _aggregate_sentence_allowed(
    sentence: str,
    *,
    chart_context: dict[str, Any],
    report: AnalysisReport | None,
    grain: str,
) -> bool:
    if grain == "product":
        return True
    lowered = sentence.lower()
    if _scatter_sentence_has_ungrounded_strength(sentence, chart_context):
        return False
    if _aggregate_peer_cross_grain_takeover(sentence, chart_context):
        return False
    known_products = _known_product_entities(report)
    visible_text = _chart_context_evidence_text(chart_context).lower()
    for entity in known_products:
        entity_lower = entity.lower()
        if entity_lower in lowered and entity_lower not in visible_text:
            return False
    unsafe_patterns = (
        r"\bcluster(?:s|ed)?\s+within\b",
        r"\bbelongs?\s+to\b",
        r"\binflates?\s+its\s+(?:cost|discount)\b",
        r"\bcost\s+or\s+discount\s+burden\b",
        r"\b(?:cost|discount|shipping|fulfillment)\s+burden\b",
        r"\broot\s+causes?\b",
        r"\bcauses?\b",
        r"\bdrives?\b",
        r"\bdirectly\s+contributes\b",
    )
    if any(re.search(pattern, lowered) for pattern in unsafe_patterns):
        return False
    return True


def _isolate_chart_grain_reflection(
    reflection: str,
    *,
    chart_context: dict[str, Any],
    report: AnalysisReport | None,
) -> str:
    if _model_diagnostic_chart_type(chart_context):
        return reflection
    grain = _chart_grain(chart_context)
    if grain not in {"category", "sub_category", "segment", "region", "country", "time", "discount_tier", "aggregate"}:
        return reflection
    safe_sentences = [
        sentence
        for sentence in _split_english_sentences(reflection)
        if _aggregate_sentence_allowed(sentence, chart_context=chart_context, report=report, grain=grain)
    ]
    if len(safe_sentences) < 2:
        return _aggregate_chart_fallback(chart_context, grain)
    return " ".join(safe_sentences[:3])


def _english_fallback_postrun_reflection(
    chart_context: dict[str, Any],
    report: AnalysisReport | None = None,
) -> str:
    chart_kind = str(chart_context.get("chart_kind") or "chart")
    chart_title = _english_context_text(chart_context.get("chart_title"), "This chart")
    context_text = " ".join(
        str(chart_context.get(key) or "")
        for key in ("chart_title", "chart_id", "business_question", "section_id", "chart_summary")
    ).lower()
    diagnostic_type = _model_diagnostic_chart_type(chart_context)
    if diagnostic_type:
        return _model_chart_fallback(chart_context, diagnostic_type)
    raw_summary = str(chart_context.get("chart_summary") or "")
    summary = _english_context_text(raw_summary, "") if _is_usable_english_chart_evidence(raw_summary) else ""
    section_id = str(chart_context.get("section_id") or "")
    module = _module_for_section(section_id, report)
    module_finding = ""
    if module is not None and module.findings:
        raw_finding = str(module.findings[0] or "")
        if _is_usable_english_chart_evidence(raw_finding):
            module_finding = _english_context_text(raw_finding, "")
    chart_kind_evidence = {
        "line": "the executed trend values available in the chart",
        "bar": "visible differences across category slices",
        "treemap": "relative contribution across visible slices",
        "pareto": "head contribution against the remaining tail",
        "heatmap": "stronger and weaker slice combinations",
        "boxplot": "median, spread, and outlier evidence",
        "scatter": "the plotted association between the measures",
        "histogram": "the center and tail of the distribution",
        "dual_axis_bar_line": "the bar and line signals in the executed chart",
    }.get(chart_kind, "the available chart evidence")
    evidence = summary or module_finding or chart_kind_evidence

    comparison = {
        "line": "Compare dated periods in the chart before using the pattern for operating follow-up.",
        "bar": "Compare the head and tail slices to judge whether sales are concentrated in a small set of objects.",
        "treemap": "Compare the largest share with the long tail to test whether the structure is overly concentrated.",
        "pareto": "Compare cumulative head contribution with the remaining tail before setting review priorities.",
        "heatmap": "Compare stronger and weaker cells to locate the slice combinations that need follow-up.",
        "boxplot": "Compare medians, spread, and outliers before drawing conclusions from the average.",
        "scatter": "Compare higher and lower value points before interpreting the pattern as an operating signal.",
        "histogram": "Compare the center and tail of the distribution before relying on the mean.",
        "dual_axis_bar_line": "Compare the bar and line signals together before treating the relationship as stable.",
    }.get(chart_kind, "Compare the strongest and weakest visible slices before turning the chart into action.")
    action = "Review the underlying records and related peer charts before making operating changes."
    module_sentence = ""
    if module_finding and module_finding != evidence:
        module_sentence = module_finding if module_finding.endswith((".", "!", "?")) else f"{module_finding}."
    parts = [
        f"{chart_title} shows {evidence}.",
        module_sentence or comparison,
        comparison,
        action,
    ]
    cleaned: list[str] = []
    for part in parts:
        text = re.sub(r"\s+", " ", part).strip()
        if text and not contains_cjk(text) and text not in cleaned:
            cleaned.append(text)
    return " ".join(cleaned[:4])


def _fallback_reflection_for_context(
    chart_context: dict[str, Any],
    report: AnalysisReport | None,
    output_language: str | None = None,
) -> str:
    if _explicit_english_output(output_language):
        return _english_fallback_postrun_reflection(chart_context, report)
    return _fallback_postrun_reflection(chart_context, report)


_UNSUPPORTED_DISCOUNT_REFLECTION_TERMS_ZH = (
    "折扣策略",
    "折扣政策",
    "高折扣订单",
    "折扣审批",
    "审批阈值",
    "折扣阈值",
    "按折扣区间",
    "折扣力度过大",
    "折扣收紧",
    "折扣回收",
    "折扣是否真正带来销售增长",
)

_UNSUPPORTED_DISCOUNT_REFLECTION_TERMS_EN = (
    "discount strategy",
    "discount policy",
    "high-discount",
    "discount approval",
    "discount threshold",
    "discount tier",
    "scoped discount tier",
    "discount-profit relationship",
    "what-if profit lift",
    "review discount tier",
)

_ALLOWED_MISSING_DISCOUNT_REFLECTION_TERMS_ZH = (
    "缺少 discount 字段",
    "未映射折扣字段",
    "当前无法进行折扣分析",
    "补充折扣字段后再验证",
)

_ALLOWED_MISSING_DISCOUNT_REFLECTION_TERMS_EN = (
    "discount field is not mapped",
    "discount is missing",
    "discount conclusions are out of scope",
    "add discount fields before",
)


def _has_discount_mapping(schema_mapping: SchemaMapping | None) -> bool:
    field_mapping = getattr(schema_mapping, "field_mapping", None)
    if not isinstance(field_mapping, dict):
        return True
    return "discount" in set(field_mapping.values())


def _is_missing_discount_limitation(text: str) -> bool:
    lowered = text.lower()
    return any(term in text for term in _ALLOWED_MISSING_DISCOUNT_REFLECTION_TERMS_ZH) or any(
        term in lowered for term in _ALLOWED_MISSING_DISCOUNT_REFLECTION_TERMS_EN
    )


def _is_unsupported_discount_reflection_without_mapping(
    reflection: str,
    schema_mapping: SchemaMapping | None,
) -> bool:
    if _has_discount_mapping(schema_mapping):
        return False
    if _is_missing_discount_limitation(reflection):
        return False
    lowered = reflection.lower()
    return any(term in reflection for term in _UNSUPPORTED_DISCOUNT_REFLECTION_TERMS_ZH) or any(
        term in lowered for term in _UNSUPPORTED_DISCOUNT_REFLECTION_TERMS_EN
    )


def _fallback_reflections_for_contexts(
    chart_contexts: list[dict[str, Any]],
    report: AnalysisReport | None,
    output_language: str | None = None,
    schema_mapping: SchemaMapping | None = None,
) -> dict[int, str]:
    reflections: dict[int, str] = {}
    for context in chart_contexts:
        cell_index = int(context.get("cell_index", -1))
        if cell_index < 0:
            continue
        reflection = _normalize_llm_reflection_markdown(
            _fallback_reflection_for_context(context, report, output_language)
        )
        if _explicit_english_output(output_language):
            guarded_reflection = _guard_english_postrun_reflection(
                reflection,
                context=context,
                report=report,
                schema_mapping=schema_mapping,
            )
            if not _is_unsupported_discount_reflection_without_mapping(guarded_reflection, schema_mapping):
                reflections[cell_index] = guarded_reflection
            continue
        if "看到什么：" not in reflection and "下一步做什么：" not in reflection:
            section_id = str(context.get("section_id", ""))
            section_guidance = {
                "product_and_category": (
                    "商品或类目之间的销售贡献与利润质量需要一起看，避免只被销售额排名带偏。",
                    "把高销售低利润商品、类目利润率差异和促销成本合并成复盘清单。"
                ),
                "discount_and_profit": (
                    "折扣区间的平均利润和亏损率需要同步判断，重点看利润是否被深折扣系统性侵蚀。",
                    "按折扣区间筛出亏损订单，再回查商品、审批和客户组合。"
                ),
                "segment_and_region": (
                    "客群与区域组合切片的强弱差异比单一维度更能定位经营责任对象。",
                    "锁定持续偏弱的组合切片，复盘折扣、产品结构和履约成本。"
                ),
                "sales_trends": (
                    "峰值、低谷和月度波动需要结合订单结构解释，避免把一次性冲高当成稳定增长。",
                    "回到峰谷月份明细，拆解商品、区域和大单贡献。"
                ),
                "metric_distributions": (
                    "长尾、异常值和均值失真会影响后续所有销售与利润判断。",
                    "同时报告中位数、P90 和极端记录，把异常订单单独复盘。"
                ),
            }.get(
                section_id,
                (
                    "图中对象之间的差异需要和同章节证据一起验证，避免孤立读图。",
                    "把最突出的对象回到明细表，确认可执行的经营动作。"
                ),
            )
            reflection = (
                f"看到什么：{reflection} "
                f"说明什么：{section_guidance[0]}"
                f"下一步做什么：{section_guidance[1]}"
            )
        if _sentence_count(reflection) < 3:
            reflection = (
                f"{reflection.rstrip('。')}。"
                "本图至少应同时看头部与尾部、峰值与低谷或销售与利润是否同步。"
                "后续动作是把图中最突出的对象回到明细数据里复盘。"
            )
        reflection = _sanitize_reflection_chart_kind_wording(
            reflection,
            str(context.get("chart_kind", "")),
        )
        if _is_unsupported_discount_reflection_without_mapping(reflection, schema_mapping):
            continue
        reflections[cell_index] = reflection
    return reflections


def _guard_english_postrun_reflection(
    reflection: str,
    *,
    context: dict[str, Any],
    report: AnalysisReport | None,
    schema_mapping: SchemaMapping | None,
) -> str:
    fallback = _fallback_reflection_for_context(context, report, "en")
    reflection = _isolate_model_chart_reflection(reflection, context)
    reflection = _isolate_chart_grain_reflection(
        reflection,
        chart_context=context,
        report=report,
    )
    return guard_english_public_narrative(
        reflection,
        report=report,
        schema_mapping=schema_mapping,
        chart_context=context,
        role="chart_commentary",
        fallback=fallback,
    )


CORE_SECTION_REFLECTION_MINIMUMS = {
    "product_and_category": 2,
    "discount_and_profit": 2,
    "segment_and_region": 1,
}


def _core_reflection_backfill_contexts(
    chart_contexts: list[dict[str, Any]],
    reflections: dict[int, str],
) -> list[dict[str, Any]]:
    reflected_by_section: dict[str, int] = {}
    reflected_cells = set(reflections)
    for context in chart_contexts:
        cell_index = int(context.get("cell_index", -1))
        if cell_index in reflected_cells:
            section_id = str(context.get("section_id", ""))
            reflected_by_section[section_id] = reflected_by_section.get(section_id, 0) + 1

    missing_contexts: list[dict[str, Any]] = []
    for section_id, minimum in CORE_SECTION_REFLECTION_MINIMUMS.items():
        current = reflected_by_section.get(section_id, 0)
        if current >= minimum:
            continue
        section_contexts = [
            context
            for context in chart_contexts
            if str(context.get("section_id", "")) == section_id
            and int(context.get("cell_index", -1)) not in reflected_cells
        ]
        needed = max(0, minimum - current)
        for context in section_contexts[:needed]:
            missing_contexts.append(context)
            reflected_cells.add(int(context.get("cell_index", -1)))
    return missing_contexts


def build_postrun_chart_reflections(
    chart_contexts: list[dict[str, Any]],
    report: AnalysisReport | None = None,
    schema_mapping: SchemaMapping | None = None,
    llm_client=None,
    max_llm_reflections: int = MAX_LLM_POSTRUN_REFLECTIONS,
    allow_fallback_reflections: bool = True,
    llm_profile: str | None = None,
    output_language: str | None = None,
) -> tuple[dict[int, str], LLMStageTrace]:
    stage = "postrun_chart_reflection"
    if not chart_contexts:
        status = "disabled" if not getattr(llm_client, "enabled", False) else "skipped"
        reason = "No executed chart outputs were collected."
        return {}, build_llm_stage_trace(
            stage=stage,
            llm_client=llm_client,
            status=status,
            reason=reason,
            attempted=False,
            applied=False,
        )

    if llm_client is None or not getattr(llm_client, "enabled", False):
        reflections = (
            _fallback_reflections_for_contexts(chart_contexts, report, output_language, schema_mapping)
            if allow_fallback_reflections
            else {}
        )
        return reflections, build_llm_stage_trace(
            stage=stage,
            llm_client=llm_client,
            status="fallback" if allow_fallback_reflections else "disabled",
            reason=(
                "LLM disabled; deterministic chart reflections were generated from "
                "chart_context, table_preview, module_summary_metrics, and module_findings."
                if allow_fallback_reflections
                else "LLM disabled and profile settings disabled deterministic fallback chart reflections."
            ),
            attempted=False,
            applied=bool(reflections),
        )

    if not hasattr(llm_client, "suggest_postrun_chart_reflection"):
        reflections = (
            _fallback_reflections_for_contexts(chart_contexts, report, output_language, schema_mapping)
            if allow_fallback_reflections
            else {}
        )
        return reflections, build_llm_stage_trace(
            stage=stage,
            llm_client=llm_client,
            status="fallback" if allow_fallback_reflections else "skipped",
            reason=(
                "LLM client does not support post-run chart reflections; deterministic "
                "chart reflections were generated from executed evidence."
                if allow_fallback_reflections
                else "LLM client does not support post-run chart reflections and profile settings disabled deterministic fallback chart reflections."
            ),
            attempted=False,
            applied=bool(reflections),
        )

    reflections: dict[int, str] = {}
    successes = 0
    failures = 0
    llm_attempts = 0
    skipped_due_to_limit = 0
    failure_reasons: list[str] = []
    section_contexts = _section_contexts_by_section(chart_contexts)
    prioritized_contexts = _prioritized_chart_contexts(chart_contexts)
    metrics_snapshot = (
        llm_client.snapshot_completion_metrics()
        if hasattr(llm_client, "snapshot_completion_metrics")
        else None
    )

    def trace_with(status: str, reason: str, *, attempted: bool, applied: bool) -> LLMStageTrace:
        subcalls = (
            llm_client.collect_completion_metrics_since(metrics_snapshot)
            if metrics_snapshot is not None and hasattr(llm_client, "collect_completion_metrics_since")
            else []
        )
        return build_aggregated_llm_stage_trace(
            stage=stage,
            llm_client=llm_client,
            status=status,
            reason=reason,
            subcalls=subcalls,
            attempted=attempted,
            applied=applied,
        )

    for position, context in enumerate(prioritized_contexts):
        cell_index = int(context["cell_index"])
        if llm_attempts >= max_llm_reflections:
            skipped_due_to_limit += 1
            continue
        try:
            llm_context = _context_for_llm(
                context,
                report,
                section_chart_contexts=section_contexts.get(str(context.get("section_id", ""))),
                output_language=output_language,
            )
            llm_attempts += 1
            payload = _call_postrun_reflection_llm(
                llm_client,
                llm_context,
                "",
                language_instruction=user_facing_language_instruction(
                    output_language if output_language is not None else "zh-CN"
                ),
            )
            reflection_markdown = _normalize_llm_reflection_markdown(
                str(payload.get("reflection_markdown", ""))
            )
            if _explicit_english_output(output_language) and reflection_markdown and contains_cjk(reflection_markdown):
                fallback_reflection = _normalize_llm_reflection_markdown(
                    _fallback_reflection_for_context(context, report, output_language)
                )
                guarded_reflection = _guard_english_postrun_reflection(
                    fallback_reflection,
                    context=context,
                    report=report,
                    schema_mapping=schema_mapping,
                )
                if not _is_unsupported_discount_reflection_without_mapping(guarded_reflection, schema_mapping):
                    reflections[cell_index] = guarded_reflection
                failures += 1
                failure_reasons.append(
                    f"language_mismatch_fallback for {context.get('chart_title', cell_index)}: English reflection contained CJK"
                )
            elif reflection_markdown and not _is_low_quality_llm_reflection(
                reflection_markdown,
                output_language=output_language,
            ):
                sanitized_reflection = _sanitize_reflection_chart_kind_wording(
                    downgrade_strong_inference_claims(_sanitize_unqualified_reflection_speculation(reflection_markdown)),
                    str(context.get("chart_kind", "")),
                )
                if _explicit_english_output(output_language):
                    sanitized_reflection = _guard_english_postrun_reflection(
                        sanitized_reflection,
                        context=context,
                        report=report,
                        schema_mapping=schema_mapping,
                    )
                if _is_unsupported_discount_reflection_without_mapping(sanitized_reflection, schema_mapping):
                    failures += 1
                    failure_reasons.append(
                        f"unsupported discount reflection for {context.get('chart_title', cell_index)}; reflection dropped"
                    )
                    continue
                reflections[cell_index] = sanitized_reflection
                successes += 1
            else:
                failures += 1
                if reflection_markdown:
                    failure_reasons.append(
                        f"low-quality reflection for {context.get('chart_title', cell_index)}; fallback hidden"
                    )
                else:
                    failure_reasons.append(
                        f"empty reflection for {context.get('chart_title', cell_index)}; fallback hidden"
                    )
        except Exception as exc:  # pragma: no cover - protected by trace + fallback behavior
            failures += 1
            failure_reasons.append(describe_llm_error(exc) + "; fallback hidden")
            unavailable_reason = getattr(llm_client, "completion_unavailable_reason", None)
            if unavailable_reason:
                remaining_count = len(prioritized_contexts[position + 1 :])
                skipped_due_to_limit += remaining_count
                failure_reasons.append(
                    "Skipped remaining chart reflections because the LLM backend is unavailable: "
                    + str(unavailable_reason)
                )
                break

    backfill_contexts = (
        _core_reflection_backfill_contexts(chart_contexts, reflections)
        if allow_fallback_reflections
        else []
    )
    if backfill_contexts:
        fallback_reflections = _fallback_reflections_for_contexts(
            backfill_contexts,
            report,
            output_language,
            schema_mapping,
        )
        reflections.update(
            {
                cell_index: _sanitize_reflection_chart_kind_wording(
                    downgrade_strong_inference_claims(
                        _sanitize_unqualified_reflection_speculation(reflection)
                    ),
                    str(
                        next(
                            (
                                context.get("chart_kind", "")
                                for context in backfill_contexts
                                if int(context.get("cell_index", -1)) == cell_index
                            ),
                            "",
                        )
                    ),
                )
                for cell_index, reflection in fallback_reflections.items()
            }
        )

    if failures == 0 and skipped_due_to_limit == 0:
        return reflections, trace_with(
            "llm_applied",
            "LLM generated chart-specific post-run reflections.",
            attempted=True,
            applied=True,
        )

    if successes > 0:
        limit_reason = (
            f"; skipped {skipped_due_to_limit} chart reflections after live LLM limit {max_llm_reflections}"
            if skipped_due_to_limit
            else ""
        )
        if skipped_due_to_limit and llm_profile:
            limit_reason += f" from llm_profile={llm_profile} max_llm_reflections={max_llm_reflections}"
        if not allow_fallback_reflections:
            profile_reason = (
                "quick profile limits LLM reflections dynamically"
                if llm_profile == "quick"
                else "full profile uses broader LLM reflection coverage"
            )
            fallback_reason = (
                f"; {llm_profile or 'selected'} profile disables deterministic fallback chart reflections; "
                "unreflected charts were left blank"
            )
        else:
            profile_reason = (
                "Some chart reflections used LLM output while missing core reflections were handled by deterministic fallback when needed"
            )
            fallback_reason = ""
        return reflections, trace_with(
            "llm_partial",
            (
                (
                    f"Some chart reflections used LLM output; {profile_reason}, disables deterministic fallback chart reflections, and leaves unreflected charts blank. "
                    if not allow_fallback_reflections
                    else f"{profile_reason}: "
                )
                + "; ".join(failure_reasons[:3])
                + limit_reason
                + fallback_reason
                + (
                    f"; deterministic fallback filled {len(backfill_contexts)} core chart reflection(s)"
                    if backfill_contexts
                    else ""
                )
            ),
            attempted=True,
            applied=True,
        )

    if backfill_contexts:
        return reflections, trace_with(
            "fallback_on_error",
            (
                "LLM could not provide enough chart reflections; deterministic fallback filled "
                f"{len(backfill_contexts)} core chart reflection(s). "
                + "; ".join(failure_reasons[:3])
            ),
            attempted=True,
            applied=True,
        )

    if not allow_fallback_reflections:
        if llm_profile == "quick":
            profile_reason = (
                "quick profile limits LLM reflections dynamically; deterministic fallback chart reflections are disabled; "
                "failed or unreflected charts are left blank. "
            )
        elif llm_profile == "full":
            profile_reason = (
                "full profile uses broader LLM reflection coverage; deterministic fallback chart reflections are disabled; "
                "failed or unreflected charts are left blank. "
            )
        else:
            profile_reason = (
                "selected profile disables deterministic fallback chart reflections; "
                "failed or unreflected charts are left blank. "
            )
    else:
        profile_reason = ""

    return reflections, trace_with(
        "fallback_on_error",
        (
            profile_reason
            + "LLM failed for all chart reflections; deterministic fallback chart reflections were not inserted: "
            + "; ".join(failure_reasons[:3])
        ),
        attempted=True,
        applied=False,
    )
