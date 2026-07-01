from __future__ import annotations

import inspect
import re
from collections.abc import Iterable

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.llm_trace import LLMStageTrace
from app.schemas.notebook_content import NotebookContentPlan, NotebookSectionContent
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.chart_selection_planner import selected_chart_metadata_by_title
from app.services.discount_profit_strategy import build_discount_profit_strategy
from app.services.llm_trace_utils import (
    build_aggregated_llm_stage_trace,
    build_llm_stage_trace,
    describe_llm_error,
)
from app.services.output_language import (
    contains_cjk,
    englishize_common_artifact_text,
    englishize_notebook_text,
    is_english_output,
    safe_english_sentence,
    user_facing_language_instruction,
)
from app.services.llm_evidence_pack import build_section_evidence_slice
from app.services.metric_distribution_strategy import build_metric_distribution_strategy
from app.services.notebook_narrative_guard import remove_unsupported_narrative
from app.services.notebook_content_fallback import (
    FallbackSectionHelpers,
    build_fallback_section_content,
)
from app.services.product_category_strategy import build_product_category_strategy
from app.services.sales_trend_strategy import build_sales_trend_strategy
from app.services.segment_region_strategy import build_segment_region_strategy


def _accepts_keyword(method, keyword: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _nb_english(output_language: str | None) -> bool:
    return output_language is not None and is_english_output(output_language)


def _nb_text(output_language: str | None, en: str, zh: str) -> str:
    return en if _nb_english(output_language) else zh


def _nb_quote(output_language: str | None, en: str, zh: str) -> str:
    return repr(_nb_text(output_language, en, zh))


def _short_english_rationale(text: str) -> str:
    fallback = "Validate the key business question for this section."
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return fallback
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    first = next((item.strip() for item in sentences if item.strip()), "")
    if not first:
        return fallback
    if len(first.split()) > 24:
        return fallback
    if not first.endswith((".", "!", "?")):
        first += "."
    return first


def _englishize_content_plan(plan: NotebookContentPlan) -> NotebookContentPlan:
    def _markdown(block: object) -> str:
        text = englishize_notebook_text(block)
        if "Section Takeaway" in text and contains_cjk(text):
            return (
                "### Section Takeaway\n\n"
                "This section compares mapped business slices and flags concrete records for follow-up review."
            )
        return text

    return NotebookContentPlan(
        sections=[
            section.model_copy(
                update={
                    "markdown_blocks": [
                        _markdown(block)
                        for block in section.markdown_blocks
                    ],
                    "code_cells": [
                        englishize_notebook_text(code)
                        for code in section.code_cells
                    ],
                }
            )
            for section in plan.sections
        ]
    )


def _original_column(schema_mapping: SchemaMapping, canonical_name: str, fallback: str) -> str:
    for original, canonical in schema_mapping.field_mapping.items():
        if canonical == canonical_name:
            return original
    return fallback


def _optional_original_column(schema_mapping: SchemaMapping, canonical_name: str) -> str | None:
    for original, canonical in schema_mapping.field_mapping.items():
        if canonical == canonical_name:
            return original
    return None


def _has_canonical_field(schema_mapping: SchemaMapping, canonical_name: str) -> bool:
    return canonical_name in schema_mapping.field_mapping.values()


def _apply_unsupported_narrative_guard(
    content: NotebookContentPlan,
    schema_mapping: SchemaMapping,
) -> tuple[NotebookContentPlan, dict[str, object]]:
    mapped_fields = set(schema_mapping.field_mapping.values())
    sections: list[NotebookSectionContent] = []
    removed_blocks: list[str] = []
    removed_terms: set[str] = set()
    for section in content.sections:
        markdown_blocks: list[str] = []
        for block in section.markdown_blocks:
            guarded, trace = remove_unsupported_narrative(block, mapped_fields=mapped_fields)
            if guarded.strip():
                markdown_blocks.append(guarded)
            if trace["unsupported_narrative_removed"]:
                removed_blocks.extend(trace["removed_blocks"])
                removed_terms.update(trace["removed_terms"])
        sections.append(
            NotebookSectionContent(
                section_id=section.section_id,
                markdown_blocks=markdown_blocks,
                code_cells=list(section.code_cells),
            )
        )
    return NotebookContentPlan(sections=sections), {
        "unsupported_narrative_removed": bool(removed_blocks),
        "removed_terms": sorted(removed_terms),
        "removed_blocks": removed_blocks,
        "reason": "removed narrative blocks referencing unavailable canonical fields" if removed_blocks else "",
    }


SECTION_TO_MODULE = {
    "metric_distributions": "metric_distribution_analysis",
    "sales_trends": "sales_trend_analysis",
    "product_and_category": "product_contribution_analysis",
    "segment_and_region": "dimension_breakdown_analysis",
    "country_market": "country_market_analysis",
    "order_structure": "order_structure_analysis",
    "discount_and_profit": "discount_profit_analysis",
    "modeling": "loss_risk_modeling",
    "forecast": "forecast_analysis",
}


_MARKDOWN_CODE_FENCE_RE = re.compile(
    r"```(?:[A-Za-z0-9_+\-.]+)?\s*\n.*?```",
    flags=re.DOTALL,
)
_UNCLOSED_MARKDOWN_CODE_FENCE_RE = re.compile(
    r"```(?:[A-Za-z0-9_+\-.]+)?\s*\n.*",
    flags=re.DOTALL,
)

_GENERATED_MOJIBAKE_REPLACEMENTS = {
    "\u935a\u52ec\u59cc\u93b5\uff45\u5c2f\u95c2\u6751\u57c4\u5a11\ufe41\u5ddd\u95b2\u5fe5\u7d30\u9a9e\u51b2\u6f4e\u9352\u2542\u9f0e\u6d93\u5e9d\u7c2d\u93b9\u71ba\u5dfc": "各折扣区间利润质量：平均利润与亏损率",
    "\u93b6\u6a3b\u58b8\u9356\u6d2a\u68ff": "折扣区间",
    "\u7edb\u682b\u6690\u705e\u509d\u70e6\u6769\u56e9\u88ab\u9429\ue1bd\u59cc\u93b5\uff49\ue5d3\u95c4\u2543\u5139\u9354\u6d98\u6d58": "策略层跳过类目折扣风险热力图",
    "\u7edb\u682b\u6690\u705e\u509d\u70e6\u6769\u56e8\u59cc\u93b5\uff45\u5c2f\u95c2\u5bf8\ue188\u7efe\u57ae\u6d58": "策略层跳过折扣区间箱线图",
    "\u9352\u2542\u9f0e\u9352\u55d7\u7af7\u59d2\u509d\ue74d": "利润分布概览",
    "\u935a\u52ed\u88ab\u9429\ue1bc\u57c4\u5a11\ufe3e\u5dfc\u7490\u3129\u567a\u7035\u89c4\u762e": "各类目利润率质量对比",
    "\u6fb6\u64ae\u5134\u935f\u55d7\u6427\u7ef1\ue21d\ue178\u7490\uff04\u5c1e Pareto \u9352\u55d7\u7af7": "头部商品累计贡献 Pareto 分布",
    "\u6942\u6a40\u6522\u935e\ue1bb\u7d86\u9352\u2542\u9f0e\u935f\u55d7\u6427\u5a13\u546d\u5d1f": "高销售低利润商品清单",
    "\u7039\u3222\u5162/\u9356\u54c4\u7159\u95bf\u20ac\u935e\ue1c0\ue582\u7035\u89c4\u762e": "客群/区域销售额对比",
    "\u6942\u6a40\u6522\u935e\ue1bb\u7d86\u9352\u2542\u9f0e\u7039\u3222\u5162\u9356\u54c4\u7159\u9352\u56e9\u5896": "高销售低利润客群区域切片",
    "\u6fb6\u64ae\u5134\u935f\u55d7\u6427\u95bf\u20ac\u935e\ue1c0\ue582\u7035\u89c4\u762e": "头部商品销售额对比",
    "\u6d93\u5d85\u6093\u9352\u56e9\u5896\u95bf\u20ac\u935e\ue1c0\ue582\u7035\u89c4\u762e": "不同切片销售额对比",
    "\u9357\u66e0\u6dee\u9352\u56e9\u5896\u95bf\u20ac\u935e\ue1c0\ue582\u7035\u89c4\u762e": "单维切片销售额对比",
    "\u7039\u70ba\u6aaf\u6d93\u5ea8\ue569\u5a34\u5b2e\u6522\u935e\ue1c0\ue582\u7035\u89c4\u762e": "实际与预测销售额对比",
    "\u95bf\u20ac\u935e\ue1c0\ue582": "销售额",
    "\u9352\u2542\u9f0e": "利润",
    "\u7edb\u682b\u6690": "策略",
}


def _module_for_section(section_id: str, report: AnalysisReport):
    module_id = SECTION_TO_MODULE.get(section_id)
    if module_id is None:
        return None
    return next((module for module in report.modules if module.module_id == module_id), None)


def _strategy_chart_set(strategy: dict[str, object] | None) -> set[str]:
    if not isinstance(strategy, dict):
        return set()
    views = strategy.get("views")
    if not isinstance(views, list):
        return set()
    return {
        str(view.get("chart"))
        for view in views
        if isinstance(view, dict) and view.get("chart")
    }


def _strategy_includes(strategy: dict[str, object] | None, chart: str) -> bool:
    selected_charts = _strategy_chart_set(strategy)
    # Only display optional strategy charts when the LLM selected them.
    return chart in selected_charts


def _selection_includes(
    chart_selection_plan: dict[str, object] | None,
    chart_id: str,
) -> bool:
    if not isinstance(chart_selection_plan, dict):
        return False
    return any(
        isinstance(item, dict) and item.get("chart_id") == chart_id
        for item in chart_selection_plan.get("selected_charts", [])
    )


def _selected_chart_ids_literal(chart_selection_plan: dict[str, object] | None) -> str:
    if not isinstance(chart_selection_plan, dict):
        return "[]"
    return repr(
        [
            str(item.get("chart_id"))
            for item in chart_selection_plan.get("selected_charts", [])
            if isinstance(item, dict) and item.get("chart_id")
        ]
    )


def _selected_chart_ids_cell(chart_selection_plan: dict[str, object] | None) -> str | None:
    selected_chart_ids_literal = _selected_chart_ids_literal(chart_selection_plan)
    if selected_chart_ids_literal == "[]":
        return None
    return "\n".join(
        [
            "# Final chart selection gate",
            f"selected_chart_ids = {selected_chart_ids_literal}",
            "selected_chart_ids",
        ]
    )


def _ensure_selected_chart_ids_cell(
    content: NotebookContentPlan,
    chart_selection_plan: dict[str, object] | None,
) -> NotebookContentPlan:
    selected_cell = _selected_chart_ids_cell(chart_selection_plan)
    if selected_cell is None or not content.sections:
        return content
    if any("selected_chart_ids =" in cell for section in content.sections for cell in section.code_cells):
        return content
    sections = list(content.sections)
    first_section = sections[0]
    sections[0] = NotebookSectionContent(
        section_id=first_section.section_id,
        markdown_blocks=list(first_section.markdown_blocks),
        code_cells=[selected_cell, *list(first_section.code_cells)],
    )
    return NotebookContentPlan(sections=sections)


def _removed_chart_ids(chart_selection_plan: dict[str, object] | None) -> set[str]:
    if not isinstance(chart_selection_plan, dict):
        return set()
    trace = chart_selection_plan.get("intent_guided_selection_trace")
    if not isinstance(trace, dict):
        trace = chart_selection_plan.get("llm_selection_trace")
    if not isinstance(trace, dict):
        return set()
    removed: set[str] = set()
    for key in ("remove_only_chart_ids", "base_removed_chart_ids"):
        values = trace.get(key, [])
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            removed.update(str(item) for item in values if str(item).strip())
    return removed


def _repair_generated_mojibake(text: str) -> str:
    repaired = text
    for bad, good in _GENERATED_MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    return repaired


def _repair_section_content_mojibake(section: NotebookSectionContent) -> NotebookSectionContent:
    return NotebookSectionContent(
        section_id=section.section_id,
        markdown_blocks=[
            _repair_generated_mojibake(block) for block in section.markdown_blocks
        ],
        code_cells=[
            _repair_generated_mojibake(cell) for cell in section.code_cells
        ],
    )


def _repair_content_plan_mojibake(content: NotebookContentPlan) -> NotebookContentPlan:
    return NotebookContentPlan(
        sections=[_repair_section_content_mojibake(section) for section in content.sections]
    )


def _hide_non_llm_chart_choices(strategy: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(strategy, dict) or strategy.get("source") == "llm_sanitized":
        return strategy
    hidden = dict(strategy)
    # Keep non-LLM fallback chart choices out of the visible notebook.
    if "views" in hidden:
        hidden["views"] = []
    if "decisions" in hidden:
        hidden["decisions"] = []
    if "relationship_views" in hidden:
        hidden["relationship_views"] = []
    hidden["hidden_reason"] = "default chart choices hidden because LLM did not decide them"
    return hidden

def _without_non_llm_chart_display_cells(code_cells: list[str]) -> list[str]:
    visible_cells: list[str] = []
    has_llm_sanitized_decision = False
    for cell in code_cells:
        if "llm_sanitized" in cell:
            has_llm_sanitized_decision = True
            visible_cells.append(cell)
            continue
        if ("fig.show()" in cell or "plt.show()" in cell) and not has_llm_sanitized_decision:
            continue
        visible_cells.append(cell)
    return visible_cells


_TITLE_RE = re.compile(r"title\s*=\s*['\"]([^'\"]+)['\"]|title\(['\"]([^'\"]+)['\"]|set_title\(f?['\"]([^'\"]+)['\"]")
_CHART_ID_RE = re.compile(r"chart_id\s*:\s*([A-Za-z0-9_\-]+)")


def _chart_titles_in_cell(cell: str) -> set[str]:
    titles: set[str] = set()
    for match in _TITLE_RE.finditer(cell):
        for group in match.groups():
            if group:
                titles.add(group.strip())
    return titles


def _selected_titles_for_section(
    chart_selection_plan: dict[str, object] | None,
    section_id: str,
) -> set[str]:
    if not isinstance(chart_selection_plan, dict):
        return set()
    return {
        str(item.get("title")).strip()
        for item in chart_selection_plan.get("selected_charts", [])
        if isinstance(item, dict)
        and item.get("section_id") == section_id
        and str(item.get("title", "")).strip()
    }


def _selected_chart_ids_for_section(
    chart_selection_plan: dict[str, object] | None,
    section_id: str,
) -> set[str]:
    if not isinstance(chart_selection_plan, dict):
        return set()
    return {
        str(item.get("chart_id")).strip()
        for item in chart_selection_plan.get("selected_charts", [])
        if isinstance(item, dict)
        and item.get("section_id") == section_id
        and str(item.get("chart_id", "")).strip()
    }


def _chart_ids_in_cell(cell: str) -> set[str]:
    return {match.group(1).strip() for match in _CHART_ID_RE.finditer(cell)}


def _split_chart_cell(cell: str) -> list[str]:
    lines = cell.splitlines()
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if "fig.show()" in line or "plt.show()" in line:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return ["\n".join(chunk).strip() for chunk in chunks if "\n".join(chunk).strip()]


def _filter_chart_display_cell(cell: str, selected_titles: set[str], selected_chart_ids: set[str]) -> str | None:
    if "fig.show()" not in cell and "plt.show()" not in cell:
        return cell
    kept_chunks: list[str] = []
    for chunk in _split_chart_cell(cell):
        if "fig.show()" not in chunk and "plt.show()" not in chunk:
            continue
        chart_ids = _chart_ids_in_cell(chunk)
        if chart_ids and chart_ids <= selected_chart_ids:
            kept_chunks.append(chunk)
            continue
    return "\n".join(kept_chunks).strip() or None


def _missing_selected_chart_render_reasons(
    *,
    selected_chart_ids: set[str],
    rendered_chart_ids: set[str],
) -> dict[str, str]:
    return {
        chart_id: "selected chart has no rendered code block with matching # chart_id marker"
        for chart_id in sorted(selected_chart_ids - rendered_chart_ids)
    }


def _apply_chart_selection_to_section(
    section: NotebookSectionContent,
    chart_selection_plan: dict[str, object] | None,
) -> NotebookSectionContent:
    selected_titles = _selected_titles_for_section(chart_selection_plan, section.section_id)
    selected_chart_ids = _selected_chart_ids_for_section(chart_selection_plan, section.section_id)
    removed_chart_ids = _removed_chart_ids(chart_selection_plan)
    if not isinstance(chart_selection_plan, dict):
        return section

    filtered_cells: list[str] = []
    rendered_chart_ids: set[str] = set()
    for cell in section.code_cells:
        filtered = _filter_chart_display_cell(cell, selected_titles, selected_chart_ids)
        if filtered:
            rendered_chart_ids.update(_chart_ids_in_cell(filtered))
            filtered_cells.append(filtered)
    rendered_chart_ids -= removed_chart_ids

    missing_reasons = _missing_selected_chart_render_reasons(
        selected_chart_ids=selected_chart_ids,
        rendered_chart_ids=rendered_chart_ids,
    )
    if missing_reasons:
        filtered_cells.append(
            "missing_selected_chart_render_reason = "
            + repr(missing_reasons)
            + "\nmissing_selected_chart_render_reason"
        )

    selected_section_ids = {
        str(item.get("section_id"))
        for item in chart_selection_plan.get("selected_charts", [])
        if isinstance(item, dict)
    }
    markdown_blocks = list(section.markdown_blocks)
    if section.section_id not in selected_section_ids:
        markdown_blocks = [
            block
            for block in markdown_blocks
            if "图表分析" not in block and "图表解读" not in block
        ]

    return NotebookSectionContent(
        section_id=section.section_id,
        markdown_blocks=markdown_blocks,
        code_cells=filtered_cells,
    )


def _chart_selection_markdown(
    chart_selection_plan: dict[str, object] | None,
    section_id: str,
    output_language: str | None = None,
) -> str | None:
    if not isinstance(chart_selection_plan, dict):
        return None
    rows = [
        item
        for item in chart_selection_plan.get("selected_charts", [])
        if isinstance(item, dict) and item.get("section_id") == section_id
    ]
    if not rows:
        return None
    english = _nb_english(output_language)
    lines = ["### Chart Selection Rationale" if english else "### 图表选择依据", ""]
    for item in rows:
        evidence_summary = str(
            item.get("business_question")
            or item.get("evidence_summary")
            or (
                "Validate the key business question for this section."
                if english
                else "验证当前 section 的关键业务判断"
            )
        ).strip()
        evidence_summary = re.sub(r"\s+", " ", evidence_summary)
        if english:
            evidence_summary = _short_english_rationale(evidence_summary)
        elif len(evidence_summary) > 72:
            evidence_summary = evidence_summary[:69].rstrip() + "..."
        title = str(item.get("title") or "").strip()
        suffix = "." if english else "。"
        lines.append(f"- {title}: {evidence_summary.rstrip('.。')}{suffix}")
    return "\n".join(lines)

def _sanitize_markdown_blocks(markdown_blocks: list[object]) -> list[str]:
    cleaned_blocks: list[str] = []
    for item in markdown_blocks:
        text = str(item)
        text = _MARKDOWN_CODE_FENCE_RE.sub("", text)
        text = _UNCLOSED_MARKDOWN_CODE_FENCE_RE.sub("", text)
        text = text.replace("```", "")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            cleaned_blocks.append(text)
    return cleaned_blocks


def _first_finding(findings: list[str], default: str) -> str:
    for item in findings:
        text = str(item).strip()
        if text:
            return text
    return default


def _format_number(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if abs(numeric) <= 1:
            numeric *= 100
        return f"{numeric:.2f}%"
    return str(value)


def _format_raw_ratios_in_text(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            return _format_percent(float(raw))
        except ValueError:
            return raw

    return re.sub(r"(?<![\d,])-?0\.\d{2,4}(?!\d)", replace, text)


def _table_rows(module, table_name: str) -> list[dict[str, object]]:
    rows = module.tables.get(table_name, [])
    return [row for row in rows if isinstance(row, dict)]


def _row_value(row: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _combined_slice_label(row: dict[str, object], *keys: str) -> str:
    values: list[str] = []
    for key in keys:
        value = _row_value(row, key)
        if value is not None and str(value).strip():
            values.append(str(value))
    return " / ".join(values)


def build_section_insight_markdown(
    *,
    section_id: str,
    module_report=None,
    dataset_profile: dict[str, object] | None = None,
    analysis_focus: str | None = None,
    evidence_pack: dict[str, object] | None = None,
    chart_contexts: list[dict[str, object]] | None = None,
    output_language: str | None = None,
) -> list[str]:
    """Build concise, evidence-grounded section narrative blocks for final notebooks."""
    del dataset_profile, analysis_focus, evidence_pack
    module = module_report
    if _nb_english(output_language):
        fallback_by_section = {
            "metric_distributions": (
                "Long-tail metrics and outliers should be separated before trend and profit interpretation.",
                "P99 clipping is a readability choice and should not hide record-level review.",
            ),
            "sales_trends": (
                "Total sales and monthly volatility should be reviewed through peak and trough periods.",
                "Follow-up review should return to the orders and products behind peak months.",
            ),
            "product_and_category": (
                "Product contribution should be reviewed together with profit quality, not sales alone.",
                "High-sales low-margin products should be prioritized for pricing and promotion checks.",
            ),
            "segment_and_region": (
                "Segment and region slices should be compared by both sales scale and profit quality.",
                "Weak slices should be traced back to records, discount, fulfillment cost, and product mix.",
            ),
            "discount_and_profit": (
                "High discount tiers should be reviewed for loss rate and margin erosion.",
                "Follow-up review should focus on high-discount low-profit records.",
            ),
            "forecast": (
                "The forecast baseline is a monitoring reference, not a production forecast.",
                "Forecast deviations should be validated with promotion, holiday, and seasonality variables.",
            ),
        }
        takeaway, commentary = fallback_by_section.get(
            section_id,
            (
                "This section identifies business objects that need follow-up review.",
                "Use the chart and table evidence to return to detailed records before taking action.",
            ),
        )
        if module is not None:
            findings = [str(item).strip() for item in getattr(module, "findings", []) if str(item).strip()]
            if findings:
                takeaway = safe_english_sentence(findings[0], fallback=takeaway)
            if len(findings) > 1:
                commentary = safe_english_sentence(findings[1], fallback=commentary)
        blocks = [f"### Section Takeaway\n\n{takeaway}"]
        for chart_context in chart_contexts or []:
            title = str(chart_context.get("chart_title") or "Selected chart").strip()
            summary = safe_english_sentence(
                chart_context.get("chart_summary") or chart_context.get("table_preview") or "",
                fallback=commentary,
            )
            blocks.append(
                "### Chart Commentary\n\n"
                f"{title} provides evidence for the selected business question. {summary}"
            )
        if not chart_contexts:
            blocks.append(f"### Chart Commentary\n\n{commentary}")
        return blocks
    if module is None:
        fallback_by_section = {
            "sales_trends": (
                "### 本节结论\n\n"
                "本节优先判断销售峰值、低谷和阶段波动是否足够明显。"
                "缺少可验证模块结果时，不把趋势解释成确定增长。"
                "后续应先补齐时间粒度、峰谷月份和对应订单明细，再决定是否做促销或库存动作。"
            ),
            "product_and_category": (
                "### 本节结论\n\n"
                "本节先判断销售是否集中在少数商品或类目。"
                "没有模块底表或缺少利润字段时，不直接假设头部商品一定贡献利润。"
                "后续如补齐利润字段，应把头部销售额、利润率和累计占比放在一起复盘，避免只追销售规模。"
            ),
            "segment_and_region": (
                "### 本节结论\n\n"
                "本节关注客群和区域切片之间的强弱差异。"
                "缺少模块结果时，只保留结构拆解方向，不把某个切片写成事实结论。"
                "后续如补齐利润字段，应补齐销售额、利润和利润率对照，再定位销售强但利润弱的组合。"
            ),
            "discount_and_profit": (
                "### 本节结论\n\n"
                "本节聚焦折扣是否侵蚀利润。"
                "没有折扣分桶和负利润占比时，不把折扣风险写成确定事实。"
                "后续应先补齐高折扣订单数、负利润率和低毛利商品清单，再制定审批阈值。"
            ),
            "forecast": (
                "### 本节结论\n\n"
                "本节只把预测作为经营基线，而不是精细预算。"
                "缺少历史窗口和预测期数时，预测结果只能用于方向判断。"
                "后续应补入季节性、促销和节假日因素，再决定库存或目标调整。"
            ),
        }
        return [fallback_by_section[section_id]] if section_id in fallback_by_section else []

    metrics = module.summary_metrics
    findings = [str(item).strip() for item in module.findings if str(item).strip()]
    lead_finding = _format_raw_ratios_in_text(
        _first_finding(findings, "当前模块结果需要结合底表继续复核。")
    )
    blocks: list[str] = []

    if section_id == "metric_distributions":
        rows = _table_rows(module, "metric_quantiles")
        metric_count = metrics.get("metric_count", len(rows))
        tail_row = rows[0] if rows else {}
        label = _row_value(tail_row, "label", "metric", "column") or "核心指标"
        p99 = _format_number(_row_value(tail_row, "p99"))
        median = _format_number(_row_value(tail_row, "median"))
        conclusion = (
            "### 本节结论\n\n"
            f"本节覆盖 `{_format_number(metric_count)}` 个核心数值指标，先判断分布是否会让均值失真。"
            f"`{label}` 的中位数约 `{median}`，P99 约 `{p99}`，头部大值和典型订单之间差距明显。"
            f"{lead_finding} 后续应把极端订单单独拆出，再进入趋势、商品和利润专题。"
        )
        blocks.append(
            "### 分布解读口径\n\n"
            "本节优先保留可执行的指标画像和长尾判断；未被图表选择器选中的箱线图、相关热力图或散点图不会在最终 notebook 中展示。"
        )
    elif section_id == "sales_trends":
        conclusion = (
            "### 本节结论\n\n"
            f"当前累计销售额约 `{_format_number(metrics.get('total_sales_amount'))}`。"
            f"峰值出现在 `{metrics.get('peak_period', 'unknown')}`，低谷出现在 `{metrics.get('trough_period', 'unknown')}`，峰谷对比比单看均值更能说明波动质量。"
            f"{lead_finding} 后续应拆分峰值月份的商品、渠道和大单，判断增长是否可复制。"
        )
    elif section_id == "product_and_category":
        top_products = _table_rows(module, "top_products")
        first = top_products[0] if top_products else {}
        second = top_products[1] if len(top_products) > 1 else {}
        first_name = _row_value(first, "Product Name", "product_name") or metrics.get("top_product", "unknown")
        first_sales = _format_number(_row_value(first, "Sales", "sales_amount"))
        first_share = _format_percent(_row_value(first, "sales_share"))
        second_name = _row_value(second, "Product Name", "product_name") or "第二梯队商品"
        conclusion = (
            "### 本节结论\n\n"
            f"当前商品数约 `{_format_number(metrics.get('distinct_products'))}`，头部商品 `{first_name}` 销售额约 `{first_sales}`，贡献约 `{first_share}`。"
            f"`{second_name}` 是直接对照，头部与第二梯队的差距决定了销售集中度风险。"
            f"{lead_finding} 后续应把头部商品的销售额和利润率一起排序，优先复盘销售高但利润不同步的 SKU。"
        )
    elif section_id == "segment_and_region":
        secondary_dimension = metrics.get("secondary_dimension")
        rows = (
            _table_rows(module, "segment_region_matrix")
            if secondary_dimension
            else []
        ) or _table_rows(module, "dimension_totals")
        weak_rows = _table_rows(module, "weak_performance_cuts")
        top = rows[0] if rows else {}
        weak = weak_rows[0] if weak_rows else {}
        top_label = _combined_slice_label(top, "Segment", "segment", "Region", "region") or metrics.get("primary_dimension", "unknown")
        weak_label = _combined_slice_label(weak, "Segment", "segment", "Region", "region") or "弱势切片"
        top_sales = _format_number(_row_value(top, "Sales", "sales_amount"))
        conclusion = (
            "### 本节结论\n\n"
            f"当前强势切片 `{top_label}` 销售额约 `{top_sales}`，弱势切片集中在 `{weak_label}`。"
            "强弱切片并列出现，说明经营差异更可能来自客群、区域或品类组合，而不是总体需求单一变化。"
            f"{lead_finding} 后续应优先复盘高销售低利润组合，检查折扣、履约成本和商品结构。"
        )
    elif section_id == "discount_and_profit":
        buckets = _table_rows(module, "discount_buckets") or _table_rows(module, "discount_profit_risk_buckets")
        negative_rate = _format_percent(metrics.get("negative_profit_rate"))
        bucket_names = "、".join(str(_row_value(row, "discount_bucket") or "") for row in buckets if _row_value(row, "discount_bucket"))
        worst = min(buckets, key=lambda row: float(row.get("avg_profit", 0))) if buckets else {}
        conclusion = (
            "### 本节结论\n\n"
            f"当前高折扣订单约 `{_format_number(metrics.get('high_discount_order_count'))}` 条，负利润占比约 `{negative_rate}`。"
            f"折扣分桶包括 `{bucket_names or '暂无分桶'}`，其中 `{_row_value(worst, 'discount_bucket') or '最低利润区间'}` 更接近尾部风险。"
            f"{lead_finding} 后续应设置高折扣审批阈值，并逐笔复盘高销售低利润订单。"
        )
    elif section_id == "forecast":
        forecast_status = str(metrics.get("forecast_status") or "")
        best_baseline = metrics.get("best_baseline")
        mae = _format_number(metrics.get("best_baseline_mae"))
        rmse = _format_number(metrics.get("best_baseline_rmse"))
        mape_value = metrics.get("best_baseline_mape")
        mape = _format_percent(mape_value)
        time_split = metrics.get("time_split")
        metric_sentence = ""
        if forecast_status == "baseline_evaluated":
            metric_sentence = (
                f"本节展示的是销售额预测 baseline，不是生产级预测；"
                f"回测采用 `{time_split or 'chronological'}` 时间切分，"
                f"当前最优 baseline 为 `{best_baseline or 'unknown'}`，"
                f"MAE `{mae}`，RMSE `{rmse}`，MAPE `{mape}`。"
            )
            if isinstance(mape_value, (int, float)) and float(mape_value) >= 100:
                metric_sentence += "MAPE 偏高，说明该数据波动较大，预测难度高，当前结果更适合作为监控参照。"
        else:
            metric_sentence = (
                "本节仅保留预测可行性说明；当前数据暂不满足稳定 baseline 回测条件，不应强行包装成预测模型。"
            )
        conclusion = (
            "### 本节结论\n\n"
            f"当前基线预测的日基线约 `{_format_number(metrics.get('baseline_sales_amount'))}`，预测窗口约 `{_format_number(metrics.get('horizon_days'))}` 天。"
            "预测线适合回答不额外干预时的方向水平，不适合直接替代预算。"
            f"{metric_sentence}"
            f"{lead_finding} 后续应加入季节性、促销和节假日变量，再判断目标是否需要上调或下修。"
        )
    else:
        conclusion = (
            "### 本节结论\n\n"
            f"{lead_finding} 当前 section 已产生可复核结果，下一步要看头部与尾部、增长与下滑或销售与利润是否同步。"
            "后续应把最突出的对象回填到底表，形成可执行的复盘清单。"
        )
    blocks.insert(0, conclusion)

    for chart_context in chart_contexts or []:
        title = str(chart_context.get("chart_title") or "当前图表")
        summary = str(chart_context.get("chart_summary") or "").strip()
        table_preview = str(chart_context.get("table_preview") or "").strip().splitlines()
        evidence = table_preview[0] if table_preview else summary
        blocks.append(
            "### 图表解读\n\n"
            f"{title} 的可用证据是 `{evidence or lead_finding}`。"
            f"{summary or lead_finding} 头部与尾部、峰值与低谷或销售与利润不同步的地方，是这张图最重要的对比。"
            "后续应把图中的关键对象回到明细表，确认是否需要调整促销、定价或商品组合。"
        )
    return blocks


def _chart_analysis_markdown(
    section_id: str,
    report: AnalysisReport,
    output_language: str | None = None,
) -> str | None:
    module = _module_for_section(section_id, report)
    insight_blocks = build_section_insight_markdown(
        section_id=section_id,
        module_report=module,
        dataset_profile=None,
        analysis_focus=None,
        evidence_pack=None,
        chart_contexts=None,
        output_language=output_language,
    )
    if insight_blocks:
        return insight_blocks[0]
    return None

def _ensure_chart_analysis(
    section_content: NotebookSectionContent,
    report: AnalysisReport,
    output_language: str | None = None,
) -> NotebookSectionContent:
    if any("本节结论" in block or "Section Takeaway" in block for block in section_content.markdown_blocks):
        return section_content

    chart_analysis = _chart_analysis_markdown(section_content.section_id, report, output_language=output_language)
    if chart_analysis is None:
        return section_content

    return NotebookSectionContent(
        section_id=section_content.section_id,
        markdown_blocks=[*section_content.markdown_blocks, chart_analysis],
        code_cells=section_content.code_cells,
    )


def _compact_narrative_blocks(
    narrative_section,
    output_language: str | None = None,
) -> list[str]:
    if narrative_section is None:
        return []
    intro = (narrative_section.intro or "").strip()
    observations = [str(item).strip() for item in narrative_section.key_observations if str(item).strip()]
    takeaway = (narrative_section.business_takeaway or "").strip()

    blocks: list[str] = []
    first_parts: list[str] = []
    if intro:
        first_parts.append(intro)
    if observations:
        if _nb_english(output_language):
            joined = "; ".join(observations[:2])
            bridge = ""
            ending = "" if joined.endswith((".", "!", "?")) else "."
        else:
            joined = "\uff1b".join(observations[:2])
            bridge = "\u7ed3\u5408\u5f53\u524d\u7ed3\u679c\uff0c" if intro else ""
            ending = "" if joined.endswith(("\u3002", "\uff01", "\uff1f", ".", "!", "?")) else "\u3002"
        first_parts.append(f"{bridge}{joined}{ending}")
    if first_parts:
        blocks.append(" ".join(first_parts))
    if takeaway:
        blocks.append(takeaway)
    return blocks

def _conclusions_section_content(
    report: AnalysisReport,
    output_language: str | None = None,
) -> NotebookSectionContent:
    trend_module = _module_for_section("sales_trends", report)
    product_module = _module_for_section("product_and_category", report)
    discount_module = _module_for_section("discount_and_profit", report)

    summary_facts: list[str] = []
    if trend_module is not None:
        value = trend_module.summary_metrics.get('total_sales_amount')
        summary_facts.append(
            f"Current total sales are approximately {_format_number(value)} ({value})"
            if _nb_english(output_language)
            else f"\u5f53\u524d\u7d2f\u8ba1\u9500\u552e\u989d\u7ea6\u4e3a {_format_number(value)} ({value})"
        )
    if product_module is not None:
        summary_facts.append(
            f"The top product is {product_module.summary_metrics.get('top_product', 'unknown')}"
            if _nb_english(output_language)
            else f"\u5934\u90e8\u5546\u54c1\u662f {product_module.summary_metrics.get('top_product', 'unknown')}"
        )
    if discount_module is not None:
        value = discount_module.summary_metrics.get('high_discount_order_count')
        summary_facts.append(
            f"High-discount records are approximately {_format_number(value)} ({value})"
            if _nb_english(output_language)
            else f"\u9ad8\u6298\u6263\u8bb0\u5f55\u7ea6 {_format_number(value)} ({value}) \u6761"
        )

    if _nb_english(output_language):
        first_paragraph = ". ".join(summary_facts) + "." if summary_facts else "The main sales analysis modules have been completed."
        paragraph_two = "Recommended actions should prioritize high-discount low-profit combinations, then review weak segments, regions, and products."
    else:
        first_paragraph = "\u3002".join(summary_facts) + "\u3002" if summary_facts else "\u5f53\u524d\u5206\u6790\u5df2\u5b8c\u6210\u4e3b\u8981\u9500\u552e\u6a21\u5757\u3002"
        paragraph_two = "\u7ecf\u8425\u5efa\u8bae\u4e0a\u5e94\u4f18\u5148\u590d\u76d8\u9ad8\u6298\u6263\u4f4e\u5229\u6da6\u7ec4\u5408\uff0c\u518d\u5b9a\u4f4d\u5f31\u52bf\u5ba2\u7fa4\u3001\u533a\u57df\u548c\u5546\u54c1\u3002"

    return NotebookSectionContent(
        section_id="conclusions",
        markdown_blocks=[first_paragraph, paragraph_two],
        code_cells=[],
    )

def _discount_profit_section(
    markdown_blocks: list[str],
    schema_mapping: SchemaMapping,
    sales_col: str,
    category_col: str,
    product_col: str,
    discount_col: str,
    profit_col: str,
    discount_profit_strategy: dict[str, object] | None = None,
    chart_selection_plan: dict[str, object] | None = None,
    output_language: str | None = None,
) -> NotebookSectionContent:
    english = _nb_english(output_language)
    mapped_fields = set(schema_mapping.field_mapping.values())
    has_discount = "discount" in mapped_fields
    has_category = "category" in mapped_fields
    has_product = "product_name" in mapped_fields
    has_segment = "segment" in mapped_fields
    has_region = "region" in mapped_fields
    segment_col = _original_column(schema_mapping, "segment", "Segment")
    region_col = _original_column(schema_mapping, "region", "Region")
    markdown = list(markdown_blocks)
    discount_strategy_literal = repr(
        discount_profit_strategy
        or {"source": "empty", "allowed_charts": [], "views": []}
    )
    selected_chart_ids_literal = _selected_chart_ids_literal(chart_selection_plan)
    discount_intro = _nb_text(
        output_language,
        "This section checks whether higher discount intensity is associated with weaker profit quality.",
        "这一节重点检查更高的折扣强度是否伴随更弱的利润质量。",
    )
    discount_what_if_note = _nb_text(
        output_language,
        "Discount tightening what-if is a scenario estimate only. It is based on discount recovery and does not represent real demand, volume, or customer behavior changes.",
        "折扣收紧 what-if 仅作情景分析：这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。",
    )
    discount_what_if_note_2 = _nb_text(
        output_language,
        "The following discount-tightening scenario estimate is based on recovered discount amount and does not represent real demand, volume, or customer behavior changes.",
        "以下为折扣收紧情景估算，基于折扣回收金额计算，不代表真实需求、销量或客户行为变化。",
    )
    discount_vs_profit_title = _nb_quote(output_language, "Discount vs. Profit Relationship", "折扣与利润散点关系")
    discount_high_hint = _nb_quote(
        output_language,
        "Set this as a discount approval review threshold and review products, customers, and authorization rules first.",
        "设为折扣审批红线，优先复盘商品、客户和授权口径。",
    )
    discount_medium_hint = _nb_quote(
        output_language,
        "Add this tier to weekly monitoring and track whether loss rate and profit margin continue to worsen.",
        "纳入周度监控，观察亏损率和利润率是否继续恶化。",
    )
    discount_low_hint = _nb_quote(
        output_language,
        "Keep this as a comparison tier; it is not the primary target for discount tightening.",
        "保留为对照区间，暂不作为折扣收紧重点。",
    )
    discount_no_risk_message = _nb_quote(
        output_language,
        "No high-risk discount tier was identified, so the discount-tightening what-if scenario was skipped.",
        "未识别到高风险折扣区间，已跳过折扣收紧 what-if 情景估算。",
    )
    discount_no_cap_message = _nb_quote(
        output_language,
        "The high-risk discount tier has no records above the target threshold, so the what-if scenario was skipped.",
        "高风险折扣区间没有可收紧到目标阈值以上的记录，已跳过 what-if 情景估算。",
    )
    discount_scenario_name = _nb_quote(output_language, "High-Risk Discount Tightening Scenario", "高风险折扣收紧情景")
    discount_scenario_note = _nb_quote(
        output_language,
        "This is a scenario estimate based on recovered discount amount and does not represent real demand, volume, or customer behavior changes.",
        "这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。",
    )
    avg_profit_label = _nb_quote(output_language, "Average Profit", "平均利润")
    loss_rate_label = _nb_quote(output_language, "Loss Rate", "亏损率")
    discount_tier_label = _nb_quote(output_language, "Discount Tier", "折扣区间")
    profit_margin_label = _nb_quote(output_language, "Profit Margin", "利润率")
    order_count_label = _nb_quote(output_language, "Order Count", "订单数")
    sales_label = _nb_quote(output_language, "Sales", "销售额")
    profit_label = _nb_quote(output_language, "Profit", "利润")
    discount_profit_quality_title = _nb_quote(
        output_language,
        "Discount Tier Profit Quality: Average Profit and Loss Rate",
        "各折扣区间利润质量：平均利润与亏损率",
    )
    discount_quality_skip_message = _nb_quote(
        output_language,
        "The strategy layer skipped the discount-tier profit quality chart.",
        "策略层跳过折扣区间利润质量图",
    )
    high_discount_product_title = _nb_quote(output_language, "High-Discount Low-Profit Products", "高折扣低利润商品")
    category_discount_heatmap_title = _nb_quote(
        output_language,
        "Category x Discount Tier Loss Rate Heatmap",
        "类目 x 折扣区间亏损率热力图",
    )
    category_label = _nb_quote(output_language, "Category", "类目")
    category_discount_skip_message = _nb_quote(
        output_language,
        "The strategy layer skipped the category discount risk heatmap.",
        "策略层跳过类目折扣风险热力图",
    )
    discount_boxplot_title = _nb_quote(output_language, "Profit Distribution by Discount Tier", "折扣区间利润率分布")
    discount_boxplot_skip_message = _nb_quote(
        output_language,
        "The strategy layer skipped the discount-tier boxplot.",
        "策略层跳过折扣区间箱线图",
    )
    no_discount_note = _nb_text(
        output_language,
        "The dataset does not include a discount field, so this section falls back to a profit overview.",
        "当前数据缺少折扣字段，因此这一节先退化为利润概览。",
    )
    profit_distribution_overview_title = _nb_quote(output_language, "Profit Distribution Overview", "利润分布概览")

    if has_discount:
        markdown.append(discount_intro)
        markdown.append(discount_what_if_note)
        markdown.append(discount_what_if_note_2)
        selected_columns = [discount_col, profit_col, sales_col]
        if has_category:
            selected_columns.append(category_col)
        if has_segment:
            selected_columns.append(segment_col)
        if has_region:
            selected_columns.append(region_col)
        if has_product:
            selected_columns.append(product_col)

        code_cells = [
            "\n".join(
                [
                    f"discount_profit_strategy = {discount_strategy_literal}",
                    f"selected_chart_ids = {selected_chart_ids_literal}",
                    "discount_view_map = {item.get('chart'): item for item in discount_profit_strategy.get('views', [])}",
                    "discount_strategy_table = pd.DataFrame(discount_profit_strategy.get('views', []))",
                    "discount_profit = clean_df[{0}].dropna()".format(repr(selected_columns)),
                    f"discount_profit['{discount_col}'] = pd.to_numeric(discount_profit['{discount_col}'], errors='coerce')",
                    f"discount_profit['{profit_col}'] = pd.to_numeric(discount_profit['{profit_col}'], errors='coerce')",
                    "discount_profit.head(), discount_strategy_table",
                ]
            ),
            "\n".join(
                [
                    "# chart_id: discount_vs_profit_scatter",
                    "if 'discount_profit_scatter' in discount_view_map or 'discount_vs_profit_scatter' in selected_chart_ids:",
                    "    fig = px.scatter(",
                    "        discount_profit,",
                    f"        x='{discount_col}',",
                    f"        y='{profit_col}',",
                    f"        size='{sales_col}',",
                    *([f"        color='{category_col}',"] if has_category else []),
                    f"        title={discount_vs_profit_title},",
                    "",
                    "    )",
                    "    fig.show()",
                    "else:",
                    "    pass",
                ]
            ),
            "\n".join(
                [
                    "discount_profit['discount_bucket'] = pd.cut(",
                    f"    discount_profit['{discount_col}'],",
                    "    bins=[-0.001, 0, 0.1, 0.2, 0.3, 1.0],",
                    "    labels=['0%', '0-10%', '10-20%', '20-30%', '30%+']",
                    ")",
                    "discount_buckets = discount_profit.groupby('discount_bucket', as_index=False, observed=False)[['{0}', '{1}']].mean()".format(
                        sales_col, profit_col
                    ),
                    "discount_buckets.head(15)",
                ]
            ),
            "\n".join(
                [
                    "discount_bucket_quality = discount_profit.groupby('discount_bucket', observed=False).agg(",
                    f"    avg_profit=('{profit_col}', 'mean'),",
                    f"    total_profit=('{profit_col}', 'sum'),",
                    f"    order_count=('{profit_col}', 'size'),",
                    f"    negative_profit_count=('{profit_col}', lambda s: (s < 0).sum()),",
                    f"    avg_sales=('{sales_col}', 'mean'),",
                    f"    total_sales=('{sales_col}', 'sum'),",
                    ").reset_index()",
                    "discount_bucket_quality['negative_profit_rate'] = discount_bucket_quality['negative_profit_count'] / discount_bucket_quality['order_count']",
                    "discount_bucket_quality['profit_margin'] = discount_bucket_quality['total_profit'] / discount_bucket_quality['total_sales'].mask(discount_bucket_quality['total_sales'] == 0)",
                    "discount_bucket_quality",
                ]
            ),
            "\n".join(
                [
                    "discount_threshold_candidates = discount_bucket_quality.assign(",
                    "    row_count=discount_bucket_quality['order_count'],",
                    "    sales_amount=discount_bucket_quality['total_sales'],",
                    ")",
                    "def _threshold_risk_level(row):",
                    "    if row['negative_profit_rate'] >= 0.5 or row['profit_margin'] < 0 or row['avg_profit'] < 0:",
                    "        return 'high'",
                    "    if row['negative_profit_rate'] >= 0.15 or row['profit_margin'] < 0.05:",
                    "        return 'medium'",
                    "    return 'low'",
                    "discount_threshold_candidates['risk_level'] = discount_threshold_candidates.apply(_threshold_risk_level, axis=1)",
                    "discount_threshold_candidates['action_hint'] = discount_threshold_candidates['risk_level'].map({",
                    f"    'high': {discount_high_hint},",
                    f"    'medium': {discount_medium_hint},",
                    f"    'low': {discount_low_hint},",
                    "})",
                    "risk_rank = {'high': 0, 'medium': 1, 'low': 2}",
                    "discount_threshold_candidates = discount_threshold_candidates.sort_values(",
                    "    by=['risk_level', 'negative_profit_rate', 'profit_margin', 'avg_profit'],",
                    "    ascending=[True, False, True, True],",
                    "    key=lambda col: col.map(risk_rank) if col.name == 'risk_level' else col,",
                    ")[['discount_bucket', 'row_count', 'order_count', 'sales_amount', 'avg_profit', 'profit_margin', 'negative_profit_rate', 'risk_level', 'action_hint']]",
                    "discount_threshold_candidates",
                ]
            ),
            "\n".join(
                [
                    "def _target_cap_for_discount_bucket(bucket):",
                    "    label = str(bucket).replace('%', '').replace('+', '').strip()",
                    "    lower_bound = label.split('-', 1)[0].strip()",
                    "    try:",
                    "        return max(float(lower_bound) / 100, 0.0)",
                    "    except (TypeError, ValueError):",
                    "        return 0.2",
                    "",
                    "high_risk_thresholds = discount_threshold_candidates[discount_threshold_candidates['risk_level'].astype(str).str.lower() == 'high']",
                    "if high_risk_thresholds.empty:",
                    "    discount_cap_what_if = pd.DataFrame()",
                    f"    print({discount_no_risk_message})",
                    "else:",
                    "    scenario_threshold = high_risk_thresholds.sort_values(",
                    "        by=['negative_profit_rate', 'profit_margin', 'avg_profit'],",
                    "        ascending=[False, True, True],",
                    "    ).iloc[0]",
                    "    high_risk_bucket = str(scenario_threshold['discount_bucket'])",
                    "    target_discount_cap = _target_cap_for_discount_bucket(high_risk_bucket)",
                    "    scenario_rows = discount_profit.copy()",
                    f"    scenario_rows['{discount_col}'] = pd.to_numeric(scenario_rows['{discount_col}'], errors='coerce').clip(lower=0, upper=1)",
                    f"    scenario_rows['{profit_col}'] = pd.to_numeric(scenario_rows['{profit_col}'], errors='coerce')",
                    f"    scenario_rows['{sales_col}'] = pd.to_numeric(scenario_rows['{sales_col}'], errors='coerce')",
                    "    scenario_rows = scenario_rows[scenario_rows['discount_bucket'].astype(str) == high_risk_bucket]",
                    f"    scenario_rows = scenario_rows[scenario_rows['{discount_col}'] > target_discount_cap]",
                    "    if scenario_rows.empty:",
                    "        discount_cap_what_if = pd.DataFrame()",
                    f"        print({discount_no_cap_message})",
                    "    else:",
                    f"        recovered_discount_amount = ((scenario_rows['{discount_col}'] - target_discount_cap).clip(lower=0) * scenario_rows['{sales_col}']).fillna(0)",
                    f"        current_profit = scenario_rows['{profit_col}'].sum()",
                    "        estimated_profit_delta = recovered_discount_amount.sum()",
                    f"        current_negative_profit_rate = (scenario_rows['{profit_col}'] < 0).mean()",
                    f"        estimated_profit = scenario_rows['{profit_col}'] + recovered_discount_amount",
                    "        estimated_negative_profit_rate_after_cap = (estimated_profit < 0).mean()",
                    "        discount_cap_what_if = pd.DataFrame([",
                    "            {",
                    f"                'scenario_name': {discount_scenario_name},",
                    "                'current_threshold': high_risk_bucket,",
                    "                'high_risk_bucket': high_risk_bucket,",
                    "                'target_discount_cap': target_discount_cap,",
                    "                'affected_row_count': int(len(scenario_rows)),",
                    f"                'affected_sales_amount': float(scenario_rows['{sales_col}'].sum()),",
                    "                'current_profit': float(current_profit),",
                    "                'estimated_profit_after_cap': float(current_profit + estimated_profit_delta),",
                    "                'estimated_profit_delta': float(estimated_profit_delta),",
                    "                'current_negative_profit_rate': float(current_negative_profit_rate),",
                    "                'estimated_negative_profit_rate_after_cap': float(estimated_negative_profit_rate_after_cap),",
                    "            }",
                    "        ])",
                    f"        print({discount_scenario_note})",
                    "discount_cap_what_if",
                ]
            ),
            "\n".join(
                [
                    "# chart_id: discount_profit_quality_bar",
                    "if 'bucket_profit_quality_bar' in discount_view_map or 'discount_profit_quality_bar' in selected_chart_ids:",
                    "    import plotly.graph_objects as go",
                    "    from plotly.subplots import make_subplots",
                    "    fig = make_subplots(specs=[[{'secondary_y': True}]])",
                    "    bucket_x = discount_threshold_candidates.sort_index()['discount_bucket'].astype(str)",
                    "    chart_thresholds = discount_threshold_candidates.sort_index()",
                    "    hover_payload = chart_thresholds[['order_count', 'sales_amount', 'profit_margin', 'negative_profit_rate']].to_numpy()",
                    "    fig.add_trace(go.Bar(",
                    "        x=bucket_x,",
                    "        y=chart_thresholds['avg_profit'],",
                    f"        name={avg_profit_label},",
                    "        customdata=hover_payload,",
                    f"        hovertemplate='{_nb_text(output_language, 'Discount Tier', '折扣区间')}=%{{x}}<br>{_nb_text(output_language, 'Average Profit', '平均利润')}=%{{y:.2f}}<br>{_nb_text(output_language, 'Order Count', '订单数')}=%{{customdata[0]}}<br>{_nb_text(output_language, 'Sales', '销售额')}=%{{customdata[1]:.2f}}<br>{_nb_text(output_language, 'Profit Margin', '利润率')}=%{{customdata[2]:.1%}}<br>{_nb_text(output_language, 'Loss Rate', '亏损率')}=%{{customdata[3]:.1%}}<extra></extra>',",
                    "    ), secondary_y=False)",
                    "    fig.add_trace(go.Scatter(",
                    "        x=bucket_x,",
                    "        y=chart_thresholds['negative_profit_rate'],",
                    f"        name={loss_rate_label},",
                    "        mode='lines+markers',",
                    "        customdata=hover_payload,",
                    f"        hovertemplate='{_nb_text(output_language, 'Discount Tier', '折扣区间')}=%{{x}}<br>{_nb_text(output_language, 'Loss Rate', '亏损率')}=%{{y:.1%}}<br>{_nb_text(output_language, 'Order Count', '订单数')}=%{{customdata[0]}}<br>{_nb_text(output_language, 'Sales', '销售额')}=%{{customdata[1]:.2f}}<br>{_nb_text(output_language, 'Profit Margin', '利润率')}=%{{customdata[2]:.1%}}<extra></extra>',",
                    "    ), secondary_y=True)",
                    "",
                    f"    fig.update_layout(title={discount_profit_quality_title}, xaxis_title={discount_tier_label})",
                    f"    fig.update_yaxes(title_text={avg_profit_label}, secondary_y=False)",
                    f"    fig.update_yaxes(title_text={loss_rate_label}, tickformat='.0%', secondary_y=True)",
                    "",
                    "    fig.show()",
                    "else:",
                    f"    print({discount_quality_skip_message})",
                ]
            ),
        ]

        if has_product:
            code_cells.append(
                "\n".join(
                    [
                        f"loss_making_products = discount_profit.groupby('{product_col}', as_index=False)[['{sales_col}', '{profit_col}']].sum().sort_values('{profit_col}').head(10)",
                        "loss_making_products",
                    ]
                )
            )
            code_cells.append(
                "\n".join(
                    [
                        f"product_profit_quality = discount_profit.groupby('{product_col}', as_index=False)[['{sales_col}', '{profit_col}', '{discount_col}']].agg({{'{sales_col}': 'sum', '{profit_col}': 'sum', '{discount_col}': 'mean'}})",
                        f"product_order_counts = discount_profit.groupby('{product_col}').size().rename('order_count').reset_index()",
                        f"product_profit_quality = product_profit_quality.merge(product_order_counts, on='{product_col}', how='left').rename(columns={{'{discount_col}': 'avg_discount'}})",
                        f"product_profit_quality['profit_margin'] = product_profit_quality['{profit_col}'] / product_profit_quality['{sales_col}'].mask(product_profit_quality['{sales_col}'] == 0)",
                        f"sales_threshold = product_profit_quality['{sales_col}'].quantile(0.75)",
                        f"high_sales_low_profit = product_profit_quality[(product_profit_quality['{sales_col}'] >= sales_threshold) & (product_profit_quality['profit_margin'] <= 0.05)].sort_values(['profit_margin', '{sales_col}'], ascending=[True, False]).head(10)",
                        "if high_sales_low_profit.empty:",
                        f"    high_sales_low_profit = product_profit_quality.sort_values(['profit_margin', '{sales_col}'], ascending=[True, False]).head(10)",
                        "high_sales_low_profit",
                    ]
                )
            )
            code_cells.append(
                "\n".join(
                    [
                        "if ('high_sales_low_profit_bar' in discount_view_map or 'high_discount_product_bar' in selected_chart_ids) and not high_sales_low_profit.empty:",
                        "    fig = px.bar(",
                        "        high_sales_low_profit,",
                        f"        x='{sales_col}',",
                        f"        y='{product_col}',",
                        "        color='profit_margin',",
                        "        orientation='h',",
                        "",
                        "        color_continuous_scale='RdYlGn',",
                        f"        title={high_discount_product_title},",
                        "    )",
                        "    fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                        "    fig.show()",
                        "else:",
                        "    pass",
                    ]
                )
            )
        elif has_category:
            code_cells.append(
                "\n".join(
                    [
                        f"loss_making_categories = discount_profit.groupby('{category_col}', as_index=False)[['{sales_col}', '{profit_col}']].sum().sort_values('{profit_col}').head(10)",
                        "loss_making_categories",
                    ]
                )
            )
        else:
            code_cells.append(
                "\n".join(
                    [
                        f"worst_profit_rows = discount_profit.sort_values('{profit_col}').head(10)[['{discount_col}', '{sales_col}', '{profit_col}']]",
                        "worst_profit_rows",
                    ]
                )
            )
        if has_category:
            code_cells.append(
                "\n".join(
                    [
                        f"category_discount_risk = discount_profit.groupby(['{category_col}', 'discount_bucket'], as_index=False, observed=False)[['{sales_col}', '{profit_col}']].sum()",
                        f"category_discount_counts = discount_profit.groupby(['{category_col}', 'discount_bucket'], observed=False).size().rename('order_count').reset_index()",
                        f"category_discount_losses = discount_profit.assign(_negative_profit=discount_profit['{profit_col}'] < 0).groupby(['{category_col}', 'discount_bucket'], observed=False)['_negative_profit'].sum().rename('negative_profit_count').reset_index()",
                        f"category_discount_risk = category_discount_risk.merge(category_discount_counts, on=['{category_col}', 'discount_bucket'], how='left').merge(category_discount_losses, on=['{category_col}', 'discount_bucket'], how='left')",
                        "category_discount_risk['negative_profit_rate'] = category_discount_risk['negative_profit_count'] / category_discount_risk['order_count']",
                        f"category_discount_risk['profit_margin'] = category_discount_risk['{profit_col}'] / category_discount_risk['{sales_col}'].mask(category_discount_risk['{sales_col}'] == 0)",
                        "category_discount_risk.sort_values(['negative_profit_rate', 'profit_margin'], ascending=[False, True]).head(15)",
                    ]
                )
            )
            code_cells.append(
                "\n".join(
                    [
                        "risk_heatmap = category_discount_risk.pivot(index='{0}', columns='discount_bucket', values='negative_profit_rate').fillna(0)".format(
                            category_col
                        ),
                        "if ('category_discount_risk_heatmap' in discount_view_map or 'discount_category_heatmap' in selected_chart_ids) and not risk_heatmap.empty:",
                        "    plt.figure(figsize=(10, max(4, len(risk_heatmap) * 0.7)))",
                        "    sns.heatmap(risk_heatmap, annot=True, fmt='.0%', cmap='RdYlGn_r')",
                        f"    plt.title({category_discount_heatmap_title})",
                        f"    plt.xlabel({discount_tier_label})",
                        f"    plt.ylabel({category_label})",
                        "    plt.tight_layout()",
                        "    plt.show()",
                        "else:",
                        f"    print({category_discount_skip_message})",
                    ]
                )
            )
        for heatmap_chart_id, dimension_col, dimension_label_en, dimension_label_zh in [
            ("discount_segment_heatmap", segment_col, "Segment", "客群"),
            ("discount_region_heatmap", region_col, "Region", "区域"),
        ]:
            if (heatmap_chart_id == "discount_segment_heatmap" and not has_segment) or (
                heatmap_chart_id == "discount_region_heatmap" and not has_region
            ):
                continue
            dimension_label = _nb_text(output_language, dimension_label_en, dimension_label_zh)
            dimension_title = _nb_text(
                output_language,
                f"{dimension_label_en} x Discount Tier Loss Rate Heatmap",
                f"{dimension_label_zh} x 折扣区间亏损率热力图",
            )
            code_cells.extend(
                [
                    "\n".join(
                        [
                            f"{heatmap_chart_id}_risk = discount_profit.groupby(['{dimension_col}', 'discount_bucket'], as_index=False, observed=False)[['{sales_col}', '{profit_col}']].sum()",
                            f"{heatmap_chart_id}_counts = discount_profit.groupby(['{dimension_col}', 'discount_bucket'], observed=False).size().rename('order_count').reset_index()",
                            f"{heatmap_chart_id}_losses = discount_profit.assign(_negative_profit=discount_profit['{profit_col}'] < 0).groupby(['{dimension_col}', 'discount_bucket'], observed=False)['_negative_profit'].sum().rename('negative_profit_count').reset_index()",
                            f"{heatmap_chart_id}_risk = {heatmap_chart_id}_risk.merge({heatmap_chart_id}_counts, on=['{dimension_col}', 'discount_bucket'], how='left').merge({heatmap_chart_id}_losses, on=['{dimension_col}', 'discount_bucket'], how='left')",
                            f"{heatmap_chart_id}_risk['negative_profit_rate'] = {heatmap_chart_id}_risk['negative_profit_count'] / {heatmap_chart_id}_risk['order_count']",
                            f"{heatmap_chart_id}_matrix = {heatmap_chart_id}_risk.pivot(index='{dimension_col}', columns='discount_bucket', values='negative_profit_rate').fillna(0)",
                            f"{heatmap_chart_id}_risk.sort_values('negative_profit_rate', ascending=False).head(15)",
                        ]
                    ),
                    "\n".join(
                        [
                            f"# chart_id: {heatmap_chart_id}",
                            f"if '{heatmap_chart_id}' in selected_chart_ids and not {heatmap_chart_id}_matrix.empty:",
                            f"    plt.figure(figsize=(10, max(4, len({heatmap_chart_id}_matrix) * 0.7)))",
                            f"    sns.heatmap({heatmap_chart_id}_matrix, annot=True, fmt='.0%', cmap='RdYlGn_r')",
                            f"    plt.title({dimension_title!r})",
                            f"    plt.xlabel({discount_tier_label})",
                            f"    plt.ylabel({dimension_label!r})",
                            "    plt.tight_layout()",
                            "    plt.show()",
                            "else:",
                            "    pass",
                        ]
                    ),
                ]
            )
        code_cells.append(
            "\n".join(
                [
                    "if 'discount_bucket_boxplot' in discount_view_map or 'profit_margin_by_discount_box' in selected_chart_ids:",
                    f"    clipped_profit = discount_profit['{profit_col}'].clip(discount_profit['{profit_col}'].quantile(0.01), discount_profit['{profit_col}'].quantile(0.99))",
                    "    boxplot_df = discount_profit.assign(_clipped_profit=clipped_profit)",
                    "    plt.figure(figsize=(10, 5))",
                    "    sns.boxplot(data=boxplot_df, x='discount_bucket', y='_clipped_profit')",
                    "",
                    f"    plt.xlabel({discount_tier_label})",
                    f"    plt.title({discount_boxplot_title})",
                    f"    plt.ylabel({profit_label})",
                    "    plt.xticks(rotation=20)",
                    "    plt.tight_layout()",
                    "    plt.show()",
                    "else:",
                    f"    print({discount_boxplot_skip_message})",
                ]
            )
        )

        return NotebookSectionContent(
            section_id="discount_and_profit",
            markdown_blocks=markdown,
            code_cells=code_cells,
        )

    markdown.append(no_discount_note)
    selected_columns = [profit_col, sales_col]
    if has_category:
        selected_columns.append(category_col)
    if has_product:
        selected_columns.append(product_col)

    code_cells = [
        "\n".join(
            [
                "profit_overview = clean_df[{0}].copy()".format(repr(selected_columns)),
                f"profit_overview['{profit_col}'] = pd.to_numeric(profit_overview['{profit_col}'], errors='coerce')",
                "profit_overview = profit_overview.dropna(subset=['{0}'])".format(profit_col),
                "profit_overview.head()",
            ]
        )
    ]

    if has_category:
        code_cells.extend(
            [
                "\n".join(
                    [
                        f"loss_by_category = profit_overview.groupby('{category_col}', as_index=False)[['{sales_col}', '{profit_col}']].sum().sort_values('{profit_col}')",
                        "loss_by_category.head(15)",
                    ]
                ),
                "\n".join(
                    [
                        "fig = px.bar(",
                        "    loss_by_category,",
                        f"    x='{category_col}',",
                        f"    y='{profit_col}',",
                        "",
                        ")",
                        "fig.show()",
                    ]
                ),
            ]
        )
    elif has_product:
        code_cells.extend(
            [
                "\n".join(
                    [
                        f"loss_by_product = profit_overview.groupby('{product_col}', as_index=False)[['{sales_col}', '{profit_col}']].sum().sort_values('{profit_col}')",
                        "loss_by_product.head(15)",
                    ]
                ),
                "\n".join(
                    [
                        "fig = px.bar(",
                        "    loss_by_product,",
                        f"    x='{product_col}',",
                        f"    y='{profit_col}',",
                        "",
                        ")",
                        "fig.show()",
                    ]
                ),
            ]
        )
    else:
        code_cells.extend(
            [
                "\n".join(
                    [
                        f"worst_profit_rows = profit_overview.sort_values('{profit_col}').head(15)[['{sales_col}', '{profit_col}']]",
                        "worst_profit_rows",
                    ]
                ),
                "\n".join(
                    [
                        "fig = px.histogram(",
                        "    profit_overview,",
                        f"    x='{profit_col}',",
                        "    nbins=30,",
                        f"    title={profit_distribution_overview_title}",
                        ")",
                        "fig.show()",
                    ]
                ),
            ]
        )

    return NotebookSectionContent(
        section_id="discount_and_profit",
        markdown_blocks=markdown,
        code_cells=code_cells,
    )


def _product_category_section(
    markdown_blocks: list[str],
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    sales_col: str,
    category_col: str,
    sub_category_col: str,
    product_col: str,
    profit_col: str,
    product_category_strategy: dict[str, object] | None = None,
    chart_selection_plan: dict[str, object] | None = None,
    output_language: str | None = None,
) -> NotebookSectionContent:
    english = _nb_english(output_language)
    mapped_fields = set(schema_mapping.field_mapping.values())
    has_product = bool({"product_name", "sku"} & mapped_fields)
    has_category = "category" in mapped_fields
    has_sub_category = "sub_category" in mapped_fields
    has_profit = "profit" in mapped_fields
    subcategory_missing_note = _nb_text(
        output_language,
        "The dataset does not include a subcategory field, so this section analyzes structure at the category level.",
        "当前数据缺少子类目字段，因此这一节先按类目层级做结构分析。",
    )
    category_profit_margin_title = _nb_quote(output_language, "Category Profit Margin Comparison", "各类目利润率质量对比")
    category_concentration_title = _nb_quote(output_language, "Category Sales Concentration", "类目销售集中度")
    top_product_sales_title = _nb_quote(output_language, "Top Product Sales Comparison", "头部商品销售额对比")
    product_pareto_title = _nb_quote(
        output_language,
        "Top Product Cumulative Contribution Pareto",
        "头部商品累计贡献 Pareto 分布",
    )
    product_bridge_title = _nb_quote(output_language, "Product Sales and Profit Bridge", "商品销售额与利润桥接")
    category_concentration_comparison_title = _nb_quote(output_language, "Category Concentration Comparison", "类目集中度对比")
    subcategory_matrix_title = _nb_quote(output_language, "Subcategory Sales-Profit Matrix", "子类目销售利润矩阵")
    top_product_profit_gap_title = _nb_quote(output_language, "Top Product Profit Gap", "头部商品利润缺口")
    category_axis_label = _nb_quote(output_language, "Category", "类目")
    sales_share_label = _nb_quote(output_language, "Sales Share", "销售占比")
    profit_gap_missing_message = _nb_quote(
        output_language,
        "Insufficient data: product, sales, or profit records are missing for the top product profit-gap chart.",
        "数据不足：缺少可用于头部商品利润缺口图的商品、销售额或利润记录。",
    )
    profit_gap_no_sales_message = _nb_quote(
        output_language,
        "Insufficient data: top products do not have valid sales, so the profit-gap chart was skipped.",
        "数据不足：头部商品没有有效销售额，跳过利润缺口图。",
    )
    product_strategy_literal = repr(
        product_category_strategy
        or {"source": "empty", "allowed_charts": [], "views": []}
    )
    local_markdown = list(markdown_blocks)
    code_cells: list[str] = [
        "\n".join(
            [
                f"product_category_strategy = {product_strategy_literal}",
                "product_view_map = {item.get('chart'): item for item in product_category_strategy.get('views', [])}",
                "product_strategy_table = pd.DataFrame(product_category_strategy.get('views', []))",
                "product_strategy_table",
            ]
        )
    ]

    if has_product:
        product_metric_columns = [sales_col]
        if has_profit:
            product_metric_columns.append(profit_col)
        code_cells.append(
            "\n".join(
                [
                    f"top_products = clean_df.groupby('{product_col}', as_index=False)[{product_metric_columns!r}].sum().sort_values('{sales_col}', ascending=False).head(15)",
                    f"top_products['sales_share'] = top_products['{sales_col}'] / top_products['{sales_col}'].sum()",
                    "top_products['cumulative_sales_share'] = top_products['sales_share'].cumsum()",
                    *(
                        [
                            f"top_products['profit_margin'] = top_products['{profit_col}'] / top_products['{sales_col}'].mask(top_products['{sales_col}'] == 0)",
                        ]
                        if has_profit
                        else []
                    ),
                    "top_products",
                ]
            )
        )

    if has_category:
        category_group_cols = [category_col, sub_category_col] if has_sub_category else [category_col]
        category_metric_columns = [sales_col]
        if has_profit:
            category_metric_columns.append(profit_col)
        if not has_sub_category:
            local_markdown.append(subcategory_missing_note)
        code_cells.append(
            "\n".join(
                [
                    f"category_sales = clean_df.groupby({category_group_cols!r}, as_index=False)[{category_metric_columns!r}].sum().sort_values('{sales_col}', ascending=False)",
                    *(
                        [
                            f"category_sales['profit_margin'] = category_sales['{profit_col}'] / category_sales['{sales_col}'].mask(category_sales['{sales_col}'] == 0)",
                        ]
                        if has_profit
                        else []
                    ),
                    "category_sales.head(20)",
                ]
            )
        )

    if has_category and has_sub_category and _strategy_includes(
        product_category_strategy,
        "category_treemap",
    ):
        code_cells.append(
            "\n".join(
                [
                    "if 'category_treemap' in product_view_map:",
                    "    fig = px.treemap(",
                    "        category_sales,",
                    "        path=['{0}', '{1}'],".format(category_col, sub_category_col),
                    "        values='{0}',".format(sales_col),
                    "",
                    "    )",
                    "    fig.show()",
                    "else:",
                    "    pass",
                ]
            )
        )

    if has_category and has_profit and (
        _strategy_includes(product_category_strategy, "category_profit_margin_bar")
        or _selection_includes(chart_selection_plan, "category_profit_margin_bar")
    ):
        code_cells.extend(
            [
                "\n".join(
                    [
                        f"category_profit_quality = clean_df.groupby('{category_col}', as_index=False)[['{sales_col}', '{profit_col}']].sum().sort_values('{profit_col}')",
                        f"category_profit_quality['profit_margin'] = category_profit_quality['{profit_col}'] / category_profit_quality['{sales_col}'].mask(category_profit_quality['{sales_col}'] == 0)",
                        "category_profit_quality",
                    ]
                ),
                "\n".join(
                    [
                        "# chart_id: category_profit_margin_bar",
                        "fig = px.bar(",
                        "    category_profit_quality.sort_values('profit_margin'),",
                        "    x='profit_margin',",
                        f"    y='{category_col}',",
                        "    orientation='h',",
                        f"    title={category_profit_margin_title},",
                        "    text='profit_margin',",
                        ")",
                        "fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')",
                        "fig.update_xaxes(tickformat='.0%')",
                        "fig.show()",
                    ]
                ),
            ]
        )

    if has_category and _selection_includes(chart_selection_plan, "category_concentration_bar"):
        code_cells.append(
            "\n".join(
                [
                    "# chart_id: category_concentration_bar",
                    "category_concentration = category_sales.copy().sort_values('{0}', ascending=False)".format(sales_col),
                    "category_concentration['sales_share'] = category_concentration['{0}'] / category_concentration['{0}'].sum()".format(sales_col),
                    "fig = px.bar(",
                    "    category_concentration.head(12),",
                    f"    x='{category_col}',",
                    "    y='sales_share',",
                    f"    title={category_concentration_title},",
                    ")",
                    "fig.update_yaxes(tickformat='.0%')",
                    f"fig.update_layout(xaxis_title={category_axis_label}, yaxis_title={sales_share_label})",
                    "fig.show()",
                ]
            )
        )

    if has_product and (
        _strategy_includes(product_category_strategy, "top_product_bar")
        or _selection_includes(chart_selection_plan, "product_category_bar")
    ):
        top_product_bar_lines = [
            "# chart_id: product_category_bar",
            "fig = px.bar(",
            "    top_products,",
            f"    x='{sales_col}',",
            f"    y='{product_col}',",
            "    orientation='h',",
            f"    title={top_product_sales_title},",
        ]
        if has_profit:
            top_product_bar_lines.extend(
                [
                    "    color='profit_margin',",
                    "    color_continuous_scale='RdYlGn',",
                ]
            )
        top_product_bar_lines.extend(
            [
                "",
                ")",
                "fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                "fig.show()",
            ]
        )
        code_cells.append("\n".join(top_product_bar_lines))

    if has_product and (
        _strategy_includes(product_category_strategy, "product_pareto")
        or _selection_includes(chart_selection_plan, "product_category_pareto")
    ):
        code_cells.append(
            "\n".join(
                [
                    "pareto_df = top_products.copy()",
                    "pareto_df['cumulative_share'] = pareto_df['{0}'].cumsum() / pareto_df['{0}'].sum()".format(
                        sales_col
                    ),
                    "fig, ax1 = plt.subplots(figsize=(10, 5))",
                    "ax1.bar(pareto_df['{0}'], pareto_df['{1}'], color='#4E79A7')".format(
                        product_col, sales_col
                    ),
                    "ax1.tick_params(axis='x', rotation=75)",
                    "ax2 = ax1.twinx()",
                    "ax2.plot(pareto_df['{0}'], pareto_df['cumulative_share'], color='#E15759', marker='o')".format(
                        product_col
                    ),
                    "ax2.axhline(0.8, color='#59A14F', linestyle='--', linewidth=1)",
                    "ax2.set_ylim(0, 1.05)",
                    f"plt.title({product_pareto_title})",
                    "plt.tight_layout()",
                    "plt.show()",
                ]
            )
        )

    if has_product and has_profit and _strategy_includes(
        product_category_strategy,
        "product_profit_bridge_scatter",
    ) or (
        has_product
        and has_profit
        and _selection_includes(chart_selection_plan, "product_profit_bridge_scatter")
    ):
        code_cells.append(
            "\n".join(
                [
                    "product_profit_bridge = top_products.copy()",
                    "fig = px.scatter(",
                    "    product_profit_bridge,",
                    f"    x='{sales_col}',",
                    f"    y='{profit_col}',",
                    "    size='sales_share',",
                    "    color='profit_margin',",
                    f"    hover_name='{product_col}',",
                    f"    title={product_bridge_title},",
                    "    color_continuous_scale='RdYlGn',",
                    ")",
                    "fig.show()",
                ]
            )
        )

    if has_category and (
        _selection_includes(chart_selection_plan, "category_concentration_bar")
        or _strategy_includes(product_category_strategy, "category_concentration_bar")
    ):
        code_cells.append(
            "\n".join(
                [
                    "category_concentration = category_sales.groupby('{0}', as_index=False)['{1}'].sum().sort_values('{1}', ascending=False)".format(
                        category_col, sales_col
                    ),
                    "category_concentration['sales_share'] = category_concentration['{0}'] / category_concentration['{0}'].sum()".format(
                        sales_col
                    ),
                    "fig = px.bar(",
                    "    category_concentration,",
                    f"    x='{category_col}',",
                    f"    y='{sales_col}',",
                    "    text='sales_share',",
                    f"    title={category_concentration_comparison_title},",
                    ")",
                    "fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')",
                    "fig.show()",
                ]
            )
        )

    if has_sub_category and has_profit and _selection_includes(chart_selection_plan, "subcategory_sales_profit_matrix"):
        code_cells.append(
            "\n".join(
                [
                    f"subcategory_matrix = clean_df.groupby('{sub_category_col}', as_index=False)[['{sales_col}', '{profit_col}']].sum().sort_values('{sales_col}', ascending=False).head(30)",
                    f"subcategory_matrix['profit_margin'] = subcategory_matrix['{profit_col}'] / subcategory_matrix['{sales_col}'].mask(subcategory_matrix['{sales_col}'] == 0)",
                    "fig = px.scatter(",
                    "    subcategory_matrix,",
                    f"    x='{sales_col}',",
                    f"    y='{profit_col}',",
                    "    color='profit_margin',",
                    f"    hover_name='{sub_category_col}',",
                    f"    title={subcategory_matrix_title},",
                    "    color_continuous_scale='RdYlGn',",
                    ")",
                    "fig.show()",
                ]
            )
        )

    if has_product and has_profit and (
        _strategy_includes(product_category_strategy, "high_sales_low_profit_bar")
        or _selection_includes(chart_selection_plan, "top_product_profit_gap_bar")
    ):
        code_cells.append(
            "\n".join(
                [
                    "# chart_id: top_product_profit_gap_bar",
                    f"profit_gap_source = clean_df[[{product_col!r}, {sales_col!r}, {profit_col!r}]].copy()",
                    f"profit_gap_source[{sales_col!r}] = pd.to_numeric(profit_gap_source[{sales_col!r}], errors='coerce')",
                    f"profit_gap_source[{profit_col!r}] = pd.to_numeric(profit_gap_source[{profit_col!r}], errors='coerce')",
                    f"profit_gap_source = profit_gap_source.dropna(subset=[{product_col!r}, {sales_col!r}, {profit_col!r}])",
                    "if profit_gap_source.empty:",
                    f"    print({profit_gap_missing_message})",
                    "else:",
                    f"    high_sales_low_profit_products = profit_gap_source.groupby({product_col!r}, as_index=False)[[{sales_col!r}, {profit_col!r}]].sum()",
                    f"    high_sales_low_profit_products = high_sales_low_profit_products[high_sales_low_profit_products[{sales_col!r}] > 0]",
                    f"    high_sales_low_profit_products['profit_margin'] = high_sales_low_profit_products[{profit_col!r}] / high_sales_low_profit_products[{sales_col!r}]",
                    f"    high_sales_low_profit_products = high_sales_low_profit_products.sort_values({sales_col!r}, ascending=False).head(20)",
                    "    if high_sales_low_profit_products.empty:",
                    f"        print({profit_gap_no_sales_message})",
                    "    else:",
                    f"        sales_threshold = high_sales_low_profit_products[{sales_col!r}].quantile(0.75)",
                    f"        focus_products = high_sales_low_profit_products[(high_sales_low_profit_products[{sales_col!r}] >= sales_threshold) & (high_sales_low_profit_products['profit_margin'] <= 0.08)].sort_values(['profit_margin', {sales_col!r}], ascending=[True, False])",
                    "        if focus_products.empty:",
                    f"            focus_products = high_sales_low_profit_products.sort_values(['profit_margin', {sales_col!r}], ascending=[True, False]).head(10)",
                    "        high_sales_low_profit_products = focus_products",
                    "        fig = px.bar(",
                    "            high_sales_low_profit_products,",
                    f"            x='{sales_col}',",
                    f"            y='{product_col}',",
                    "            color='profit_margin',",
                    "            orientation='h',",
                    f"            title={top_product_profit_gap_title},",
                    "            color_continuous_scale='RdYlGn',",
                    "        )",
                    "        fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                    "        fig.show()",
                ]
            )
        )

    chart_analysis = _chart_analysis_markdown(
        "product_and_category",
        report,
        output_language=output_language,
    )
    if chart_analysis:
        local_markdown.append(chart_analysis)
    return NotebookSectionContent(
        section_id="product_and_category",
        markdown_blocks=local_markdown,
        code_cells=code_cells,
    )


def _segment_region_section(
    *,
    markdown_blocks: list[str],
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    sales_col: str,
    quantity_col: str,
    profit_col: str,
    segment_region_strategy: dict[str, object] | None = None,
    chart_selection_plan: dict[str, object] | None = None,
    output_language: str | None = None,
) -> NotebookSectionContent:
    english = output_language is not None and is_english_output(output_language)
    primary_dimension, secondary_dimension = _pick_notebook_dimensions(schema_mapping)
    has_profit = _has_canonical_field(schema_mapping, "profit")
    has_quantity = _has_canonical_field(schema_mapping, "quantity")
    local_markdown = list(markdown_blocks)
    segment_margin_heatmap_note = (
        "Profit-margin heatmap helps locate weak segment-region combinations."
        if english
        else "利润率热力图用于定位利润质量偏弱的客群区域组合。"
    )
    segment_margin_heatmap_focus_title = (
        "Segment x Region Profit Margin Heatmap (Focused Slices)"
        if english
        else "客群 x 区域利润率热力图（重点切片）"
    )
    segment_margin_heatmap_title = (
        "Segment x Region Profit Margin Heatmap"
        if english
        else "客群 x 区域利润率热力图"
    )
    dense_heatmap_message = (
        "The heatmap matrix is dense, so only top sales rows and columns are shown and cell annotations are disabled for readability."
        if english
        else "热力图原始矩阵 {heatmap_original_shape} 较密，已展示销售额较高的 Top N 行列，并关闭单元格数字标注以提高可读性。"
    )
    segment_scatter_note = (
        "The scatter plot compares sales scale and profit to identify high-sales but weak-margin combinations."
        if english
        else "散点图进一步对照销售规模和利润，识别销售不低但利润偏弱的组合。"
    )
    weak_slices_note = (
        "The final table lists weak slices for follow-up review across category, discount, and fulfillment-cost details."
        if english
        else "最后列出弱势组合切片，作为后续下钻类目、折扣和履约成本的对象清单。"
    )
    weak_slices_title = (
        "High-Sales Low-Profit Segment-Region Slices"
        if english
        else "高销售低利润客群区域切片"
    )
    region_sales_profit_title = (
        "Sales and Profit by Region"
        if english
        else "区域销售与利润对比"
    )
    segment_summary_note = _nb_text(
        output_language,
        "This section aggregates sales, profit, and order count by available slices before comparing strong and weak combinations.",
        "本节先按可用切片汇总销售、利润和订单数，再判断强势与弱势组合是否来自同一类经营对象。",
    )
    segment_sales_profit_title = _nb_quote(
        output_language,
        "Sales and Profit by Segment",
        "客群销售与利润对比",
    )
    segment_region_sales_title = _nb_quote(
        output_language,
        "Sales by Segment and Region",
        "客群/区域销售额对比",
    )
    two_dimensional_slice_note = _nb_text(
        output_language,
        "Two-dimensional slices test whether sales are concentrated in specific segment and region combinations.",
        "双维切片用于确认销售额是否集中在特定客群与区域组合。",
    )

    if primary_dimension is None:
        local_markdown.append(
            "When profit or a second dimension is unavailable, this section falls back to sales comparison and single-dimension segment review."
            if english
            else "\u5f53\u524d\u7f3a\u5c11\u5229\u6da6\u5b57\u6bb5\u6216\u7b2c\u4e8c\u4e2a\u7ef4\u5ea6\u65f6\uff0c\u5148\u505a\u9500\u552e\u989d\u5bf9\u6bd4\u548c\u5355\u7ef4\u5ba2\u7fa4\u5206\u6790\u3002"
        )
        return NotebookSectionContent(
            section_id="segment_and_region",
            markdown_blocks=local_markdown,
            code_cells=[],
        )

    primary_col = _original_column(schema_mapping, primary_dimension, "Segment")
    secondary_col = (
        _original_column(schema_mapping, secondary_dimension, "Region")
        if secondary_dimension
        else None
    )
    group_cols = [primary_col] + ([secondary_col] if secondary_col else [])
    metric_columns = [sales_col]
    if has_profit:
        metric_columns.append(profit_col)
    if has_quantity:
        metric_columns.append(quantity_col)

    strategy_literal = repr(
        segment_region_strategy
        or {"source": "empty", "allowed_charts": [], "views": []}
    )
    code_cells = [
        "\n".join(
            [
                f"segment_region_strategy = {strategy_literal}",
                "segment_view_map = {item.get('chart'): item for item in segment_region_strategy.get('views', [])}",
                "segment_strategy_table = pd.DataFrame(segment_region_strategy.get('views', []))",
                "segment_strategy_table",
            ]
        )
    ]

    if secondary_col:
        local_markdown.append(
            (
                ""
                if has_profit
                else (
                    "Missing profit field; this section is downgraded to sales-only segment and region comparison."
                    if english
                    else "缺少利润字段，本节降级为客群/区域的销售额表现对比，不输出利润率或利润质量判断。"
                )
            )
        )
    else:
        local_markdown.append(
            (
                ""
                if has_profit
                else (
                    "Missing profit field; this section keeps a single-dimension sales comparison until profit is available."
                    if english
                    else "缺少利润字段，本节先保留单维度销售额表现对比，后续补齐利润后再判断低利润组合。"
                )
            )
        )

    code_cells.append(
        "\n".join(
            [
                f"segment_group_cols = {group_cols!r}",
                f"segment_metric_cols = {metric_columns!r}",
                (
                    "segment_region = clean_df.groupby(segment_group_cols, as_index=False, observed=False)"
                    "[segment_metric_cols].sum()"
                ),
                (
                    "segment_region_counts = clean_df.groupby(segment_group_cols, observed=False)"
                    ".size().reset_index(name='order_count')"
                ),
                "segment_region = segment_region.merge(segment_region_counts, on=segment_group_cols, how='left')",
                f"segment_region['sales_share'] = segment_region['{sales_col}'] / segment_region['{sales_col}'].sum()",
                (
                    f"segment_region['profit_margin'] = segment_region['{profit_col}'] "
                    f"/ segment_region['{sales_col}'].mask(segment_region['{sales_col}'] == 0)"
                    if has_profit
                    else (
                        "# Profit field is unavailable, so profit_margin is not calculated."
                        if english
                        else "# 当前缺少利润字段，暂不计算 profit_margin"
                    )
                ),
                "segment_region['slice_label'] = segment_region[segment_group_cols].astype(str).agg(' / '.join, axis=1)",
                f"segment_region.sort_values('{sales_col}', ascending=False).head(20)",
            ]
        )
    )

    code_cells.append(
        "\n".join(
            [
                f"primary_group_cols = ['{primary_col}']",
                (
                    "segment_profit_quality = clean_df.groupby(primary_group_cols, as_index=False, observed=False)"
                    "[segment_metric_cols].sum()"
                ),
                (
                    "segment_profit_counts = clean_df.groupby(primary_group_cols, observed=False)"
                    ".size().reset_index(name='order_count')"
                ),
                "segment_profit_quality = segment_profit_quality.merge(segment_profit_counts, on=primary_group_cols, how='left')",
                f"segment_profit_quality['sales_share'] = segment_profit_quality['{sales_col}'] / segment_profit_quality['{sales_col}'].sum()",
                (
                    f"segment_profit_quality['profit_margin'] = segment_profit_quality['{profit_col}'] "
                    f"/ segment_profit_quality['{sales_col}'].mask(segment_profit_quality['{sales_col}'] == 0)"
                    if has_profit
                    else (
                        "# Profit field is unavailable, so profit_margin is not calculated."
                        if english
                        else "# 当前缺少利润字段，暂不计算 profit_margin"
                    )
                ),
                f"segment_profit_quality.sort_values('{sales_col}', ascending=False)",
            ]
        )
    )

    if not has_profit or not secondary_col:
        local_markdown.append(
            "When profit or a second dimension is unavailable, this section falls back to sales comparison and single-dimension segment review."
            if english
            else "\u5f53\u524d\u7f3a\u5c11\u5229\u6da6\u5b57\u6bb5\u6216\u7b2c\u4e8c\u4e2a\u7ef4\u5ea6\u65f6\uff0c\u5148\u505a\u9500\u552e\u989d\u5bf9\u6bd4\u548c\u5355\u7ef4\u5ba2\u7fa4\u5206\u6790\u3002"
        )
    else:
        local_markdown.append(segment_summary_note)
    if (
        _strategy_includes(segment_region_strategy, "dimension_sales_bar")
        or _selection_includes(chart_selection_plan, "segment_region_sales_bar")
        or _selection_includes(chart_selection_plan, "segment_sales_profit_bar")
    ):
        dimension_bar_chart_id = (
            "segment_sales_profit_bar"
            if _selection_includes(chart_selection_plan, "segment_sales_profit_bar")
            else "segment_region_sales_bar"
        )
        dimension_bar_title = (
            segment_sales_profit_title
            if _selection_includes(chart_selection_plan, "segment_sales_profit_bar")
            else segment_region_sales_title
        )
        dimension_bar_lines = [
            f"# chart_id: {dimension_bar_chart_id}",
            "fig = px.bar(",
            f"    segment_profit_quality.sort_values('{sales_col}', ascending=True),",
            f"    x='{sales_col}',",
            f"    y='{primary_col}',",
            "    orientation='h',",
            f"    title={dimension_bar_title},",
        ]
        if has_profit:
            dimension_bar_lines.extend(
                [
                    "    color='profit_margin',",
                    "    color_continuous_scale='RdYlGn',",
                    "",
                ]
            )
        dimension_bar_lines.extend(
            [
                ")",
                "fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                "fig.show()",
            ]
        )
        code_cells.append("\n".join(dimension_bar_lines))

    if secondary_col:
        local_markdown.append(two_dimensional_slice_note)
        if _strategy_includes(segment_region_strategy, "segment_region_sales_heatmap"):
            code_cells.append(
                "\n".join(
                    [
                        (
                            f"sales_heatmap_df = segment_region.pivot(index='{primary_col}', "
                            f"columns='{secondary_col}', values='{sales_col}').fillna(0)"
                        ),
                        "plt.figure(figsize=(10, 5))",
                        "sns.heatmap(sales_heatmap_df, annot=True, fmt='.0f', cmap='YlGnBu')",
                        "",
                        "plt.tight_layout()",
                        "plt.show()",
                    ]
                )
            )

        if has_profit:
            local_markdown.append(segment_margin_heatmap_note)
            if _strategy_includes(
                segment_region_strategy, "segment_region_profit_margin_heatmap"
            ) or _selection_includes(chart_selection_plan, "segment_region_margin_heatmap"):
                code_cells.append(
                    "\n".join(
                        [
                            "# chart_id: segment_region_margin_heatmap",
                            (
                                f"margin_heatmap_df = segment_region.pivot(index='{primary_col}', "
                                f"columns='{secondary_col}', values='profit_margin').fillna(0)"
                            ),
                            "heatmap_original_shape = margin_heatmap_df.shape",
                            "heatmap_too_dense = heatmap_original_shape[0] * heatmap_original_shape[1] > 64 or heatmap_original_shape[0] > 8 or heatmap_original_shape[1] > 8",
                            "if heatmap_too_dense:",
                            "    top_rows = segment_region.groupby('{0}')['{1}'].sum().sort_values(ascending=False).head(8).index".format(primary_col, sales_col),
                            "    top_cols = segment_region.groupby('{0}')['{1}'].sum().sort_values(ascending=False).head(8).index".format(secondary_col, sales_col),
                            "    margin_heatmap_render = margin_heatmap_df.loc[margin_heatmap_df.index.intersection(top_rows), margin_heatmap_df.columns.intersection(top_cols)]",
                            "else:",
                            "    margin_heatmap_render = margin_heatmap_df",
                            "annotation_enabled = not heatmap_too_dense",
                            "heatmap_readability_trace = {'chart_id': 'segment_region_margin_heatmap', 'heatmap_readability_mode': 'top_n_no_annotation' if heatmap_too_dense else 'raw', 'original_shape': list(heatmap_original_shape), 'rendered_shape': list(margin_heatmap_render.shape), 'top_n_applied': bool(heatmap_too_dense), 'annotation_enabled': annotation_enabled, 'reason': 'Top N slices with hover/color only for readability' if heatmap_too_dense else 'raw heatmap'}",
                            "plt.figure(figsize=(max(10, margin_heatmap_render.shape[1] * 1.2), max(5, margin_heatmap_render.shape[0] * 0.55)))",
                            "sns.heatmap(margin_heatmap_render, annot=annotation_enabled, fmt='.1%', cmap='RdYlGn', center=0)",
                            f"plt.title({segment_margin_heatmap_focus_title!r} if heatmap_too_dense else {segment_margin_heatmap_title!r})",
                            "plt.xticks(rotation=35, ha='right')",
                            "plt.yticks(rotation=0)",
                            "plt.tight_layout()",
                            "plt.show()",
                            "if heatmap_too_dense:",
                            f"    print(f{dense_heatmap_message!r})",
                        ]
                    )
                )

            local_markdown.append(segment_scatter_note)
            if _strategy_includes(segment_region_strategy, "segment_region_bubble"):
                code_cells.append(
                    "\n".join(
                        [
                            "segment_region_bubble = segment_region.copy()",
                            "fig = px.scatter(",
                            "    segment_region_bubble,",
                            f"    x='{sales_col}',",
                            f"    y='{profit_col}',",
                            f"    size='{sales_col}',",
                            "    color='profit_margin',",
                            "    hover_name='slice_label',",
                            "    color_continuous_scale='RdYlGn',",
                            "",
                            ")",
                            "fig.show()",
                        ]
                    )
                )

    if has_profit:
        local_markdown.append(weak_slices_note)
        code_cells.append(
            "\n".join(
                [
                    "segment_weak_base = segment_region.copy()",
                    f"segment_sales_threshold = segment_weak_base['{sales_col}'].quantile(0.5)",
                    (
                        "weak_segment_region = segment_weak_base[(segment_weak_base['{0}'] >= segment_sales_threshold) "
                        "& ((segment_weak_base['{1}'] < 0) | (segment_weak_base['profit_margin'] <= 0.08))].copy()"
                    ).format(sales_col, profit_col),
                    "if weak_segment_region.empty:",
                    (
                        "    weak_segment_region = segment_weak_base.sort_values("
                        "['profit_margin', '{0}', '{1}'], ascending=[True, True, False]).head(10).copy()"
                    ).format(profit_col, sales_col),
                    "else:",
                    (
                        "    weak_segment_region = weak_segment_region.sort_values("
                        "['profit_margin', '{0}', '{1}'], ascending=[True, True, False]).head(10).copy()"
                    ).format(profit_col, sales_col),
                    "weak_segment_region",
                ]
            )
        )
        if _strategy_includes(segment_region_strategy, "weak_segment_region_bar") or _selection_includes(
            chart_selection_plan, "segment_region_low_margin_table_or_bar"
        ):
            code_cells.append(
                "\n".join(
                    [
                        "# chart_id: segment_region_low_margin_table_or_bar",
                        "fig = px.bar(",
                        "    weak_segment_region,",
                        f"    x='{sales_col}',",
                        "    y='slice_label',",
                        "    color='profit_margin',",
                        "    orientation='h',",
                        "    color_continuous_scale='RdYlGn',",
                        f"    title={weak_slices_title!r},",
                        ")",
                        "fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                        "fig.show()",
                    ]
                )
            )

    if has_profit and _selection_includes(chart_selection_plan, "region_sales_profit_bar"):
        region_col = _original_column(schema_mapping, "region", "Region")
        code_cells.extend(
            [
                "\n".join(
                    [
                        f"region_profit_quality = clean_df.groupby('{region_col}', as_index=False)[['{sales_col}', '{profit_col}']].sum().sort_values('{sales_col}', ascending=False)",
                        f"region_profit_quality['profit_margin'] = region_profit_quality['{profit_col}'] / region_profit_quality['{sales_col}'].mask(region_profit_quality['{sales_col}'] == 0)",
                        "region_profit_quality",
                    ]
                ),
                "\n".join(
                    [
                        "# chart_id: region_sales_profit_bar",
                        "fig = px.bar(",
                        "    region_profit_quality.sort_values('{0}', ascending=True),".format(sales_col),
                        f"    x='{sales_col}',",
                        f"    y='{region_col}',",
                        "    color='profit_margin',",
                        "    orientation='h',",
                        "    color_continuous_scale='RdYlGn',",
                        f"    title={region_sales_profit_title!r},",
                        ")",
                        "fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                        "fig.show()",
                    ]
                ),
            ]
        )

    chart_analysis = _chart_analysis_markdown(
        "segment_and_region",
        report,
        output_language=output_language,
    )
    if chart_analysis:
        local_markdown.append(chart_analysis)
    return NotebookSectionContent(
        section_id="segment_and_region",
        markdown_blocks=local_markdown,
        code_cells=code_cells,
    )


def _pick_notebook_dimensions(schema_mapping: SchemaMapping) -> tuple[str | None, str | None]:
    priority = ("segment", "region", "category", "state", "city", "country")
    available = [item for item in priority if item in schema_mapping.field_mapping.values()]
    primary = available[0] if available else None
    secondary = available[1] if len(available) > 1 else None
    return primary, secondary


def _fallback_section_content(
    section_id: str,
    narrative: NotebookNarrative,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    metric_distribution_strategy: dict[str, object] | None = None,
    sales_trend_strategy: dict[str, object] | None = None,
    product_category_strategy: dict[str, object] | None = None,
    segment_region_strategy: dict[str, object] | None = None,
    discount_profit_strategy: dict[str, object] | None = None,
    chart_selection_plan: dict[str, object] | None = None,
    output_language: str | None = None,
) -> NotebookSectionContent:
    return build_fallback_section_content(
        section_id=section_id,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        metric_distribution_strategy=metric_distribution_strategy,
        sales_trend_strategy=sales_trend_strategy,
        product_category_strategy=product_category_strategy,
        segment_region_strategy=segment_region_strategy,
        discount_profit_strategy=discount_profit_strategy,
        chart_selection_plan=chart_selection_plan,
        output_language=output_language,
        helpers=FallbackSectionHelpers(
            compact_narrative_blocks=_compact_narrative_blocks,
            original_column=_original_column,
            optional_original_column=_optional_original_column,
            nb_english=_nb_english,
            nb_text=_nb_text,
            nb_quote=_nb_quote,
            module_for_section=_module_for_section,
            chart_selection_markdown=_chart_selection_markdown,
            apply_chart_selection_to_section=_apply_chart_selection_to_section,
            chart_analysis_markdown=_chart_analysis_markdown,
            discount_profit_section=_discount_profit_section,
            product_category_section=_product_category_section,
            segment_region_section=_segment_region_section,
            selected_chart_ids_literal=_selected_chart_ids_literal,
            selection_includes=_selection_includes,
            without_non_llm_chart_display_cells=_without_non_llm_chart_display_cells,
            build_section_insight_markdown=build_section_insight_markdown,
            conclusions_section_content=_conclusions_section_content,
        ),
    )


def _fallback_content(
    outline: NotebookOutline,
    narrative: NotebookNarrative,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    analysis_plan: AnalysisPlan,
    llm_client=None,
    chart_selection_plan: dict[str, object] | None = None,
    evidence_pack: dict[str, object] | None = None,
    output_language: str | None = None,
) -> NotebookContentPlan:
    has_metric_distribution_section = any(
        section.section_id == "metric_distributions" for section in outline.sections
    )
    metric_distribution_strategy = (
        build_metric_distribution_strategy(
            _module_for_section("metric_distributions", report),
            llm_client=llm_client,
        )
        if has_metric_distribution_section
        else None
    )
    has_sales_trend_section = any(section.section_id == "sales_trends" for section in outline.sections)
    sales_trend_strategy = (
        build_sales_trend_strategy(
            _module_for_section("sales_trends", report),
            llm_client=llm_client,
        )
        if has_sales_trend_section
        else None
    )
    has_product_category_section = any(
        section.section_id == "product_and_category" for section in outline.sections
    )
    product_category_strategy = (
        build_product_category_strategy(
            _module_for_section("product_and_category", report),
            llm_client=llm_client,
        )
        if has_product_category_section
        else None
    )
    has_segment_region_section = any(
        section.section_id == "segment_and_region" for section in outline.sections
    )
    segment_region_strategy = (
        build_segment_region_strategy(
            _module_for_section("segment_and_region", report),
            llm_client=llm_client,
        )
        if has_segment_region_section
        else None
    )
    has_discount_profit_section = any(
        section.section_id == "discount_and_profit" for section in outline.sections
    )
    discount_profit_strategy = (
        build_discount_profit_strategy(
            _module_for_section("discount_and_profit", report),
            llm_client=llm_client,
        )
        if has_discount_profit_section
        else None
    )
    metric_distribution_strategy = _hide_non_llm_chart_choices(metric_distribution_strategy)
    sales_trend_strategy = _hide_non_llm_chart_choices(sales_trend_strategy)
    product_category_strategy = _hide_non_llm_chart_choices(product_category_strategy)
    segment_region_strategy = _hide_non_llm_chart_choices(segment_region_strategy)
    discount_profit_strategy = _hide_non_llm_chart_choices(discount_profit_strategy)
    return NotebookContentPlan(
        sections=[
            _fallback_section_content(
                section.section_id,
                narrative,
                report,
                schema_mapping,
                metric_distribution_strategy=(
                    metric_distribution_strategy
                    if section.section_id == "metric_distributions"
                    else None
                ),
                sales_trend_strategy=(
                    sales_trend_strategy
                    if section.section_id == "sales_trends"
                    else None
                ),
                product_category_strategy=(
                    product_category_strategy
                    if section.section_id == "product_and_category"
                    else None
                ),
                segment_region_strategy=(
                    segment_region_strategy
                    if section.section_id == "segment_and_region"
                    else None
                ),
                discount_profit_strategy=(
                    discount_profit_strategy
                    if section.section_id == "discount_and_profit"
                    else None
                ),
                chart_selection_plan=chart_selection_plan,
                output_language=output_language,
            )
            for section in outline.sections
        ]
    )


def _narrative_map(narrative: NotebookNarrative) -> dict[str, object]:
    return {section.section_id: section for section in narrative.sections}


def _section_module_ids(section_id: str) -> list[str]:
    mapping = {
        "data_cleaning": ["data_quality_check"],
        "sales_trends": ["sales_trend_analysis"],
        "product_and_category": ["product_contribution_analysis"],
        "segment_and_region": ["dimension_breakdown_analysis"],
        "discount_and_profit": ["discount_profit_analysis"],
        "modeling": ["loss_risk_modeling"],
        "forecast": ["forecast_analysis"],
        "conclusions": [
            "sales_trend_analysis",
            "product_contribution_analysis",
            "dimension_breakdown_analysis",
            "discount_profit_analysis",
            "forecast_analysis",
        ],
    }
    return mapping.get(section_id, [])


def _section_is_llm_eligible(section_id: str, llm_client=None) -> bool:
    eligible_sections = {
        "sales_trends",
        "product_and_category",
        "discount_and_profit",
        "forecast",
    }
    if section_id in {"segment_and_region", "country_market", "order_structure"}:
        # These sections already have stronger deterministic fallbacks, so only
        # ask the LLM to enrich them when a usable client is available.
        return bool(
            llm_client is not None
            and getattr(llm_client, "enabled", False)
            and getattr(llm_client, "large_section_payloads_supported", True)
        )
    return section_id in eligible_sections


def _section_report_context(
    section_id: str,
    report: AnalysisReport,
    evidence_pack: dict[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(evidence_pack, dict):
        return {
            "section_id": section_id,
            "evidence_pack": build_section_evidence_slice(evidence_pack, section_id),
        }
    module_ids = set(_section_module_ids(section_id))
    modules = [
        {
            "module_id": module.module_id,
            "title": module.title,
            "chart_type": module.chart_type,
            "summary_metrics": module.summary_metrics,
            "tables": {
                table_name: rows[:5]
                for table_name, rows in list(module.tables.items())[:8]
            },
            "findings": module.findings[:8],
            "warnings": module.warnings[:5],
        }
        for module in report.modules
        if module.module_id in module_ids
    ]
    return {
        "dataset_type": report.dataset_type,
        "summary": report.summary,
        "modules": modules,
    }


def _sanitize_section_content(
    payload: dict[str, object], fallback_section: NotebookSectionContent
) -> NotebookSectionContent:
    section_id = str(payload.get("section_id") or fallback_section.section_id).strip()
    if section_id != fallback_section.section_id:
        return fallback_section

    markdown_blocks = payload.get("markdown_blocks", [])
    if not isinstance(markdown_blocks, list):
        return fallback_section
    cleaned_markdown_blocks = _sanitize_markdown_blocks(markdown_blocks)
    if not cleaned_markdown_blocks:
        return fallback_section

    return NotebookSectionContent(
        section_id=fallback_section.section_id,
        markdown_blocks=cleaned_markdown_blocks,
        code_cells=fallback_section.code_cells,
    )


def _sanitize_section_content_model(
    section_content: NotebookSectionContent,
    fallback_section: NotebookSectionContent,
) -> NotebookSectionContent:
    if section_content.section_id != fallback_section.section_id:
        return fallback_section
    if not section_content.markdown_blocks:
        return fallback_section
    cleaned_markdown_blocks = _sanitize_markdown_blocks(list(section_content.markdown_blocks))
    if not cleaned_markdown_blocks:
        return fallback_section
    return NotebookSectionContent(
        section_id=fallback_section.section_id,
        markdown_blocks=cleaned_markdown_blocks,
        code_cells=fallback_section.code_cells,
    )


def build_notebook_content(
    outline: NotebookOutline,
    narrative: NotebookNarrative,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    analysis_plan: AnalysisPlan,
    llm_client=None,
    chart_selection_plan: dict[str, object] | None = None,
    evidence_pack: dict[str, object] | None = None,
) -> NotebookContentPlan:
    content, _ = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=llm_client,
        chart_selection_plan=chart_selection_plan,
        evidence_pack=evidence_pack,
    )
    return content


def build_notebook_content_with_trace(
    outline: NotebookOutline,
    narrative: NotebookNarrative,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    analysis_plan: AnalysisPlan,
    llm_client=None,
    chart_selection_plan: dict[str, object] | None = None,
    evidence_pack: dict[str, object] | None = None,
    output_language: str | None = None,
) -> tuple[NotebookContentPlan, LLMStageTrace]:
    metrics_snapshot = (
        llm_client.snapshot_completion_metrics()
        if llm_client is not None
        and getattr(llm_client, "enabled", False)
        and hasattr(llm_client, "snapshot_completion_metrics")
        else None
    )
    fallback = _repair_content_plan_mojibake(
        _fallback_content(
            outline,
            narrative,
            report,
            schema_mapping,
            analysis_plan,
            llm_client=llm_client,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
        )
    )
    fallback = _ensure_selected_chart_ids_cell(fallback, chart_selection_plan)
    fallback, narrative_guard_trace = _apply_unsupported_narrative_guard(fallback, schema_mapping)
    if output_language is not None and is_english_output(output_language):
        fallback = _englishize_content_plan(fallback)

    if llm_client is None or not getattr(llm_client, "enabled", False):
        trace = build_llm_stage_trace(
            stage="notebook_content",
            llm_client=llm_client,
            status="disabled",
            reason=(
                "LLM is disabled, so notebook sections used deterministic content blocks."
                + (
                    f" Narrative guard removed unsupported terms: {', '.join(narrative_guard_trace['removed_terms'])}."
                    if narrative_guard_trace["unsupported_narrative_removed"]
                    else ""
                )
            ),
        )
        return fallback, trace

    suggest_content = getattr(llm_client, "suggest_notebook_section_content", None)
    if suggest_content is None:
        subcalls = (
            llm_client.collect_completion_metrics_since(metrics_snapshot)
            if metrics_snapshot is not None and hasattr(llm_client, "collect_completion_metrics_since")
            else []
        )
        return fallback, build_aggregated_llm_stage_trace(
            stage="notebook_content",
            llm_client=llm_client,
            status="skipped",
            reason="The configured LLM client does not provide notebook section content generation.",
            subcalls=subcalls,
        )

    narrative_lookup = _narrative_map(narrative)
    built_sections: list[NotebookSectionContent] = []
    applied_count = 0
    no_change_count = 0
    fallback_count = 0
    skipped_count = 0
    error_messages: list[str] = []

    for section in outline.sections:
        fallback_section = next(
            item for item in fallback.sections if item.section_id == section.section_id
        )
        if not _section_is_llm_eligible(section.section_id, llm_client=llm_client):
            built_sections.append(fallback_section)
            skipped_count += 1
            continue
        try:
            args = [
                section,
                narrative_lookup.get(section.section_id),
                _section_report_context(section.section_id, report, evidence_pack),
                schema_mapping,
                analysis_plan,
                fallback_section.model_dump(),
            ]
            kwargs = {}
            if _accepts_keyword(suggest_content, "language_instruction"):
                kwargs["language_instruction"] = user_facing_language_instruction(output_language)
            llm_section = suggest_content(*args, **kwargs)
        except Exception as exc:
            built_sections.append(fallback_section)
            fallback_count += 1
            error_messages.append(f"{section.section_id}: {describe_llm_error(exc, limit=120)}")
            continue

        if isinstance(llm_section, NotebookSectionContent):
            result_section = _sanitize_section_content_model(llm_section, fallback_section)
        elif isinstance(llm_section, dict):
            result_section = _sanitize_section_content(llm_section, fallback_section)
        else:
            built_sections.append(fallback_section)
            fallback_count += 1
            error_messages.append(f"{section.section_id}: invalid_payload")
            continue

        result_section = _ensure_chart_analysis(result_section, report, output_language=output_language)
        fallback_section = _ensure_chart_analysis(fallback_section, report, output_language=output_language)

        if result_section == fallback_section:
            no_change_count += 1
        else:
            applied_count += 1
        built_sections.append(_apply_chart_selection_to_section(result_section, chart_selection_plan))

    if applied_count == 0 and no_change_count == 0:
        subcalls = (
            llm_client.collect_completion_metrics_since(metrics_snapshot)
            if metrics_snapshot is not None and hasattr(llm_client, "collect_completion_metrics_since")
            else []
        )
        return fallback, build_aggregated_llm_stage_trace(
            stage="notebook_content",
            llm_client=llm_client,
            status="fallback_on_error",
            reason=(
                "Notebook section content generation fell back to deterministic content blocks. "
                + ("; ".join(error_messages[:3]) if error_messages else "No enriched sections were produced.")
            ),
            subcalls=subcalls,
            attempted=True,
        )

    result = _ensure_selected_chart_ids_cell(
        _repair_content_plan_mojibake(NotebookContentPlan(sections=built_sections)),
        chart_selection_plan,
    )
    result, narrative_guard_trace = _apply_unsupported_narrative_guard(result, schema_mapping)
    if output_language is not None and is_english_output(output_language):
        result = _englishize_content_plan(result)
    if fallback_count == 0 and skipped_count == 0:
        status = "llm_applied"
        reason = (
            f"LLM enriched {applied_count} notebook section(s)"
            + (f" and validated {no_change_count} section(s) without structural changes." if no_change_count else ".")
        )
    else:
        status = "llm_partial"
        reason = (
            f"LLM enriched {applied_count} section(s), validated {no_change_count} unchanged section(s), fell back on {fallback_count} section(s), and skipped {skipped_count} section(s). "
            + ("; ".join(error_messages[:3]) if error_messages else "")
        ).strip()

    subcalls = (
        llm_client.collect_completion_metrics_since(metrics_snapshot)
        if metrics_snapshot is not None and hasattr(llm_client, "collect_completion_metrics_since")
        else []
    )
    if narrative_guard_trace["unsupported_narrative_removed"]:
        reason = (
            reason
            + f" Narrative guard removed unsupported terms: {', '.join(narrative_guard_trace['removed_terms'])}."
        )
    trace = build_aggregated_llm_stage_trace(
        stage="notebook_content",
        llm_client=llm_client,
        status=status,
        reason=reason,
        subcalls=subcalls,
        attempted=True,
        applied=True,
    )
    return result, trace
