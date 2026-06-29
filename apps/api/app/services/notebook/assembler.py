from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from nbformat.v4 import new_code_cell, new_markdown_cell

from app.core.config import get_settings
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook.markdown_sanitizer import (
    remove_template_discourse_markers,
    sanitize_final_markdown_text,
)
from app.services.notebook.modeling_renderer import render_modeling_section_cells
from app.services.notebook.output_policy import (
    apply_notebook_output_policy,
    compact_section_conclusion_markdown,
    is_compact_notebook_mode,
    normalize_notebook_output_mode,
    suppress_step_intro_for_section,
)
from app.services.notebook.report_utils import module_map, narrative_map
from app.services.notebook.section_renderer import (
    code_step_intro,
    prepare_final_section_markdown_blocks,
    render_section_cells,
    section_conclusion_markdown,
    section_has_llm_chart_decision,
)
from app.services.notebook.summary_builder import (
    build_final_conclusion_markdown,
    build_kaggle_analysis_brief_markdown,
)
from app.services.notebook_toolset import new_analysis_notebook, save_notebook
from app.services.notebook_narrative_guard import remove_unsupported_narrative
from app.services.final_synthesis_guard import guard_english_public_narrative
from app.services.output_language import (
    ENGLISH_SECTION_TITLES,
    contains_cjk,
    englishize_common_artifact_text,
    englishize_notebook_text,
    is_english_output,
)


TOC_ITEMS = [
    ("分析目标", "section-title_and_goal"),
    ("数据集与字段说明", "section-dataset_and_schema"),
    ("数据清洗与预处理", "section-data_cleaning"),
    ("指标分布与异常值分析", "section-metric_distributions"),
    ("销售趋势分析", "section-sales_trends"),
    ("商品与类目分析", "section-product_and_category"),
    ("客群与区域分析", "section-segment_and_region"),
    ("订单结构分析", "section-order_structure"),
    ("折扣与利润分析", "section-discount_and_profit"),
    ("建模分析", "section-modeling"),
    ("结论与行动建议", "section-conclusions"),
]


def _toc_items(output_language: str | None = None) -> list[tuple[str, str]]:
    if not is_english_output(output_language):
        return TOC_ITEMS
    return [
        (ENGLISH_SECTION_TITLES.get(anchor.replace("section-", ""), (title, ""))[0], anchor)
        for title, anchor in TOC_ITEMS
    ]


def build_toc_markdown(
    items: list[tuple[str, str]] | None = None,
    output_language: str | None = None,
) -> str:
    lines = ["## Table of Contents" if is_english_output(output_language) else "## 目录", ""]
    lines.extend(f"- [{title}](#{anchor})" for title, anchor in (items or _toc_items(output_language)))
    return "\n".join(lines)


def with_anchor(markdown: str, anchor: str) -> str:
    if not markdown.strip():
        return markdown
    marker = f'<a id="{anchor}"></a>'
    if marker in markdown:
        return markdown
    return f"{markdown}\n\n{marker}"


def section_anchor(section_id: str) -> str:
    return f"section-{section_id}"


def _is_empty_action_plan_markdown(markdown: str) -> bool:
    lines = [line.strip() for line in str(markdown or "").splitlines() if line.strip()]
    if not lines or lines[0] not in {"## Action Plan", "# Action Plan"}:
        return False
    ignorable = {
        '<a id="action-plan"></a>',
        "| priority | issue | evidence | action | required_fields |",
        "| --- | --- | --- | --- | --- |",
    }
    return all(line in ignorable for line in lines[1:])


def _remove_action_plan_toc_link(markdown: str) -> str:
    lines = [
        line
        for line in str(markdown or "").splitlines()
        if line.strip() != "- [Action Plan](#action-plan)"
    ]
    return "\n".join(lines).strip()


def _final_conclusion_markdown(
    section: Any | None,
    *,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None,
    analysis_focus: dict[str, Any] | None,
    evidence_pack: dict[str, Any] | None,
    narrative_section: Any | None = None,
    content_section: Any | None = None,
    modeling_outcome: dict[str, Any] | None = None,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    final_synthesis: dict[str, Any] | None = None,
    output_language: str | None = None,
) -> str:
    observations = list(getattr(narrative_section, "key_observations", []) or [])
    intro = getattr(narrative_section, "intro", None)
    takeaway = getattr(narrative_section, "business_takeaway", None)
    has_narrative = bool(intro or observations or takeaway)
    content_blocks = [] if has_narrative else list(getattr(content_section, "markdown_blocks", []) or [])
    markdown = build_final_conclusion_markdown(
        report,
        schema_mapping,
        dataset_profile,
        analysis_focus,
        evidence_pack,
        narrative_intro=intro,
        narrative_observations=observations,
        narrative_takeaway=takeaway,
        content_blocks=content_blocks,
        modeling_outcome=modeling_outcome,
        modeling_outcome_interpretation=modeling_outcome_interpretation,
        final_synthesis=final_synthesis,
        output_language=output_language,
    )
    anchor = section_anchor(getattr(section, "section_id", "conclusions"))
    return with_anchor(markdown, anchor)


def add_notebook_toc(
    notebook: Any,
    report: AnalysisReport,
    output_language: str | None = None,
) -> None:
    if notebook.cells and getattr(notebook.cells[0], "cell_type", None) == "markdown":
        notebook.cells[0].source = with_anchor(str(notebook.cells[0].source), "notebook-top")
    existing_markdown = "\n\n".join(
        str(cell.source)
        for cell in notebook.cells
        if getattr(cell, "cell_type", None) == "markdown"
    )
    toc_source_items = _toc_items(output_language)
    toc_items = [
        (title, anchor)
        for title, anchor in toc_source_items
        if f'<a id="{anchor}"></a>' in existing_markdown
        or anchor == "section-title_and_goal"
    ]
    insert_index = 1
    for index, cell in enumerate(notebook.cells):
        if getattr(cell, "cell_type", None) == "markdown" and "## Notebook 分析摘要" in str(cell.source):
            insert_index = index + 1
            break
    notebook.cells.insert(insert_index, new_markdown_cell(build_toc_markdown(toc_items, output_language)))
    has_goal_anchor = any(
        getattr(cell, "cell_type", None) == "markdown"
        and '<a id="section-title_and_goal"></a>' in str(cell.source)
        for cell in notebook.cells
    )
    if not has_goal_anchor:
        goal_markdown = with_anchor(
            "\n".join(
                [
                    "## Analysis Goal" if is_english_output(output_language) else "## 分析目标",
                    "",
                    (
                        f"This notebook analyzes the `{report.dataset_type}` dataset and turns sales, profit, discount, and priority slices into actionable business judgment."
                        if is_english_output(output_language)
                        else f"本报告围绕 `{report.dataset_type}` 数据集展开，目标是把销售、利润、折扣和重点切片转化为可执行的经营判断。"
                    ),
                ]
            ),
            "section-title_and_goal",
        )
        notebook.cells.insert(insert_index + 1, new_markdown_cell(goal_markdown))


def _apply_final_narrative_guard(
    notebook: Any,
    *,
    mapped_fields: set[str],
    schema_mapping: SchemaMapping | None,
) -> dict[str, Any]:
    guarded_cells: list[Any] = []
    removed_any = False
    dropped_empty_markdown_cells = 0
    dropped_empty_action_plan_cells = 0
    for cell in notebook.cells:
        if getattr(cell, "cell_type", None) != "markdown":
            guarded_cells.append(cell)
            continue
        guarded_source, trace = remove_unsupported_narrative(
            str(cell.source),
            mapped_fields=mapped_fields,
        )
        sanitized = sanitize_final_markdown_text(guarded_source, schema_mapping)
        removed_any = removed_any or bool(trace.get("unsupported_narrative_removed"))
        if _is_empty_action_plan_markdown(sanitized):
            dropped_empty_action_plan_cells += 1
            continue
        if not sanitized.strip():
            dropped_empty_markdown_cells += 1
            continue
        cell.source = sanitized
        guarded_cells.append(cell)
    if dropped_empty_action_plan_cells:
        for cell in guarded_cells:
            if getattr(cell, "cell_type", None) == "markdown" and "#action-plan" in str(cell.source):
                cell.source = _remove_action_plan_toc_link(str(cell.source))
    notebook.cells = guarded_cells
    return {
        "unsupported_narrative_removed": removed_any,
        "dropped_empty_markdown_cells_count": dropped_empty_markdown_cells,
        "dropped_empty_action_plan_cells_count": dropped_empty_action_plan_cells,
    }


def _guard_english_notebook_public_line(line: str, schema_mapping: SchemaMapping) -> str:
    bullet_match = re.match(r"^(\s*(?:[-*+]|\d+[.)])\s+)(.+)$", line)
    prefix = ""
    text = line
    if bullet_match:
        prefix = bullet_match.group(1)
        text = bullet_match.group(2)
    guarded = guard_english_public_narrative(
        text,
        schema_mapping=schema_mapping,
        role="notebook_markdown",
        fallback="Review the relationship against order-level evidence before broader operating changes.",
    )
    return f"{prefix}{guarded}" if guarded else ""


def _guard_english_notebook_public_markdown(markdown: str, schema_mapping: SchemaMapping) -> str:
    guarded_lines: list[str] = []
    in_code_fence = False
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            guarded_lines.append(line)
            continue
        if (
            in_code_fence
            or not stripped
            or stripped.startswith("#")
            or stripped.startswith("<a id=")
            or stripped.startswith("|")
            or stripped.startswith("Task ID:")
            or (stripped.startswith("- [") and "](#" in stripped)
        ):
            guarded_lines.append(line)
            continue
        guarded_lines.append(_guard_english_notebook_public_line(line, schema_mapping))
    return "\n".join(guarded_lines).strip()


def _englishize_notebook_cells(notebook: Any, schema_mapping: SchemaMapping) -> None:
    for cell in notebook.cells:
        if getattr(cell, "cell_type", None) == "markdown":
            cell.source = _guard_english_notebook_public_markdown(
                englishize_notebook_text(str(cell.source)),
                schema_mapping,
            )
        elif getattr(cell, "cell_type", None) == "code":
            cell.source = englishize_notebook_text(str(cell.source))


def _english_step_intro(section_title: str) -> str:
    return f"Run the next analysis step for {section_title} and inspect the generated output."


def _english_section_takeaway(section_title: str) -> str:
    return (
        "### Section Takeaway\n\n"
        f"This section provides supporting evidence for {section_title}. "
        "Validate the chart and table outputs with business context before acting."
    )


def _english_chart_commentary(section_title: str) -> str:
    return (
        "### Chart Commentary\n\n"
        f"Use the chart and table outputs in this section to review {section_title}. "
        "Compare the strongest and weakest slices before turning the result into an action."
    )


def build_notebook(
    task_id: str,
    output_dir: Path,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    plan: AnalysisPlan,
    outline: NotebookOutline | None = None,
    narrative: NotebookNarrative | None = None,
    content_plan: NotebookContentPlan | None = None,
    dataset_profile: dict[str, Any] | None = None,
    analysis_focus: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
    modeling_interpretation: dict[str, str] | None = None,
    modeling_opportunity_decision: dict[str, str] | None = None,
    modeling_outcome: dict[str, Any] | None = None,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    final_synthesis: dict[str, Any] | None = None,
    notebook_output_mode: str | None = None,
    output_language: str | None = None,
) -> Path:
    resolved_output_mode = normalize_notebook_output_mode(
        notebook_output_mode if notebook_output_mode is not None else get_settings().notebook_output_mode
    )
    if content_plan is not None and outline is not None:
        notebook = new_analysis_notebook(outline.title if outline.title else None, task_id=task_id)
        notebook.cells.append(new_markdown_cell(build_kaggle_analysis_brief_markdown(
            report,
            dataset_profile,
            analysis_focus,
            evidence_pack,
            final_synthesis=final_synthesis,
            output_language=output_language,
        )))
        content_lookup = {section.section_id: section for section in content_plan.sections}
        module_lookup = module_map(report)
        narrative_lookup = narrative_map(narrative)
        for section in outline.sections:
            if section.section_id == "conclusions":
                notebook.cells.append(
                    new_markdown_cell(
                        _final_conclusion_markdown(
                            section,
                            report=report,
                            schema_mapping=schema_mapping,
                            dataset_profile=dataset_profile,
                            analysis_focus=analysis_focus,
                            evidence_pack=evidence_pack,
                            narrative_section=narrative_lookup.get("conclusions"),
                            content_section=content_lookup.get("conclusions"),
                            modeling_outcome=modeling_outcome,
                            modeling_outcome_interpretation=modeling_outcome_interpretation,
                            final_synthesis=final_synthesis,
                            output_language=output_language,
                        )
                    )
                )
                continue
            if section.section_id == "modeling":
                modeling_cells = render_modeling_section_cells(
                    module_lookup.get("loss_risk_modeling"),
                    modeling_interpretation=modeling_interpretation,
                    modeling_opportunity_decision=modeling_opportunity_decision,
                    modeling_outcome=modeling_outcome,
                    forecast_module=module_lookup.get("forecast_analysis"),
                    modeling_outcome_interpretation=modeling_outcome_interpretation,
                    output_language=output_language,
                )
                if modeling_cells and getattr(modeling_cells[0], "cell_type", None) == "markdown":
                    modeling_cells[0].source = with_anchor(str(modeling_cells[0].source), section_anchor(section.section_id))
                notebook.cells.extend(modeling_cells)
                continue
            section_content = content_lookup.get(section.section_id)
            if section_content is None:
                continue
            section_content = apply_notebook_output_policy(
                section_content,
                notebook_output_mode=resolved_output_mode,
            )
            intro_blocks, analysis_blocks, section_conclusions = prepare_final_section_markdown_blocks(
                section_content.markdown_blocks,
                section,
                schema_mapping,
                output_language=output_language,
            )
            if intro_blocks:
                notebook.cells.append(
                    new_markdown_cell(
                        with_anchor(
                            f"## {section.title}\n\n" + "\n\n".join(intro_blocks),
                            section_anchor(section.section_id),
                        )
                    )
                )
            previous_step_intro = ""
            for code_cell in section_content.code_cells:
                if not suppress_step_intro_for_section(section.section_id, resolved_output_mode):
                    step_intro = code_step_intro(section.section_id, code_cell, output_language=output_language)
                    if is_english_output(output_language) and step_intro:
                        step_intro = englishize_notebook_text(step_intro)
                        if contains_cjk(step_intro):
                            step_intro = _english_step_intro(section.title)
                    if step_intro and step_intro != previous_step_intro:
                        notebook.cells.append(new_markdown_cell(step_intro))
                        previous_step_intro = step_intro
                notebook.cells.append(new_code_cell(code_cell))
            if analysis_blocks and section_has_llm_chart_decision(section_content.code_cells):
                for analysis_block in analysis_blocks:
                    notebook.cells.append(new_markdown_cell(analysis_block))
            section_conclusion = section_conclusions[0] if section_conclusions else section_conclusion_markdown(section, report, schema_mapping, output_language=output_language)
            if section_conclusion:
                if is_english_output(output_language):
                    section_conclusion = englishize_notebook_text(section_conclusion)
                    if contains_cjk(section_conclusion):
                        section_conclusion = ""
                section_conclusion = compact_section_conclusion_markdown(
                    section_conclusion,
                    resolved_output_mode,
                )
                section_conclusion = sanitize_final_markdown_text(section_conclusion, schema_mapping)
                section_conclusion = remove_template_discourse_markers(section_conclusion)
                notebook.cells.append(new_markdown_cell(section_conclusion))
    elif outline is None:
        notebook = new_analysis_notebook("Sales Analysis Notebook", task_id=task_id)
        notebook.cells.extend(
            [
            new_markdown_cell(build_kaggle_analysis_brief_markdown(
                report,
                dataset_profile,
                analysis_focus,
                evidence_pack,
                final_synthesis=final_synthesis,
                output_language=output_language,
            )),
            new_markdown_cell(
                "## Schema Mapping\n\n```json\n"
                + schema_mapping.model_dump_json(indent=2)
                + "\n```"
            ),
            new_markdown_cell(
                "## Analysis Plan\n\n```json\n" + plan.model_dump_json(indent=2) + "\n```"
            ),
            new_code_cell("import pandas as pd\ndf = pd.read_csv('raw.csv')\ndf.head()"),
            ]
        )

        for module in report.modules:
            notebook.cells.append(new_markdown_cell(f"## {module.title}"))
            notebook.cells.append(
                new_markdown_cell(
                    "### Findings\n\n" + "\n".join(f"- {item}" for item in module.findings)
                )
            )
            notebook.cells.append(
                new_code_cell(
                    "# Structured module output\n"
                    f"module_output = {module.model_dump()!r}\n"
                    "module_output"
                )
            )

        notebook.cells.append(
            new_markdown_cell(
                _final_conclusion_markdown(
                    None,
                    report=report,
                    schema_mapping=schema_mapping,
                    dataset_profile=dataset_profile,
                    analysis_focus=analysis_focus,
                    evidence_pack=evidence_pack,
                    modeling_outcome=modeling_outcome,
                    modeling_outcome_interpretation=modeling_outcome_interpretation,
                    final_synthesis=final_synthesis,
                    output_language=output_language,
                )
            )
        )
    else:
        notebook = new_analysis_notebook(outline.title if outline.title else None, task_id=task_id)
        notebook.cells.append(new_markdown_cell(build_kaggle_analysis_brief_markdown(
            report,
            dataset_profile,
            analysis_focus,
            evidence_pack,
            final_synthesis=final_synthesis,
            output_language=output_language,
        )))
        module_lookup = module_map(report)
        narrative_lookup = narrative_map(narrative)
        for section in outline.sections:
            if section.section_id == "conclusions":
                notebook.cells.append(
                    new_markdown_cell(
                        _final_conclusion_markdown(
                            section,
                            report=report,
                            schema_mapping=schema_mapping,
                            dataset_profile=dataset_profile,
                            analysis_focus=analysis_focus,
                            evidence_pack=evidence_pack,
                            narrative_section=narrative_lookup.get("conclusions"),
                            modeling_outcome=modeling_outcome,
                            modeling_outcome_interpretation=modeling_outcome_interpretation,
                            final_synthesis=final_synthesis,
                            output_language=output_language,
                        )
                    )
                )
                continue
            section_cells = render_section_cells(
                section,
                module_lookup,
                narrative_lookup,
                report,
                schema_mapping,
                plan,
                modeling_interpretation=modeling_interpretation,
                modeling_outcome=modeling_outcome,
                modeling_outcome_interpretation=modeling_outcome_interpretation,
                output_language=output_language,
            )
            if section_cells and getattr(section_cells[0], "cell_type", None) == "markdown":
                section_cells[0].source = with_anchor(str(section_cells[0].source), section_anchor(section.section_id))
            notebook.cells.extend(section_cells)

    add_notebook_toc(notebook, report, output_language=output_language)
    mapped_fields = set(schema_mapping.field_mapping.values())
    _apply_final_narrative_guard(
        notebook,
        mapped_fields=mapped_fields,
        schema_mapping=schema_mapping,
    )
    if is_english_output(output_language):
        _englishize_notebook_cells(notebook, schema_mapping)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "analysis.ipynb"
    return save_notebook(notebook, path)
