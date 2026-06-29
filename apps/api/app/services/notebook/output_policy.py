from __future__ import annotations

import re

from app.schemas.notebook_content import NotebookSectionContent


NOTEBOOK_OUTPUT_MODES = {"compact", "full", "debug"}

COMPACT_SECTION_IDS = {
    "dataset_and_schema",
    "data_cleaning",
    "metric_distributions",
    "sales_trends",
    "product_and_category",
    "segment_and_region",
    "order_structure",
}

_SECTION_LIMITS = {
    "dataset_and_schema": (1, 0),
    "data_cleaning": (2, 0),
    "metric_distributions": (2, 1),
    "sales_trends": (1, 1),
    "product_and_category": (2, 1),
    "segment_and_region": (2, 1),
    "order_structure": (2, 1),
}

_HEAD_TAIL_RE = re.compile(r"\.(head|tail)\((?:1[1-9]|[2-9]\d+)\)")
_NLARGEST_RE = re.compile(r"\.nlargest\((?:1[1-9]|[2-9]\d+),")
_NSMALLEST_RE = re.compile(r"\.nsmallest\((?:1[1-9]|[2-9]\d+),")
_DOTTED_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.")
_BARE_DATA_ARG_RE = re.compile(r"\b(?:px|sns)\.[A-Za-z_][A-Za-z0-9_]*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,")
_ASSIGNMENT_TEMPLATE = r"(^|\n)\s*{name}\s*(?:=|\[)"
_IGNORED_REFERENCE_NAMES = {
    "ax",
    "fig",
    "go",
    "np",
    "pd",
    "pio",
    "plt",
    "px",
    "sns",
}


def normalize_notebook_output_mode(mode: str | None) -> str:
    normalized = str(mode or "compact").strip().lower()
    return normalized if normalized in NOTEBOOK_OUTPUT_MODES else "compact"


def is_compact_notebook_mode(mode: str | None) -> bool:
    return normalize_notebook_output_mode(mode) == "compact"


def strip_compact_markdown_heading(markdown: str, heading: str, mode: str | None) -> str:
    if not is_compact_notebook_mode(mode):
        return markdown
    lines = str(markdown or "").splitlines()
    if not lines or lines[0].strip() != heading:
        return markdown
    while len(lines) > 1 and not lines[1].strip():
        lines.pop(1)
    return "\n".join(lines[1:]).strip()


def compact_section_conclusion_markdown(markdown: str, mode: str | None) -> str:
    body = strip_compact_markdown_heading(markdown, "### 本节结论", mode)
    if not is_compact_notebook_mode(mode):
        return body
    if body.startswith("本节结论："):
        return body
    paragraphs = re.split(r"(\n\s*\n)", body, maxsplit=1)
    if not paragraphs or not paragraphs[0].strip():
        return body
    paragraphs[0] = "本节结论：" + paragraphs[0].lstrip()
    return "".join(paragraphs).strip()


def compacted_section_ids() -> set[str]:
    return set(COMPACT_SECTION_IDS)


def suppress_step_intro_for_section(section_id: str, notebook_output_mode: str | None) -> bool:
    return is_compact_notebook_mode(notebook_output_mode) and section_id in COMPACT_SECTION_IDS


def apply_notebook_output_policy(
    section_content: NotebookSectionContent,
    *,
    notebook_output_mode: str,
) -> NotebookSectionContent:
    mode = normalize_notebook_output_mode(notebook_output_mode)
    if mode != "compact" or section_content.section_id not in COMPACT_SECTION_IDS:
        return section_content

    return NotebookSectionContent(
        section_id=section_content.section_id,
        markdown_blocks=list(section_content.markdown_blocks),
        code_cells=_compact_code_cells(
            section_content.section_id,
            list(section_content.code_cells),
        ),
    )


def _compact_code_cells(section_id: str, code_cells: list[str]) -> list[str]:
    if not code_cells:
        return []

    table_budget, chart_budget = _SECTION_LIMITS.get(section_id, (2, 1))
    selected_indices: list[int] = []

    required_indices = set(_required_indices(section_id, code_cells))
    required_indices.update(
        index for index, cell in enumerate(code_cells) if _is_chart_cell(cell) and _has_chart_id_marker(cell)
    )
    for index in sorted(required_indices):
        if index not in selected_indices:
            selected_indices.append(index)

    non_chart_count = sum(1 for index in selected_indices if not _is_chart_cell(code_cells[index]))
    chart_count = sum(1 for index in selected_indices if _is_chart_cell(code_cells[index]))

    for index, cell in enumerate(code_cells):
        if index in selected_indices:
            continue
        if _is_chart_cell(cell):
            if chart_count >= chart_budget:
                continue
            selected_indices.append(index)
            chart_count += 1
            continue
        if non_chart_count < table_budget:
            selected_indices.append(index)
            non_chart_count += 1

    dependency_indices = _dependency_indices_for_selected_charts(code_cells, selected_indices)
    selected_indices = sorted(set(selected_indices) | dependency_indices)
    selected_indices = _trim_optional_table_cells(
        code_cells,
        selected_indices,
        table_budget=table_budget,
        protected_indices=required_indices | dependency_indices,
    )
    if not selected_indices:
        selected_indices = [0]
    return [_limit_display_rows(code_cells[index]) for index in selected_indices]


def _required_indices(section_id: str, code_cells: list[str]) -> list[int]:
    markers_by_section = {
        "dataset_and_schema": ["pd.read_csv"],
        "data_cleaning": ["clean_df = df.copy()", "missing_summary"],
        "metric_distributions": ["metric_strategy", "metric_quantiles", "describe()"],
        "sales_trends": ["monthly_sales ="],
        "product_and_category": ["top_products ="],
        "segment_and_region": ["segment_region =", "segment_view =", "top_segment_region"],
        "order_structure": ["order_structure", "order_summary", "order_lines", "order_id_col"],
    }
    markers = markers_by_section.get(section_id, [])
    indices: list[int] = []
    for marker in markers:
        for index, cell in enumerate(code_cells):
            if marker in cell:
                indices.append(index)
                break
    return indices


def _is_chart_cell(code_cell: str) -> bool:
    return "fig.show()" in code_cell or "plt.show()" in code_cell


def _has_chart_id_marker(code_cell: str) -> bool:
    return "# chart_id:" in code_cell


def _dependency_indices_for_selected_charts(code_cells: list[str], selected_indices: list[int]) -> set[int]:
    dependencies: set[int] = set()
    for index in selected_indices:
        if not _is_chart_cell(code_cells[index]):
            continue
        for dependency_index in _chart_dependency_indices(code_cells, chart_index=index):
            dependencies.add(dependency_index)
    return dependencies


def _trim_optional_table_cells(
    code_cells: list[str],
    selected_indices: list[int],
    *,
    table_budget: int,
    protected_indices: set[int],
) -> list[int]:
    selected = set(selected_indices)
    while True:
        table_indices = sorted(index for index in selected if not _is_chart_cell(code_cells[index]))
        if len(table_indices) <= table_budget:
            break
        optional = [index for index in table_indices if index not in protected_indices]
        if not optional:
            break
        selected.remove(optional[-1])
    return sorted(selected)


def _chart_dependency_indices(code_cells: list[str], *, chart_index: int) -> list[int]:
    chart_cell = code_cells[chart_index]
    referenced_names = {
        name
        for name in _DOTTED_NAME_RE.findall(chart_cell)
        if name not in _IGNORED_REFERENCE_NAMES
    }
    referenced_names.update(
        name for name in _BARE_DATA_ARG_RE.findall(chart_cell) if name not in _IGNORED_REFERENCE_NAMES
    )
    dependency_indices: list[int] = []
    for name in referenced_names:
        assignment_re = re.compile(_ASSIGNMENT_TEMPLATE.format(name=re.escape(name)))
        for index in range(chart_index - 1, -1, -1):
            if assignment_re.search(code_cells[index]):
                dependency_indices.append(index)
                break
    return dependency_indices


def _limit_display_rows(code_cell: str) -> str:
    limited = _HEAD_TAIL_RE.sub(lambda match: f".{match.group(1)}(10)", code_cell)
    limited = _NLARGEST_RE.sub(".nlargest(10,", limited)
    limited = _NSMALLEST_RE.sub(".nsmallest(10,", limited)
    return limited
