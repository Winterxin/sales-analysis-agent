from __future__ import annotations

import nbformat

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan, NotebookSectionContent
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook.assembler import build_notebook, build_toc_markdown
from app.services.notebook.section_renderer import (
    prepare_final_section_markdown_blocks,
    section_conclusion_markdown,
)
from app.services.notebook.markdown_sanitizer import trim_markdown_paragraph


def test_english_notebook_omits_generic_filler_when_section_has_no_analysis_blocks(tmp_path) -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=0,
        summary=[],
        modules=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
    )
    outline = NotebookOutline(
        title="Sales Data Analysis Notebook",
        sections=[
            NotebookSection(
                section_id="custom_sales_check",
                title="Sales Check",
                purpose="Review sales output.",
            )
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="custom_sales_check",
                markdown_blocks=[],
                code_cells=["print('sales check')"],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="tatest-api-key",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(analysis_plan=["sales_check"], chart_preferences={}),
        outline=outline,
        content_plan=content_plan,
        output_language="en",
    )

    nb = nbformat.read(notebook_path, as_version=4)
    combined = "\n\n".join(str(cell.source) for cell in nb.cells)

    assert "Sales Check" in combined
    assert "print('sales check')" in combined
    assert "### Chart Commentary" not in combined
    assert "This section provides supporting evidence" not in combined


def test_english_section_renderer_suppresses_generic_takeaway_without_real_finding() -> None:
    section = NotebookSection(
        section_id="segment_and_region",
        title="Segment and Region Analysis",
        purpose="Review segment and region output.",
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount"},
        confidence=0.9,
    )
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=0,
        summary=[],
        modules=[],
    )

    conclusion = section_conclusion_markdown(section, report, schema_mapping, output_language="en")
    _intro_blocks, _analysis_blocks, conclusion_blocks = prepare_final_section_markdown_blocks(
        ["### 本节结论\n\n暂无明确业务结论。"],
        section,
        schema_mapping,
        output_language="en",
    )
    combined = "\n\n".join(item for item in [conclusion, *conclusion_blocks] if item)

    assert conclusion is None
    assert conclusion_blocks == []
    assert "### Section Takeaway" not in combined
    assert "mapped fields and module outputs define the current review scope" not in combined
    assert "This section identifies the records and business objects that need follow-up review" not in combined


def test_english_section_renderer_preserves_real_takeaway_content() -> None:
    section = NotebookSection(
        section_id="segment_and_region",
        title="Segment and Region Analysis",
        purpose="Review segment and region output.",
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount", "Segment": "segment", "Region": "region"},
        confidence=0.9,
    )
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="Segment and Region Analysis",
                chart_type="bar",
                findings=["West region contributed the largest sales amount in the current period."],
            )
        ],
    )

    conclusion = section_conclusion_markdown(section, report, schema_mapping, output_language="en")
    _intro_blocks, _analysis_blocks, conclusion_blocks = prepare_final_section_markdown_blocks(
        ["### Business Takeaway\n\nSales concentration is highest in the top three SKUs."],
        section,
        schema_mapping,
        output_language="en",
    )
    combined = "\n\n".join(item for item in [conclusion, *conclusion_blocks] if item)

    assert "### Section Takeaway" in combined
    assert "West region contributed the largest sales amount in the current period." in combined
    assert "Sales concentration is highest in the top three SKUs." in combined
    assert "mapped fields and module outputs define the current review scope" not in combined


def test_english_section_renderer_filters_template_intro_and_uses_specific_intro() -> None:
    section = NotebookSection(
        section_id="product_and_category",
        title="Product and Category Analysis",
        purpose="Summarize key findings, risks, and next actions.",
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount", "Product": "product_name"},
        confidence=0.9,
    )

    intro_blocks, _analysis_blocks, _conclusion_blocks = prepare_final_section_markdown_blocks(
        [
            "This section reviews product and category analysis using the mapped sales dataset and available module evidence.",
            "Section purpose: summarize key findings, risks, and next actions.",
        ],
        section,
        schema_mapping,
        output_language="en",
    )

    combined = "\n\n".join(intro_blocks)

    assert "This section reviews" not in combined
    assert "Section purpose:" not in combined
    assert "mapped sales dataset" not in combined
    assert "available module evidence" not in combined
    assert "Compare product and category contribution" in combined


def test_english_markdown_trim_uses_word_boundary_without_ellipsis() -> None:
    text = (
        "Product concentration spans 424 SKUs and the top records should be reviewed "
        "before expanding promotion budgets across the long tail."
    )

    trimmed = trim_markdown_paragraph(text, max_chars=70)

    assert trimmed.endswith(".")
    assert "…" not in trimmed
    assert "distri." not in trimmed
    assert not trimmed.endswith("revie.")


def test_chinese_section_renderer_keeps_existing_takeaway_behavior() -> None:
    section = NotebookSection(
        section_id="segment_and_region",
        title="客群与区域分析",
        purpose="复盘客群与区域表现。",
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount"},
        confidence=0.9,
    )
    report = AnalysisReport(
        task_id="task-zh-takeaway",
        dataset_type="sales_transaction",
        module_count=0,
        summary=[],
        modules=[],
    )

    conclusion = section_conclusion_markdown(section, report, schema_mapping, output_language="zh-CN")

    assert conclusion is not None
    assert "### 本节结论" in conclusion
    assert "复盘客群与区域表现" in conclusion


def test_english_toc_uses_generic_modeling_label_without_loss_risk_suffix() -> None:
    english_toc = build_toc_markdown(output_language="en")
    chinese_toc = build_toc_markdown(output_language="zh-CN")

    assert "Modeling Analysis" in english_toc
    assert "Modeling Analysis: Loss Risk Review" not in english_toc
    assert "建模分析" in chinese_toc
