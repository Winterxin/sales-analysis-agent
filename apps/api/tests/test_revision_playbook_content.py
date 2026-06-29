from __future__ import annotations

import json

from nbformat.v4 import new_markdown_cell, new_notebook

from app.schemas.notebook_revision import NotebookRevisionDecision, NotebookRevisionPlan
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_toolset import save_notebook


def test_revision_markdown_contains_business_playbook_guidance(tmp_path) -> None:
    from app.services.notebook_agent_loop import apply_notebook_revision_plan

    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 结论与行动建议"),
        ]
    )
    notebook_path = save_notebook(notebook, tmp_path / "analysis.ipynb")

    apply_notebook_revision_plan(
        notebook_path=notebook_path,
        output_path=notebook_path,
        revision_plan=NotebookRevisionPlan(
            decisions=[
                NotebookRevisionDecision(
                    revision_key="discount_bucket_loss_rate",
                    reason="高折扣负利润明显，需要继续拆到折扣区间。",
                )
            ]
        ),
        report=AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "Discount": "discount",
                "Profit": "profit",
                "Sales": "sales_amount",
            },
            confidence=1.0,
        ),
    )

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    combined_markdown = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "markdown"
    )

    assert "分析问题" in combined_markdown
    assert "业务价值" in combined_markdown
    assert "阅读方式" in combined_markdown
    assert "折扣区间" in combined_markdown
    assert "亏损率" in combined_markdown


def test_weak_segment_revision_uses_same_combined_slice_from_prior_section(tmp_path) -> None:
    from app.services.notebook_agent_loop import apply_notebook_revision_plan

    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 结论与行动建议"),
        ]
    )
    notebook_path = save_notebook(notebook, tmp_path / "analysis.ipynb")

    apply_notebook_revision_plan(
        notebook_path=notebook_path,
        output_path=notebook_path,
        revision_plan=NotebookRevisionPlan(
            decisions=[
                NotebookRevisionDecision(
                    revision_key="weak_segment_category_cross",
                    reason="Consumer / Central 是前文识别的弱势组合切片，需要继续拆到类目。",
                )
            ]
        ),
        report=AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "Segment": "segment",
                "Region": "region",
                "Category": "category",
                "Sales": "sales_amount",
                "Profit": "profit",
            },
            confidence=1.0,
        ),
    )

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    combined_code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "weak_slice_label" in combined_code
    assert "segment_col, region_col" in combined_code
    assert "weak_segment_category[region_col] == weak_region" in combined_code


def test_discount_revision_chart_title_matches_average_profit_and_loss_rate(tmp_path) -> None:
    from app.services.notebook_agent_loop import apply_notebook_revision_plan

    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 结论与行动建议"),
        ]
    )
    notebook_path = save_notebook(notebook, tmp_path / "analysis.ipynb")

    apply_notebook_revision_plan(
        notebook_path=notebook_path,
        output_path=notebook_path,
        revision_plan=NotebookRevisionPlan(
            decisions=[
                NotebookRevisionDecision(
                    revision_key="discount_bucket_loss_rate",
                    reason="高折扣负利润明显，需要继续拆到折扣区间。",
                )
            ]
        ),
        report=AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "Discount": "discount",
                "Profit": "profit",
                "Sales": "sales_amount",
            },
            confidence=1.0,
        ),
    )

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    combined_code = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "平均利润与亏损率" in combined_code
    assert "y='avg_profit'" in combined_code
    assert "ax2.plot" in combined_code
