from __future__ import annotations

import json
from pathlib import Path

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from app.schemas.notebook_revision import NotebookRevisionDecision, NotebookRevisionPlan
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_toolset import save_notebook


def _report() -> AnalysisReport:
    return AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0)


def _schema_mapping() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Discount": "discount",
            "Profit": "profit",
            "Sales": "sales_amount",
        },
        confidence=1.0,
    )


def test_apply_notebook_revision_plan_inserts_before_conclusions(tmp_path: Path) -> None:
    from app.services.notebook_agent_loop import apply_notebook_revision_plan

    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 折扣与利润分析"),
            new_code_cell("clean_df = None"),
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
                    reason="高折扣负利润明显，需要继续补充分析。",
                )
            ]
        ),
        report=_report(),
        schema_mapping=_schema_mapping(),
    )

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_sources = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "markdown"
    ]

    revision_index = next(
        index for index, source in enumerate(markdown_sources) if "## 追加分析：折扣区间亏损率" in source
    )
    conclusions_index = next(
        index for index, source in enumerate(markdown_sources) if "## 结论与行动建议" in source
    )

    assert revision_index < conclusions_index
