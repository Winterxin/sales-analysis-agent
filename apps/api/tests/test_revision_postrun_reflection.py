from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.notebook_revision import NotebookRevisionDecision, NotebookRevisionPlan
from app.services.notebook_postrun_reflection import extract_postrun_chart_contexts


def test_extract_postrun_chart_contexts_recognizes_revision_section_titles(tmp_path: Path) -> None:
    from app.services.notebook_agent_loop import augment_outline_with_revision_sections

    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 追加分析：折扣区间亏损率"),
            new_code_cell(
                "fig = px.bar(discount_loss_rate, x='discount_bucket', y='loss_rate', title='各折扣区间亏损率与平均利润')\n"
                "fig.show()",
                outputs=[
                    nbformat.v4.new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [{"type": "bar", "x": ["20-30%", "30%+"], "y": [0.32, 0.61]}],
                                "layout": {"title": {"text": "各折扣区间亏损率与平均利润"}},
                            }
                        },
                    )
                ],
                execution_count=7,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    outline = NotebookOutline(
        title="销售数据分析 Notebook",
        sections=[
            NotebookSection(
                section_id="discount_and_profit",
                title="折扣与利润分析",
                purpose="评估折扣与利润的关系。",
            )
        ],
    )
    revision_plan = NotebookRevisionPlan(
        decisions=[
            NotebookRevisionDecision(
                revision_key="discount_bucket_loss_rate",
                reason="高折扣负利润明显，需要继续拆解亏损率。",
            )
        ]
    )

    augmented_outline = augment_outline_with_revision_sections(outline, revision_plan)
    contexts = extract_postrun_chart_contexts(notebook_path, augmented_outline)

    assert len(contexts) == 1
    assert contexts[0]["section_id"] == "discount_and_profit"
    assert contexts[0]["section_title"] == "追加分析：折扣区间亏损率"
    assert contexts[0]["chart_title"] == "各折扣区间亏损率与平均利润"
