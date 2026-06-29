from __future__ import annotations

import json
from pathlib import Path

from app.core.config import Settings
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan, NotebookSectionContent
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.notebook_revision import NotebookRevisionDecision, NotebookRevisionPlan
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_agent_loop import apply_notebook_revision_plan
from app.services.notebook_builder import build_notebook
from app.services.notebook_postrun_reflection import apply_postrun_chart_reflections
from app.services.notebook_toolset import save_notebook
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def _schema_mapping() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Product": "product_name",
            "Category": "category",
            "Segment": "segment",
            "Region": "region",
            "Order ID": "order_id",
        },
        confidence=0.9,
    )


def _report() -> AnalysisReport:
    return AnalysisReport(
        task_id="task-output-mode",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["总销售额为 1000，折扣利润需要复盘。"],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                summary_metrics={"negative_profit_rate": 0.2},
                tables={
                    "discount_threshold_candidates": [
                        {"discount_bucket": "30%+", "negative_profit_rate": 0.7}
                    ],
                    "discount_cap_what_if": [
                        {
                            "scenario_name": "高风险折扣收紧情景估算",
                            "target_discount_cap": 0.3,
                            "estimated_profit_delta": 1200.0,
                        }
                    ],
                },
                findings=["高折扣区间利润质量偏弱。"],
            ),
            ModuleReport(
                module_id="loss_risk_modeling",
                title="亏损风险建模",
                chart_type="model",
                summary_metrics={
                    "best_model": "LogisticRegression",
                    "best_recall": 0.9,
                    "best_f1": 0.8,
                    "best_roc_auc": 0.86,
                    "target_positive_rate": 0.2,
                    "train_rows": 80,
                    "test_rows": 20,
                    "threshold_default": 0.5,
                    "true_positive_count": 8,
                    "false_positive_count": 3,
                    "false_negative_count": 1,
                    "true_negative_count": 8,
                },
                tables={
                    "model_comparison": [
                        {
                            "model": "LogisticRegression",
                            "precision": 0.75,
                            "recall": 0.9,
                            "f1": 0.8,
                            "roc_auc": 0.86,
                        }
                    ],
                    "threshold_analysis": [
                        {"threshold": 0.4, "precision": 0.7, "recall": 0.95, "f1": 0.81}
                    ],
                    "feature_importance_grouped": [
                        {"feature_group": "discount", "total_importance": 0.42}
                    ],
                    "confusion_matrix": [
                        {"actual": "loss", "predicted": "loss", "count": 8}
                    ],
                    "high_risk_examples": [
                        {
                            "row_index": 1,
                            "loss_probability": 0.91,
                            "actual_is_loss": 1,
                            "predicted_is_loss": 1,
                            "discount": 0.4,
                            "sales_amount": 100,
                        }
                    ],
                },
                findings=["模型可用于亏损记录人工复核优先级排序。"],
            ),
        ],
    )


def _outline() -> NotebookOutline:
    return NotebookOutline(
        title="Output Mode Notebook",
        sections=[
            NotebookSection(section_id="dataset_and_schema", title="数据集与字段说明", purpose="确认字段。"),
            NotebookSection(section_id="data_cleaning", title="数据清洗与预处理", purpose="清洗数据。"),
            NotebookSection(section_id="metric_distributions", title="指标分布与异常值分析", purpose="看分布。"),
            NotebookSection(section_id="sales_trends", title="销售趋势分析", purpose="看趋势。"),
            NotebookSection(section_id="product_and_category", title="商品与类目分析", purpose="看商品。"),
            NotebookSection(section_id="segment_and_region", title="客群与区域分析", purpose="看切片。"),
            NotebookSection(section_id="order_structure", title="订单结构分析", purpose="看订单。"),
            NotebookSection(section_id="discount_and_profit", title="折扣与利润分析", purpose="看折扣。"),
            NotebookSection(section_id="modeling", title="建模分析", purpose="看模型。"),
            NotebookSection(section_id="conclusions", title="结论与行动建议", purpose="总结。"),
        ],
    )


def _section(section_id: str, *, code_count: int) -> NotebookSectionContent:
    code_cells = [
        f"{section_id}_table_{index} = clean_df.head({15 + index})\n{section_id}_table_{index}"
        for index in range(code_count)
    ]
    code_cells.append(
        f"{section_id}_chart_data = clean_df.head(20)\n"
        f"fig = px.bar({section_id}_chart_data, x='Sales', y='Profit', title='{section_id} chart')\n"
        "fig.show()"
    )
    return NotebookSectionContent(
        section_id=section_id,
        markdown_blocks=[
            f"{section_id} intro.",
            f"### 本节结论\n\n{section_id} conclusion.",
        ],
        code_cells=code_cells,
    )


def _content_plan() -> NotebookContentPlan:
    compact_targets = [
        "dataset_and_schema",
        "data_cleaning",
        "metric_distributions",
        "sales_trends",
        "product_and_category",
        "segment_and_region",
        "order_structure",
    ]
    sections = [_section(section_id, code_count=3) for section_id in compact_targets]
    sections.append(
        NotebookSectionContent(
            section_id="discount_and_profit",
            markdown_blocks=[
                "折扣章节导入。",
                "### 本节结论\n\n首先，折扣收紧需要静态估算边界。",
            ],
            code_cells=[
                "discount_threshold_candidates = pd.DataFrame([{'discount_bucket': '30%+'}])\ndiscount_threshold_candidates",
                "discount_cap_what_if = pd.DataFrame([{'estimated_profit_delta': 1200}])\ndiscount_cap_what_if",
                "print('what-if Markdown 限制说明：不代表真实需求、销量或客户行为变化')",
            ],
        )
    )
    sections.append(NotebookSectionContent(section_id="conclusions", markdown_blocks=["总结。"], code_cells=[]))
    return NotebookContentPlan(sections=sections)


def _build(tmp_path: Path, mode: str | None = None) -> dict[str, object]:
    path = build_notebook(
        task_id="task-output-mode",
        output_dir=tmp_path,
        report=_report(),
        schema_mapping=_schema_mapping(),
        plan=AnalysisPlan(),
        outline=_outline(),
        narrative=None,
        content_plan=_content_plan(),
        notebook_output_mode=mode,
        output_language="zh-CN",
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    return "".join(cell.get("source", []))


def _combined(cells: list[dict[str, object]], cell_type: str | None = None) -> str:
    return "\n\n".join(
        _source(cell)
        for cell in cells
        if cell_type is None or cell.get("cell_type") == cell_type
    )


def test_notebook_output_mode_defaults_to_compact() -> None:
    assert Settings().notebook_output_mode == "compact"


def test_full_output_mode_keeps_original_front_half_code_cells(tmp_path: Path) -> None:
    data = _build(tmp_path / "full", mode="full")
    code = _combined(data["cells"], "code")

    assert code.count("_table_0 =") == 7
    assert code.count("_table_1 =") == 7
    assert code.count("_table_2 =") == 7
    assert code.count("fig.show()") >= 7


def test_compact_output_mode_reduces_front_half_density_and_keeps_required_sections(tmp_path: Path) -> None:
    compact = _build(tmp_path / "compact", mode="compact")
    full = _build(tmp_path / "full", mode="full")
    compact_code_cells = [cell for cell in compact["cells"] if cell.get("cell_type") == "code"]
    full_code_cells = [cell for cell in full["cells"] if cell.get("cell_type") == "code"]
    combined = _combined(compact["cells"])

    assert len(compact_code_cells) < len(full_code_cells)
    assert "## 目录" in combined
    assert "## Action Plan" not in combined
    assert "discount_cap_what_if" in combined
    assert "不代表真实需求、销量或客户行为变化" in combined
    assert "## 建模分析：亏损风险识别" in combined
    assert "## 结论与行动建议" in combined
    assert "### 主要结论" in combined
    assert "### 行动建议" in combined
    assert "_table_2" not in combined
    assert ".head(10)" in combined


def test_toc_does_not_list_modeling_without_modeling_body(tmp_path: Path) -> None:
    outline = NotebookOutline(
        title="No Modeling Notebook",
        sections=[
            NotebookSection(section_id="dataset_and_schema", title="数据集与字段说明", purpose="确认字段。"),
            NotebookSection(section_id="data_cleaning", title="数据清洗与预处理", purpose="清洗数据。"),
            NotebookSection(section_id="sales_trends", title="销售趋势分析", purpose="看趋势。"),
            NotebookSection(section_id="conclusions", title="结论与行动建议", purpose="总结。"),
        ],
    )
    content = NotebookContentPlan(
        sections=[
            NotebookSectionContent(section_id="dataset_and_schema", markdown_blocks=["字段说明。"], code_cells=[]),
            NotebookSectionContent(section_id="data_cleaning", markdown_blocks=["清洗说明。"], code_cells=[]),
            NotebookSectionContent(section_id="sales_trends", markdown_blocks=["趋势说明。"], code_cells=[]),
            NotebookSectionContent(section_id="conclusions", markdown_blocks=["总结。"], code_cells=[]),
        ]
    )

    path = build_notebook(
        task_id="task-output-mode",
        output_dir=tmp_path / "no-modeling-toc",
        report=_report(),
        schema_mapping=_schema_mapping(),
        plan=AnalysisPlan(),
        outline=outline,
        narrative=None,
        content_plan=content,
        notebook_output_mode="full",
        output_language="zh-CN",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    markdown_cells = [_source(cell) for cell in data["cells"] if cell.get("cell_type") == "markdown"]
    toc = next(source for source in markdown_cells if source.startswith("## 目录"))
    combined = "\n\n".join(markdown_cells)

    assert "建模分析" not in toc
    assert "## 建模分析：亏损风险识别" not in combined


def test_compact_output_mode_guards_strong_inference_in_markdown(tmp_path: Path) -> None:
    content_plan = _content_plan()
    for section in content_plan.sections:
        if section.section_id == "discount_and_profit":
            section.markdown_blocks = [
                "折扣章节导入。",
                "### 本节结论\n\n折扣审批失控是核心原因，必然导致利润质量恶化。",
            ]
    path = build_notebook(
        task_id="task-output-mode",
        output_dir=tmp_path / "compact-guard",
        report=_report(),
        schema_mapping=_schema_mapping(),
        plan=AnalysisPlan(),
        outline=_outline(),
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="compact",
        output_language="zh-CN",
    )
    compact = json.loads(path.read_text(encoding="utf-8"))
    markdown = _combined(compact["cells"], "markdown")

    assert "本节结论：" in markdown
    assert "### 图表解读" not in markdown
    assert "折扣审批失控" not in markdown
    assert "核心原因" not in markdown
    assert "必然导致" not in markdown
    assert "可能的重要风险信号" in markdown


def test_compact_output_mode_does_not_compress_modeling_section(tmp_path: Path) -> None:
    compact = _build(tmp_path / "compact", mode="compact")
    full = _build(tmp_path / "full", mode="full")

    compact_modeling = [
        _source(cell)
        for cell in compact["cells"]
        if "## 建模分析" in _source(cell)
        or "threshold_analysis" in _source(cell)
        or "feature_importance_grouped" in _source(cell)
        or "confusion_matrix" in _source(cell)
    ]
    full_modeling = [
        _source(cell)
        for cell in full["cells"]
        if "## 建模分析" in _source(cell)
        or "threshold_analysis" in _source(cell)
        or "feature_importance_grouped" in _source(cell)
        or "confusion_matrix" in _source(cell)
    ]
    compact_text = "\n\n".join(compact_modeling)

    assert compact_modeling == full_modeling
    assert "LogisticRegression" in compact_text
    assert "threshold_analysis" in compact_text
    assert "feature_importance_grouped" in compact_text
    assert "confusion_matrix" in compact_text


def test_compact_output_mode_keeps_chart_dependency_cells(tmp_path: Path) -> None:
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="segment_and_region",
                markdown_blocks=["切片导入。", "### 本节结论\n\n切片结论。"],
                code_cells=[
                    "segment_view = clean_df.groupby('Segment', as_index=False)['Sales'].sum()\nsegment_view",
                    "weak_segments = clean_df.head(20)\nweak_segments",
                    (
                        "region_profit_quality = clean_df.groupby('Region', as_index=False)[['Sales', 'Profit']].sum()\n"
                        "region_profit_quality['profit_margin'] = region_profit_quality['Profit'] / region_profit_quality['Sales'].mask(region_profit_quality['Sales'] == 0)\n"
                        "region_profit_quality"
                    ),
                    (
                        "fig = px.bar(\n"
                        "    region_profit_quality.sort_values('Sales', ascending=True),\n"
                        "    x='Sales',\n"
                        "    y='Region',\n"
                        "    color='profit_margin',\n"
                        "    title='区域销售与利润对比',\n"
                        ")\n"
                        "fig.show()"
                    ),
                ],
            )
        ]
    )

    path = build_notebook(
        task_id="task-output-mode",
        output_dir=tmp_path,
        report=_report(),
        schema_mapping=_schema_mapping(),
        plan=AnalysisPlan(),
        outline=NotebookOutline(
            title="Dependency Notebook",
            sections=[
                NotebookSection(section_id="segment_and_region", title="客群与区域分析", purpose="看切片。")
            ],
        ),
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="compact",
        output_language="zh-CN",
    )

    combined_code = _combined(json.loads(path.read_text(encoding="utf-8"))["cells"], "code")

    assert "region_profit_quality =" in combined_code
    assert "region_profit_quality.sort_values" in combined_code
    assert "weak_segments =" not in combined_code


def test_compact_output_mode_suppresses_front_half_step_intros_but_keeps_titles_and_conclusions(
    tmp_path: Path,
) -> None:
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="dataset_and_schema",
                markdown_blocks=["字段说明总览。", "### 本节结论\n\n字段映射可进入分析。"],
                code_cells=[
                    "import pandas as pd\ndf = pd.read_csv('raw.csv')\ndf.head()",
                    "df.info()",
                    "df.describe(include='all')",
                ],
            ),
            NotebookSectionContent(
                section_id="sales_trends",
                markdown_blocks=["趋势章节总览。", "### 本节结论\n\n销售趋势存在阶段波动。"],
                code_cells=[
                    "monthly_sales = clean_df.groupby('Order Date', as_index=False)['Sales'].sum()\nmonthly_sales.head(12)",
                    "fig = px.line(monthly_sales, x='Order Date', y='Sales', title='按月销售额趋势')\nfig.show()",
                ],
            ),
        ]
    )
    outline = NotebookOutline(
        title="Step Intro Notebook",
        sections=[
            NotebookSection(section_id="dataset_and_schema", title="数据集与字段说明", purpose="确认字段。"),
            NotebookSection(section_id="sales_trends", title="销售趋势分析", purpose="看趋势。"),
        ],
    )

    compact = build_notebook(
        task_id="task-step-intro",
        output_dir=tmp_path / "compact",
        report=_report(),
        schema_mapping=_schema_mapping(),
        plan=AnalysisPlan(),
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="compact",
        output_language="zh-CN",
    )
    full = build_notebook(
        task_id="task-step-intro",
        output_dir=tmp_path / "full",
        report=_report(),
        schema_mapping=_schema_mapping(),
        plan=AnalysisPlan(),
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="full",
        output_language="zh-CN",
    )

    compact_text = _combined(json.loads(compact.read_text(encoding="utf-8"))["cells"])
    full_text = _combined(json.loads(full.read_text(encoding="utf-8"))["cells"])
    step_intro_markers = [
        "先导入分析所需库并读取原始数据",
        "接着检查字段类型和缺失情况",
        "再看描述统计",
        "先按月汇总销售额和销量",
        "再看按月销售额趋势图",
    ]

    assert sum(marker in compact_text for marker in step_intro_markers) == 0
    assert sum(marker in full_text for marker in step_intro_markers) > 0
    for title in ["## 数据集与字段说明", "## 销售趋势分析"]:
        assert title in compact_text
    for conclusion in ["字段映射可进入分析", "销售趋势存在阶段波动"]:
        assert conclusion in compact_text


def test_compact_output_mode_removes_section_conclusion_heading_only(tmp_path: Path) -> None:
    compact = _build(tmp_path / "compact", mode="compact")
    full = _build(tmp_path / "full", mode="full")
    compact_text = _combined(compact["cells"])
    full_text = _combined(full["cells"])

    assert "### 本节结论" not in compact_text
    assert "本节结论：折扣收紧需要静态估算边界。" in compact_text
    assert "dataset_and_schema conclusion." in compact_text
    assert "折扣收紧需要静态估算边界。" in compact_text
    assert "首先，折扣收紧" not in compact_text
    assert "首先，折扣收紧" not in full_text
    assert "### 本节结论" in full_text


def test_compact_postrun_reflection_removes_chart_heading_only(tmp_path: Path) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 销售趋势分析"),
            new_code_cell("fig.show()"),
        ]
    )
    compact_path = save_notebook(notebook, tmp_path / "compact.ipynb")
    full_path = save_notebook(notebook, tmp_path / "full.ipynb")
    reflection = "首先，2014-11 销售额达到高点，随后回落，说明旺季后需要复盘库存和促销节奏。"

    apply_postrun_chart_reflections(
        notebook_path=compact_path,
        output_path=compact_path,
        reflections_by_chart_cell={1: reflection},
        notebook_output_mode="compact",
    )
    apply_postrun_chart_reflections(
        notebook_path=full_path,
        output_path=full_path,
        reflections_by_chart_cell={1: reflection},
        notebook_output_mode="full",
    )

    compact_text = _combined(json.loads(compact_path.read_text(encoding="utf-8"))["cells"])
    full_text = _combined(json.loads(full_path.read_text(encoding="utf-8"))["cells"])

    assert "### 图表解读" not in compact_text
    assert "2014-11 销售额达到高点" in compact_text
    assert "复盘库存和促销节奏" in compact_text
    assert "首先，2014-11" not in compact_text
    assert "首先，2014-11" not in full_text
    assert "### 图表解读" in full_text


def test_output_modes_hide_capability_matrix(
    tmp_path: Path,
) -> None:
    compact = _build(tmp_path / "compact", mode="compact")
    full = _build(tmp_path / "full", mode="full")
    compact_text = _combined(compact["cells"])
    full_text = _combined(full["cells"])

    assert "## Analysis Capability Matrix" not in compact_text
    assert "Capability Matrix" not in compact_text
    assert "简版 Capability Matrix" not in compact_text
    assert "完整能力矩阵可在 full/debug 模式查看" not in compact_text
    assert "data_limit_or_alternative" not in compact_text
    assert "## Analysis Capability Matrix" not in full_text
    assert "Capability Matrix" not in full_text
    assert "| capability | status | required_fields | data_limit_or_alternative |" not in full_text


def test_compact_output_mode_limits_revision_append_sections_and_full_keeps_all(
    tmp_path: Path,
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("# Notebook"),
            new_markdown_cell("## 结论与行动建议"),
        ]
    )
    compact_path = save_notebook(notebook, tmp_path / "compact.ipynb")
    full_path = save_notebook(notebook, tmp_path / "full.ipynb")
    revision_plan = NotebookRevisionPlan(
        decisions=[
            NotebookRevisionDecision(
                revision_key="discount_bucket_loss_rate",
                reason="高折扣负利润明显，需要继续补充分析。",
            ),
            NotebookRevisionDecision(
                revision_key="top_product_profit_bridge",
                reason="头部商品需要核查销售额和利润是否同步。",
            ),
        ]
    )

    apply_notebook_revision_plan(
        notebook_path=compact_path,
        output_path=compact_path,
        revision_plan=revision_plan,
        report=_report(),
        schema_mapping=_schema_mapping(),
        notebook_output_mode="compact",
    )
    apply_notebook_revision_plan(
        notebook_path=full_path,
        output_path=full_path,
        revision_plan=revision_plan,
        report=_report(),
        schema_mapping=_schema_mapping(),
        notebook_output_mode="full",
    )

    compact_payload = json.loads(compact_path.read_text(encoding="utf-8"))
    full_payload = json.loads(full_path.read_text(encoding="utf-8"))
    compact_text = _combined(compact_payload["cells"])
    full_text = _combined(full_payload["cells"])
    compact_code_cells = [cell for cell in compact_payload["cells"] if cell.get("cell_type") == "code"]

    assert compact_text.count("## 追加分析：") == 1
    assert full_text.count("## 追加分析：") == 2
    assert "## 追加分析：折扣区间亏损率" in compact_text
    assert "## 追加分析：头部商品销售-利润桥接" not in compact_text
    assert "### 追加分析口径" not in compact_text
    assert len(compact_code_cells) <= 2
