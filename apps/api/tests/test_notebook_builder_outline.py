from __future__ import annotations

import json
from pathlib import Path

from nbformat.v4 import new_markdown_cell, new_notebook

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan, NotebookSectionContent
from app.schemas.notebook_narrative import NotebookNarrative, NotebookSectionNarrative
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_builder import build_notebook
from app.services.notebook.assembler import _apply_final_narrative_guard
from app.services.notebook_planner import build_section_priority

MOJIBAKE_MARKERS = ("????", "\ufffd", "鍥捐〃", "绛栫暐")


def test_final_narrative_guard_drops_empty_action_plan_and_toc_link() -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 目录\n\n- [Action Plan](#action-plan)\n- [结论与行动建议](#section-conclusions)"),
            new_markdown_cell("## Action Plan\n\n<a id=\"action-plan\"></a>"),
            new_markdown_cell("## 结论与行动建议\n\n保留结论。"),
        ]
    )

    trace = _apply_final_narrative_guard(notebook, mapped_fields=set(), schema_mapping=None)
    combined = "\n\n".join(str(cell.source) for cell in notebook.cells)

    assert trace["dropped_empty_action_plan_cells_count"] == 1
    assert "## Action Plan" not in combined
    assert "#action-plan" not in combined
    assert "结论与行动建议" in combined


def test_notebook_builder_respects_outline_section_order(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["Summary item."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["趋势发现。"],
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                findings=["商品发现。"],
            ),
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    outline = NotebookOutline(
        title="Rich Notebook",
        sections=[
            NotebookSection(
                section_id="title_and_goal",
                title="Notebook Goal",
                purpose="State the objective.",
            ),
            NotebookSection(
                section_id="sales_trends",
                title="Trend Story",
                purpose="Explain the trend.",
            ),
            NotebookSection(
                section_id="conclusions",
                title="Closing Notes",
                purpose="Wrap up.",
            ),
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="title_and_goal",
                intro="This notebook explains the business goal.",
                key_observations=["Goal observation."],
                business_takeaway="Goal takeaway.",
                followup_question="What business metric matters most?",
            ),
            NotebookSectionNarrative(
                section_id="sales_trends",
                intro="Trend intro.",
                key_observations=["Trend observation."],
                business_takeaway="Trend takeaway.",
                followup_question="What drove the peak?",
            ),
            NotebookSectionNarrative(
                section_id="conclusions",
                intro="Conclusion intro.",
                key_observations=["Conclusion observation."],
                business_takeaway="Conclusion takeaway.",
                followup_question="What should be done next?",
            ),
        ],
        suggested_followups=["discount_and_profit"],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="title_and_goal",
                markdown_blocks=["This notebook explains the business goal."],
                code_cells=[],
            ),
            NotebookSectionContent(
                section_id="sales_trends",
                markdown_blocks=[
                    "Trend intro.",
                    "### Business Takeaway\n\nTrend takeaway.",
                    "### 图表分析\n\n#### 现象与判断\n- 趋势有阶段波动。\n\n#### 经营含义\n- 需要确认波动来源。",
                ],
                code_cells=[
                    "trend_df = df.groupby('Order Date', as_index=False)['Sales'].sum()",
                    "trend_df.head()",
                ],
            ),
            NotebookSectionContent(
                section_id="conclusions",
                markdown_blocks=["Conclusion intro.", "### Business Takeaway\n\nConclusion takeaway."],
                code_cells=[],
            ),
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=narrative,
        content_plan=content_plan,
    )

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
    ]

    assert markdown_cells[0].startswith("# Rich Notebook")
    assert any("## Trend Story" in cell for cell in markdown_cells)
    assert any("## Conclusions and Recommendations" in cell or "## 结论与行动建议" in cell for cell in markdown_cells)
    assert any("### Main Conclusions" in cell or "### 主要结论" in cell for cell in markdown_cells)
    assert any("### Recommendations" in cell or "### 行动建议" in cell for cell in markdown_cells)
    assert any("Trend intro." in cell for cell in markdown_cells)
    assert any("Trend takeaway." in cell for cell in markdown_cells)
    assert not any("图表分析" in cell for cell in markdown_cells)
    code_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    ]
    assert any("groupby('Order Date'" in cell for cell in code_cells)


def test_content_plan_final_conclusion_prefers_llm_narrative_conclusions(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=0,
        summary=["DETERMINISTIC_FALLBACK_SUMMARY_SHOULD_NOT_RENDER"],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
        confidence=0.9,
    )
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="conclusions", title="结论与行动建议", purpose="总结。")],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="conclusions",
                intro="LLM 结论：当前核心经营矛盾是折扣增长与利润质量背离。",
                key_observations=[
                    "主要风险对象是高折扣且低利润的商品、类目和组合切片。",
                    "图表和建模结果可以把人工复核优先级落到具体订单和业务对象。",
                ],
                business_takeaway="下一步应先建立高风险订单复核标签，再校准折扣审批阈值。",
                followup_question="不应出现在最终结论。",
            )
        ]
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="conclusions",
                markdown_blocks=["模板结论：只拼接指标。"],
                code_cells=[],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=outline,
        narrative=narrative,
        content_plan=content_plan,
    )

    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    combined = "\n\n".join(markdown_cells)
    conclusion_cell = next(
        cell
        for cell in markdown_cells
        if cell.startswith("## Conclusions and Recommendations") or cell.startswith("## 结论与行动建议")
    )

    assert "## Conclusions and Recommendations" in combined or "## 结论与行动建议" in combined
    assert "### Main Conclusions" in combined or "### 主要结论" in combined
    assert "模板结论：只拼接指标" not in combined
    assert "DETERMINISTIC_FALLBACK_SUMMARY_SHOULD_NOT_RENDER" in conclusion_cell
    assert "不应出现在最终结论" not in conclusion_cell
    assert "### Main Conclusions" in conclusion_cell or "### 主要结论" in conclusion_cell
    assert "### Recommendations" in conclusion_cell or "### 行动建议" in conclusion_cell


def test_focus_risk_items_uses_discount_risks_when_discount_and_profit_available() -> None:
    from app.services.notebook.action_plan_builder import focus_risk_items

    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣利润分析",
                chart_type="scatter",
                warnings=["高折扣订单存在利润侵蚀风险。"],
            )
        ],
    )

    risks = focus_risk_items(
        report,
        {"has_discount": True, "has_profit": True},
        ["discount_erosion_focus"],
    )

    assert "高折扣订单存在利润侵蚀风险。" in risks


def test_notebook_builder_does_not_duplicate_chart_analysis_blocks(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Summary item."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="趋势分析",
                chart_type="line",
                findings=["趋势发现。"],
                summary_metrics={"total_sales_amount": 1000},
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    outline = NotebookOutline(
        title="Rich Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="Trend Story",
                purpose="Explain the trend.",
            )
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="sales_trends",
                markdown_blocks=[
                    "Trend intro.",
                    "### 图表分析\n\n#### 现象与判断\n- 这是分析块。\n\n#### 经营含义\n- 这是经营含义。",
                ],
                code_cells=["trend_df = df.groupby('Order Date', as_index=False)['Sales'].sum()"],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="full",
    )

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    combined = "\n\n".join(markdown_cells)
    assert combined.count("### 图表分析") == 0


def test_notebook_builder_hides_template_chart_analysis_until_llm_reflection(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Summary item."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="趋势分析",
                chart_type="line",
                findings=["趋势发现。"],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    outline = NotebookOutline(
        title="Rich Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="Trend Story",
                purpose="Explain the trend.",
            )
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="sales_trends",
                markdown_blocks=[
                    "Trend intro.",
                    "### 图表分析\n\n这是一段模板化的预执行分析，不应该出现在最终 notebook。",
                ],
                code_cells=["trend_df = df.groupby('Order Date', as_index=False)['Sales'].sum()"],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="full",
    )

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    combined = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "Trend intro." in combined
    assert "### 图表分析" not in combined
    assert "模板化的预执行分析" not in combined


def test_notebook_builder_does_not_emit_legacy_chart_explanation_blocks(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Summary item."],
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                findings=["头部商品贡献集中。"],
                summary_metrics={
                    "distinct_products": 1850,
                    "top_product": "Canon imageCLASS 2200 Advanced Copier",
                },
                tables={
                    "top_products": [
                        {
                            "Product Name": "Canon imageCLASS 2200 Advanced Copier",
                            "Sales": 61599.824,
                        }
                    ]
                },
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Product Name": "product_name", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    outline = NotebookOutline(
        title="Rich Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="商品与类目分析",
                purpose="Explain the product structure.",
            )
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="product_and_category",
                markdown_blocks=["Product intro."],
                code_cells=[
                    "top_products = clean_df.groupby('Product Name', as_index=False)['Sales'].sum().head(15)",
                    "fig = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top 商品销售额对比')\nfig.show()",
                ],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="full",
    )

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    combined = "\n\n".join(markdown_cells)
    assert "图后解读：" not in combined


def test_notebook_builder_final_markdown_has_no_mojibake(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Summary item."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["趋势发现。"],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    outline = NotebookOutline(
        title="Rich Notebook",
        sections=[
            NotebookSection(
                section_id="dataset_and_schema",
                title="数据集与字段",
                purpose="确认数据结构。",
            ),
            NotebookSection(
                section_id="sales_trends",
                title="销售趋势",
                purpose="解释趋势。",
            ),
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="dataset_and_schema",
                markdown_blocks=["读取数据。"],
                code_cells=[
                    "import pandas as pd\ndf = pd.read_csv('raw.csv')\ndf.head()",
                    "df.info()",
                    "df.describe(include='all')",
                ],
            ),
            NotebookSectionContent(
                section_id="sales_trends",
                markdown_blocks=["趋势说明。"],
                code_cells=[
                    "trend_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'monthly_line'}]}",
                    "monthly_sales = clean_df.groupby('Order Date', as_index=False)['Sales'].sum()",
                ],
            ),
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="full",
    )

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    combined_markdown = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "## Sales Trends" in combined_markdown
    assert not any(marker in combined_markdown for marker in MOJIBAKE_MARKERS)


def test_notebook_builder_shows_chart_analysis_only_after_llm_chart_decision(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Summary item."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["趋势发现。"],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    outline = NotebookOutline(
        title="Rich Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="销售趋势",
                purpose="解释趋势。",
            )
        ],
    )
    chart_analysis = "### 图表分析\n\nLLM 决策后的趋势分析。"

    fallback_path = build_notebook(
        task_id="task-fallback",
        output_dir=tmp_path / "fallback",
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=None,
        content_plan=NotebookContentPlan(
            sections=[
                NotebookSectionContent(
                    section_id="sales_trends",
                    markdown_blocks=["趋势说明。", chart_analysis],
                    code_cells=[
                        "trend_strategy = {'source': 'deterministic_fallback', 'views': []}",
                    ],
                )
            ]
        ),
    )
    fallback_text = fallback_path.read_text(encoding="utf-8")
    assert "LLM 决策后的趋势分析" not in fallback_text

    llm_path = build_notebook(
        task_id="task-llm",
        output_dir=tmp_path / "llm",
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=None,
        content_plan=NotebookContentPlan(
            sections=[
                NotebookSectionContent(
                    section_id="sales_trends",
                    markdown_blocks=["趋势说明。", chart_analysis],
                    code_cells=[
                        "trend_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'monthly_line'}]}",
                    ],
                )
            ]
        ),
    )
    llm_text = llm_path.read_text(encoding="utf-8")
    assert "LLM 决策后的趋势分析" not in llm_text


def test_notebook_builder_adds_management_summary_action_plan_and_capability_matrix(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=3,
        summary=[
            "2024 年总销售额达到 1,250,000，其中 West 区域贡献 42%。",
            "Canon imageCLASS 2200 Advanced Copier 销售额 61,599.82，但利润率只有 4.8%。",
            "Consumer 客群贡献 51% 销售额，是当前核心客群。",
        ],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["2024 年 11 月销售额 186,000，是全年峰值。"],
                summary_metrics={"total_sales_amount": 1250000, "peak_month": "2024-11"},
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                findings=["Canon imageCLASS 2200 Advanced Copier 销售额 61,599.82。"],
                summary_metrics={"top_product": "Canon imageCLASS 2200 Advanced Copier"},
                tables={
                    "top_products": [
                        {
                            "Product Name": "Canon imageCLASS 2200 Advanced Copier",
                            "Sales": 61599.82,
                            "Profit": 2950.0,
                        }
                    ]
                },
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                findings=["20% 以上折扣订单平均利润为 -18.4。"],
                warnings=["高折扣区间存在利润侵蚀风险。"],
                summary_metrics={"high_discount_avg_profit": -18.4},
            ),
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Product Name": "product_name",
            "Region": "region",
            "Segment": "segment",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    outline = NotebookOutline(
        title="Rich Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="销售趋势",
                purpose="解释销售趋势。",
            ),
            NotebookSection(
                section_id="discount_and_profit",
                title="折扣与利润",
                purpose="解释折扣风险。",
            ),
            NotebookSection(
                section_id="conclusions",
                title="结论与行动建议",
                purpose="总结行动。",
            ),
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="sales_trends",
                markdown_blocks=["趋势说明。"],
                code_cells=[
                    "trend_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'monthly_line'}]}",
                ],
            ),
            NotebookSectionContent(
                section_id="discount_and_profit",
                markdown_blocks=["折扣说明。"],
                code_cells=[
                    "discount_profit_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'discount_vs_profit'}]}",
                ],
            ),
            NotebookSectionContent(
                section_id="conclusions",
                markdown_blocks=["结论说明。"],
                code_cells=[],
            ),
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=plan,
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="full",
    )

    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    combined = "\n\n".join(markdown_cells)

    assert "## Notebook Analysis Summary" in combined
    assert "## Table of Contents" in combined
    assert "#section-title_and_goal" in combined
    assert "#section-discount_and_profit" in combined
    assert "#capability-matrix" not in combined
    assert combined.count('<a id="section-title_and_goal"></a>') == 1
    assert '<a id="section-discount_and_profit"></a>' in combined
    assert '<a id="capability-matrix"></a>' not in combined
    assert combined.index("## Table of Contents") < combined.index("## Notebook Analysis Summary") < combined.index("## Sales Trends")
    assert "<a id=\"section-title_and_goal\"></a>" not in markdown_cells[0]
    assert "## Kaggle-style Analysis Summary" not in combined
    assert "## Management Summary" not in combined
    assert combined.count("Key Findings") >= 1
    assert combined.count("Recommendations") >= 1
    assert "Canon imageCLASS 2200 Advanced Copier" in combined
    assert "## Analysis Capability Matrix" not in combined
    assert "Capability Matrix" not in combined
    assert "## Action Plan" not in combined
    assert "| priority | issue | evidence | action | required_fields |" not in combined
    assert "review" in combined.lower() or "validate" in combined.lower()
    assert "Key Findings" in combined
    assert "这一节不只看总体规模" not in combined
    assert "下一步应检查 折扣审批、成本和订单级利润字段" not in combined
    all_cells_text = "\n\n".join(
        "".join(cell.get("source", [])) for cell in data["cells"]
    )
    strategy_index = all_cells_text.index("discount_profit_strategy")
    conclusion_index = all_cells_text.find("### Section Conclusion", strategy_index)
    if conclusion_index != -1:
        assert strategy_index < conclusion_index
    assert "Section analysis question" not in combined
    assert "Why this question matters" not in combined
    assert "Chart/table notes" not in combined
    assert "Business meaning" not in combined
    assert not any(marker.lower() in combined.lower() for marker in ["unknown", "none", "nan", "????", "\ufffd"])


def test_notebook_builder_capability_matrix_explains_missing_fields_without_unknown(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["总销售额为 88,000，当前数据缺少利润和折扣字段。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["总销售额为 88,000。"],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    outline = NotebookOutline(
        title="Limited Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="销售趋势",
                purpose="解释趋势。",
            )
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="sales_trends",
                markdown_blocks=["趋势说明。"],
                code_cells=[],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=outline,
        narrative=None,
        content_plan=content_plan,
    )

    combined = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "Capability Matrix" not in combined
    assert "Capability Matrix" not in combined
    assert "## Conclusions and Recommended Actions" not in combined
    assert not any(marker.lower() in combined.lower() for marker in ["unknown", "none", "nan"])


def test_notebook_builder_rewrites_generated_unknown_values_as_data_limits(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["总销售额为 88,000。"],
        modules=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    outline = NotebookOutline(
        title="Guarded Notebook",
        sections=[
            NotebookSection(
                section_id="conclusions",
                title="结论与行动建议",
                purpose="总结。",
            )
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="conclusions",
                markdown_blocks=["高折扣记录约 unknown (None) 条。"],
                code_cells=[],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=outline,
        narrative=None,
        content_plan=content_plan,
    )

    combined = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "High-discount records" not in combined
    assert "unknown" not in combined.lower()
    assert "none" not in combined.lower()


def test_management_summary_does_not_prioritize_loss_when_negative_profit_rate_is_zero(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["当前共有 0 条利润为负的记录，需要优先排查亏损来源。"],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="利润质量分析",
                chart_type="bar",
                findings=[
                    "当前共有 0 条利润为负的记录，需要优先排查亏损来源。",
                    "利润率分布显示不同类目存在结构差异。",
                ],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Product": "product_name",
        },
        confidence=0.9,
    )
    outline = NotebookOutline(
        title="Profit Notebook",
        sections=[
            NotebookSection(
                section_id="conclusions",
                title="结论与行动建议",
                purpose="总结。",
            )
        ],
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="conclusions",
                markdown_blocks=["结论。"],
                code_cells=[],
            )
        ]
    )

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=outline,
        narrative=None,
        content_plan=content_plan,
        dataset_profile={"has_profit": True, "negative_profit_rate": 0.0},
        analysis_focus={"selected_focuses": ["profit_quality_focus"]},
    )

    combined = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "优先排查亏损来源" not in combined
    assert "negative profit" in combined.lower() or "profit" in combined.lower()


def test_online_retail_summary_and_action_avoid_profit_discount_language(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Country=United Kingdom，__sales_amount=8,187,806.36，Quantity=4,263,829"],
        modules=[
            ModuleReport(
                module_id="country_market_analysis",
                title="国家市场分析",
                chart_type="bar",
                findings=["销售额最高国家是 United Kingdom，销售占比约 84.0%。"],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "InvoiceNo": "order_id",
            "CustomerID": "customer_id",
            "Country": "country",
            "Description": "product_name",
            "Quantity": "quantity",
            "UnitPrice": "unit_price",
            "__sales_amount": "sales_amount",
        },
        confidence=0.9,
    )
    outline = NotebookOutline(
        title="Retail Notebook",
        sections=[
            NotebookSection(section_id="country_market", title="国家市场", purpose="市场结构。"),
            NotebookSection(section_id="order_structure", title="订单结构", purpose="订单结构。"),
        ],
    )
    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=outline,
        narrative=None,
        content_plan=NotebookContentPlan(sections=[]),
        dataset_profile={
            "has_profit": False,
            "has_discount": False,
            "country_count": 3,
            "top_country_sales_share": 0.84,
            "repeat_customer_rate": 0.7,
            "line_per_order_avg": 20.9,
        },
        analysis_focus={
            "selected_focuses": [
                "country_market_focus",
                "customer_order_structure_focus",
                "product_concentration_focus",
            ]
        },
    )

    combined = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "负利润" not in combined
    assert "亏损率" not in combined
    assert "折扣侵蚀" not in combined


def test_build_section_priority_uses_focuses_to_split_core_support_skipped() -> None:
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Segment": "segment",
            "Product Name": "product_name",
            "Country": "country",
        },
        confidence=0.9,
    )
    plan = AnalysisPlan(
        analysis_plan=[
            "metric_distribution_analysis",
            "sales_trend_analysis",
            "product_contribution_analysis",
            "dimension_breakdown_analysis",
            "country_market_analysis",
            "discount_profit_analysis",
        ]
    )

    priority = build_section_priority(
        schema_mapping=schema_mapping,
        analysis_plan=plan,
        dataset_profile={"country_count": 1, "has_country": True, "has_profit": True, "has_discount": True},
        analysis_focus={
            "selected_focuses": [
                "discount_erosion_focus",
                "profit_quality_focus",
                "segment_region_focus",
                "product_concentration_focus",
            ],
            "support_focuses": ["trend_volatility_focus"],
            "skipped_focuses": ["country_market_focus"],
        },
    )

    assert priority["core_sections"] == [
        "discount_and_profit",
        "segment_and_region",
        "product_and_category",
    ]
    assert "sales_trends" in priority["support_sections"]
    assert "country_market" in priority["skipped_sections"]
    assert "country_count=1" in priority["section_reasons"]["country_market"]


def test_notebook_builder_uses_evidence_pack_and_formats_ratios(tmp_path: Path) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["负利润记录占比为 0.1872，30%+ 折扣桶亏损率为 0.9777。"],
        modules=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Category": "category",
        },
        confidence=0.9,
    )
    evidence_pack = {
        "dataset_signature": {"dominant_story": "discount_loss", "shape_type": "superstore_like"},
        "focus_evidence": {
            "discount_erosion_focus": [
                {
                    "evidence_id": "negative_profit_rate",
                    "formatted": "18.72%",
                    "business_meaning": "负利润记录占比较高，应优先检查折扣与利润质量。",
                    "strength": 0.8,
                }
            ]
        },
        "distinctive_facts": [
            {
                "fact_id": "negative_profit_rate",
                "text": "负利润记录占比为 18.72%，折扣风险需要优先进入经营复盘。",
                "strength": 0.8,
                "source": "dataset_profile",
            }
        ],
        "limitations": [],
    }
    path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=NotebookOutline(
            title="Evidence Notebook",
            sections=[NotebookSection(section_id="conclusions", title="结论", purpose="总结。")],
        ),
        narrative=None,
        content_plan=NotebookContentPlan(sections=[]),
        dataset_profile={"negative_profit_rate": 0.1872},
        analysis_focus={"selected_focuses": ["discount_erosion_focus"]},
        evidence_pack=evidence_pack,
    )

    combined = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "0.1872" not in combined
    assert "0.9777" not in combined
    assert "negative_profit_rate" not in combined
    assert "Evidence Notebook" in combined


def test_notebook_builder_adds_kaggle_style_analysis_brief_and_dedupes_actions(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["总销售额为 1,250,000。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1250000.0},
                findings=["月度销售波动明显。"],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣利润",
                chart_type="bar",
                summary_metrics={
                    "total_profit_amount": 180000.0,
                    "negative_profit_rate": 0.1872,
                },
                findings=["高折扣记录利润质量偏弱。"],
            ),
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order ID": "order_id",
            "Customer ID": "customer_id",
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Product": "product_name",
            "Category": "category",
            "Segment": "segment",
            "Region": "region",
        },
        confidence=0.9,
    )
    evidence_pack = {
        "dataset_signature": {"dominant_story": "discount_loss"},
        "focus_evidence": {
            "discount_erosion_focus": [
                {
                    "evidence_id": "negative_profit_rate",
                    "business_meaning": "负利润记录占比为 18.72%，应优先检查折扣与利润质量。",
                }
            ],
            "profit_quality_focus": [
                {
                    "evidence_id": "negative_profit_rate",
                    "business_meaning": "负利润记录占比为 18.72%，应优先检查折扣与利润质量。",
                },
                {
                    "evidence_id": "profit_margin_spread",
                    "business_meaning": "利润率跨度为 77.00%，说明利润质量存在切片差异。",
                },
            ],
            "segment_region_focus": [
                {
                    "evidence_id": "weak_segment_region_slice",
                    "business_meaning": "弱势组合切片已经出现高销售低利润特征。",
                }
            ],
        },
        "distinctive_facts": [
            {"fact_id": "negative_profit_rate", "text": "负利润记录占比为 18.72%。"},
            {"fact_id": "profit_margin_spread", "text": "利润率跨度为 77.00%。"},
            {"fact_id": "monthly_volatility", "text": "月度波动率为 51.00%。"},
        ],
    }

    notebook_path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=NotebookOutline(
            title="Sales Notebook",
            sections=[NotebookSection(section_id="conclusions", title="结论", purpose="总结。")],
        ),
        narrative=None,
        content_plan=NotebookContentPlan(sections=[]),
        dataset_profile={
            "has_profit": True,
            "has_discount": True,
            "order_count": 5000,
            "customer_count": 793,
            "negative_profit_rate": 0.1872,
            "profit_margin_spread": 0.77,
            "monthly_volatility": 0.51,
        },
        analysis_focus={
            "selected_focuses": [
                "discount_erosion_focus",
                "profit_quality_focus",
                "segment_region_focus",
                "product_concentration_focus",
            ]
        },
        evidence_pack=evidence_pack,
    )

    markdown_cells = [
        "".join(cell.get("source", []))
        for cell in json.loads(notebook_path.read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "markdown"
    ]
    combined = "\n\n".join(markdown_cells)
    brief_index = next(index for index, cell in enumerate(markdown_cells) if "## Notebook Analysis Summary" in cell)
    if "## Conclusions and Recommendations" in combined:
        conclusion = combined.split("## Conclusions and Recommendations", maxsplit=1)[1]
    else:
        conclusion = combined.split("## 结论与行动建议", maxsplit=1)[1]

    assert brief_index >= 1
    assert "## Management Summary" not in combined
    assert "## Kaggle-style Analysis Summary" not in combined
    brief = markdown_cells[brief_index]
    assert "Dataset type judgment" not in brief
    assert "Agent-detected analysis goal" not in brief
    assert "Core analysis thread" not in brief
    assert "### KPI" in brief
    assert "### Key Findings" in brief
    assert "### Recommended Actions" not in brief
    assert "sales" in combined.lower()
    assert "profit" in combined.lower()
    assert "sales" in combined.lower()
    assert "[negative_profit_rate]" not in combined
    assert "profit" in combined.lower()
    assert "## Action Plan" not in combined
    assert conclusion.lower().count("approval") <= 1
    assert "high-sales low-profit" in conclusion.lower() or "profit" in conclusion.lower()
    assert "sales" in conclusion.lower()
    assert "profit" in conclusion.lower()


def test_final_notebook_hides_chart_selection_basis_and_keeps_one_late_section_conclusion(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0)
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount"},
        confidence=0.9,
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="metric_distributions",
                markdown_blocks=[
                    (
                        "本节用销售额分布判断长尾和异常值是否会影响均值判断。"
                        "这段导入后面如果还有很多解释，也应该在最终正文里被压缩。"
                    ),
                    "### 图表选择依据\n\n- 核心数值指标分布画像：ranking_factors=sales_amount, evidence_ids=e1, selection_source=llm。",
                    "### 本节结论\n\nLLM 结论：销售额 P99 明显高于中位数，应把极端订单单独复盘。",
                ],
                code_cells=[
                    "metric_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'metric_sales_distribution'}]}",
                    "sales_hist = clean_df['Sales'].describe()\nsales_hist",
                ],
            )
        ]
    )

    path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=NotebookOutline(
            title="Evidence Notebook",
            sections=[
                NotebookSection(section_id="metric_distributions", title="指标分布", purpose="解释分布。")
            ],
        ),
        narrative=None,
        content_plan=content_plan,
        notebook_output_mode="full",
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    cells_text = ["".join(cell.get("source", [])) for cell in data["cells"]]
    combined = "\n\n".join(cells_text)
    section_start = combined.index("## Metric Distributions") if "## Metric Distributions" in combined else combined.index("## 指标分布")
    code_index = combined.index("sales_hist =")
    conclusion_index = combined.find("### Section Conclusion", section_start)
    if conclusion_index == -1:
        conclusion_index = combined.find("### 本节结论", section_start)

    assert "## Notebook Analysis Summary" in combined
    assert "## Kaggle-style Analysis Summary" not in combined
    assert "## Management Summary" not in combined
    assert "### Chart Selection Rationale" not in combined
    assert "ranking_factors" not in combined
    assert "evidence_ids" not in combined
    assert "selection_source" not in combined
    assert "Question this section answers:" not in combined
    assert combined.count("### Section Conclusion") + combined.count("### 本节结论") <= 1
    if conclusion_index != -1:
        assert code_index < conclusion_index
    assert "sales_hist =" in combined
    assert "Key Evidence" not in combined


def test_final_notebook_hides_followup_question_and_bounds_speculation(
    tmp_path: Path,
) -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                findings=["高折扣订单利润偏弱。"],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount", "Discount": "discount", "Profit": "profit"},
        confidence=0.9,
    )
    content_plan = NotebookContentPlan(
        sections=[
            NotebookSectionContent(
                section_id="discount_and_profit",
                markdown_blocks=[
                    "折扣章节导入。Follow-up Question: 哪些审批流程存在漏洞？",
                    "### 本节结论\n\n审批流程存在漏洞，供应链成本过高，促销 ROI 不足，客户生命周期价值不足。",
                ],
                code_cells=[
                    "discount_profit_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'discount_vs_profit'}]}",
                    "discount_profit = clean_df[['Discount', 'Profit']]\ndiscount_profit.head()",
                ],
            )
        ]
    )

    path = build_notebook(
        task_id="task-1",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(),
        outline=NotebookOutline(
            title="Evidence Notebook",
            sections=[
                NotebookSection(section_id="discount_and_profit", title="折扣与利润分析", purpose="解释折扣。")
            ],
        ),
        narrative=None,
        content_plan=content_plan,
    )

    combined = path.read_text(encoding="utf-8")
    assert "Follow-up Question" not in combined
    assert "审批流程存在漏洞" not in combined
    assert "供应链成本过高" not in combined
    assert "促销 ROI 不足" not in combined
    assert "客户生命周期价值不足" not in combined
    assert "Follow-up Question" not in combined
