from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_postrun_reflection import (
    apply_postrun_chart_reflections,
    build_postrun_chart_reflections,
    compress_chart_reflection_text,
    extract_postrun_chart_contexts,
    _context_for_llm,
    _model_diagnostic_chart_type,
    _infer_chart_kind_from_code,
)
from app.services.llm_profile_policy import build_llm_profile_policy, resolve_max_postrun_reflections


def test_extract_postrun_chart_contexts_collects_chart_outputs_from_executed_notebook(
    tmp_path: Path,
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 商品与类目分析"),
            new_code_cell(
                "top_products = clean_df.groupby('Product Name', as_index=False)['Sales'].sum().head(5)\n"
                "top_products",
                outputs=[
                    new_output(
                        "execute_result",
                        data={
                            "text/plain": (
                                "                 Product Name     Sales\n"
                                "0  Canon imageCLASS 2200 Advanced Copier  61599.8240"
                            )
                        },
                        execution_count=8,
                    )
                ],
                execution_count=8,
            ),
            new_code_cell(
                "fig = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top 商品销售额对比')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [{"type": "bar", "name": "销售额"}],
                                "layout": {"title": {"text": "Top 商品销售额对比"}},
                            }
                        },
                    )
                ],
                execution_count=9,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    outline = NotebookOutline(
        title="销售数据分析 Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="商品与类目分析",
                purpose="识别商品与类目结构。",
            )
        ],
    )

    contexts = extract_postrun_chart_contexts(notebook_path, outline)

    assert len(contexts) == 1
    context = contexts[0]
    assert context["section_id"] == "product_and_category"
    assert context["section_title"] == "商品与类目分析"
    assert context["chart_title"] == "Top 商品销售额对比"
    assert "Canon imageCLASS 2200 Advanced Copier" in context["table_preview"]
    assert context["output_formats"] == ["application/vnd.plotly.v1+json"]
    assert context["chart_kind"] == "bar"
    assert "Canon imageCLASS 2200 Advanced Copier" in context["chart_summary"]


def test_extract_postrun_chart_contexts_matches_section_heading_even_with_intro_text(
    tmp_path: Path,
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 商品与类目分析\n\n这一节先看商品结构，再看图。"),
            new_code_cell(
                "fig = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top 商品销售额对比')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [{"type": "bar", "name": "销售额"}],
                                "layout": {"title": {"text": "Top 商品销售额对比"}},
                            }
                        },
                    )
                ],
                execution_count=9,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    outline = NotebookOutline(
        title="销售数据分析 Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="商品与类目分析",
                purpose="识别商品与类目结构。",
            )
        ],
    )

    contexts = extract_postrun_chart_contexts(notebook_path, outline)

    assert len(contexts) == 1
    assert contexts[0]["section_title"] == "商品与类目分析"


def test_extract_postrun_chart_contexts_infers_matplotlib_chart_kind_from_code(tmp_path: Path) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 折扣与利润分析"),
            new_code_cell(
                "plt.figure(figsize=(10, 5))\n"
                "sns.boxplot(data=discount_profit, x='discount_bucket', y='Profit')\n"
                "plt.title('不同折扣区间利润分布')\n"
                "plt.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "image/png": "abc123",
                            "text/plain": "<Figure size 1000x500 with 1 Axes>",
                        },
                    )
                ],
                execution_count=21,
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
                purpose="识别高折扣风险。",
            )
        ],
    )

    contexts = extract_postrun_chart_contexts(notebook_path, outline)

    assert len(contexts) == 1
    assert contexts[0]["chart_kind"] == "boxplot"
    assert "折扣区间" in contexts[0]["chart_summary"]


def test_extract_postrun_chart_contexts_treats_px_line_scatter_trace_as_line(tmp_path: Path) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 销售趋势分析"),
            new_code_cell(
                "fig = px.line(monthly_sales, x='order_month', y='Sales', title='按月销售额趋势')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [
                                    {
                                        "type": "scatter",
                                        "mode": "lines",
                                        "x": ["2014-01", "2014-03", "2015-07"],
                                        "y": [4526.0, 55691.009, 23648.287],
                                    }
                                ],
                                "layout": {"title": {"text": "按月销售额趋势"}},
                            }
                        },
                    )
                ],
                execution_count=18,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    outline = NotebookOutline(
        title="销售数据分析 Notebook",
        sections=[
            NotebookSection(
                section_id="sales_trends",
                title="销售趋势分析",
                purpose="观察销售额随时间变化。",
            )
        ],
    )

    contexts = extract_postrun_chart_contexts(notebook_path, outline)

    assert len(contexts) == 1
    assert contexts[0]["chart_kind"] == "line"
    assert "2014-03" in contexts[0]["chart_summary"]
    assert "2014-01" in contexts[0]["chart_summary"]


def test_extract_postrun_chart_contexts_treats_go_scatter_lines_as_line_and_markers_as_scatter(
    tmp_path: Path,
) -> None:
    line_notebook = new_notebook(
        cells=[
            new_markdown_cell("## Sales Trend Analysis"),
            new_code_cell(
                "fig = go.Figure()\n"
                "fig.add_trace(go.Scatter(x=monthly_sales['order_month'], y=monthly_sales['Sales'], mode='lines+markers'))\n"
                "fig.update_layout(title='Sales Trend with Rolling Average')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [
                                    {
                                        "type": "scatter",
                                        "mode": "lines+markers",
                                        "x": ["2014-01", "2014-02", "2014-03"],
                                        "y": [120.0, 180.0, 150.0],
                                    }
                                ],
                                "layout": {"title": {"text": "Sales Trend with Rolling Average"}},
                            }
                        },
                    )
                ],
                execution_count=18,
            ),
        ]
    )
    marker_notebook = new_notebook(
        cells=[
            new_markdown_cell("## Discount and Profit Analysis"),
            new_code_cell(
                "fig = go.Figure()\n"
                "fig.add_trace(go.Scatter(x=discount_profit['Discount'], y=discount_profit['Profit'], mode='markers'))\n"
                "fig.update_layout(title='Discount vs. Profit Relationship')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [
                                    {
                                        "type": "scatter",
                                        "mode": "markers",
                                        "x": [0.0, 0.2, 0.3],
                                        "y": [66.34, -12.0, -105.91],
                                    }
                                ],
                                "layout": {"title": {"text": "Discount vs. Profit Relationship"}},
                            }
                        },
                    )
                ],
                execution_count=19,
            ),
        ]
    )
    line_path = tmp_path / "line.executed.ipynb"
    marker_path = tmp_path / "marker.executed.ipynb"
    line_path.write_text(nbformat.writes(line_notebook), encoding="utf-8")
    marker_path.write_text(nbformat.writes(marker_notebook), encoding="utf-8")
    line_outline = NotebookOutline(
        title="Sales Data Analysis Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trend Analysis", purpose="Review trend.")],
    )
    marker_outline = NotebookOutline(
        title="Sales Data Analysis Notebook",
        sections=[
            NotebookSection(section_id="discount_and_profit", title="Discount and Profit Analysis", purpose="Review relationship.")
        ],
    )

    line_contexts = extract_postrun_chart_contexts(line_path, line_outline)
    marker_contexts = extract_postrun_chart_contexts(marker_path, marker_outline)

    assert len(line_contexts) == 1
    assert line_contexts[0]["chart_kind"] == "line"
    reflections, _trace = build_postrun_chart_reflections(
        chart_contexts=line_contexts,
        report=None,
        schema_mapping=None,
        llm_client=None,
        allow_fallback_reflections=True,
        output_language="en",
    )
    reflection = next(iter(reflections.values()))
    assert "trend" in reflection.lower() or "dated periods" in reflection.lower() or "rolling baseline" in reflection.lower()
    assert "visible relationship between the plotted measures" not in reflection
    assert "high-value and low-value clusters" not in reflection
    assert "scatter" not in reflection.lower()

    assert len(marker_contexts) == 1
    assert marker_contexts[0]["chart_kind"] == "scatter"


def test_infer_chart_kind_treats_plotly_secondary_bar_scatter_as_dual_axis() -> None:
    source = (
        "fig = make_subplots(specs=[[{'secondary_y': True}]])\n"
        "fig.add_trace(go.Bar(x=discount_profit['bucket'], y=discount_profit['avg_profit']), secondary_y=False)\n"
        "fig.add_trace(go.Scatter(x=discount_profit['bucket'], y=discount_profit['loss_rate']), secondary_y=True)\n"
        "fig.show()"
    )

    assert _infer_chart_kind_from_code(source) == "dual_axis_bar_line"


def test_extract_postrun_chart_contexts_extracts_matplotlib_pareto_title_and_kind(
    tmp_path: Path,
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 商品与类目分析"),
            new_code_cell(
                "fig, ax1 = plt.subplots(figsize=(10, 5))\n"
                "ax1.bar(pareto_df['Product Name'], pareto_df['Sales'])\n"
                "ax2 = ax1.twinx()\n"
                "ax2.plot(pareto_df['Product Name'], pareto_df['cumulative_share'])\n"
                "plt.title('头部商品帕累托分布')\n"
                "plt.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "image/png": "abc123",
                            "text/plain": "<Figure size 1000x500 with 2 Axes>",
                        },
                    )
                ],
                execution_count=32,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    outline = NotebookOutline(
        title="销售数据分析 Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="商品与类目分析",
                purpose="识别商品贡献结构。",
            )
        ],
    )

    contexts = extract_postrun_chart_contexts(notebook_path, outline)

    assert len(contexts) == 1
    assert contexts[0]["chart_title"] == "头部商品帕累托分布"
    assert contexts[0]["chart_kind"] == "pareto"
    assert "帕累托" in contexts[0]["chart_summary"]


def test_extract_postrun_chart_contexts_uses_table_preview_relevant_to_chart_code(
    tmp_path: Path,
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 商品与类目分析"),
            new_code_cell(
                "top_products = clean_df.groupby('Product Name', as_index=False)['Sales'].sum().sort_values('Sales', ascending=False).head(15)\n"
                "top_products",
                outputs=[
                    new_output(
                        "execute_result",
                        data={
                            "text/plain": (
                                "                          Product Name     Sales\n"
                                "0  Canon imageCLASS 2200 Advanced Copier  61599.8240"
                            )
                        },
                        execution_count=8,
                    )
                ],
                execution_count=8,
            ),
            new_code_cell(
                "category_sales = clean_df.groupby(['Category', 'Sub-Category'], as_index=False)['Sales'].sum()\n"
                "category_sales.head(20)",
                outputs=[
                    new_output(
                        "execute_result",
                        data={
                            "text/plain": (
                                "          Category Sub-Category      Sales\n"
                                "8  Office Supplies    Fasteners  3024.2800"
                            )
                        },
                        execution_count=9,
                    )
                ],
                execution_count=9,
            ),
            new_code_cell(
                "fig = px.treemap(category_sales, path=['Category', 'Sub-Category'], values='Sales', title='类目销售结构')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [{"type": "treemap"}],
                                "layout": {"title": {"text": "类目销售结构"}},
                            }
                        },
                    )
                ],
                execution_count=10,
            ),
            new_code_cell(
                "fig = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top 商品销售额对比')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [
                                    {
                                        "type": "bar",
                                        "x": [61599.8240],
                                        "y": ["Canon imageCLASS 2200 Advanced Copier"],
                                        "orientation": "h",
                                    }
                                ],
                                "layout": {"title": {"text": "Top 商品销售额对比"}},
                            }
                        },
                    )
                ],
                execution_count=11,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    outline = NotebookOutline(
        title="销售数据分析 Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="商品与类目分析",
                purpose="识别商品贡献结构。",
            )
        ],
    )

    contexts = extract_postrun_chart_contexts(notebook_path, outline)

    top_products_context = next(
        context for context in contexts if context["chart_title"] == "Top 商品销售额对比"
    )
    assert "Canon imageCLASS 2200 Advanced Copier" in top_products_context["table_preview"]
    assert "Fasteners" not in top_products_context["table_preview"]


def test_apply_postrun_chart_reflections_inserts_markdown_after_chart_cells(tmp_path: Path) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_markdown_cell("## 商品与类目分析"),
            new_code_cell(
                "fig = px.bar(top_products, x='Sales', y='Product Name', orientation='h', title='Top 商品销售额对比')\n"
                "fig.show()",
                outputs=[
                    new_output(
                        "display_data",
                        data={
                            "application/vnd.plotly.v1+json": {
                                "data": [{"type": "bar"}],
                                "layout": {"title": {"text": "Top 商品销售额对比"}},
                            }
                        },
                    )
                ],
                execution_count=9,
            ),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    output_path = tmp_path / "analysis.final.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    apply_postrun_chart_reflections(
        notebook_path=notebook_path,
        output_path=output_path,
        reflections_by_chart_cell={
            2: "该图显示头部商品销售额明显集中，Canon imageCLASS 2200 Advanced Copier 处于领先位置。"
        },
    )

    rendered = json.loads(output_path.read_text(encoding="utf-8"))
    cells = rendered["cells"]
    assert cells[3]["cell_type"] == "markdown"
    combined_markdown = "".join(cells[3]["source"])
    assert "图表解读" in combined_markdown


def test_apply_postrun_chart_reflections_skips_empty_reflection_cells(tmp_path: Path) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_code_cell("fig.show()"),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    output_path = tmp_path / "analysis.final.ipynb"
    nbformat.write(notebook, notebook_path)

    apply_postrun_chart_reflections(
        notebook_path=notebook_path,
        output_path=output_path,
        reflections_by_chart_cell={1: "", 2: "   "},
    )

    rendered = nbformat.read(output_path, as_version=4)
    assert len(rendered.cells) == 2


def test_apply_postrun_chart_reflections_removes_template_markers(tmp_path: Path) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("# 销售数据分析 Notebook"),
            new_code_cell("fig.show()"),
        ]
    )
    notebook_path = tmp_path / "analysis.executed.ipynb"
    output_path = tmp_path / "analysis.final.ipynb"
    nbformat.write(notebook, notebook_path)

    apply_postrun_chart_reflections(
        notebook_path=notebook_path,
        output_path=output_path,
        reflections_by_chart_cell={
            1: (
                "看到什么：30%+ 区间亏损率达到 75%。"
                "说明什么：折扣利润质量明显分化。"
                "下一步做什么：复盘高折扣订单。"
            )
        },
    )

    rendered = nbformat.read(output_path, as_version=4)
    markdown = "\n".join(str(cell.get("source", "")) for cell in rendered.cells)
    assert "看到什么" not in markdown
    assert "说明什么" not in markdown
    assert "下一步做什么" not in markdown
    assert "30%+" in markdown
    assert "75%" in markdown


def test_apply_postrun_chart_reflections_compresses_long_llm_text_and_bounds_speculation(
    tmp_path: Path,
) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## 折扣与利润分析"),
            new_code_cell(
                "fig = px.scatter(discount_profit, x='Discount', y='Profit', title='折扣与利润散点关系')\nfig.show()"
            ),
        ]
    )
    input_path = tmp_path / "input.ipynb"
    output_path = tmp_path / "output.ipynb"
    nbformat.write(notebook, input_path)
    long_reflection = (
        "折扣与利润呈明显负相关，折扣 30% 区间平均利润为 -18.4，亏损率达到 75%。"
        "这说明高折扣订单已经成为利润风险。"
        "同时还可以推断审批流程存在漏洞，供应链成本过高，促销 ROI 不足，客户生命周期价值不足。"
        "结合其他图表还可以继续展开商品、区域、客户、订单结构、活动策略、定价体系和运营流程等很多背景，"
        "这些扩展说明会让一段图表解读变得很长很难读。"
        "后续应优先复盘高折扣订单的商品和客户组合。"
    )

    apply_postrun_chart_reflections(
        input_path,
        output_path,
        {1: long_reflection},
    )

    rendered = nbformat.read(output_path, as_version=4)
    markdown = "\n\n".join(
        "".join(cell.get("source", []))
        for cell in rendered.cells
        if cell.get("cell_type") == "markdown"
    )
    inserted = markdown.split("### 图表解读", maxsplit=1)[1]
    chinese_chars = len([char for char in inserted if "\u4e00" <= char <= "\u9fff"])

    assert chinese_chars <= 280
    assert "-18.4" in inserted
    assert "75%" in inserted
    assert "审批流程存在漏洞" not in inserted
    assert "供应链成本过高" not in inserted
    assert "促销 ROI 不足" not in inserted
    assert "客户生命周期价值不足" not in inserted
    assert "需结合审批记录进一步验证折扣流程是否存在漏洞" in inserted


def test_compress_chart_reflection_keeps_three_sentence_reading_shape() -> None:
    text = (
        "STATUS 图显示 Shipped 占据主要订单数，说明销售额主要来自已发货订单，"
        "后续应复盘不同订单规模的客户服务和复购差异，判断是否存在优化空间。"
    )

    compressed = compress_chart_reflection_text(text)

    sentence_count = len([part for part in re.split(r"[。！？.!?\n]+", compressed) if part.strip()])
    assert sentence_count >= 3
    assert "Shipped" in compressed


class FakePostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"
    last_context = None

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        self.last_context = chart_context
        return {
            "reflection_markdown": (
                f"{chart_context['chart_title']} 已经显示出头部商品集中，"
                "其中 Canon imageCLASS 2200 Advanced Copier 的销售额约 61,599.824，需要和利润表一起复核。"
                "头部 SKU 高于第二梯队时，销售额集中但利润未必同步集中，当前增长质量可能被少数高收入低毛利商品稀释。"
                "下一步应把 Top 商品的销售额、利润和折扣放在同一张明细表里排序。"
            )
        }


def test_build_postrun_chart_reflections_uses_chart_context_and_llm() -> None:
    chart_contexts = [
        {
            "section_id": "product_and_category",
            "section_title": "商品与类目分析",
            "chart_title": "Top 商品销售额对比",
            "table_preview": "Canon imageCLASS 2200 Advanced Copier 61599.8240",
            "code_source": "fig = px.bar(...)",
            "cell_index": 3,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "头部对象是 Canon imageCLASS 2200 Advanced Copier，对应数值约为 61599.824",
        }
    ]
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["销售集中度较高。"],
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                findings=["头部商品贡献集中。"],
                summary_metrics={"top_product": "Canon imageCLASS 2200 Advanced Copier"},
                tables={
                    "top_products": [
                        {
                            "Product Name": "Canon imageCLASS 2200 Advanced Copier",
                            "Sales": 61599.824,
                        },
                        {
                            "Product Name": "Fellowes PB500 Electric Punch Plastic Comb Binding Machine",
                            "Sales": 27453.384,
                        },
                    ]
                },
            )
        ],
    )
    llm = FakePostrunReflectionLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        llm_client=llm,
    )

    assert trace.status == "llm_applied"
    assert 3 in reflections
    assert "Top 商品销售额对比" in reflections[3]
    assert llm.last_context["module_summary_metrics"]["top_product"] == "Canon imageCLASS 2200 Advanced Copier"
    assert "top_products" in llm.last_context["module_tables_preview"]
    assert (
        "Canon imageCLASS 2200 Advanced Copier"
        in llm.last_context["module_tables_preview"]["top_products"]
    )
    assert llm.last_context["module_findings"][0] == "头部商品贡献集中。"


class SectionContextCapturingLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.contexts = []

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        self.contexts.append(chart_context)
        return {
            "reflection_markdown": (
                f"{chart_context['chart_title']} 不能只单独看图形变化，"
                "还要和同一章节的其它图表一起判断销售结构是否健康。"
                "当前上下文给出了 Canon imageCLASS 2200 Advanced Copier 约 61,599.824 和类目结构两个信号，"
                "因此下一步应把头部商品、类目贡献和利润表现放在同一张复盘表里交叉验证。"
            )
        }


def test_build_postrun_chart_reflections_sends_section_peer_context_to_llm() -> None:
    chart_contexts = [
        {
            "section_id": "product_and_category",
            "section_title": "商品与类目分析",
            "chart_title": "类目销售结构",
            "table_preview": "Technology Phones 330007.0540",
            "code_source": "fig = px.treemap(...)",
            "cell_index": 10,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "treemap",
            "chart_summary": "Technology Phones 贡献最高，销售额约 330007.054。",
        },
        {
            "section_id": "product_and_category",
            "section_title": "商品与类目分析",
            "chart_title": "Top 商品销售额对比",
            "table_preview": "Canon imageCLASS 2200 Advanced Copier 61599.8240",
            "code_source": "fig = px.bar(...)",
            "cell_index": 11,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "头部对象是 Canon imageCLASS 2200 Advanced Copier，销售额约 61599.824。",
        },
    ]
    llm = SectionContextCapturingLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
    )

    assert trace.status == "llm_applied"
    assert set(reflections) == {10, 11}
    assert llm.contexts[0]["section_chart_count"] == 2
    assert llm.contexts[0]["section_chart_position"] == 1
    assert llm.contexts[1]["section_chart_position"] == 2
    assert llm.contexts[0]["section_peer_charts"][1]["chart_title"] == "Top 商品销售额对比"
    assert llm.contexts[1]["section_peer_charts"][0]["chart_title"] == "类目销售结构"


class DualAxisScatterWordingLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"
    last_context = None

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        self.last_context = chart_context
        return {
            "reflection_markdown": (
                "散点图清晰揭示了 30%+ 折扣区间平均利润偏低，亏损率同步升高。"
                "散点图显示高折扣订单的盈利质量已经明显弱于 0-10% 和 10-20% 区间。"
                "下一步应把该区间拆到商品和客户层面复盘。"
            )
        }


def test_dual_axis_bar_line_context_and_reflection_avoid_scatter_wording() -> None:
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": "平均利润与亏损率组合图",
            "table_preview": "30%+ -45.8 18.72%",
            "code_source": (
                "fig = make_subplots(specs=[[{'secondary_y': True}]])\n"
                "fig.add_trace(go.Bar(x=buckets, y=avg_profit), secondary_y=False)\n"
                "fig.add_trace(go.Scatter(x=buckets, y=loss_rate), secondary_y=True)"
            ),
            "cell_index": 18,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "dual_axis_bar_line",
            "chart_summary": "30%+ 区间平均利润最低，亏损率最高。",
        }
    ]
    llm = DualAxisScatterWordingLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
    )

    assert trace.status == "llm_applied"
    assert "散点图" not in reflections[18]
    assert "双轴图" in reflections[18]
    assert "双轴柱线组合图" in llm.last_context["analysis_focus"]
    assert "不是散点图" in llm.last_context["chart_kind_instruction"]


class CountingPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        self.calls.append(len(self.calls))
        return {
            "reflection_markdown": (
                "Executed evidence shows item A is around 123, which is large enough to treat this "
                "slice as a concrete operating signal rather than a decorative chart. The head slice "
                "is stronger than the tail slice, so the section should treat concentration as a real "
                "risk signal. Compared with the peer charts in the same section, the next action is to "
                "inspect whether discount, low margin, and product concentration are appearing together."
            )
        }


def test_build_postrun_chart_reflections_limits_live_llm_calls() -> None:
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": f"图表 {index}",
            "table_preview": "A 123",
            "code_source": "fig = px.bar(...)",
            "cell_index": index,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "A 项数值约 123。",
        }
        for index in range(5)
    ]
    llm = CountingPostrunReflectionLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
        max_llm_reflections=2,
    )

    assert llm.calls == [0, 1]
    assert set(reflections) == {0, 1}
    assert trace.status == "llm_partial"
    assert "limit" in trace.reason


class FailingPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        raise RuntimeError("LLM unavailable for this chart")


def test_quick_postrun_reflections_disable_deterministic_fallback() -> None:
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": f"折扣图表 {index}",
            "table_preview": "30%+ -46.25 75%",
            "code_source": "fig = px.bar(...)",
            "cell_index": index,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "30%+ 区间平均利润约 -46.25，亏损率约 75%。",
        }
        for index in range(3)
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=FailingPostrunReflectionLLM(),
        max_llm_reflections=3,
        allow_fallback_reflections=False,
        llm_profile="quick",
    )

    assert reflections == {}
    assert trace.status == "fallback_on_error"
    assert trace.applied is False
    assert "quick profile" in trace.reason
    assert "deterministic fallback chart reflections are disabled" in trace.reason


def test_full_postrun_reflections_disable_deterministic_fallback_on_llm_failure() -> None:
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": f"折扣图表 {index}",
            "table_preview": "30%+ -46.25 75%",
            "code_source": "fig = px.bar(...)",
            "cell_index": index,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "30%+ 区间平均利润约 -46.25，亏损率约 75%。",
        }
        for index in range(3)
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=FailingPostrunReflectionLLM(),
        max_llm_reflections=3,
        allow_fallback_reflections=False,
        llm_profile="full",
    )

    assert reflections == {}
    assert trace.status == "fallback_on_error"
    assert trace.applied is False
    assert "full profile uses broader LLM reflection coverage" in trace.reason
    assert "deterministic fallback chart reflections are disabled" in trace.reason
    assert "left blank" in trace.reason
    assert "deterministic fallback filled" not in trace.reason
    assert "handled by deterministic fallback" not in trace.reason


def test_quick_postrun_reflection_limit_uses_dynamic_profile_count() -> None:
    policy = build_llm_profile_policy("quick")
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": f"图表 {index}",
            "table_preview": "A 123",
            "code_source": "fig = px.bar(...)",
            "cell_index": index,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "A 项数值约 123。",
        }
        for index in range(8)
    ]
    llm = CountingPostrunReflectionLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
        max_llm_reflections=resolve_max_postrun_reflections(policy, len(chart_contexts)),
        allow_fallback_reflections=False,
        llm_profile="quick",
    )

    assert llm.calls == [0, 1, 2, 3]
    assert len(reflections) == 4
    assert "llm_profile=quick" in trace.reason
    assert "max_llm_reflections=4" in trace.reason
    assert "handled by deterministic fallback" not in trace.reason
    assert "deterministic fallback filled" not in trace.reason
    assert "quick profile" in trace.reason
    assert "deterministic fallback" in trace.reason
    assert "leaves unreflected charts blank" in trace.reason


def test_full_postrun_reflection_limit_leaves_unreflected_charts_blank() -> None:
    policy = build_llm_profile_policy("full")
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": f"图表 {index}",
            "table_preview": "A 123",
            "code_source": "fig = px.bar(...)",
            "cell_index": index,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "A 项数值约 123。",
        }
        for index in range(10)
    ]
    llm = CountingPostrunReflectionLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
        max_llm_reflections=resolve_max_postrun_reflections(policy, len(chart_contexts)),
        allow_fallback_reflections=False,
        llm_profile="full",
    )

    assert llm.calls == list(range(8))
    assert set(reflections) == set(range(8))
    assert 8 not in reflections
    assert 9 not in reflections
    assert "full profile uses broader LLM reflection coverage" in trace.reason
    assert "unreflected charts were left blank" in trace.reason
    assert "deterministic fallback filled" not in trace.reason
    assert "handled by deterministic fallback" not in trace.reason


class MetricsPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.metrics: list[dict[str, object]] = []

    @property
    def last_completion_metrics(self) -> dict[str, object]:
        return dict(self.metrics[-1]) if self.metrics else {}

    def snapshot_completion_metrics(self) -> int:
        return len(self.metrics)

    def collect_completion_metrics_since(self, snapshot_index: int) -> list[dict[str, object]]:
        return [dict(item) for item in self.metrics[snapshot_index:]]

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        self.metrics.append(
            {
                "stage": "postrun_chart_reflection",
                "cache_status": "bypassed",
                "prompt_chars": 100 + len(self.metrics),
                "response_chars": 80,
                "remote_elapsed_ms": 12.5,
                "saved_ms": 0,
                "cache_key": None,
                "error_type": None,
                "model": "fake-model",
                "source": "test",
            }
        )
        return {
            "reflection_markdown": (
                f"{chart_context['chart_title']} 显示 30%+ 区间亏损率达到 75%，"
                "同时和同节折扣区间对比一起看，低折扣区间与高折扣区间的利润质量已经明显分化。"
                "折扣审批失控是核心原因，并必然导致利润质量恶化，但这更需要回到明细里核对商品、客户和订单规模。"
                "下一步应把高折扣订单拆到商品和客户层面复盘，优先确认亏损是否集中在少数 SKU 或大额订单。"
            )
        }


def test_build_postrun_chart_reflections_aggregates_trace_and_guards_strong_claims() -> None:
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": f"折扣图 {index}",
            "table_preview": "30%+ 75%",
            "code_source": "fig = px.bar(...)",
            "cell_index": index,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "30%+ 区间亏损率 75%。",
        }
        for index in range(2)
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=MetricsPostrunReflectionLLM(),
        max_llm_reflections=2,
    )

    assert set(reflections) == {0, 1}
    assert trace.status == "llm_applied"
    assert trace.subcall_count == 2
    assert trace.aggregate_prompt_chars == 201
    assert trace.aggregate_remote_elapsed_ms == 25.0
    assert all("折扣审批失控" not in item for item in reflections.values())
    assert all("核心原因" not in item for item in reflections.values())
    assert all("可能的重要风险信号" in item for item in reflections.values())


def test_build_postrun_chart_reflections_backfills_core_sections_after_llm_budget() -> None:
    chart_contexts = [
        {
            "section_id": section_id,
            "section_title": section_title,
            "chart_title": chart_title,
            "table_preview": table_preview,
            "code_source": "fig = px.bar(...)",
            "cell_index": index,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": chart_summary,
        }
        for index, (section_id, section_title, chart_title, table_preview, chart_summary) in enumerate(
            [
                ("product_and_category", "商品与类目分析", "类目利润率质量对比", "A 123 4.8%", "A 类目销售额 123，利润率 4.8%。"),
                ("product_and_category", "商品与类目分析", "高销售低利润商品", "B 456 1.2%", "B 商品销售额 456，利润率 1.2%。"),
                ("discount_and_profit", "折扣与利润分析", "折扣与利润散点关系", "0.3 -18.4", "折扣 0.3 的利润约 -18.4。"),
                ("discount_and_profit", "折扣与利润分析", "各折扣区间利润质量：平均利润与亏损率", "30% 75%", "折扣区间亏损率 75%。"),
                ("segment_and_region", "客群与区域分析", "客群区域组合利润率", "Consumer / Central 12.3%", "组合切片利润率 12.3%。"),
            ]
        )
    ]
    llm = CountingPostrunReflectionLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
        max_llm_reflections=1,
    )

    assert llm.calls == [0]
    assert set(reflections) == {0, 1, 2, 3, 4}
    assert trace.status == "llm_partial"
    assert "deterministic fallback" in trace.reason


class PriorityCapturingPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.chart_titles: list[str] = []

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        self.chart_titles.append(str(chart_context["chart_title"]))
        return {
            "reflection_markdown": (
                f"{chart_context['chart_title']} 的执行结果已经给出足够明确的经营信号："
                "头部对象约 61,599.824，利润质量和同节图表之间存在可以交叉验证的结构差异。"
                "因此这一张图应该优先被解释，而不是让普通辅助图先消耗有限的 LLM 调用预算。"
                "下一步应先复盘该对象的销售额和利润是否同步。"
            )
        }


def test_build_postrun_chart_reflections_prioritizes_business_critical_charts() -> None:
    chart_contexts = [
        {
            "section_id": "metric_distributions",
            "section_title": "指标分布分析",
            "chart_title": "销量分布",
            "table_preview": "Quantity mean 3.79",
            "code_source": "sns.histplot(clean_df['Quantity'])",
            "cell_index": 1,
            "output_formats": ["image/png"],
            "chart_kind": "histogram",
            "chart_summary": "销量主要集中在 1 到 5 件。",
        },
        {
            "section_id": "product_and_category",
            "section_title": "商品与类目分析",
            "chart_title": "头部商品累计贡献 Pareto 分布",
            "table_preview": "Canon imageCLASS 2200 Advanced Copier 61599.8240",
            "code_source": "ax1.bar(...); ax2.plot(...); plt.title('头部商品累计贡献 Pareto 分布')",
            "cell_index": 2,
            "output_formats": ["image/png"],
            "chart_kind": "pareto",
            "chart_summary": "头部商品贡献明显集中。",
        },
        {
            "section_id": "segment_and_region",
            "section_title": "客群与区域分析",
            "chart_title": "区域销售额对比",
            "table_preview": "West 725457.8245",
            "code_source": "fig = px.bar(...)",
            "cell_index": 3,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "West 销售额最高。",
        },
    ]
    llm = PriorityCapturingPostrunReflectionLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
        max_llm_reflections=1,
    )

    assert llm.chart_titles == ["头部商品累计贡献 Pareto 分布"]
    assert set(reflections) == {2, 3}
    assert trace.status == "llm_partial"


class LowQualityPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        return {
            "reflection_markdown": (
                "这张图对应的是当前图。结合当前执行结果，最值得先看的不是图形样式，"
                "而是图里是否已经出现明显集中或分化。后续应该继续分析。"
            )
        }


def test_build_postrun_chart_reflections_rejects_boilerplate_llm_output() -> None:
    chart_contexts = [
        {
            "section_id": "sales_trends",
            "section_title": "销售趋势分析",
            "chart_title": "按月销售额趋势",
            "table_preview": "2014-03 55691.0090",
            "code_source": "fig = px.line(...)",
            "cell_index": 18,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "高点出现在 2014-03（约 55691.009），低点出现在 2014-01（约 4526.0）。",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=LowQualityPostrunReflectionLLM(),
    )

    assert trace.status == "fallback_on_error"
    assert reflections == {}
    assert "low-quality reflection" in trace.reason


def test_build_postrun_chart_reflections_uses_evidence_fallback_when_llm_is_disabled() -> None:
    chart_contexts = [
        {
            "section_id": "sales_trends",
            "section_title": "销售趋势分析",
            "chart_title": "按月销售额趋势",
            "table_preview": "2014-03 55691.0090",
            "code_source": "fig = px.line(...)",
            "cell_index": 18,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "高点出现在 2014-03（约 55691.009），低点出现在 2014-01（约 4526.0）",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=None,
    )

    assert trace.status == "fallback"
    assert 18 in reflections
    reflection = reflections[18]
    assert "2014-03" in reflection
    assert "2014-01" in reflection
    assert "高点" in reflection and "低点" in reflection
    assert len([part for part in reflection.replace("！", "。").replace("？", "。").split("。") if part.strip()]) >= 3
    assert "如果图表显示" not in reflection


class EnglishCjkPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0
        self.contexts: list[dict[str, object]] = []
        self.language_instructions: list[str | None] = []

    def suggest_postrun_chart_reflection(
        self,
        chart_context,
        fallback_markdown,
        language_instruction=None,
    ):
        self.calls += 1
        self.contexts.append(chart_context)
        self.language_instructions.append(language_instruction)
        return {
            "reflection_markdown": (
                "这张图显示 2014-03 销售额达到 55,691，明显高于 2014-01 的 4,526。"
                "高点和低点之间的差距说明销售节奏存在波动。"
                "下一步应复核该月份的促销、订单和库存明细。"
            )
        }


class UnsafeModelChartReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self, reflection: str) -> None:
        self.reflection = reflection
        self.contexts: list[dict[str, object]] = []

    def suggest_postrun_chart_reflection(
        self,
        chart_context,
        fallback_markdown,
        language_instruction=None,
    ):
        self.contexts.append(chart_context)
        return {"reflection_markdown": self.reflection}


def test_english_model_threshold_reflection_isolated_from_discount_and_what_if_evidence() -> None:
    llm = UnsafeModelChartReflectionLLM(
        "Raising the threshold reduces recall and conflicts with the discount tier's 97.81% negative-profit rate. "
        "It weakens the $51,490 discount-cap profit-improvement scenario."
    )
    chart_contexts = [
        {
            "section_id": "modeling",
            "section_title": "Modeling Analysis",
            "chart_title": "Model Review Threshold Trade-off",
            "business_question": "How should threshold choices affect review workload?",
            "table_preview": "threshold recall precision review_load\n0.40 0.92 0.64 120",
            "code_source": "threshold_df[['threshold', 'recall', 'precision', 'review_load']]",
            "cell_index": 31,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "threshold recall precision review_load false positives false negatives",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        schema_mapping=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[31]
    assert trace.status == "llm_applied"
    assert "discount tier" not in reflection.lower()
    assert "discount cap" not in reflection.lower()
    assert "profit improvement" not in reflection.lower()
    assert "$" not in reflection
    assert "threshold" in reflection.lower()
    assert sum(term in reflection.lower() for term in ("recall", "precision", "review workload", "review capacity")) >= 2
    assert "manual review" in reflection.lower() or "review capacity" in reflection.lower()
    assert not re.search(r"[\u3400-\u9fff]", reflection)
    assert "…" not in reflection


def test_english_confusion_matrix_reflection_isolated_from_product_margin_evidence() -> None:
    llm = UnsafeModelChartReflectionLLM(
        "The confusion matrix shows TP, FP, FN, and TN counts for the loss-risk classifier. "
        "It proves that the model can isolate the most damaging SKU, which has a -80% margin. "
        "Use the matrix to compare false positives and false negatives before manual review."
    )
    chart_contexts = [
        {
            "section_id": "modeling",
            "section_title": "Modeling Analysis",
            "chart_title": "Confusion Matrix",
            "business_question": "How many loss-risk orders are correctly identified?",
            "table_preview": "TP FP FN TN\n42 8 5 120",
            "code_source": "ConfusionMatrixDisplay.from_predictions(y_test, y_pred)",
            "cell_index": 32,
            "output_formats": ["image/png"],
            "chart_kind": "heatmap",
            "chart_summary": "TP FP FN TN recall precision false positives false negatives",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        schema_mapping=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[32]
    assert trace.status == "llm_applied"
    for phrase in ("sku", "margin", "product", "category", "region", "proves", "$"):
        assert phrase not in reflection.lower()
    assert any(term in reflection.lower() for term in ("tp", "fp", "fn", "tn", "false positive", "false negative"))
    assert "manual review" in reflection.lower() or "review threshold" in reflection.lower()
    assert not re.search(r"[\u3400-\u9fff]", reflection)
    assert "…" not in reflection


def test_english_postrun_reflection_rejects_cjk_and_uses_english_fallback() -> None:
    chart_contexts = [
        {
            "section_id": "sales_trends",
            "section_title": "Sales Trend Analysis",
            "chart_title": "Monthly Sales Trend",
            "table_preview": "2014-03 55691.0090\n2014-01 4526.0",
            "code_source": "fig = px.line(...)",
            "cell_index": 18,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "Peak appears in 2014-03 at about 55,691; trough appears in 2014-01 at about 4,526.",
        }
    ]
    llm = EnglishCjkPostrunReflectionLLM()

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    assert llm.calls == 1
    assert llm.language_instructions
    assert "English" in str(llm.language_instructions[0])
    assert trace.status == "fallback_on_error"
    assert "language_mismatch_fallback" in trace.reason
    assert "English reflection contained CJK" in trace.reason
    assert 18 in reflections
    assert "2014-03" in reflections[18]
    assert "2014-01" in reflections[18]
    assert not re.search(r"[\u3400-\u9fff]", reflections[18])
    assert "请优先" not in str(llm.contexts[0].get("analysis_focus"))
    assert "trend" in str(llm.contexts[0].get("analysis_focus")).lower()


class ChinesePostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_postrun_chart_reflection(
        self,
        chart_context,
        fallback_markdown,
        language_instruction=None,
    ):
        return {
            "reflection_markdown": (
                "这张图显示 2014-03 销售额达到 55,691，明显高于 2014-01 的 4,526。"
                "高点和低点之间的差距说明销售节奏存在波动。"
                "下一步应复核该月份的促销、订单和库存明细。"
            )
        }


def test_chinese_postrun_reflection_keeps_chinese_llm_output() -> None:
    chart_contexts = [
        {
            "section_id": "sales_trends",
            "section_title": "销售趋势分析",
            "chart_title": "按月销售额趋势",
            "table_preview": "2014-03 55691.0090\n2014-01 4526.0",
            "code_source": "fig = px.line(...)",
            "cell_index": 18,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "高点出现在 2014-03（约 55691.009），低点出现在 2014-01（约 4526.0）。",
        }
    ]

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=None,
        llm_client=ChinesePostrunReflectionLLM(),
        output_language="zh-CN",
    )

    assert trace.status == "llm_applied"
    assert "这张图显示" in reflections[18]


class UnsafeFreshGroundingReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self, reflection: str) -> None:
        self.reflection = reflection
        self.contexts: list[dict[str, object]] = []

    def suggest_postrun_chart_reflection(
        self,
        chart_context,
        fallback_markdown,
        language_instruction=None,
    ):
        self.contexts.append(chart_context)
        return {"reflection_markdown": self.reflection}


def test_english_category_chart_rejects_cross_grain_product_story() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "Cubify CubeX 3D Printer likely clusters within Furniture and inflates its cost or discount burden. "
        "The category chart compares profit-margin differences across visible category slices. "
        "Review weaker category slices against the underlying records before assigning product-level causes."
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="Product Analysis",
                chart_type="bar",
                findings=["Cubify CubeX 3D Printer belongs to Technology."],
                tables={
                    "top_products": [
                        {
                            "Product Name": "Cubify CubeX 3D Printer",
                            "Category": "Technology",
                            "Profit": -3839.99,
                        }
                    ]
                },
            )
        ],
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "product_and_category",
                "section_title": "Product and Category Analysis",
                "chart_title": "Category Profit Margin Comparison",
                "chart_id": "category_profit_margin",
                "business_question": "Which categories have weaker profit margin?",
                "table_preview": "Category profit_margin\nFurniture -0.08\nTechnology 0.12",
                "code_source": "category_profit = clean_df.groupby('Category')['Profit'].sum()",
                "cell_index": 41,
                "output_formats": ["application/vnd.plotly.v1+json"],
                "chart_kind": "bar",
                "chart_summary": "Compares profit-margin differences across category slices.",
            }
        ],
        report=report,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[41]
    assert trace.status == "llm_applied"
    for phrase in ("Cubify CubeX", "cluster within", "cost burden", "discount burden"):
        assert phrase.lower() not in reflection.lower()
    assert "category" in reflection.lower()
    assert "profit-margin" in reflection.lower() or "profit margin" in reflection.lower()


def test_english_category_chart_rejects_peer_product_scope_takeover() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "Category Profit Margin Comparison compares profit-margin differences across visible category slices. "
        "Review the visible category margin gap before assigning operating causes. "
        "The concurrent Top Product Profit Gap chart highlights individual SKU performance and specific Furniture items, "
        "likely compounds this issue by pinpointing Storage and Supplies sub-category drags down margins."
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="Product Analysis",
                chart_type="bar",
                tables={
                    "top_products": [
                        {
                            "Product Name": "Canon Storage Cabinet",
                            "Sub-Category": "Storage",
                            "Profit": -1200.0,
                        }
                    ]
                },
            )
        ],
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "product_and_category",
                "section_title": "Product and Category Analysis",
                "chart_title": "Category Profit Margin Comparison",
                "chart_id": "category_profit_margin",
                "business_question": "Which categories have weaker profit margin?",
                "table_preview": "Category Sales Profit profit_margin\nFurniture 1000 -80 -0.08\nTechnology 1200 144 0.12",
                "code_source": "category_profit = clean_df.groupby('Category')['Profit'].sum()",
                "cell_index": 43,
                "output_formats": ["application/vnd.plotly.v1+json"],
                "chart_kind": "bar",
                "chart_summary": "Compares profit-margin differences across category slices.",
            }
        ],
        report=report,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[43]
    lowered = reflection.lower()
    assert trace.status == "llm_applied"
    for phrase in (
        "top product",
        "sku",
        "individual",
        "specific furniture items",
        "storage",
        "supplies",
        "sub-category",
        "compounds",
        "drags down",
        "canon storage cabinet",
    ):
        assert phrase not in lowered
    assert "category" in lowered
    assert "profit-margin" in lowered or "profit margin" in lowered


def test_english_product_chart_preserves_visible_product_entity() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "Cubify CubeX 3D Printer has weaker profit margin than peer products in the visible product comparison. "
        "Review the product-level records before changing broader category rules. "
        "Use the visible product comparison as the scope for follow-up."
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "product_and_category",
                "section_title": "Product and Category Analysis",
                "chart_title": "Product Profit Margin Comparison",
                "chart_id": "product_profit_margin",
                "business_question": "Which products have weaker profit margin?",
                "table_preview": "Product Name profit_margin\nCubify CubeX 3D Printer -0.42\nPrinter Stand 0.08",
                "code_source": "product_profit = clean_df.groupby('Product Name')['Profit'].sum()",
                "cell_index": 42,
                "output_formats": ["application/vnd.plotly.v1+json"],
                "chart_kind": "bar",
                "chart_summary": "Compares profit margin across visible product slices.",
            }
        ],
        report=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[42]
    assert trace.status == "llm_applied"
    assert "Cubify CubeX 3D Printer" in reflection
    assert "visible product comparison" in reflection


def test_english_scatter_reflection_rejects_ungrounded_correlation_strength() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "The scatter plot reveals a clear negative correlation (r = -0.2189) between discount and profit. "
        "Review the observed association in the plotted records before broader pricing changes. "
        "Use the visible scatter pattern as scoped evidence only."
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "discount_and_profit",
                "section_title": "Discount and Profit Analysis",
                "chart_title": "Discount vs. Profit Relationship",
                "chart_id": "discount_profit_scatter",
                "business_question": "How do discount and profit relate in plotted records?",
                "table_preview": "Discount Profit\n0.10 12.0\n0.30 -3.0",
                "code_source": "fig = px.scatter(discount_profit, x='Discount', y='Profit')",
                "cell_index": 44,
                "output_formats": ["application/vnd.plotly.v1+json"],
                "chart_kind": "scatter",
                "chart_summary": "Shows plotted discount and profit records.",
            }
        ],
        report=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[44]
    lowered = reflection.lower()
    assert trace.status == "llm_applied"
    for phrase in ("clear negative correlation", "strongly correlated", "highly correlated", "-0.2189"):
        assert phrase not in lowered
    assert "association" in lowered or "plotted" in lowered
    assert not reflection.rstrip().endswith(("and", "or", "with", "to", "of", "for", "in", "at", "by", "from"))


def test_english_threshold_reflection_rejects_cross_evidence_and_invented_numbers() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "Raising the threshold reduces recall. "
        "The inflection point appears near 60-70% recall and conflicts with the discount-profit relationship of -0.2189."
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "modeling",
                "section_title": "Modeling Analysis",
                "chart_title": "Model Review Threshold Trade-off",
                "chart_id": "model_threshold_tradeoff",
                "business_question": "How should threshold choices affect review workload?",
                "table_preview": "threshold recall precision review_load\n0.40 0.92 0.64 120",
                "code_source": "display(high_discount_examples[['Category','discount','Profit']])",
                "cell_index": 51,
                "output_formats": ["application/vnd.plotly.v1+json"],
                "chart_kind": "line",
                "chart_summary": "threshold recall precision review_load false positives false negatives",
            }
        ],
        report=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[51]
    assert trace.status == "llm_applied"
    for phrase in ("60-70%", "60–70%", "-0.2189", "discount", "profit", "conflicts with"):
        assert phrase not in reflection
    assert "threshold" in reflection.lower()
    assert sum(term in reflection.lower() for term in ("recall", "review capacity", "false-positive", "false-negative", "manual review")) >= 2


def test_english_confusion_matrix_rejects_cross_chart_business_evidence() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "False negatives likely hide the true scale of losses in high-discount segments. "
        "Given the discount-profit correlation of -0.2189, adjust the threshold to reduce profit losses."
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "modeling",
                "section_title": "Modeling Analysis",
                "chart_title": "Confusion Matrix",
                "chart_id": "loss_risk_confusion_matrix",
                "business_question": "How many loss-risk orders are correctly identified?",
                "table_preview": "TP FP FN TN\n42 8 5 120",
                "code_source": "display(false_negative_loss_examples[['Product Name','Discount','Profit']])",
                "cell_index": 52,
                "output_formats": ["image/png"],
                "chart_kind": "heatmap",
                "chart_summary": "TP FP FN TN recall precision false positives false negatives",
            }
        ],
        report=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[52]
    assert trace.status == "llm_applied"
    for phrase in ("discount", "-0.2189", "profit", "segments", "true scale", "adjust the threshold to reduce profit losses"):
        assert phrase not in reflection.lower()
    assert sum(term in reflection.lower() for term in ("confusion matrix", "false positives", "false negatives", "review threshold", "manual review")) >= 2


def test_english_confusion_matrix_at_default_threshold_precedes_threshold_fallback() -> None:
    chart_context = {
        "section_id": "modeling",
        "section_title": "Modeling Analysis",
        "chart_title": "Confusion Matrix at Default Threshold",
        "chart_id": "loss_risk_confusion_matrix",
        "business_question": "How many loss-risk orders are correctly identified?",
        "table_preview": "TP FP FN TN threshold\n42 8 5 120 0.50",
        "code_source": (
            "default_threshold = 0.5\n"
            "threshold = default_threshold\n"
            "ConfusionMatrixDisplay.from_predictions(y_test, y_pred)"
        ),
        "cell_index": 55,
        "output_formats": ["image/png"],
        "chart_kind": "heatmap",
        "chart_summary": "TP FP FN TN false positives false negatives at the default threshold",
    }

    assert _model_diagnostic_chart_type(chart_context) == "confusion_matrix"

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[chart_context],
        report=None,
        llm_client=None,
        allow_fallback_reflections=True,
        output_language="en",
    )

    reflection = reflections[55].lower()
    assert trace.status == "fallback"
    assert "correctly identified loss-risk orders" in reflection
    assert "missed orders" in reflection
    assert "extra review load" in reflection
    assert "false positives" in reflection or "false-positive" in reflection
    assert "false negatives" in reflection or "false-negative" in reflection
    assert "manual review" in reflection or "manual-review" in reflection
    assert "compares model recall with review workload across threshold choices" not in reflection
    assert "across threshold choices" not in reflection


def test_english_confusion_matrix_without_structured_evidence_uses_deterministic_fallback() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "The high true negative count shows the model is overly aggressive. "
        "The concerning false positives create a significant review workload. "
        "If deployed in its current state, the model could strain operations."
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "modeling",
                "section_title": "Modeling Analysis",
                "chart_title": "Confusion Matrix at Default Threshold",
                "chart_id": "loss_risk_confusion_matrix",
                "business_question": "How many loss-risk orders are correctly identified?",
                "table_preview": "Figure size 640x480; confusion matrix rendered as an image.",
                "code_source": "ConfusionMatrixDisplay.from_predictions(y_test, y_pred)",
                "cell_index": 57,
                "output_formats": ["image/png"],
                "chart_kind": "heatmap",
                "chart_summary": "Confusion matrix image for the default threshold.",
            }
        ],
        report=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[57].lower()
    assert trace.status == "llm_applied"
    for phrase in (
        "high true negative count",
        "concerning",
        "overly aggressive",
        "significant review workload",
        "deployed",
        "current state",
    ):
        assert phrase not in reflection
    assert "missed orders" in reflection
    assert "extra review load" in reflection
    assert "false positives" in reflection or "false-positive" in reflection
    assert "false negatives" in reflection or "false-negative" in reflection
    assert "manual review" in reflection or "manual-review" in reflection


def test_english_confusion_matrix_with_structured_evidence_preserves_safe_numbers() -> None:
    llm = UnsafeFreshGroundingReflectionLLM(
        "The confusion matrix shows FP 8 as a concerning false-positive count. "
        "The confusion matrix lists TP 42, FP 8, FN 5, and TN 120 for the current validation records. "
        "Review false positives and false negatives before changing the review threshold. "
        "Use the model only for manual review prioritization."
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=[
            {
                "section_id": "modeling",
                "section_title": "Modeling Analysis",
                "chart_title": "Confusion Matrix at Default Threshold",
                "chart_id": "loss_risk_confusion_matrix",
                "business_question": "How many loss-risk orders are correctly identified?",
                "table_preview": "TP FP FN TN\n42 8 5 120",
                "code_source": "ConfusionMatrixDisplay.from_predictions(y_test, y_pred)",
                "cell_index": 58,
                "output_formats": ["image/png"],
                "chart_kind": "heatmap",
                "chart_summary": "TP 42 FP 8 FN 5 TN 120",
            }
        ],
        report=None,
        llm_client=llm,
        allow_fallback_reflections=False,
        output_language="en",
    )

    reflection = reflections[58]
    lowered = reflection.lower()
    assert trace.status == "llm_applied"
    for value in ("42", "8", "5", "120"):
        assert value in reflection
    for phrase in ("high", "concerning", "overly aggressive", "deployed", "discount", "profit"):
        assert phrase not in lowered
    assert "lists TP 42" in reflection
    assert "correctly identified loss-risk orders with missed orders" not in reflection


def test_english_feature_importance_identity_precedes_threshold_code_keywords() -> None:
    chart_context = {
        "section_id": "modeling",
        "section_title": "Modeling Analysis",
        "chart_title": "Feature Importance Ranking",
        "chart_id": "loss_risk_feature_importance",
        "business_question": "Which predictive signals are most visible in the model?",
        "table_preview": "feature importance\nDiscount 0.42\nSales 0.25",
        "code_source": "threshold = 0.5\nthreshold_df = pd.DataFrame()\nplot_feature_importance(model)",
        "cell_index": 56,
        "output_formats": ["image/png"],
        "chart_kind": "bar",
        "chart_summary": "Feature importance ranks predictive signals used by the model.",
    }

    assert _model_diagnostic_chart_type(chart_context) == "feature_importance"


def test_english_model_diagnostic_context_hides_raw_code_and_cross_chart_evidence() -> None:
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Discount-profit correlation is -0.2189."],
        modules=[
            ModuleReport(
                module_id="modeling_analysis",
                title="Modeling",
                chart_type="heatmap",
                findings=["High-discount segments had weak profit."],
                summary_metrics={"recall": 0.92, "precision": 0.64},
                tables={"examples": [{"Product Name": "Example Product", "Category": "Furniture"}]},
            )
        ],
    )
    context = _context_for_llm(
        {
            "section_id": "modeling",
            "section_title": "Modeling Analysis",
            "chart_title": "Confusion Matrix",
            "chart_id": "loss_risk_confusion_matrix",
            "business_question": "How many loss-risk orders are correctly identified?",
            "table_preview": "TP FP FN TN\n42 8 5 120",
            "code_source": (
                "ConfusionMatrixDisplay.from_predictions(y_test, y_pred)\n"
                "display(False Negative Loss Examples[['sales_amount','category','sub_category','segment','region','Discount']])"
            ),
            "cell_index": 53,
            "output_formats": ["image/png"],
            "chart_kind": "heatmap",
            "chart_summary": "TP FP FN TN recall precision false positives false negatives",
            "model_metrics": {"recall": 0.92, "precision": 0.64},
            "model_metric_evidence": "Validation recall and precision are available.",
            "axis_labels": {"x": "Predicted label", "y": "Actual label"},
            "axis_metadata": {"x_values": ["Predicted positive", "Predicted negative"]},
            "label_metadata": {"positive_label": "loss-risk order"},
        },
        report=report,
        section_chart_contexts=None,
        output_language="en",
    )
    serialized = json.dumps(context, ensure_ascii=False)
    allowed_keys = {
        "chart_title",
        "chart_kind",
        "model_diagnostic_chart_type",
        "chart_summary",
        "table_preview",
        "model_metrics",
        "model_metric_evidence",
        "axis_labels",
        "axis_metadata",
        "label_metadata",
        "analysis_focus",
        "section_review_instruction",
    }
    forbidden_keys = {
        "code_source",
        "module_summary_metrics",
        "module_findings",
        "module_tables_preview",
        "report_summary",
        "section_peer_charts",
        "section_chart_count",
        "section_chart_position",
        "section_id",
        "section_title",
        "image_urls",
        "output_formats",
    }

    assert context["model_diagnostic_chart_type"] == "confusion_matrix"
    assert context["chart_title"] == "Confusion Matrix"
    assert set(context) <= allowed_keys
    assert forbidden_keys.isdisjoint(context)
    for required_key in (
        "chart_title",
        "chart_kind",
        "model_diagnostic_chart_type",
        "chart_summary",
        "table_preview",
        "model_metrics",
        "model_metric_evidence",
        "axis_labels",
        "axis_metadata",
        "label_metadata",
        "analysis_focus",
        "section_review_instruction",
    ):
        assert required_key in context
    for phrase in (
        "sales_amount",
        "category",
        "sub_category",
        "segment",
        "region",
        "High-risk Examples",
        "False Negative Loss Examples",
        "Discount",
        "Discount-profit correlation",
        "High-discount segments",
        "module_findings",
        "module_tables_preview",
        "report_summary",
    ):
        assert phrase not in serialized


def test_english_non_model_context_keeps_existing_code_source() -> None:
    context = _context_for_llm(
        {
            "section_id": "sales_trends",
            "section_title": "Sales Trend Analysis",
            "chart_title": "Monthly Sales Trend",
            "chart_id": "monthly_sales_trend",
            "business_question": "How did sales move over time?",
            "table_preview": "month Sales\n2024-01 100",
            "code_source": "fig = px.line(monthly_sales, x='month', y='Sales')",
            "cell_index": 54,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "Monthly sales trend.",
        },
        report=None,
        section_chart_contexts=None,
        output_language="en",
    )

    assert context.get("model_diagnostic_chart_type") is None
    assert "px.line" in str(context.get("code_source"))


def test_apply_postrun_chart_reflections_uses_language_specific_heading(tmp_path: Path) -> None:
    notebook = new_notebook(
        cells=[
            new_markdown_cell("## Sales Trend Analysis"),
            new_code_cell("fig.show()"),
        ]
    )
    notebook_path = tmp_path / "analysis.ipynb"
    notebook_path.write_text(nbformat.writes(notebook), encoding="utf-8")

    apply_postrun_chart_reflections(
        notebook_path=notebook_path,
        output_path=notebook_path,
        reflections_by_chart_cell={
            1: "Sales peaked at 55,691 in 2014-03. It was higher than the 4,526 trough in 2014-01. Review promotion and order details before action."
        },
        output_language="en",
    )

    markdown = "\n\n".join(
        str(cell.source)
        for cell in nbformat.read(notebook_path, as_version=4).cells
        if cell.cell_type == "markdown"
    )
    assert "### Chart Commentary" in markdown
    assert "### 图表解读" not in markdown
    assert not re.search(r"[\u3400-\u9fff]", markdown)


def test_build_postrun_chart_reflections_fallback_uses_module_metrics_and_tables() -> None:
    chart_contexts = [
        {
            "section_id": "discount_and_profit",
            "section_title": "折扣与利润分析",
            "chart_title": "Profit Distribution by Discount Bucket",
            "table_preview": "0% 120.4\n30%+ -45.8",
            "code_source": "sns.boxplot(data=discount_profit, x='discount_bucket', y='Profit')",
            "cell_index": 56,
            "output_formats": ["image/png"],
            "chart_kind": "boxplot",
            "chart_summary": "用于比较不同折扣区间下利润分布的中位数、波动范围和异常点。",
        }
    ]
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="boxplot",
                findings=[],
                summary_metrics={
                    "high_discount_order_count": 1393,
                    "negative_profit_rate": 0.1872,
                },
                tables={
                    "discount_buckets": [
                        {"discount_bucket": "0%", "avg_profit": 120.4},
                        {"discount_bucket": "30%+", "avg_profit": -45.8},
                    ]
                },
            )
        ],
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        llm_client=None,
    )

    assert trace.status == "fallback"
    assert 56 in reflections
    reflection = reflections[56]
    assert "1,393" in reflection or "1393" in reflection
    assert "18.72%" in reflection
    assert "30%+" in reflection
    assert "-45.80" in reflection or "-45.8" in reflection
    assert len([part for part in reflection.replace("！", "。").replace("？", "。").split("。") if part.strip()]) >= 3


def test_build_postrun_chart_reflections_forecast_fallback_when_llm_disabled() -> None:
    chart_contexts = [
        {
            "section_id": "forecast",
            "section_title": "预测与展望",
            "chart_title": "实际与预测销售额对比",
            "table_preview": "2018-01 93351.355667\n2018-02 93351.355667",
            "code_source": (
                "baseline_value = float(forecast_source['Sales'].tail(3).mean())\n"
                "forecast_df['forecast_sales'] = baseline_value\n"
                "fig = px.line(combined_plot_df, x='order_month', y='Sales', color='series_name')"
            ),
            "cell_index": 62,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "line",
            "chart_summary": "主要观察时间趋势和波动节奏。",
        }
    ]
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="forecast_analysis",
                title="预测分析",
                chart_type="line",
                findings=[],
                summary_metrics={
                    "baseline_sales_amount": 2172.98,
                    "horizon_days": 28,
                },
                tables={"forecast": [{"order_month": "2018-01", "forecast_sales": 2172.98}]},
            )
        ],
    )

    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=chart_contexts,
        report=report,
        llm_client=None,
    )

    assert trace.status == "fallback"
    assert 62 in reflections
    reflection = reflections[62]
    assert "28" in reflection
    assert "2,172.98" in reflection or "2172.98" in reflection
    assert "2018-01" in reflection or "2018-02" in reflection


class FixedPostrunReflectionLLM:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def __init__(self, reflection: str) -> None:
        self.reflection = reflection

    def suggest_postrun_chart_reflection(self, chart_context, fallback_markdown):
        return {"reflection_markdown": self.reflection}


def _single_postrun_chart_context() -> list[dict[str, object]]:
    return [
        {
            "section_id": "product_and_category",
            "section_title": "商品与类目分析",
            "chart_title": "Top 商品销售额对比",
            "table_preview": "A 1000\nB 800",
            "code_source": "fig = px.bar(top_products, x='Sales', y='Product')",
            "cell_index": 91,
            "output_formats": ["application/vnd.plotly.v1+json"],
            "chart_kind": "bar",
            "chart_summary": "头部商品销售额较高。",
        }
    ]


def test_postrun_reflection_drops_current_discount_claims_when_discount_missing() -> None:
    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=_single_postrun_chart_context(),
        report=None,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        llm_client=FixedPostrunReflectionLLM(
            "看到什么：图表显示 A 商品销售额为 1000，B 商品销售额为 800，销售集中度较高。"
            "说明什么：需要复核折扣策略和折扣审批阈值，避免按折扣区间误判利润质量。"
            "下一步做什么：对高折扣订单做专项分析，并把 Top 商品回到明细复核。"
        ),
        allow_fallback_reflections=False,
    )

    assert trace.status == "fallback_on_error"
    assert reflections == {}


def test_postrun_reflection_keeps_missing_discount_limitation_when_discount_missing() -> None:
    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=_single_postrun_chart_context(),
        report=None,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        llm_client=FixedPostrunReflectionLLM(
            "看到什么：图表显示 A 商品销售额为 1000，B 商品销售额为 800，销售集中度较高。"
            "说明什么：缺少 discount 字段，当前无法进行折扣分析，不能直接判断折扣影响。"
            "下一步做什么：补充折扣字段后再验证，并先按商品利润质量复盘。"
        ),
        allow_fallback_reflections=False,
    )

    assert trace.status == "llm_applied"
    assert 91 in reflections
    assert "缺少 discount 字段" in reflections[91]


def test_postrun_reflection_keeps_cost_pricing_fulfillment_without_discount() -> None:
    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=_single_postrun_chart_context(),
        report=None,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        llm_client=FixedPostrunReflectionLLM(
            "看到什么：图表显示 A 商品销售额为 1000，B 商品销售额为 800，头部商品销售额较高。"
            "说明什么：需要结合成本、定价和履约成本判断利润质量，不能只看销售额。"
            "下一步做什么：回到商品明细复核，并优先检查低利润商品。"
        ),
        allow_fallback_reflections=False,
    )

    assert trace.status == "llm_applied"
    assert 91 in reflections
    assert "成本、定价和履约成本" in reflections[91]


def test_postrun_reflection_keeps_discount_strategy_when_discount_mapped() -> None:
    reflections, trace = build_postrun_chart_reflections(
        chart_contexts=_single_postrun_chart_context(),
        report=None,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        llm_client=FixedPostrunReflectionLLM(
            "看到什么：图表显示 A 商品销售额为 1000，B 商品销售额为 800，销售集中度较高。"
            "说明什么：需要复核折扣策略和折扣审批阈值，避免按折扣区间误判利润质量。"
            "下一步做什么：对高折扣订单做专项分析，并把 Top 商品回到明细复核。"
        ),
        allow_fallback_reflections=False,
    )

    assert trace.status == "llm_applied"
    assert 91 in reflections
    assert "折扣策略" in reflections[91]
