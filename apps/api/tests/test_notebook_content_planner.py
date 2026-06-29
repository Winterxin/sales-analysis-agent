from __future__ import annotations

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan, NotebookSectionContent
from app.schemas.notebook_narrative import NotebookNarrative, NotebookSectionNarrative
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_content_planner import (
    _apply_chart_selection_to_section,
    _ensure_selected_chart_ids_cell,
    _filter_chart_display_cell,
    _fallback_content,
    _missing_selected_chart_render_reasons,
    _without_non_llm_chart_display_cells,
    build_notebook_content,
    build_notebook_content_with_trace,
)
from app.services.notebook.output_policy import apply_notebook_output_policy
from app.services.notebook.section_renderer import (
    code_step_intro,
    prepare_final_section_markdown_blocks,
    section_conclusion_markdown,
    section_intro_markdown,
)


def test_chart_display_cells_are_hidden_only_without_llm_sanitized_decision() -> None:
    code_cells = [
        "discount_profit_strategy = {'source': 'deterministic_fallback', 'views': [{'chart': 'discount_vs_profit'}]}",
        "fig = px.scatter(discount_profit, x='Discount', y='Profit')\nfig.show()",
        "discount_profit_strategy = {'source': 'llm_sanitized', 'views': [{'chart': 'discount_bucket_boxplot'}]}",
        "plt.figure(figsize=(8, 4))\nsns.boxplot(data=discount_profit, x='discount_bucket', y='Profit')\nplt.show()",
    ]

    visible_cells = _without_non_llm_chart_display_cells(code_cells)

    combined = "\n\n".join(visible_cells)
    assert "px.scatter" not in combined
    assert "deterministic_fallback" in combined
    assert "sns.boxplot" in combined
    assert "llm_sanitized" in combined


def test_notebook_section_renderer_generates_english_source_text_without_changing_chinese_default() -> None:
    section = NotebookSection(
        section_id="discount_and_profit",
        title="Discount and Profit Analysis",
        purpose="Assess margin risk.",
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount", "Profit": "profit"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=0,
        summary=[],
        modules=[],
    )

    intro_en = section_intro_markdown(section, output_language="en")
    conclusion_en = section_conclusion_markdown(section, report, schema_mapping, output_language="en")
    step_intro_en = code_step_intro(
        "discount_and_profit",
        "discount_profit = clean_df.copy()\nfig = px.scatter(discount_profit)",
        output_language="en",
    )
    intro_blocks_en, _analysis_blocks, conclusion_blocks_en = prepare_final_section_markdown_blocks(
        ["### 本节结论\n\n折扣风险需要结合利润记录复核。"],
        section,
        schema_mapping,
        output_language="en",
    )

    english_text = "\n".join(
        item
        for item in [intro_en, conclusion_en, step_intro_en, *intro_blocks_en, *conclusion_blocks_en]
        if item
    )
    assert "Assess whether higher discounts are eroding profit quality" in english_text
    assert "### Section Takeaway" not in english_text
    assert "Prepare the discount and profit base table" in english_text
    assert "This section identifies the records and business objects that need follow-up review" not in english_text
    assert "mapped fields and module outputs define the current review scope" not in english_text
    for forbidden in ("本节结论", "折扣", "利润", "复盘", "数据限制说明"):
        assert forbidden not in english_text

    assert "本节围绕折扣与利润关系展开" in (section_intro_markdown(section) or "")


def test_notebook_content_planner_generates_english_source_labels_for_extended_sections() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(section_id="metric_distributions", title="Metric Distributions", purpose="Profile metrics."),
            NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trend."),
            NotebookSection(section_id="product_and_category", title="Product and Category", purpose="Analyze product mix."),
            NotebookSection(section_id="country_market", title="Country Market", purpose="Analyze markets."),
            NotebookSection(section_id="order_structure", title="Order Structure", purpose="Analyze order structure."),
            NotebookSection(section_id="discount_and_profit", title="Discount and Profit", purpose="Analyze margin."),
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Profit": "profit",
            "Discount": "discount",
            "Product": "product_name",
            "Category": "category",
            "Sub-Category": "sub_category",
            "PRODUCTLINE": "productline",
            "COUNTRY": "country",
            "Customer": "customer_id",
            "Order ID": "order_id",
            "Unit Price": "unit_price",
            "DEALSIZE": "deal_size",
            "STATUS": "order_status",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0, summary=[], modules=[])
    analysis_plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    chart_selection_plan = {
        "selected_charts": [
            {"section_id": "metric_distributions", "chart_id": "quantity_distribution", "title": "Quantity Distribution"},
            {"section_id": "sales_trends", "chart_id": "sales_trends_monthly_line", "title": "Monthly Sales Trend"},
            {"section_id": "product_and_category", "chart_id": "top_product_profit_gap_bar", "title": "Top Product Profit Gap"},
            {"section_id": "product_and_category", "chart_id": "productline_sales_bar", "title": "Product Line Sales"},
            {"section_id": "country_market", "chart_id": "country_sales_bar", "title": "Sales by Country"},
            {"section_id": "order_structure", "chart_id": "order_status_breakdown", "title": "Sales Structure by Status"},
            {"section_id": "order_structure", "chart_id": "unit_price_quantity_scatter", "title": "Unit Price vs Quantity"},
            {"section_id": "discount_and_profit", "chart_id": "discount_profit_quality_bar", "title": "Discount Tier Profit Quality"},
            {"section_id": "discount_and_profit", "chart_id": "discount_vs_profit_scatter", "title": "Discount vs Profit"},
        ]
    }

    content = _fallback_content(
        outline,
        narrative,
        report,
        schema_mapping,
        analysis_plan,
        chart_selection_plan=chart_selection_plan,
        output_language="en",
    )

    combined = "\n\n".join(
        block
        for section in content.sections
        for block in [*section.markdown_blocks, *section.code_cells]
    )
    for forbidden in (
        "图表选择依据",
        "验证当前 section 的关键业务判断",
        "折扣收紧",
        "高风险折扣区间",
        "策略层跳过",
        "订单金额分布",
        "销售额结构",
        "单价与数量关系",
        "年月销售热力图",
        "滚动均线",
        "平均日销售额",
        "长尾分布",
        "截尾展示",
        "低频高额客户",
        "销售额",
        "销量",
        "产品线",
        "交易规模",
    ):
        assert forbidden not in combined
    for expected in (
        "Chart Selection Rationale",
        "Discount vs. Profit Relationship",
        "Top Product Profit Gap",
        "Product Line Sales Contribution",
        "Sales by Country",
        "Sales Structure by Status",
        "Unit Price vs. Quantity",
        "Sales Trend with Rolling Average",
    ):
        assert expected in combined


def test_multi_chart_cell_filtering_keeps_only_selected_chart_code() -> None:
    section = NotebookSectionContent(
        section_id="order_structure",
        markdown_blocks=[],
        code_cells=[
            "status_table = clean_df.groupby(order_status_col, as_index=False)[sales_col].sum()",
            "\n".join(
                [
                    "# chart_id: order_status_breakdown",
                    "fig = px.bar(status_table, x=order_status_col, y=sales_col, title='STATUS 销售额结构')",
                    "fig.show()",
                    "# chart_id: status_deal_size_stacked_bar",
                    "fig = px.bar(status_deal_size, x=order_status_col, y=sales_col, color=deal_size_col, title='STATUS x 交易规模销售结构')",
                    "fig.show()",
                ]
            ),
        ],
    )

    filtered = _apply_chart_selection_to_section(
        section,
        {
            "selected_charts": [
                {
                    "section_id": "order_structure",
                    "chart_id": "order_status_breakdown",
                    "title": "STATUS 销售额结构",
                }
            ],
            "rejected_charts": [
                {
                    "section_id": "order_structure",
                    "chart_id": "status_deal_size_stacked_bar",
                    "title": "STATUS x 交易规模销售结构",
                    "reason": "redundant insight type",
                }
            ],
        },
    )

    combined = "\n\n".join(filtered.code_cells)
    assert "STATUS 销售额结构" in combined
    assert "STATUS x 交易规模销售结构" not in combined


def test_chart_display_filter_prefers_chart_id_when_title_differs() -> None:
    cell = "\n".join(
        [
            "# chart_id: category_profit_margin_bar",
            "fig = px.bar(category_profit_quality, title='各类目利润率质量对比')",
            "fig.show()",
        ]
    )

    filtered = _filter_chart_display_cell(
        cell,
        selected_titles={"类目利润率质量对比"},
        selected_chart_ids={"category_profit_margin_bar"},
    )

    assert filtered is not None
    assert "各类目利润率质量对比" in filtered


def test_multi_chart_cell_filtering_uses_per_chunk_chart_id() -> None:
    cell = "\n".join(
        [
            "# chart_id: sales_trends_monthly_line",
            "fig = px.line(monthly_sales, title='销售趋势与滚动均线')",
            "fig.show()",
            "# chart_id: rolling_volatility_line",
            "fig = px.line(monthly_sales, title='滚动波动')",
            "fig.show()",
        ]
    )

    filtered = _filter_chart_display_cell(
        cell,
        selected_titles=set(),
        selected_chart_ids={"rolling_volatility_line"},
    )

    assert filtered is not None
    assert "rolling_volatility_line" in filtered
    assert "sales_trends_monthly_line" not in filtered


def test_chart_filter_drops_unselected_chart_even_when_title_matches() -> None:
    cell = "\n".join(
        [
            "# chart_id: removed_chart",
            "fig = px.bar(data, title='Selected Title')",
            "fig.show()",
        ]
    )

    filtered = _filter_chart_display_cell(
        cell,
        selected_titles={"Selected Title"},
        selected_chart_ids={"kept_chart"},
    )

    assert filtered is None


def test_chart_filter_drops_unmarked_chart_display_cells() -> None:
    cell = "fig = px.histogram(invoice_values, title='发票金额分布')\nfig.show()"

    filtered = _filter_chart_display_cell(
        cell,
        selected_titles={"发票金额分布"},
        selected_chart_ids={"invoice_value_distribution"},
    )

    assert filtered is None


def test_apply_chart_selection_records_missing_selected_chart_reason() -> None:
    section = NotebookSectionContent(
        section_id="order_structure",
        markdown_blocks=[],
        code_cells=["order_id_col = 'InvoiceNo'"],
    )

    filtered = _apply_chart_selection_to_section(
        section,
        {
            "selected_charts": [
                {"section_id": "order_structure", "chart_id": "invoice_value_distribution"}
            ],
        },
    )

    combined = "\n\n".join(filtered.code_cells)
    assert "missing_selected_chart_render_reason" in combined
    assert "invoice_value_distribution" in combined


def test_sales_sample_style_selection_renders_only_final_selected_markers() -> None:
    section = NotebookSectionContent(
        section_id="segment_and_region",
        markdown_blocks=[],
        code_cells=[
            "# chart_id: sales_trends_monthly_line\nfig = px.line(monthly_sales)\nfig.show()",
            "# chart_id: segment_sales_profit_bar\nfig = px.bar(segment_profit_quality)\nfig.show()",
            "# chart_id: product_category_bar\nfig = px.bar(top_products, title='商品类目销售额')\nfig.show()",
            "# chart_id: profit_distribution_if_available\nfig = px.histogram(profit_df)\nfig.show()",
            "# chart_id: segment_region_sales_bar\nfig = px.bar(segment_region)\nfig.show()",
        ],
    )

    filtered = _apply_chart_selection_to_section(
        section,
        {
            "selected_charts": [
                {"section_id": "segment_and_region", "chart_id": "sales_trends_monthly_line"},
                {"section_id": "segment_and_region", "chart_id": "segment_sales_profit_bar"},
            ],
            "intent_guided_selection_trace": {
                "remove_only_chart_ids": ["product_category_bar", "profit_distribution_if_available"],
                "base_removed_chart_ids": ["segment_region_sales_bar"],
            },
        },
    )

    combined = "\n\n".join(filtered.code_cells)
    assert "# chart_id: sales_trends_monthly_line" in combined
    assert "# chart_id: segment_sales_profit_bar" in combined
    assert "product_category_bar" not in combined
    assert "profit_distribution_if_available" not in combined
    assert "segment_region_sales_bar" not in combined
    assert "missing_selected_chart_render_reason" not in combined


def test_online_retail_style_selection_keeps_invoice_distribution_only() -> None:
    section = NotebookSectionContent(
        section_id="order_structure",
        markdown_blocks=[],
        code_cells=[
            "# chart_id: invoice_value_distribution\nfig = px.histogram(invoice_values)\nfig.show()",
            "# chart_id: quantity_distribution\nfig = px.histogram(quantity_distribution_df)\nfig.show()",
            "# chart_id: sales_trends_monthly_line\nfig = px.line(monthly_sales)\nfig.show()",
        ],
    )

    filtered = _apply_chart_selection_to_section(
        section,
        {
            "selected_charts": [
                {"section_id": "order_structure", "chart_id": "invoice_value_distribution"}
            ],
            "llm_selection_trace": {
                "remove_only_chart_ids": ["quantity_distribution"],
                "base_removed_chart_ids": ["sales_trends_monthly_line"],
            },
        },
    )

    combined = "\n\n".join(filtered.code_cells)
    assert "# chart_id: invoice_value_distribution" in combined
    assert "quantity_distribution" not in combined
    assert "sales_trends_monthly_line" not in combined
    assert "missing_selected_chart_render_reason" not in combined


def test_missing_selected_chart_reason_ignores_rendered_selected_markers() -> None:
    reasons = _missing_selected_chart_render_reasons(
        selected_chart_ids={"invoice_value_distribution"},
        rendered_chart_ids={"invoice_value_distribution"},
    )

    assert reasons == {}


def test_content_plan_gets_global_selected_chart_ids_cell_when_missing() -> None:
    content = _ensure_selected_chart_ids_cell(
        NotebookContentPlan(
            sections=[
                NotebookSectionContent(
                    section_id="dataset_and_schema",
                    markdown_blocks=[],
                    code_cells=["df = pd.read_csv('raw.csv')"],
                )
            ]
        ),
        {
            "selected_charts": [
                {"chart_id": "country_sales_bar"},
                {"chart_id": "invoice_value_distribution"},
            ]
        },
    )

    first_cell = content.sections[0].code_cells[0]
    assert "selected_chart_ids = ['country_sales_bar', 'invoice_value_distribution']" in first_cell


def test_compact_output_policy_keeps_explicit_selected_chart_id_cells() -> None:
    section = NotebookSectionContent(
        section_id="product_and_category",
        markdown_blocks=[],
        code_cells=[
            "top_products = clean_df.groupby('Product Name').sum()\ntop_products",
            "# chart_id: category_profit_margin_bar\nfig = px.bar(category_profit_quality)\nfig.show()",
            "# chart_id: product_category_bar\nfig = px.bar(top_products)\nfig.show()",
            "# chart_id: top_product_profit_gap_bar\nfig = px.bar(high_sales_low_profit_products)\nfig.show()",
        ],
    )

    compact = apply_notebook_output_policy(section, notebook_output_mode="compact")
    combined = "\n\n".join(compact.code_cells)

    assert "category_profit_margin_bar" in combined
    assert "product_category_bar" in combined
    assert "top_product_profit_gap_bar" in combined


def test_compact_output_policy_keeps_bare_dataframe_dependencies_for_selected_charts() -> None:
    section = NotebookSectionContent(
        section_id="segment_and_region",
        markdown_blocks=[],
        code_cells=[
            "segment_region = clean_df.groupby('Region').sum()\nsegment_region",
            "weak_segment_region = segment_region.copy()\nweak_segment_region",
            "# chart_id: segment_region_low_margin_table_or_bar\nfig = px.bar(\n    weak_segment_region,\n    x='Sales',\n    y='slice_label',\n)\nfig.show()",
        ],
    )

    compact = apply_notebook_output_policy(section, notebook_output_mode="compact")
    combined = "\n\n".join(compact.code_cells)

    assert "weak_segment_region = segment_region.copy()" in combined
    assert "segment_region_low_margin_table_or_bar" in combined


def test_notebook_content_planner_builds_real_analysis_code_cells() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(section_id="dataset_and_schema", title="Dataset and Schema", purpose="Explain schema."),
            NotebookSection(section_id="data_cleaning", title="Data Cleaning", purpose="Clean data."),
            NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends."),
            NotebookSection(section_id="product_and_category", title="Product and Category", purpose="Analyze mix."),
            NotebookSection(section_id="segment_and_region", title="Segment and Region", purpose="Analyze cuts."),
            NotebookSection(section_id="discount_and_profit", title="Discount and Profit", purpose="Analyze margin."),
            NotebookSection(section_id="forecast", title="Forecast", purpose="Project trend."),
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="sales_trends",
                intro="Trend intro.",
                key_observations=["Peak period: 2014-03-18"],
                business_takeaway="Trend takeaway.",
                followup_question="What changed before the peak?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Product Name": "product_name",
            "Segment": "segment",
            "Region": "region",
            "Discount": "discount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=4,
        summary=["Summary."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["覆盖 1237 个时间粒度。"],
                summary_metrics={"total_sales_amount": 2297200.86},
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "data_quality_check",
            "sales_trend_analysis",
            "product_contribution_analysis",
            "dimension_breakdown_analysis",
            "forecast_analysis",
        ],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    code_lookup = {section.section_id: section.code_cells for section in content.sections}
    markdown_lookup = {section.section_id: section.markdown_blocks for section in content.sections}
    assert any("pd.read_csv" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("DATE_FORMATS" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("parse_sales_dates" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("read_csv_with_encoding_fallback" in cell for cell in code_lookup["dataset_and_schema"])
    assert not any("df = pd.read_csv('raw.csv')" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("format='%d-%b-%y'" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("plt.rcParams['font.sans-serif']" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("matplotlib.font_manager" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("axes.unicode_minus" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("pio.templates.default" in cell for cell in code_lookup["dataset_and_schema"])
    assert any("parse_sales_dates" in cell for cell in code_lookup["data_cleaning"])
    assert not any("px.line" in cell for cell in code_lookup["sales_trends"])
    assert not any("px.treemap" in cell for cell in code_lookup["product_and_category"])
    assert not any("sns.heatmap" in cell for cell in code_lookup["segment_and_region"])
    assert not any("px.scatter" in cell for cell in code_lookup["discount_and_profit"])
    assert not any("sns.boxplot" in cell for cell in code_lookup["discount_and_profit"])
    assert any("forecast_df" in cell for cell in code_lookup["forecast"])
    assert any("'views': []" in cell for cell in code_lookup["sales_trends"])
    assert any("'views': []" in cell for cell in code_lookup["product_and_category"])
    assert not any("Top 商品销售额对比" in cell for cell in code_lookup["product_and_category"])
    assert any("'views': []" in cell for cell in code_lookup["discount_and_profit"])
    assert not any("Profit Distribution by Discount Bucket" in cell for cell in code_lookup["discount_and_profit"])
    assert any(block.startswith("### 本节结论") and "下一步追问" not in block for block in markdown_lookup["sales_trends"])
    assert any(block.startswith("### 本节结论") and "下一步追问" not in block for block in markdown_lookup["product_and_category"])
    assert any(block.startswith("### 本节结论") and "下一步追问" not in block for block in markdown_lookup["segment_and_region"])
    assert any(block.startswith("### 本节结论") and "下一步追问" not in block for block in markdown_lookup["discount_and_profit"])
    assert any(block.startswith("### 本节结论") and "下一步追问" not in block for block in markdown_lookup["forecast"])
    assert all("下一步追问" not in block for block in markdown_lookup["sales_trends"])
    assert all("下一步追问" not in block for block in markdown_lookup["discount_and_profit"])
    assert all("当前更值得关注的是" not in block for block in markdown_lookup["sales_trends"])
    assert all("值得关注" not in block for blocks in markdown_lookup.values() for block in blocks)


def test_notebook_content_planner_builds_metric_distribution_section() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="metric_distributions",
                title="Metric Distributions",
                purpose="Profile numeric metrics.",
            )
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Discount": "discount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="metric_distribution_analysis",
                title="Metric Distribution",
                chart_type="histogram",
                findings=["Sales and profit distributions show visible long tails."],
                summary_metrics={"metric_count": 4},
                tables={
                    "metric_quantiles": [
                        {
                            "metric": "sales_amount",
                            "column": "Sales",
                            "label": "Sales",
                            "count": 9994,
                            "mean": 229.86,
                            "median": 54.49,
                            "p90": 572.706,
                            "p95": 956.98,
                            "p99": 2481.6946,
                            "min": 0.44,
                            "max": 22638.48,
                            "skew": 12.97,
                            "mean_median_ratio": 4.22,
                            "p99_median_ratio": 45.55,
                            "non_negative": True,
                        },
                        {
                            "metric": "profit",
                            "column": "Profit",
                            "label": "Profit",
                            "count": 9994,
                            "mean": 28.65,
                            "median": 8.67,
                            "p90": 89.28,
                            "p95": 168.47,
                            "p99": 580.66,
                            "min": -6599.98,
                            "max": 8399.98,
                            "skew": 7.56,
                            "mean_median_ratio": 3.3,
                            "p99_median_ratio": 66.97,
                            "non_negative": False,
                        }
                    ],
                    "metric_outliers": [
                        {
                            "metric": "sales_amount",
                            "label": "Sales",
                            "outlier_ratio": 0.1167,
                        },
                        {
                            "metric": "profit",
                            "label": "Profit",
                            "outlier_ratio": 0.1882,
                        }
                    ],
                    "relationship_candidates": [
                        {
                            "x_metric": "discount",
                            "x_column": "Discount",
                            "x_label": "折扣",
                            "y_metric": "profit",
                            "y_column": "Profit",
                            "y_label": "利润",
                            "correlation": -0.62,
                            "abs_correlation": 0.62,
                            "business_priority": 1.0,
                            "observation_count": 9994,
                            "color_metric": "sales_amount",
                            "color_column": "Sales",
                            "color_label": "销售额",
                        },
                        {
                            "x_metric": "sales_amount",
                            "x_column": "Sales",
                            "x_label": "销售额",
                            "y_metric": "profit",
                            "y_column": "Profit",
                            "y_label": "利润",
                            "correlation": 0.41,
                            "abs_correlation": 0.41,
                            "business_priority": 0.8,
                            "observation_count": 9994,
                            "color_metric": "discount",
                            "color_column": "Discount",
                            "color_label": "折扣",
                        },
                    ],
                },
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["metric_distribution_analysis"],
        chart_preferences={"metric_distribution_analysis": "histogram"},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    assert len(content.sections) == 1
    assert content.sections[0].section_id == "metric_distributions"
    assert any("metric_profile" in cell for cell in content.sections[0].code_cells)
    assert not any("plt.subplots(2, 2" in cell for cell in content.sections[0].code_cells)
    assert not any("sns.boxplot" in cell for cell in content.sections[0].code_cells)
    assert any("metric_strategy" in cell for cell in content.sections[0].code_cells)
    assert any("'decisions': []" in cell for cell in content.sections[0].code_cells)
    assert not any("np.log1p" in cell for cell in content.sections[0].code_cells)
    assert not any("clip(upper=" in cell for cell in content.sections[0].code_cells)
    assert not any("ax.axvline" in cell for cell in content.sections[0].code_cells)
    assert not any("quantile(0.01)" in cell for cell in content.sections[0].code_cells)
    assert not any("showfliers=False" in cell for cell in content.sections[0].code_cells)
    assert not any("P1-P99截尾箱线图" in cell for cell in content.sections[0].code_cells)
    assert not any("correlation_matrix" in cell for cell in content.sections[0].code_cells)
    assert not any("sns.heatmap" in cell for cell in content.sections[0].code_cells)
    assert any("'relationship_views': []" in cell for cell in content.sections[0].code_cells)
    assert not any("sns.scatterplot" in cell for cell in content.sections[0].code_cells)
    assert len(content.sections[0].markdown_blocks) >= 2


def test_notebook_content_planner_renders_quantity_distribution_safely() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="metric_distributions",
                title="Metric Distributions",
                purpose="Profile numeric metrics.",
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Quantity": "quantity", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="metric_distribution_analysis",
                title="Metric Distribution",
                chart_type="histogram",
                summary_metrics={"metric_count": 2},
            )
        ],
    )
    chart_selection_plan = {
        "selected_charts": [
            {
                "chart_id": "quantity_distribution",
                "section_id": "metric_distributions",
                "title": "订单行数量分布",
            }
        ],
        "rejected_charts": [],
    }

    content = build_notebook_content(
        outline=outline,
        narrative=NotebookNarrative(sections=[], suggested_followups=[]),
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=AnalysisPlan(analysis_plan=["metric_distribution_analysis"], chart_preferences={}, reasoning_summary=[]),
        chart_selection_plan=chart_selection_plan,
    )

    combined = "\n\n".join(content.sections[0].code_cells)
    assert "quantity_distribution_df" in combined
    assert "pd.to_numeric(clean_df[quantity_col], errors='coerce')" in combined
    assert "dropna()" in combined
    assert "nunique() > 1" in combined
    assert "订单行数量分布" in combined
    assert "labels={'quantity_value': '数量', 'record_count': '记录数'}" in combined
    assert "有效数量数据不足，跳过订单行数量分布图。" in combined


def test_notebook_content_planner_builds_sales_trend_strategy_section() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
        },
        confidence=0.95,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend",
                chart_type="line",
                summary_metrics={
                    "daily_grain_count": 420,
                    "monthly_grain_count": 14,
                    "year_count": 2,
                    "has_multi_year": True,
                },
                tables={
                    "monthly_totals": [
                        {"order_month": "2024-01", "Sales": 3800.0, "Quantity": 95},
                        {"order_month": "2024-02", "Sales": 4200.0, "Quantity": 102},
                    ],
                    "weekday_profile": [
                        {"weekday": "Monday", "weekday_cn": "周一", "avg_sales": 158.0, "avg_quantity": 2.8}
                    ],
                    "year_month_totals": [
                        {"year": 2024, "month": 1, "month_name": "Jan", "Sales": 3800.0},
                        {"year": 2025, "month": 1, "month_name": "Jan", "Sales": 4500.0},
                    ],
                },
                findings=[],
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["sales_trend_analysis"],
        chart_preferences={"sales_trend_analysis": "line"},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    assert content.sections[0].section_id == "sales_trends"
    assert any("trend_strategy" in cell for cell in content.sections[0].code_cells)
    assert any("rolling(window=30" in cell for cell in content.sections[0].code_cells)
    assert not any("make_subplots" in cell for cell in content.sections[0].code_cells)
    assert any("weekday_profile" in cell for cell in content.sections[0].code_cells)
    assert not any("sns.heatmap" in cell for cell in content.sections[0].code_cells)
    assert any("'views': []" in cell for cell in content.sections[0].code_cells)


class SalesTrendStrategyOnlyLLMClient:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_sales_trend_strategy(self, payload):
        return {
            "views": [
                {
                    "chart": "daily_rolling_line",
                    "reason": "Show daily trend and rolling mean.",
                    "rolling_window_days": 30,
                }
            ]
        }


class ProductCategoryStrategyOnlyLLMClient:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_product_category_strategy(self, payload):
        return {
            "views": [
                {
                    "chart": "product_pareto",
                    "reason": "Use Pareto to inspect product concentration.",
                }
            ]
        }


def test_product_pareto_title_and_font_setup_are_not_mojibake() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="dataset_and_schema",
                title="Dataset",
                purpose="Load data.",
            ),
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category",
                purpose="Analyze product concentration.",
            ),
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Product Name": "product_name",
            "Category": "category",
            "Sales": "sales_amount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                findings=["头部商品贡献集中。"],
                tables={
                    "top_products": [
                        {
                            "Product Name": "A",
                            "Sales": 1000.0,
                            "Profit": 50.0,
                            "sales_share": 0.18,
                            "cumulative_sales_share": 0.18,
                        }
                    ]
                },
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["product_contribution_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content, _ = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=ProductCategoryStrategyOnlyLLMClient(),
    )

    combined_code = "\n\n".join(
        cell for section in content.sections for cell in section.code_cells
    )
    assert "头部商品累计贡献 Pareto 分布" in combined_code
    assert "澶撮儴" not in combined_code
    assert "鍟嗗搧" not in combined_code
    assert "绛栫暐" not in combined_code


def test_notebook_content_planner_initializes_daily_rolling_plotly_figure() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend",
                chart_type="line",
                summary_metrics={"daily_grain_count": 60},
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["sales_trend_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content, trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=SalesTrendStrategyOnlyLLMClient(),
    )

    combined_code = "\n\n".join(content.sections[0].code_cells)
    figure_init_index = combined_code.find("fig = make_subplots")
    add_trace_index = combined_code.find("fig.add_trace")

    assert trace.status == "skipped"
    assert figure_init_index != -1
    assert add_trace_index != -1
    assert figure_init_index < add_trace_index


def test_notebook_content_planner_hides_default_charts_without_llm_decision() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(section_id="metric_distributions", title="Metric Distributions", purpose="Profile metrics."),
            NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends."),
            NotebookSection(section_id="product_and_category", title="Product and Category", purpose="Analyze mix."),
            NotebookSection(section_id="segment_and_region", title="Segment and Region", purpose="Analyze cuts."),
            NotebookSection(section_id="discount_and_profit", title="Discount and Profit", purpose="Analyze margin."),
            NotebookSection(section_id="forecast", title="Forecast", purpose="Project trend."),
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Product Name": "product_name",
            "Category": "category",
            "Segment": "segment",
            "Region": "region",
            "Discount": "discount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=5,
        summary=[],
        modules=[
            ModuleReport(
                module_id="metric_distribution_analysis",
                title="Metrics",
                chart_type="histogram",
                summary_metrics={"metric_count": 2},
                tables={
                    "metric_quantiles": [
                        {"metric": "sales_amount", "column": "Sales", "label": "Sales", "median": 50.0, "p99": 500.0}
                    ]
                },
            ),
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend",
                chart_type="line",
                summary_metrics={"daily_grain_count": 30, "monthly_grain_count": 3},
                tables={"monthly_totals": [{"order_month": "2024-01", "Sales": 100.0}]},
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="Product",
                chart_type="bar",
                summary_metrics={"distinct_products": 3},
                tables={"top_products": [{"Product Name": "A", "Sales": 100.0}]},
            ),
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="Dimension",
                chart_type="bar",
                summary_metrics={"primary_dimension": "Segment", "secondary_dimension": "Region"},
                tables={"segment_region_matrix": [{"Segment": "Consumer", "Region": "West", "Sales": 100.0}]},
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="Discount",
                chart_type="scatter",
                summary_metrics={"high_discount_order_count": 2},
                tables={"discount_profit_risk_buckets": [{"discount_bucket": "30%+", "avg_profit": -5.0}]},
            ),
            ModuleReport(
                module_id="forecast_analysis",
                title="Forecast",
                chart_type="line",
                summary_metrics={"baseline_sales_amount": 100.0, "horizon_days": 28},
            ),
        ],
    )
    analysis_plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=None,
    )

    code_by_section = {
        section.section_id: "\n\n".join(section.code_cells)
        for section in content.sections
    }
    assert "'decisions': []" in code_by_section["metric_distributions"]
    assert "plt.show()" not in code_by_section["metric_distributions"]
    assert "'source': 'deterministic_fallback'" in code_by_section["sales_trends"]
    assert "'views': []" in code_by_section["sales_trends"]
    assert "'views': []" in code_by_section["product_and_category"]
    assert "'views': []" in code_by_section["segment_and_region"]
    assert "'views': []" in code_by_section["discount_and_profit"]
    assert "fig.show()" not in code_by_section["sales_trends"]
    assert "fig.show()" not in code_by_section["product_and_category"]
    assert "fig.show()" not in code_by_section["segment_and_region"]
    assert "fig.show()" not in code_by_section["discount_and_profit"]
    assert "fig.show()" not in code_by_section["forecast"]


class FakeNotebookContentLLMClient:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        return {
            "section_id": section.section_id,
            "markdown_blocks": [
                "Custom trend markdown.",
                "This is a sharper analysis narrative.",
            ],
            "code_cells": [
                "trend_df = df.groupby('Order Date', as_index=False)['Sales'].sum()",
                "trend_df.head()",
            ],
        }


class FallbackNotebookContentLLMClient:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        result = dict(fallback_section)
        result["markdown_blocks"] = list(result.get("markdown_blocks", [])) + [
            "LLM 保留本地代码并只优化章节说明。"
        ]
        return result


class SameNotebookContentLLMClient:
    enabled = True
    source = "test"
    configured_model = "fake-model"

    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        return fallback_section


class UnsafeNotebookContentLLMClient(FakeNotebookContentLLMClient):
    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        return {
            "section_id": section.section_id,
            "markdown_blocks": [
                "LLM 识别到商品销售贡献高度集中，需要进一步核查头部商品是否同步贡献利润。",
            ],
            "code_cells": [
                "required_cols = ['product_name', 'category', 'sales_amount', 'quantity']\n"
                "missing_cols = [c for c in required_cols if c not in clean_df.columns]\n"
                "if missing_cols:\n"
                "    raise KeyError(missing_cols)"
            ],
        }


class CodeFenceNotebookContentLLMClient(FakeNotebookContentLLMClient):
    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        return {
            "section_id": section.section_id,
            "markdown_blocks": [
                "销售趋势值得重点观察。\n\n"
                "```python\n"
                "import pandas as pd\n"
                "trend_df = clean_df.copy()\n"
                "trend_df.head()\n"
                "```\n\n"
                "这段结论应该保留，但代码不能混进 markdown。"
            ],
            "code_cells": ["print('LLM code must be ignored')"],
        }


def test_notebook_content_planner_accepts_llm_section_content_but_sanitizes_invalid_sections() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="sales_trends",
                intro="Trend intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[],
    )
    analysis_plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=FakeNotebookContentLLMClient(),
    )

    assert len(content.sections) == 1
    assert content.sections[0].section_id == "sales_trends"
    assert content.sections[0].markdown_blocks[0] == "Custom trend markdown."


def test_notebook_content_planner_strips_code_fences_from_llm_markdown() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="sales_trends",
                intro="Trend intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[],
    )
    analysis_plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=CodeFenceNotebookContentLLMClient(),
    )

    combined_markdown = "\n\n".join(content.sections[0].markdown_blocks)
    combined_code = "\n\n".join(content.sections[0].code_cells)
    assert "销售趋势值得重点观察" in combined_markdown
    assert "这段结论应该保留" in combined_markdown
    assert "```" not in combined_markdown
    assert "import pandas" not in combined_markdown
    assert "trend_df = clean_df.copy()" not in combined_markdown
    assert "print('LLM code must be ignored')" not in combined_code
    assert "monthly_sales" in combined_code


def test_notebook_content_planner_preserves_verified_code_when_llm_returns_unsafe_code() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category",
                purpose="Analyze product mix.",
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="product_and_category",
                intro="Product intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Product Name": "product_name",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Sales": "sales_amount",
            "Quantity": "quantity",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["product_contribution_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content, trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=UnsafeNotebookContentLLMClient(),
    )

    section = content.sections[0]
    combined_code = "\n\n".join(section.code_cells)
    assert section.markdown_blocks[0].startswith("LLM 识别到商品销售贡献")
    assert "required_cols = ['product_name'" not in combined_code
    assert "Product Name" in combined_code
    assert "Sales" in combined_code
    assert trace.status == "llm_applied"


def test_notebook_content_planner_builds_compact_conclusions_with_specific_data_points() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="conclusions", title="Conclusions", purpose="Wrap up.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Product Name": "product_name",
            "Segment": "segment",
            "Region": "region",
            "Discount": "discount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=4,
        summary=[
            "发现 0 条时间缺失记录。",
            "共覆盖 1237 个时间粒度。",
            "Top 商品为 Canon imageCLASS 2200 Advanced Copier。",
            "已按 segment x region 完成双维拆解。",
            "当前共有 1393 条高折扣记录，其中负利润记录占比为 0.1872。",
        ],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                findings=["销售表现存在阶段性波动。"],
                summary_metrics={"total_sales_amount": 2297200.86},
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                findings=["头部商品贡献集中。"],
                summary_metrics={"top_product": "Canon imageCLASS 2200 Advanced Copier"},
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                findings=["高折扣订单的利润质量偏弱。"],
                summary_metrics={
                    "high_discount_order_count": 1393,
                    "negative_profit_rate": 0.1872,
                },
            ),
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "sales_trend_analysis",
            "product_contribution_analysis",
            "discount_profit_analysis",
        ],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    section = content.sections[0]
    combined = "\n\n".join(section.markdown_blocks)
    assert "2297200.86" in combined
    assert "Canon imageCLASS 2200 Advanced Copier" in combined
    assert "1393" in combined
    assert "建议" in combined
    assert "后续追问" not in combined


def test_notebook_content_planner_degrades_discount_section_to_profit_overview_when_discount_missing() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="discount_and_profit",
                title="Discount and Profit",
                purpose="Analyze margin quality.",
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="discount_and_profit",
                intro="Margin intro.",
                key_observations=["Profit is available but discount is missing."],
                business_takeaway="Focus on profit quality first.",
                followup_question="Which categories are losing money?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Category": "category",
            "Product Name": "product_name",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Profit needs review."],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["discount_profit_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    section = content.sections[0]
    assert section.section_id == "discount_and_profit"
    assert any("利润概览" in block for block in section.markdown_blocks)
    assert len(section.code_cells) >= 2
    assert all("Discount" not in cell for cell in section.code_cells)
    assert any("profit" in cell.lower() for cell in section.code_cells)


def test_notebook_content_planner_handles_profit_only_dataset_without_product_or_category() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="discount_and_profit",
                title="Discount and Profit",
                purpose="Analyze margin quality.",
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="discount_and_profit",
                intro="利润章节。",
                key_observations=["只有利润字段可用。"],
                business_takeaway="先看利润分布。",
                followup_question="哪些记录利润最差？",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Profit needs review."],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["discount_profit_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    section = content.sections[0]
    assert section.section_id == "discount_and_profit"
    assert all("Product Name" not in cell for cell in section.code_cells)
    assert all("Category" not in cell for cell in section.code_cells)
    assert any("profit_overview" in cell for cell in section.code_cells)


def test_notebook_content_planner_degrades_segment_region_section_to_single_dimension_when_region_missing() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Analyze customer and geography.",
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="segment_and_region",
                intro="Segment intro.",
                key_observations=["Only segment is available."],
                business_takeaway="Single-dimension view still helps locate weak customer groups.",
                followup_question="Which segment has the weakest profit quality?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Segment": "segment",
            "Sales": "sales_amount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Segment view only."],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["dimension_breakdown_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    section = content.sections[0]
    assert section.section_id == "segment_and_region"
    assert any("单维客群分析" in block for block in section.markdown_blocks)
    assert len(section.code_cells) >= 2
    assert all("Region" not in cell for cell in section.code_cells)
    assert any("Segment" in cell for cell in section.code_cells)


def test_notebook_content_planner_degrades_product_section_when_subcategory_missing() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category",
                purpose="Analyze assortment performance.",
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="product_and_category",
                intro="Product intro.",
                key_observations=["Sub-category is unavailable."],
                business_takeaway="Category-level structure is still useful.",
                followup_question="Which products dominate category sales?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Product Name": "product_name",
            "Category": "category",
            "Sales": "sales_amount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Category available."],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["product_contribution_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    section = content.sections[0]
    assert section.section_id == "product_and_category"
    assert any("类目层级" in block for block in section.markdown_blocks)
    assert all("Sub-Category" not in cell for cell in section.code_cells)
    assert any("Category" in cell for cell in section.code_cells)


def test_notebook_content_planner_uses_allowed_geographic_dimensions_for_segment_region_section() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Analyze dimensional performance.",
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="segment_and_region",
                intro="Dimension intro.",
                key_observations=["Use state and region."],
                business_takeaway="Available dimensions should drive the chapter.",
                followup_question="Which region-state cut is weakest?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Region": "region",
            "State": "state",
            "Sales": "sales_amount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Channel and region available."],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["dimension_breakdown_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    section = content.sections[0]
    assert any("Region" in cell for cell in section.code_cells)
    assert any("State" in cell for cell in section.code_cells)
    assert all("Channel" not in cell for cell in section.code_cells)


def test_notebook_content_planner_degrades_segment_region_metrics_when_profit_missing() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Analyze dimensional performance.",
            )
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="segment_and_region",
                intro="Dimension intro.",
                key_observations=["Profit is unavailable."],
                business_takeaway="Sales-only dimension review is still useful.",
                followup_question="Which region-state cut contributes most sales?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Region": "region",
            "State": "state",
            "Sales": "sales_amount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Profit missing."],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["dimension_breakdown_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    section = content.sections[0]
    assert any("缺少利润字段" in block or "销售额表现" in block for block in section.markdown_blocks)
    assert all("Profit" not in cell for cell in section.code_cells)


def test_notebook_content_planner_uses_chinese_fallback_labels_for_enriched_sections() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="discount_and_profit",
                title="Discount and Profit",
                purpose="Analyze margin quality.",
            ),
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Analyze customer and geography.",
            ),
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category",
                purpose="Analyze assortment performance.",
            ),
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="discount_and_profit",
                intro="折扣与利润章节。",
                key_observations=["存在高折扣订单。"],
                business_takeaway="需要评估利润侵蚀。",
                followup_question="哪些折扣区间最危险？",
            ),
            NotebookSectionNarrative(
                section_id="segment_and_region",
                intro="客群与区域章节。",
                key_observations=["只有单个维度可用。"],
                business_takeaway="先做降级分析。",
                followup_question="哪些切片需要关注？",
            ),
            NotebookSectionNarrative(
                section_id="product_and_category",
                intro="商品与类目章节。",
                key_observations=["缺少子类目。"],
                business_takeaway="先看类目结构。",
                followup_question="哪些商品贡献最高？",
            ),
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Category": "category",
            "Product Name": "product_name",
            "Segment": "segment",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["Summary."],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["discount_profit_analysis", "dimension_breakdown_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )
    markdown_lookup = {section.section_id: section.markdown_blocks for section in content.sections}

    assert any("本节结论" in block for block in markdown_lookup["discount_and_profit"])
    assert any("本节结论" in block for block in markdown_lookup["segment_and_region"])
    assert any("本节结论" in block for block in markdown_lookup["product_and_category"])


def test_notebook_content_planner_reports_applied_trace_when_llm_content_is_used() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="sales_trends",
                intro="Trend intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[],
    )
    analysis_plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])

    content, trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=FakeNotebookContentLLMClient(),
    )

    assert content.sections[0].markdown_blocks[0] == "Custom trend markdown."
    assert trace.stage == "notebook_content"
    assert trace.status == "llm_applied"
    assert trace.llm_enabled is True
    assert trace.attempted is True
    assert trace.applied is True


def test_notebook_content_trace_treats_valid_unchanged_llm_output_as_applied() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
    )
    report = AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0)

    _content, trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=AnalysisPlan(),
        llm_client=SameNotebookContentLLMClient(),
    )

    assert trace.status == "llm_applied"
    assert "validated 1 section" in trace.reason
    assert "fell back" not in trace.reason


def test_discount_profit_quality_chart_plots_avg_profit_and_loss_rate() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="discount_and_profit",
                title="Discount and Profit",
                purpose="Analyze margin.",
            )
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                findings=["折扣区间利润质量分化。"],
                tables={
                    "discount_profit_risk_buckets": [
                        {
                            "discount_bucket": "高折扣区间",
                            "avg_profit": -46.25,
                            "negative_profit_rate": 0.75,
                            "order_count": 4,
                        }
                    ]
                },
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["discount_profit_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content, _trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=FallbackNotebookContentLLMClient(),
        chart_selection_plan={
            "selected_charts": [
                {
                    "section_id": "discount_and_profit",
                    "chart_id": "discount_profit_quality_bar",
                    "title": "各折扣区间利润质量：平均利润与亏损率",
                }
            ]
        },
    )

    combined_code = "\n\n".join(content.sections[0].code_cells)
    combined_markdown = "\n\n".join(content.sections[0].markdown_blocks)
    assert "discount_threshold_candidates" in combined_code
    assert "discount_cap_what_if" in combined_code
    assert "risk_level" in combined_code
    assert "action_hint" in combined_code
    assert "这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。" in (
        combined_code + combined_markdown
    )
    assert "以下为折扣收紧情景估算，基于折扣回收金额计算，不代表真实需求、销量或客户行为变化。" in (
        combined_code + combined_markdown
    )
    assert "各折扣区间利润质量：平均利润与亏损率" in combined_code
    assert "go.Bar" in combined_code
    assert "avg_profit" in combined_code
    assert "go.Scatter" in combined_code
    assert "negative_profit_rate" in combined_code or "loss_rate" in combined_code
    assert "secondary_y=True" in combined_code
    assert "order_count" in combined_code
    assert "negative_profit_count" in combined_code
    assert "total_profit" in combined_code
    assert "total_sales" in combined_code
    assert "当前图表用于观察" not in combined_code
    assert "如果图表显示" not in combined_code


def test_order_sales_chart_candidates_render_when_selected() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(section_id="product_and_category", title="商品类目", purpose="产品线分析。"),
            NotebookSection(section_id="country_market", title="国家市场", purpose="国家分析。"),
            NotebookSection(section_id="order_structure", title="订单结构", purpose="订单分析。"),
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "ORDERDATE": "order_datetime",
            "ORDERNUMBER": "order_id",
            "PRODUCTLINE": "productline",
            "DEALSIZE": "deal_size",
            "STATUS": "order_status",
            "COUNTRY": "country",
            "QUANTITYORDERED": "quantity",
            "SALES": "sales_amount",
        },
        confidence=0.9,
    )
    report = AnalysisReport(task_id="task-1", dataset_type="sales_transaction", module_count=0)
    selected = [
        ("product_and_category", "productline_sales_donut", "PRODUCTLINE 销售额占比"),
        ("product_and_category", "productline_monthly_trend", "PRODUCTLINE 月度销售趋势"),
        ("product_and_category", "productline_deal_size_heatmap", "PRODUCTLINE x 交易规模销售热力图"),
        ("country_market", "country_sales_donut", "国家销售额占比"),
        ("country_market", "country_productline_heatmap", "国家 x PRODUCTLINE 销售热力图"),
        ("order_structure", "deal_size_order_count_bar", "DEALSIZE 订单数对比"),
        ("order_structure", "deal_size_avg_order_value_bar", "DEALSIZE 平均订单金额对比"),
        ("order_structure", "status_order_count_bar", "STATUS 订单数对比"),
        ("order_structure", "status_deal_size_stacked_bar", "STATUS x 交易规模销售结构"),
    ]

    content, _trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=NotebookNarrative(sections=[], suggested_followups=[]),
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=AnalysisPlan(),
        llm_client=FallbackNotebookContentLLMClient(),
        chart_selection_plan={
            "selected_charts": [
                {"section_id": section_id, "chart_id": chart_id, "title": title}
                for section_id, chart_id, title in selected
            ]
        },
    )

    combined_code = "\n\n".join(cell for section in content.sections for cell in section.code_cells)
    for _section_id, _chart_id, title in selected:
        assert title in combined_code
    assert "# chart_id: productline_deal_size_heatmap" in combined_code


class RecordingNotebookContentLLMClient(FakeNotebookContentLLMClient):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        self.calls.append(section.section_id)
        return {
            "section_id": section.section_id,
            "markdown_blocks": [f"Custom content for {section.section_id}."],
            "code_cells": fallback_section["code_cells"],
        }


class StrategyMetricsNotebookContentLLMClient(RecordingNotebookContentLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.metrics: list[dict[str, object]] = []
        self.last_completion_metrics: dict[str, object] = {}

    def snapshot_completion_metrics(self) -> int:
        return len(self.metrics)

    def collect_completion_metrics_since(self, snapshot: int) -> list[dict[str, object]]:
        return self.metrics[snapshot:]

    def _record(self, stage: str, prompt_chars: int, remote_elapsed_ms: float) -> None:
        metric = {
            "stage": stage,
            "cache_status": "miss",
            "prompt_chars": prompt_chars,
            "response_chars": 20,
            "remote_elapsed_ms": remote_elapsed_ms,
            "elapsed_ms": remote_elapsed_ms + 1,
            "saved_ms": 0.0,
            "cache_enabled": True,
        }
        self.metrics.append(metric)
        self.last_completion_metrics = metric

    def suggest_sales_trend_strategy(self, payload):
        self._record("sales_trend_strategy", 50, 5.0)
        return {
            "views": [
                {
                    "chart": "daily_rolling_line",
                    "reason": "Show daily trend and rolling mean.",
                    "rolling_window_days": 30,
                }
            ]
        }

    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        self._record("notebook_content", 70, 7.0)
        return super().suggest_notebook_section_content(
            section,
            narrative_section,
            report_context,
            schema_mapping,
            analysis_plan,
            fallback_section,
        )


def test_notebook_content_planner_calls_llm_per_section() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends."),
            NotebookSection(section_id="discount_and_profit", title="Discount and Profit", purpose="Analyze margin."),
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="sales_trends",
                intro="Trend intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            ),
            NotebookSectionNarrative(
                section_id="discount_and_profit",
                intro="Discount intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            ),
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Discount": "discount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        summary=[],
        modules=[],
    )
    analysis_plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    llm_client = RecordingNotebookContentLLMClient()

    content, trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=llm_client,
    )

    assert llm_client.calls == ["sales_trends", "discount_and_profit"]
    assert [section.section_id for section in content.sections] == ["sales_trends", "discount_and_profit"]
    assert trace.status in {"llm_applied", "llm_partial"}


def test_notebook_content_trace_includes_fallback_strategy_subcalls() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={},
                tables={
                    "daily_sales": [
                        {"Order Date": "2024-01-01", "Sales": 100.0},
                        {"Order Date": "2024-01-02", "Sales": 120.0},
                    ]
                },
            )
        ],
    )
    llm_client = StrategyMetricsNotebookContentLLMClient()

    _content, trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=AnalysisPlan(analysis_plan=["sales_trend_analysis"]),
        llm_client=llm_client,
    )

    assert trace.subcall_count == 2
    assert trace.aggregate_prompt_chars == 120
    assert [subcall["stage"] for subcall in trace.subcalls] == [
        "sales_trend_strategy",
        "notebook_content",
    ]


class CapturingReportContextLLMClient(FakeNotebookContentLLMClient):
    def __init__(self) -> None:
        self.report_context = None

    def suggest_notebook_section_content(
        self,
        section,
        narrative_section,
        report_context,
        schema_mapping,
        analysis_plan,
        fallback_section,
    ):
        self.report_context = report_context
        return super().suggest_notebook_section_content(
            section,
            narrative_section,
            report_context,
            schema_mapping,
            analysis_plan,
            fallback_section,
        )


def test_notebook_content_prompt_uses_compact_section_report_context() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends.")],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order Date": "order_datetime", "Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["summary"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend",
                chart_type="line",
                tables={"daily_totals": [{"day": index, "sales": index * 10} for index in range(20)]},
                chart_payload={"large": "x" * 10000},
                findings=[f"finding {index}" for index in range(20)],
            )
        ],
    )
    analysis_plan = AnalysisPlan(analysis_plan=["sales_trend_analysis"], chart_preferences={}, reasoning_summary=[])
    llm_client = CapturingReportContextLLMClient()

    build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=llm_client,
    )

    assert llm_client.report_context is not None
    module = llm_client.report_context["modules"][0]
    assert "chart_payload" not in module
    assert len(module["tables"]["daily_totals"]) == 5
    assert len(module["findings"]) == 8


class CliproxyNotebookContentLLMClient(RecordingNotebookContentLLMClient):
    source = "cliproxy"


def test_notebook_content_planner_skips_segment_region_llm_on_cliproxy() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(section_id="segment_and_region", title="Segment and Region", purpose="Analyze cuts."),
            NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Analyze trends."),
        ],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="segment_and_region",
                intro="Segment intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            ),
            NotebookSectionNarrative(
                section_id="sales_trends",
                intro="Trend intro.",
                key_observations=[],
                business_takeaway="Takeaway.",
                followup_question="Question?",
            ),
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Order Date": "order_datetime",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Segment": "segment",
            "Region": "region",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=2,
        summary=[],
        modules=[],
    )
    analysis_plan = AnalysisPlan(analysis_plan=[], chart_preferences={}, reasoning_summary=[])
    llm_client = CliproxyNotebookContentLLMClient()

    content, trace = build_notebook_content_with_trace(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=llm_client,
    )

    assert llm_client.calls == ["sales_trends"]
    assert [section.section_id for section in content.sections] == ["segment_and_region", "sales_trends"]
    assert trace.status == "llm_partial"
    assert "skipped 1 section" in trace.reason


def test_notebook_content_planner_chart_analysis_uses_table_context_and_readable_metrics() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(section_id="product_and_category", title="Product and Category", purpose="Analyze mix."),
            NotebookSection(section_id="segment_and_region", title="Segment and Region", purpose="Analyze cuts."),
            NotebookSection(section_id="discount_and_profit", title="Discount and Profit", purpose="Analyze margin."),
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Product Name": "product_name",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Segment": "segment",
            "Region": "region",
            "Discount": "discount",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=3,
        summary=[],
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品贡献分析",
                chart_type="bar",
                findings=["Top 商品为 Canon imageCLASS 2200 Advanced Copier。"],
                summary_metrics={
                    "distinct_products": 1850,
                    "top_product": "Canon imageCLASS 2200 Advanced Copier",
                },
                tables={
                    "top_products": [
                        {
                            "Product Name": "Canon imageCLASS 2200 Advanced Copier",
                            "Sales": 61599.824,
                            "sales_share": 0.026815166694633505,
                        },
                        {
                            "Product Name": "Fellowes PB500 Electric Punch Plastic Comb Binding Machine with Manual Bind",
                            "Sales": 27453.384,
                            "sales_share": 0.01195079824078368,
                        },
                    ]
                },
            ),
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="客群与区域分析",
                chart_type="stacked_bar",
                findings=[
                    "已按 segment x region 完成双维拆解。",
                    "segment 维度下 Home Office 的利润表现最弱，值得重点复盘。",
                ],
                summary_metrics={
                    "primary_dimension": "segment",
                    "secondary_dimension": "region",
                    "group_count": 3,
                },
                tables={
                    "dimension_totals": [
                        {"Segment": "Consumer", "Sales": 1161401.345, "Profit": 134119.2092},
                        {"Segment": "Corporate", "Sales": 706146.3668, "Profit": 91979.134},
                        {"Segment": "Home Office", "Sales": 429653.1485, "Profit": 60298.6785},
                    ],
                    "weak_performance_cuts": [
                        {"Segment": "Home Office", "Sales": 429653.1485, "Profit": 60298.6785}
                    ],
                },
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                findings=[
                    "当前共有 1393 条高折扣记录，其中负利润记录占比为 0.1872。",
                    "30%+ 折扣区间平均利润最低。",
                ],
                summary_metrics={
                    "negative_profit_order_count": 1871,
                    "high_discount_order_count": 1393,
                    "negative_profit_rate": 0.1872,
                },
                tables={
                    "discount_buckets": [
                        {
                            "discount_bucket": "10-20%",
                            "avg_profit": 24.738823806956052,
                            "order_count": 3709,
                        },
                        {
                            "discount_bucket": "20-30%",
                            "avg_profit": -45.67963612334801,
                            "order_count": 227,
                        },
                        {
                            "discount_bucket": "30%+",
                            "avg_profit": -107.20993018867925,
                            "order_count": 1166,
                        },
                    ]
                },
            ),
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=[
            "product_contribution_analysis",
            "dimension_breakdown_analysis",
            "discount_profit_analysis",
        ],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    markdown_lookup = {
        section.section_id: "\n\n".join(section.markdown_blocks)
        for section in content.sections
    }

    product_markdown = markdown_lookup["product_and_category"]
    assert "2.68%" in product_markdown
    assert "Canon imageCLASS 2200 Advanced Copier" in product_markdown
    assert "Fellowes PB500 Electric Punch Plastic Comb Binding Machine with Manual Bind" in product_markdown

    segment_markdown = markdown_lookup["segment_and_region"]
    assert "Consumer" in segment_markdown
    assert "Home Office" in segment_markdown
    assert "1,161,401.34" in segment_markdown

    discount_markdown = markdown_lookup["discount_and_profit"]
    assert "18.72%" in discount_markdown
    assert "20-30%" in discount_markdown
    assert "30%+" in discount_markdown
    assert "0.1872" not in discount_markdown


def test_notebook_content_planner_uses_discount_profit_strategy_views() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="discount_and_profit",
                title="Discount and Profit",
                purpose="Analyze margin risk.",
            )
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Category": "category",
            "Product Name": "product_name",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                findings=["30%+ 折扣区间平均利润最低。"],
                summary_metrics={
                    "negative_profit_order_count": 4,
                    "high_discount_order_count": 4,
                    "negative_profit_rate": 0.57,
                    "worst_discount_bucket": "30%+",
                    "high_sales_low_profit_count": 2,
                },
                tables={
                    "discount_profit_risk_buckets": [
                        {
                            "discount_bucket": "30%+",
                            "avg_profit": -46.25,
                            "negative_profit_rate": 0.75,
                            "profit_margin": -0.05,
                            "order_count": 4,
                        }
                    ],
                    "high_sales_low_profit_items": [
                        {
                            "Product Name": "F",
                            "Sales": 1000.0,
                            "Profit": -90.0,
                            "profit_margin": -0.09,
                            "avg_discount": 0.45,
                        }
                    ],
                    "category_discount_risk": [
                        {
                            "Category": "Furniture",
                            "discount_bucket": "30%+",
                            "negative_profit_rate": 0.67,
                            "profit_margin": -0.06,
                            "Sales": 3100.0,
                        }
                    ],
                },
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["discount_profit_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    combined_code = "\n\n".join(content.sections[0].code_cells)

    assert "discount_profit_strategy" in combined_code
    assert "discount_view_map" in combined_code
    assert "'views': []" in combined_code
    assert "if 'bucket_profit_quality_bar'" not in combined_code
    assert "high_sales_low_profit" in combined_code
    assert "category_discount_risk" in combined_code
    assert "make_subplots" not in combined_code
    assert "fig.show()" not in combined_code


def test_notebook_content_planner_uses_product_category_strategy_views() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category",
                purpose="Analyze assortment quality.",
            )
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Product Name": "product_name",
            "Category": "category",
            "Sub-Category": "sub_category",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品与类目分析",
                chart_type="bar",
                findings=["头部商品贡献集中。"],
                summary_metrics={
                    "distinct_products": 120,
                    "top_1_sales_share": 0.18,
                    "top_10_sales_share": 0.64,
                    "products_to_80pct_sales": 22,
                },
                tables={
                    "top_products": [
                        {
                            "Product Name": "A",
                            "Sales": 1000.0,
                            "Profit": 50.0,
                            "sales_share": 0.18,
                            "cumulative_sales_share": 0.18,
                            "profit_margin": 0.05,
                        }
                    ],
                    "category_sales": [
                        {
                            "Category": "Tech",
                            "Sub-Category": "Phones",
                            "Sales": 3200.0,
                            "Profit": 500.0,
                            "profit_margin": 0.1562,
                        }
                    ],
                    "category_profit_quality": [
                        {
                            "Category": "Furniture",
                            "Sales": 1600.0,
                            "Profit": -80.0,
                            "profit_margin": -0.05,
                        }
                    ],
                    "high_sales_low_profit_products": [
                        {
                            "Product Name": "A",
                            "Sales": 1000.0,
                            "Profit": 50.0,
                            "profit_margin": 0.05,
                        }
                    ],
                },
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["product_contribution_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    combined_code = "\n\n".join(content.sections[0].code_cells)

    assert "product_category_strategy" in combined_code
    assert "product_view_map" in combined_code
    assert "category_profit_quality" not in combined_code
    assert "fig.show()" not in combined_code
    assert "'views': []" in combined_code


def test_top_product_profit_gap_chart_cell_is_self_contained() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category",
                purpose="Analyze product profit gaps.",
            )
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Product Name": "product_name",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品与类目分析",
                chart_type="bar",
                findings=[],
                tables={},
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["product_contribution_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )
    chart_selection_plan = {
        "selected_charts": [
            {
                "chart_id": "top_product_profit_gap_bar",
                "section_id": "product_and_category",
                "title": "头部商品利润缺口",
            }
        ]
    }

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        chart_selection_plan=chart_selection_plan,
    )

    chart_cells = [
        cell for cell in content.sections[0].code_cells if "title='头部商品利润缺口'" in cell
    ]
    assert chart_cells
    cell = chart_cells[0]
    assert "# chart_id: top_product_profit_gap_bar" in cell
    assert "high_sales_low_profit_products =" in cell
    assert ".groupby(" in cell
    assert cell.index("high_sales_low_profit_products =") < cell.index("fig = px.bar(")


def test_category_profit_margin_bar_cell_has_chart_id_marker() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="product_and_category",
                title="Product and Category",
                purpose="Analyze category margin.",
            )
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Category": "category",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["product_contribution_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )
    chart_selection_plan = {
        "selected_charts": [
            {
                "chart_id": "category_profit_margin_bar",
                "section_id": "product_and_category",
                "title": "类目利润率质量对比",
            }
        ]
    }

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        chart_selection_plan=chart_selection_plan,
    )

    combined = "\n\n".join(content.sections[0].code_cells)
    assert "# chart_id: category_profit_margin_bar" in combined


def test_notebook_content_planner_uses_segment_region_strategy_views() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[
            NotebookSection(
                section_id="segment_and_region",
                title="Segment and Region",
                purpose="Analyze segment and region cuts.",
            )
        ],
    )
    narrative = NotebookNarrative(sections=[], suggested_followups=[])
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Segment": "segment",
            "Region": "region",
            "Sales": "sales_amount",
            "Quantity": "quantity",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="客群与区域分析",
                chart_type="stacked_bar",
                findings=["已按 segment x region 完成双维拆解。"],
                summary_metrics={
                    "primary_dimension": "segment",
                    "secondary_dimension": "region",
                    "high_sales_low_profit_cut_count": 2,
                },
                tables={
                    "segment_region_matrix": [
                        {
                            "Segment": "Home Office",
                            "Region": "West",
                            "Sales": 750.0,
                            "Profit": -120.0,
                            "profit_margin": -0.16,
                        }
                    ],
                    "primary_profit_quality": [
                        {
                            "Segment": "Home Office",
                            "Sales": 1550.0,
                            "Profit": -100.0,
                            "profit_margin": -0.0645,
                        }
                    ],
                    "weak_performance_cuts": [
                        {
                            "Segment": "Home Office",
                            "Region": "West",
                            "Sales": 750.0,
                            "Profit": -120.0,
                            "profit_margin": -0.16,
                        }
                    ],
                },
            )
        ],
    )
    analysis_plan = AnalysisPlan(
        analysis_plan=["dimension_breakdown_analysis"],
        chart_preferences={},
        reasoning_summary=[],
    )

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    combined_code = "\n\n".join(content.sections[0].code_cells)

    assert "segment_region_strategy" in combined_code
    assert "segment_view_map" in combined_code
    assert "segment_profit_quality" in combined_code
    assert "segment_region_bubble" in combined_code
    assert "weak_segment_region" in combined_code
    assert "segment_region_profit_margin_heatmap" in combined_code
    assert "observed=False" in combined_code


def test_notebook_content_planner_compacts_narrative_without_template_bridge() -> None:
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="segment_and_region", title="Segment and Region", purpose="Analyze cuts.")],
    )
    narrative = NotebookNarrative(
        sections=[
            NotebookSectionNarrative(
                section_id="segment_and_region",
                intro="本节比较客群和区域两个维度下的表现差异，帮助识别结构性的强弱切片。",
                key_observations=[
                    "Consumer 是销售额和利润最高的分群。",
                    "Home Office 的利润表现最弱。",
                ],
                business_takeaway="双维拆解更容易暴露出某些区域中的弱势客群，适合进一步制定差异化动作。",
                followup_question="哪些区域里的某类客群销售不差，但利润持续偏弱？",
            )
        ],
        suggested_followups=[],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Segment": "segment",
            "Region": "region",
            "Sales": "sales_amount",
            "Profit": "profit",
        },
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[],
        modules=[
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="客群与区域分析",
                chart_type="stacked_bar",
                summary_metrics={
                    "primary_dimension": "segment",
                    "secondary_dimension": "region",
                },
                findings=["已按 segment x region 完成双维拆解。"],
            )
        ],
    )
    analysis_plan = AnalysisPlan(analysis_plan=["dimension_breakdown_analysis"], chart_preferences={}, reasoning_summary=[])

    content = build_notebook_content(
        outline=outline,
        narrative=narrative,
        report=report,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
    )

    combined = "\n\n".join(content.sections[0].markdown_blocks)
    assert "当前更值得关注的是" not in combined
    assert "结合当前结果" in combined
    assert "Consumer 是销售额和利润最高的分群" in combined
