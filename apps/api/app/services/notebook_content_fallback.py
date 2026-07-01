from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.schemas.notebook_content import NotebookSectionContent
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping


@dataclass(frozen=True)
class FallbackSectionHelpers:
    compact_narrative_blocks: Callable[..., list[str]]
    original_column: Callable[..., str]
    optional_original_column: Callable[..., str | None]
    nb_english: Callable[..., bool]
    nb_text: Callable[..., str]
    nb_quote: Callable[..., str]
    module_for_section: Callable[..., object]
    chart_selection_markdown: Callable[..., str]
    apply_chart_selection_to_section: Callable[..., NotebookSectionContent]
    chart_analysis_markdown: Callable[..., str]
    discount_profit_section: Callable[..., NotebookSectionContent]
    product_category_section: Callable[..., NotebookSectionContent]
    segment_region_section: Callable[..., NotebookSectionContent]
    selected_chart_ids_literal: Callable[..., str]
    selection_includes: Callable[..., bool]
    without_non_llm_chart_display_cells: Callable[..., list[str]]
    build_section_insight_markdown: Callable[..., list[str]]
    conclusions_section_content: Callable[..., NotebookSectionContent]


def build_fallback_section_content(
    section_id: str,
    narrative: NotebookNarrative,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    metric_distribution_strategy: dict[str, object] | None = None,
    sales_trend_strategy: dict[str, object] | None = None,
    product_category_strategy: dict[str, object] | None = None,
    segment_region_strategy: dict[str, object] | None = None,
    discount_profit_strategy: dict[str, object] | None = None,
    chart_selection_plan: dict[str, object] | None = None,
    output_language: str | None = None,
    *,
    helpers: FallbackSectionHelpers,
) -> NotebookSectionContent:
    _compact_narrative_blocks = helpers.compact_narrative_blocks
    _original_column = helpers.original_column
    _optional_original_column = helpers.optional_original_column
    _nb_english = helpers.nb_english
    _nb_text = helpers.nb_text
    _nb_quote = helpers.nb_quote
    _module_for_section = helpers.module_for_section
    _chart_selection_markdown = helpers.chart_selection_markdown
    _apply_chart_selection_to_section = helpers.apply_chart_selection_to_section
    _chart_analysis_markdown = helpers.chart_analysis_markdown
    _discount_profit_section = helpers.discount_profit_section
    _product_category_section = helpers.product_category_section
    _segment_region_section = helpers.segment_region_section
    _selected_chart_ids_literal = helpers.selected_chart_ids_literal
    _selection_includes = helpers.selection_includes
    _without_non_llm_chart_display_cells = helpers.without_non_llm_chart_display_cells
    build_section_insight_markdown = helpers.build_section_insight_markdown
    _conclusions_section_content = helpers.conclusions_section_content
    narrative_map = {section.section_id: section for section in narrative.sections}
    narrative_section = narrative_map.get(section_id)

    markdown_blocks: list[str] = _compact_narrative_blocks(
        narrative_section,
        output_language=output_language,
    )

    order_date_col = _original_column(schema_mapping, "order_datetime", "Order Date")
    sales_col = _original_column(schema_mapping, "sales_amount", "Sales")
    quantity_col = _optional_original_column(schema_mapping, "quantity")
    category_col = _original_column(schema_mapping, "category", "Category")
    sub_category_col = _original_column(schema_mapping, "sub_category", "Sub-Category")
    product_col = _original_column(
        schema_mapping,
        "product_name",
        _original_column(schema_mapping, "sku", "Product Name"),
    )
    discount_col = _original_column(schema_mapping, "discount", "Discount")
    profit_col = _original_column(schema_mapping, "profit", "Profit")
    country_col = _original_column(schema_mapping, "country", "Country")
    customer_col = _original_column(schema_mapping, "customer_id", "CustomerID")
    order_id_col = _original_column(schema_mapping, "order_id", "InvoiceNo")
    unit_price_col = _original_column(schema_mapping, "unit_price", "UnitPrice")
    deal_size_col = _original_column(schema_mapping, "deal_size", "DEALSIZE")
    order_status_col = _original_column(schema_mapping, "order_status", "STATUS")

    if section_id == "dataset_and_schema":
        return _build_dataset_and_schema_fallback_section(
            section_id=section_id,
            report=report,
            output_language=output_language,
            _chart_analysis_markdown=_chart_analysis_markdown,
            markdown_blocks=markdown_blocks,
        )
    if section_id == "data_cleaning":
        return _build_data_cleaning_fallback_section(
            section_id=section_id,
            report=report,
            schema_mapping=schema_mapping,
            output_language=output_language,
            _original_column=_original_column,
            _chart_analysis_markdown=_chart_analysis_markdown,
            markdown_blocks=markdown_blocks,
            order_date_col=order_date_col,
            sales_col=sales_col,
            quantity_col=quantity_col,
        )
    if section_id == "metric_distributions":
        return _build_metric_distributions_fallback_section(
            section_id=section_id,
            report=report,
            schema_mapping=schema_mapping,
            metric_distribution_strategy=metric_distribution_strategy,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
            _nb_text=_nb_text,
            _nb_quote=_nb_quote,
            _module_for_section=_module_for_section,
            _chart_selection_markdown=_chart_selection_markdown,
            _apply_chart_selection_to_section=_apply_chart_selection_to_section,
            _chart_analysis_markdown=_chart_analysis_markdown,
            _selected_chart_ids_literal=_selected_chart_ids_literal,
            build_section_insight_markdown=build_section_insight_markdown,
            markdown_blocks=markdown_blocks,
            sales_col=sales_col,
            quantity_col=quantity_col,
            discount_col=discount_col,
            profit_col=profit_col,
        )
    if section_id == "sales_trends":
        return _build_sales_trends_fallback_section(
            section_id=section_id,
            report=report,
            sales_trend_strategy=sales_trend_strategy,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
            _nb_english=_nb_english,
            _nb_quote=_nb_quote,
            _chart_selection_markdown=_chart_selection_markdown,
            _apply_chart_selection_to_section=_apply_chart_selection_to_section,
            _chart_analysis_markdown=_chart_analysis_markdown,
            _selected_chart_ids_literal=_selected_chart_ids_literal,
            markdown_blocks=markdown_blocks,
            order_date_col=order_date_col,
            sales_col=sales_col,
            quantity_col=quantity_col,
        )
    if section_id == "product_and_category":
        return _build_product_and_category_fallback_section(
            section_id=section_id,
            report=report,
            schema_mapping=schema_mapping,
            product_category_strategy=product_category_strategy,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
            _original_column=_original_column,
            _nb_text=_nb_text,
            _nb_quote=_nb_quote,
            _chart_selection_markdown=_chart_selection_markdown,
            _apply_chart_selection_to_section=_apply_chart_selection_to_section,
            _product_category_section=_product_category_section,
            markdown_blocks=markdown_blocks,
            order_date_col=order_date_col,
            sales_col=sales_col,
            quantity_col=quantity_col,
            category_col=category_col,
            sub_category_col=sub_category_col,
            product_col=product_col,
            profit_col=profit_col,
            deal_size_col=deal_size_col,
        )
    if section_id == "segment_and_region":
        return _build_segment_and_region_fallback_section(
            section_id=section_id,
            report=report,
            schema_mapping=schema_mapping,
            segment_region_strategy=segment_region_strategy,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
            _chart_selection_markdown=_chart_selection_markdown,
            _apply_chart_selection_to_section=_apply_chart_selection_to_section,
            _segment_region_section=_segment_region_section,
            markdown_blocks=markdown_blocks,
            sales_col=sales_col,
            quantity_col=quantity_col,
            profit_col=profit_col,
        )
    if section_id == "country_market":
        return _build_country_market_fallback_section(
            section_id=section_id,
            schema_mapping=schema_mapping,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
            _original_column=_original_column,
            _nb_text=_nb_text,
            _nb_quote=_nb_quote,
            _chart_selection_markdown=_chart_selection_markdown,
            _apply_chart_selection_to_section=_apply_chart_selection_to_section,
            _selection_includes=_selection_includes,
            markdown_blocks=markdown_blocks,
            order_date_col=order_date_col,
            sales_col=sales_col,
            country_col=country_col,
            customer_col=customer_col,
            order_id_col=order_id_col,
        )
    if section_id == "order_structure":
        return _build_order_structure_fallback_section(
            section_id=section_id,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
            _nb_text=_nb_text,
            _nb_quote=_nb_quote,
            _chart_selection_markdown=_chart_selection_markdown,
            _apply_chart_selection_to_section=_apply_chart_selection_to_section,
            markdown_blocks=markdown_blocks,
            sales_col=sales_col,
            quantity_col=quantity_col,
            customer_col=customer_col,
            order_id_col=order_id_col,
            unit_price_col=unit_price_col,
            deal_size_col=deal_size_col,
            order_status_col=order_status_col,
        )
    if section_id == "discount_and_profit":
        return _build_discount_and_profit_fallback_section(
            section_id=section_id,
            report=report,
            schema_mapping=schema_mapping,
            discount_profit_strategy=discount_profit_strategy,
            chart_selection_plan=chart_selection_plan,
            output_language=output_language,
            _chart_selection_markdown=_chart_selection_markdown,
            _apply_chart_selection_to_section=_apply_chart_selection_to_section,
            _chart_analysis_markdown=_chart_analysis_markdown,
            _discount_profit_section=_discount_profit_section,
            _without_non_llm_chart_display_cells=_without_non_llm_chart_display_cells,
            markdown_blocks=markdown_blocks,
            sales_col=sales_col,
            category_col=category_col,
            product_col=product_col,
            discount_col=discount_col,
            profit_col=profit_col,
        )
    if section_id == "modeling":
        return _build_modeling_fallback_section(
            section_id=section_id,
            report=report,
            _module_for_section=_module_for_section,
            markdown_blocks=markdown_blocks,
        )
    if section_id == "conclusions":
        return _build_conclusions_fallback_section(
            report=report,
            output_language=output_language,
            _conclusions_section_content=_conclusions_section_content,
        )
    if section_id == "forecast":
        return _build_forecast_fallback_section(
            section_id=section_id,
            report=report,
            output_language=output_language,
            _chart_analysis_markdown=_chart_analysis_markdown,
            markdown_blocks=markdown_blocks,
            sales_col=sales_col,
        )

    return NotebookSectionContent(
        section_id=section_id,
        markdown_blocks=markdown_blocks,
        code_cells=[],
    )


def _build_dataset_and_schema_fallback_section(
    *,
    section_id,
    report,
    output_language,
    _chart_analysis_markdown,
    markdown_blocks,
) -> NotebookSectionContent:
    code_cells = [
        "\n".join(
            [
                "import pandas as pd",
                "import numpy as np",
                "import warnings",
                "import plotly.express as px",
                "import plotly.io as pio",
                "import seaborn as sns",
                "import matplotlib.pyplot as plt",
                "import matplotlib.font_manager as fm",
                "",
                "CHINESE_FONT_CANDIDATES = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']",
                "available_fonts = {font.name for font in fm.fontManager.ttflist}",
                "chosen_font = next((font for font in CHINESE_FONT_CANDIDATES if font in available_fonts), 'DejaVu Sans')",
                "plt.rcParams['font.family'] = [chosen_font]",
                "plt.rcParams['font.sans-serif'] = [chosen_font, *CHINESE_FONT_CANDIDATES]",
                "plt.rcParams['axes.unicode_minus'] = False",
                "sns.set_theme(style='whitegrid', font=chosen_font)",
                "pio.templates.default = 'plotly_white'",
                "",
                "DATE_FORMATS = ['%d-%b-%y', '%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d/%m/%Y']",
                "",
                "def parse_sales_dates(series):",
                "    sample = series.dropna().astype(str)",
                "    preferred_parsed = pd.to_datetime(sample, format='%d-%b-%y', errors='coerce')",
                "    if not sample.empty and preferred_parsed.notna().mean() >= 0.7:",
                "        return pd.to_datetime(series, format='%d-%b-%y', errors='coerce')",
                "    for fmt in DATE_FORMATS:",
                "        parsed_sample = pd.to_datetime(sample, format=fmt, errors='coerce')",
                "        if not sample.empty and parsed_sample.notna().mean() >= 0.7:",
                "            return pd.to_datetime(series, format=fmt, errors='coerce')",
                "    with warnings.catch_warnings():",
                "        warnings.simplefilter('ignore', UserWarning)",
                "        return pd.to_datetime(series, errors='coerce')",
                "",
                "def read_csv_with_encoding_fallback(path):",
                "    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']",
                "    last_error = None",
                "    for encoding in encodings:",
                "        try:",
                "            return pd.read_csv(path, encoding=encoding)",
                "        except UnicodeDecodeError as exc:",
                "            last_error = exc",
                "    return pd.read_csv(path, encoding='latin1', encoding_errors='replace')",
                "",
                "df = read_csv_with_encoding_fallback('raw.csv')",
                "df.head()",
            ]
        ),
        "df.info()",
        "df.describe(include='all').transpose().head(20)",
    ]
    chart_analysis = _chart_analysis_markdown(section_id, report, output_language=output_language)
    if chart_analysis:
        markdown_blocks.append(chart_analysis)
    return NotebookSectionContent(section_id=section_id, markdown_blocks=markdown_blocks, code_cells=code_cells)


def _build_data_cleaning_fallback_section(
    *,
    section_id,
    report,
    schema_mapping,
    output_language,
    _original_column,
    _chart_analysis_markdown,
    markdown_blocks,
    order_date_col,
    sales_col,
    quantity_col,
) -> NotebookSectionContent:
    ship_date_col = _original_column(schema_mapping, "ship_datetime", "")
    code_cells = [
        "\n".join(
            [
                f"df['{order_date_col}'] = parse_sales_dates(df['{order_date_col}'])",
                *(
                    [f"df['{ship_date_col}'] = parse_sales_dates(df['{ship_date_col}'])"]
                    if ship_date_col
                    else []
                ),
                "missing_summary = df.isnull().sum().sort_values(ascending=False)",
                "duplicate_rows = int(df.duplicated().sum())",
                "missing_summary.head(15), duplicate_rows",
            ]
        ),
        "\n".join(
            [
                "clean_df = df.copy()",
                f"clean_df['{sales_col}'] = pd.to_numeric(clean_df['{sales_col}'], errors='coerce')",
                *(
                    [f"clean_df['{quantity_col}'] = pd.to_numeric(clean_df['{quantity_col}'], errors='coerce')"]
                    if quantity_col
                    else []
                ),
                "clean_df = clean_df.dropna(subset=['{0}', '{1}'])".format(order_date_col, sales_col),
                "clean_df.head()",
            ]
        ),
    ]
    chart_analysis = _chart_analysis_markdown(section_id, report, output_language=output_language)
    if chart_analysis:
        markdown_blocks.append(chart_analysis)
    return NotebookSectionContent(section_id=section_id, markdown_blocks=markdown_blocks, code_cells=code_cells)


def _build_metric_distributions_fallback_section(
    *,
    section_id,
    report,
    schema_mapping,
    metric_distribution_strategy,
    chart_selection_plan,
    output_language,
    _nb_text,
    _nb_quote,
    _module_for_section,
    _chart_selection_markdown,
    _apply_chart_selection_to_section,
    _chart_analysis_markdown,
    _selected_chart_ids_literal,
    build_section_insight_markdown,
    markdown_blocks,
    sales_col,
    quantity_col,
    discount_col,
    profit_col,
) -> NotebookSectionContent:
    metric_columns = {
        col: label
        for col, label in [
            (sales_col, _nb_text(output_language, "Sales", "销售额")),
            *(([(quantity_col, "Quantity")] if quantity_col else [])),
            (discount_col, _nb_text(output_language, "Discount", "折扣")),
            (profit_col, _nb_text(output_language, "Profit", "利润")),
        ]
        if col in schema_mapping.field_mapping
        or col in schema_mapping.field_mapping.keys()
        or col in {"Sales", "Discount", "Profit"}
    }
    metric_column_literal = repr(metric_columns)
    metric_strategy_literal = repr(
        metric_distribution_strategy
        or {"source": "empty", "allowed_charts": [], "decisions": [], "relationship_views": []}
    )
    selected_chart_ids_literal = _selected_chart_ids_literal(chart_selection_plan)
    local_markdown = list(markdown_blocks)
    if not local_markdown:
        local_markdown.append(
            ""
        )
    metric_profile_title = _nb_quote(
        output_language,
        "Core Metric Distribution Profile: Strategy-Based Display Scale",
        "核心数值指标分布画像：按策略选择展示尺度",
    )
    metric_no_p99_message = _nb_quote(
        output_language,
        "No P99 clipped distribution view was triggered.",
        "当前未触发 P99 截尾分布视角。",
    )
    metric_indicator_label = _nb_quote(output_language, "Metric", "指标")
    metric_value_label = _nb_quote(output_language, "Value", "数值")
    metric_min_label = _nb_quote(output_language, "Raw Min", "原始最小值")
    metric_max_label = _nb_quote(output_language, "Raw Max", "原始最大值")
    metric_no_boxplot_message = _nb_quote(
        output_language,
        "No numeric metric is available for the boxplot.",
        "当前没有可用于箱线图的数值指标。",
    )
    metric_correlation_title = _nb_quote(
        output_language,
        "Core Metric Correlation Heatmap",
        "核心数值指标相关性热力图",
    )
    metric_relationship_title = _nb_quote(
        output_language,
        "Core Metric Relationship Scatterplots: Correlation Direction and Risk Bands",
        "核心指标关系散点图：确认相关方向与风险簇",
    )
    metric_sales_distribution_clipped_title = _nb_quote(
        output_language,
        "Core Metric Distribution Profile (P99 Clipped)",
        "核心数值指标分布画像（P99截尾）",
    )
    metric_sales_distribution_title = _nb_quote(
        output_language,
        "Core Metric Distribution Profile: Strategy-Based Display Scale",
        "核心数值指标分布画像：按策略选择展示尺度",
    )
    metric_sales_log_title = _nb_quote(output_language, "Sales Log Distribution", "销售额对数分布")
    metric_sales_long_tail_message = _nb_quote(
        output_language,
        "This distribution is long-tailed, so the chart clips at P99 for readability; clipped records={distribution_readability_trace['clipped_count']}.",
        "该分布为长尾分布，图中按 P99={p99:.2f} 截尾展示主体区间；截尾记录数={distribution_readability_trace['clipped_count']}。",
    )
    quantity_insufficient_message = _nb_quote(
        output_language,
        "Insufficient valid quantity data; the order-line quantity distribution chart was skipped.",
        "有效数量数据不足，跳过订单行数量分布图。",
    )
    quantity_distribution_clipped_title = _nb_quote(
        output_language,
        "Order-Line Quantity Distribution (P99 Clipped)",
        "订单行数量分布（P99截尾）",
    )
    quantity_distribution_title = _nb_quote(output_language, "Order-Line Quantity Distribution", "订单行数量分布")
    quantity_label = _nb_quote(output_language, "Quantity", "数量")
    record_count_label = _nb_quote(output_language, "Record Count", "记录数")
    quantity_long_tail_message = _nb_quote(
        output_language,
        "This quantity distribution is long-tailed, so the chart clips at P99 for readability; clipped records={distribution_readability_trace['clipped_count']}, share={distribution_readability_trace['clipped_ratio']:.2%}.",
        "该数量分布存在长尾，图中按 P99={p99:.2f} 截尾展示主体区间；截尾记录数={distribution_readability_trace['clipped_count']}，占比={distribution_readability_trace['clipped_ratio']:.2%}。",
    )
    profit_distribution_clipped_title = _nb_quote(
        output_language,
        "Profit Distribution (P99 Clipped)",
        "利润分布（P99截尾）",
    )
    profit_distribution_title = _nb_quote(output_language, "Profit Distribution", "利润分布")
    profit_long_tail_message = _nb_quote(
        output_language,
        "This profit distribution is long-tailed, so the chart clips at P99 for readability; clipped records={distribution_readability_trace['clipped_count']}.",
        "该利润分布存在长尾，图中按 P99={p99:.2f} 截尾展示主体区间；截尾记录数={distribution_readability_trace['clipped_count']}。",
    )
    code_cells = [
        "\n".join(
            [
                f"metric_columns = {metric_column_literal}",
                f"metric_strategy = {metric_strategy_literal}",
                f"selected_chart_ids = {selected_chart_ids_literal}",
                f"sales_col = {sales_col!r}",
                f"quantity_col = {quantity_col!r}",
                f"profit_col = {profit_col!r}",
                "available_metric_columns = [col for col in metric_columns if col in clean_df.columns]",
                "metric_profile = clean_df[available_metric_columns].apply(pd.to_numeric, errors='coerce')",
                "metric_summary = metric_profile.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).transpose()",
                "metric_summary['missing_count'] = metric_profile.isna().sum()",
                "metric_strategy_table = pd.DataFrame(metric_strategy.get('decisions', []))",
                "relationship_view_table = pd.DataFrame(metric_strategy.get('relationship_views', []))",
                "metric_summary, metric_strategy_table, relationship_view_table",
            ]
        ),
        "\n".join(
            [
                "fig, axes = plt.subplots(2, 2, figsize=(12, 8))",
                "axes = axes.flatten()",
                "strategy_by_column = {item.get('column'): item for item in metric_strategy.get('decisions', [])}",
                "for ax, col in zip(axes, available_metric_columns):",
                "    series = metric_profile[col].dropna()",
                "    decision = strategy_by_column.get(col, {})",
                "    charts = set(decision.get('charts', []))",
                "    label = metric_columns[col]",
                "    is_log_view = 'log_histogram' in charts and (series >= 0).all()",
                "    if is_log_view:",
                "        plot_series = np.log1p(series.clip(lower=0))",
                "        sns.histplot(plot_series, kde=True, ax=ax, color='#4E79A7')",
                "        ax.set_title(f'{label} log1p distribution')",
                "        ax.set_xlabel(f'log1p({label})')",
                "    elif 'p99_clipped_histogram' in charts:",
                "        clip_upper = series.quantile(0.99)",
                "        plot_series = series.clip(upper=clip_upper)",
                "        sns.histplot(plot_series, kde=True, ax=ax, color='#4E79A7')",
                f"        ax.set_title(f'{{label}} P99 {_nb_text(output_language, 'clipped distribution', '截尾分布')}')",
                "        ax.set_xlabel(label)",
                "    else:",
                "        plot_series = series",
                "        sns.histplot(plot_series, kde=True, ax=ax, color='#4E79A7')",
                f"        ax.set_title(f'{{label}} {_nb_text(output_language, 'raw distribution', '原始分布')}')",
                "        ax.set_xlabel(label)",
                "    if 'quantile_markers' in charts and not series.empty:",
                "        quantiles = {'P25': series.quantile(0.25), 'P50': series.quantile(0.5), 'P75': series.quantile(0.75)}",
                "        colors = {'P25': '#59A14F', 'P50': '#F28E2B', 'P75': '#E15759'}",
                "        for q_name, q_value in quantiles.items():",
                "            color = colors.get(q_name, '#666666')",
                "            marker_value = np.log1p(max(q_value, 0)) if is_log_view else q_value",
                "            ax.axvline(marker_value, color=color, linestyle='--', linewidth=1, label=q_name)",
                "        ax.legend(fontsize=8)",
                "for ax in axes[len(available_metric_columns):]:",
                "    ax.set_visible(False)",
                f"plt.suptitle({metric_profile_title}, y=1.02)",
                "plt.tight_layout()",
                "plt.show()",
            ]
        ),
        "\n".join(
            [
                "p99_columns = [item.get('column') for item in metric_strategy.get('decisions', []) if 'p99_clipped_histogram' in item.get('charts', []) and item.get('column') in metric_profile.columns]",
                "if p99_columns:",
                "    fig, axes = plt.subplots(1, len(p99_columns), figsize=(6 * len(p99_columns), 4))",
                "    axes = np.atleast_1d(axes)",
                "    for ax, col in zip(axes, p99_columns):",
                "        series = metric_profile[col].dropna()",
                "        clip_upper = series.quantile(0.99)",
                "        clipped = series.clip(upper=clip_upper)",
                "        sns.histplot(clipped, kde=True, ax=ax, color='#76B7B2')",
                "",
                "        ax.axvline(series.quantile(0.9), color='#F28E2B', linestyle='--', label='P90')",
                "        ax.axvline(clip_upper, color='#59A14F', linestyle='--', label='P99')",
                "",
                "        ax.set_xlabel(metric_columns[col])",
                "        ax.legend(fontsize=8)",
                "    plt.tight_layout()",
                "    plt.show()",
                "else:",
                f"    print({metric_no_p99_message})",
            ]
        ),
        "\n".join(
            [
                "boxplot_columns = [item.get('column') for item in metric_strategy.get('decisions', []) if 'boxplot' in item.get('charts', []) and item.get('column') in metric_profile.columns]",
                "if not boxplot_columns:",
                "    boxplot_columns = available_metric_columns",
                "boxplot_frames = []",
                "clip_rows = []",
                "for col in boxplot_columns:",
                "    series = metric_profile[col].dropna()",
                "    if series.empty:",
                "        continue",
                "    clip_lower = series.quantile(0.01)",
                "    clip_upper = series.quantile(0.99)",
                "    label = metric_columns[col]",
                "    clipped_series = series.clip(lower=clip_lower, upper=clip_upper)",
                "",
                f"    clip_rows.append({{{metric_indicator_label}: label, 'P1': clip_lower, 'P99': clip_upper, {metric_min_label}: series.min(), {metric_max_label}: series.max()}})",
                f"    boxplot_frames.append(pd.DataFrame({{{metric_indicator_label}: label, {metric_value_label}: clipped_series}}))",
                "if boxplot_frames:",
                "    metric_long = pd.concat(boxplot_frames, ignore_index=True)",
                "    metric_clip_ranges = pd.DataFrame(clip_rows)",
                f"    plt.figure(figsize=(10, max(4, 0.7 * metric_long[{metric_indicator_label}].nunique())))",
                f"    sns.boxplot(data=metric_long, x={metric_value_label}, y={metric_indicator_label}, color='#4E79A7')",
                "",
                f"    if (metric_long[{metric_value_label}] < 0).any() and (metric_long[{metric_value_label}] > 0).any():",
                "        plt.axvline(0, color='#E15759', linestyle='--', linewidth=1)",
                "",
                "",
                f"    plt.ylabel({metric_indicator_label})",
                "    plt.tight_layout()",
                "    plt.show()",
                "    metric_clip_ranges",
                "else:",
                f"    print({metric_no_boxplot_message})",
            ]
        ),
        "\n".join(
            [
                "relationship_views = metric_strategy.get('relationship_views', [])",
                "relationship_view_map = {item.get('chart'): item for item in relationship_views}",
                "strategy_by_metric = {item.get('metric'): item for item in metric_strategy.get('decisions', [])}",
                "heatmap_view = relationship_view_map.get('correlation_heatmap', {})",
                "heatmap_metrics = [metric for metric in heatmap_view.get('metrics', []) if strategy_by_metric.get(metric, {}).get('column') in metric_profile.columns]",
                "heatmap_columns = [strategy_by_metric[metric]['column'] for metric in heatmap_metrics]",
                "if len(heatmap_columns) >= 2:",
                "    correlation_matrix = metric_profile[heatmap_columns].corr(numeric_only=True)",
                "    correlation_matrix.index = [metric_columns[col] for col in heatmap_columns]",
                "    correlation_matrix.columns = [metric_columns[col] for col in heatmap_columns]",
                "    plt.figure(figsize=(8, 6))",
                "    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f')",
                f"    plt.title({metric_correlation_title})",
                "    plt.tight_layout()",
                "    plt.show()",
                "    correlation_matrix",
                "else:",
                "    pass",
            ]
        ),
        "\n".join(
            [
                "relationship_view_map = {item.get('chart'): item for item in metric_strategy.get('relationship_views', [])}",
                "scatter_relationships = relationship_view_map.get('scatter_relationships', {}).get('pairs', [])",
                "if scatter_relationships:",
                "    fig, axes = plt.subplots(1, len(scatter_relationships), figsize=(6 * len(scatter_relationships), 5))",
                "    axes = np.atleast_1d(axes)",
                "    for ax, pair in zip(axes, scatter_relationships):",
                "        x_col = pair.get('x_column')",
                "        y_col = pair.get('y_column')",
                "        color_col = pair.get('color_column')",
                "        if not x_col or not y_col or x_col not in clean_df.columns or y_col not in clean_df.columns:",
                "            ax.set_visible(False)",
                "            continue",
                "        plot_columns = list(dict.fromkeys([x_col, y_col] + ([color_col] if color_col and color_col in clean_df.columns else [])))",
                "        relationship_df = clean_df[plot_columns].apply(pd.to_numeric, errors='coerce').dropna(subset=[x_col, y_col]).copy()",
                "        if relationship_df.empty:",
                "            ax.set_visible(False)",
                "            continue",
                "        if color_col and color_col in relationship_df.columns:",
                "            sns.scatterplot(data=relationship_df, x=x_col, y=y_col, hue=color_col, palette='viridis', alpha=0.55, ax=ax, legend=False)",
                "        else:",
                "            sns.scatterplot(data=relationship_df, x=x_col, y=y_col, alpha=0.55, ax=ax)",
                "        x_label = pair.get('x_label') or metric_columns.get(x_col, x_col)",
                "        y_label = pair.get('y_label') or metric_columns.get(y_col, y_col)",
                "        corr_value = pair.get('correlation')",
                "        corr_text = f' corr={corr_value:.2f}' if isinstance(corr_value, (int, float)) else ''",
                "        ax.set_title(f'{x_label} vs {y_label}{corr_text}')",
                "        ax.set_xlabel(x_label)",
                "        ax.set_ylabel(y_label)",
                f"    plt.suptitle({metric_relationship_title}, y=1.02)",
                "    plt.tight_layout()",
                "    plt.show()",
                "else:",
                "    pass",
            ]
        ),
    ]
    selected_metric_cells = []
    if isinstance(chart_selection_plan, dict):
        selected_metric_cells = [
            "\n".join(
                [
                    "if 'metric_sales_distribution' in selected_chart_ids and sales_col in clean_df.columns:",
                    "    metric_sales_values = pd.to_numeric(clean_df[sales_col], errors='coerce').dropna()",
                    "    p99 = metric_sales_values.quantile(0.99) if not metric_sales_values.empty else None",
                    "    median = metric_sales_values.median() if not metric_sales_values.empty else None",
                    "    max_value = metric_sales_values.max() if not metric_sales_values.empty else None",
                    "    long_tail = bool(p99 and ((max_value > p99 * 2) or (median and median > 0 and p99 / median > 10)))",
                    "    metric_sales_plot = metric_sales_values[metric_sales_values <= p99] if long_tail else metric_sales_values",
                    "    distribution_readability_trace = {'chart_id': 'metric_sales_distribution', 'distribution_readability_mode': 'p99_clipped' if long_tail else 'raw', 'p99': float(p99) if p99 is not None else None, 'max': float(max_value) if max_value is not None else None, 'median': float(median) if median is not None else None, 'clipped_count': int((metric_sales_values > p99).sum()) if long_tail else 0, 'clipped_ratio': float((metric_sales_values > p99).mean()) if long_tail else 0.0, 'reason': 'P99 clipped for readability' if long_tail else 'raw distribution'}",
                    f"    title = {metric_sales_distribution_clipped_title} if long_tail else {metric_sales_distribution_title}",
                    "    fig = px.histogram(metric_sales_plot.to_frame(name=sales_col), x=sales_col, nbins=40, title=title)",
                    "    fig.show()",
                    "    if long_tail:",
                    f"        print(f{metric_sales_long_tail_message})",
                    "else:",
                    "    pass",
                ]
            ),
            "\n".join(
                [
                    "if 'sales_amount_log_distribution' in selected_chart_ids and sales_col in clean_df.columns:",
                    "    sales_log_df = clean_df[[sales_col]].copy()",
                    "    sales_log_df['sales_log1p'] = np.log1p(pd.to_numeric(sales_log_df[sales_col], errors='coerce').clip(lower=0))",
                    f"    fig = px.histogram(sales_log_df, x='sales_log1p', nbins=40, title={metric_sales_log_title})",
                    "    fig.show()",
                    "else:",
                    "    pass",
                ]
            ),
            "\n".join(
                [
                    "# chart_id: quantity_distribution",
                    "if 'quantity_distribution' in selected_chart_ids and quantity_col in clean_df.columns:",
                    "    quantity_distribution_df = pd.to_numeric(clean_df[quantity_col], errors='coerce').dropna().to_frame(name='quantity_value')",
                    "    quantity_distribution_valid = len(quantity_distribution_df) >= 5 and quantity_distribution_df['quantity_value'].nunique() > 1",
                    "    if not quantity_distribution_valid:",
                    f"        print({quantity_insufficient_message})",
                    "    else:",
                    "        quantity_values = quantity_distribution_df['quantity_value']",
                    "        p99 = quantity_values.quantile(0.99)",
                    "        median = quantity_values.median()",
                    "        max_value = quantity_values.max()",
                    "        long_tail = bool(p99 and ((max_value > p99 * 2) or (median and median > 0 and p99 / median > 10)))",
                    "        quantity_plot_df = quantity_distribution_df[quantity_distribution_df['quantity_value'] <= p99].copy() if long_tail else quantity_distribution_df.copy()",
                    "        distribution_readability_trace = {'chart_id': 'quantity_distribution', 'distribution_readability_mode': 'p99_clipped' if long_tail else 'raw', 'p99': float(p99), 'max': float(max_value), 'median': float(median), 'clipped_count': int((quantity_values > p99).sum()) if long_tail else 0, 'clipped_ratio': float((quantity_values > p99).mean()) if long_tail else 0.0, 'reason': 'P99 clipped for readability' if long_tail else 'raw distribution'}",
                    "        fig = px.histogram(",
                    "            quantity_plot_df,",
                    "            x='quantity_value',",
                    "            nbins=40,",
                    f"            title={quantity_distribution_clipped_title} if long_tail else {quantity_distribution_title},",
                    f"            labels={{'quantity_value': {quantity_label}, 'record_count': {record_count_label}}},",
                    "        )",
                    f"        fig.update_layout(xaxis_title={quantity_label}, yaxis_title={record_count_label})",
                    "        fig.show()",
                    "        if long_tail:",
                    f"            print(f{quantity_long_tail_message})",
                    "else:",
                    "    pass",
                ]
            ),
            "\n".join(
                [
                    "# chart_id: profit_distribution_if_available",
                    "if 'profit_distribution_if_available' in selected_chart_ids and profit_col in clean_df.columns:",
                    "    profit_values = pd.to_numeric(clean_df[profit_col], errors='coerce').dropna()",
                    "    p99 = profit_values.quantile(0.99) if not profit_values.empty else None",
                    "    median = profit_values.median() if not profit_values.empty else None",
                    "    max_value = profit_values.max() if not profit_values.empty else None",
                    "    long_tail = bool(p99 and ((max_value > p99 * 2) or (median and median > 0 and p99 / median > 10)))",
                    "    profit_plot = profit_values[profit_values <= p99] if long_tail else profit_values",
                    "    distribution_readability_trace = {'chart_id': 'profit_distribution_if_available', 'distribution_readability_mode': 'p99_clipped' if long_tail else 'raw', 'p99': float(p99) if p99 is not None else None, 'max': float(max_value) if max_value is not None else None, 'median': float(median) if median is not None else None, 'clipped_count': int((profit_values > p99).sum()) if long_tail else 0, 'clipped_ratio': float((profit_values > p99).mean()) if long_tail else 0.0, 'reason': 'P99 clipped for readability' if long_tail else 'raw distribution'}",
                    f"    fig = px.histogram(profit_plot.to_frame(name=profit_col), x=profit_col, nbins=40, title={profit_distribution_clipped_title} if long_tail else {profit_distribution_title})",
                    "    fig.show()",
                    "    if long_tail:",
                    f"        print(f{profit_long_tail_message})",
                    "else:",
                    "    pass",
                ]
            ),
        ]
    # 指标分布节默认保留底表；只有 chart_selection_plan 选中的分布图才展示。
    code_cells = code_cells[:1] + selected_metric_cells
    chart_analysis = _chart_analysis_markdown(section_id, report, output_language=output_language)
    if chart_analysis:
        local_markdown.append(chart_analysis)
        metric_module = _module_for_section(section_id, report)
        for extra_block in build_section_insight_markdown(
            section_id=section_id,
            module_report=metric_module,
            output_language=output_language,
        )[1:]:
            local_markdown.append(extra_block)
    selection_markdown = _chart_selection_markdown(chart_selection_plan, section_id, output_language=output_language)
    if selection_markdown:
        local_markdown.append(selection_markdown)
    return _apply_chart_selection_to_section(
        NotebookSectionContent(
            section_id=section_id,
            markdown_blocks=local_markdown,
            code_cells=code_cells,
        ),
        chart_selection_plan,
    )


def _build_sales_trends_fallback_section(
    *,
    section_id,
    report,
    sales_trend_strategy,
    chart_selection_plan,
    output_language,
    _nb_english,
    _nb_quote,
    _chart_selection_markdown,
    _apply_chart_selection_to_section,
    _chart_analysis_markdown,
    _selected_chart_ids_literal,
    markdown_blocks,
    order_date_col,
    sales_col,
    quantity_col,
) -> NotebookSectionContent:
    selected_chart_ids_literal = _selected_chart_ids_literal(chart_selection_plan)
    trend_strategy_literal = repr(
        sales_trend_strategy
        or {"source": "empty", "allowed_charts": [], "views": []}
    )
    trend_value_columns_literal = repr([sales_col] + ([quantity_col] if quantity_col else []))
    weekday_label_map_literal = repr(
        {
            "Monday": "Monday",
            "Tuesday": "Tuesday",
            "Wednesday": "Wednesday",
            "Thursday": "Thursday",
            "Friday": "Friday",
            "Saturday": "Saturday",
            "Sunday": "Sunday",
        }
        if _nb_english(output_language)
        else {
            "Monday": "周一",
            "Tuesday": "周二",
            "Wednesday": "周三",
            "Thursday": "周四",
            "Friday": "周五",
            "Saturday": "周六",
            "Sunday": "周日",
        }
    )
    daily_sales_label = _nb_quote(output_language, "Daily Sales", "日销售额")
    rolling_average_label = _nb_quote(output_language, "Rolling Average", "滚动均线")
    sales_trend_title = _nb_quote(output_language, "Sales Trend with Rolling Average", "销售趋势与滚动均线")
    rolling_volatility_name = _nb_quote(output_language, "30-Day Volatility", "30日波动率")
    rolling_volatility_title = _nb_quote(
        output_language,
        "Rolling Volatility of Daily Sales Change",
        "销售日变化滚动波动率",
    )
    rolling_volatility_skip_message = _nb_quote(
        output_language,
        "Insufficient data points; rolling_volatility_line was skipped.",
        "数据点不足，跳过 rolling_volatility_line。",
    )
    weekday_label = _nb_quote(output_language, "Weekday", "星期")
    average_daily_sales_label = _nb_quote(output_language, "Average Daily Sales", "平均日销售额")
    year_month_heatmap_title = _nb_quote(output_language, "Year-Month Sales Heatmap", "年月销售热力图")
    month_label = _nb_quote(output_language, "Month", "月份")
    year_label = _nb_quote(output_language, "Year", "年份")
    recent_growth_title = _nb_quote(output_language, "Recent Growth Rate Comparison", "近期增长率对比")
    month_columns_literal = repr(
        [f"Month {month}" for month in range(1, 13)]
        if _nb_english(output_language)
        else [f"{month}月" for month in range(1, 13)]
    )
    code_cells = [
        "\n".join(
            [
                f"trend_strategy = {trend_strategy_literal}",
                f"selected_chart_ids = {selected_chart_ids_literal}",
                f"quantity_col = {quantity_col!r}",
                f"trend_value_columns = {trend_value_columns_literal}",
                "trend_strategy_table = pd.DataFrame(trend_strategy.get('views', []))",
                "trend_df = clean_df.copy()",
                f"trend_df['order_day'] = trend_df['{order_date_col}'].dt.floor('D')",
                f"trend_df['order_month'] = trend_df['{order_date_col}'].dt.to_period('M').astype(str)",
                "trend_view_map = {item.get('chart'): item for item in trend_strategy.get('views', [])}",
                "daily_sales = trend_df.groupby('order_day', as_index=False)[trend_value_columns].sum().sort_values('order_day')",
                "rolling_window = int((trend_view_map.get('daily_rolling_line') or {}).get('rolling_window_days', 30))",
                f"daily_sales['sales_rolling_mean'] = daily_sales['{sales_col}'].rolling(window=30, min_periods=1).mean() if rolling_window == 30 else daily_sales['{sales_col}'].rolling(window=rolling_window, min_periods=1).mean()",
                "if quantity_col and quantity_col in daily_sales.columns:",
                "    daily_sales['quantity_rolling_mean'] = daily_sales[quantity_col].rolling(window=30, min_periods=1).mean() if rolling_window == 30 else daily_sales[quantity_col].rolling(window=rolling_window, min_periods=1).mean()",
                "monthly_sales = trend_df.groupby('order_month', as_index=False)[trend_value_columns].sum().sort_values('order_month')",
                "monthly_sales['order_month_dt'] = pd.to_datetime(monthly_sales['order_month'])",
                "weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']",
                f"weekday_label_map = {weekday_label_map_literal}",
                f"weekday_agg = {{'avg_sales': ({sales_col!r}, 'mean')}}",
                "if quantity_col and quantity_col in daily_sales.columns:",
                "    weekday_agg['avg_quantity'] = (quantity_col, 'mean')",
                "weekday_profile = daily_sales.assign(weekday=daily_sales['order_day'].dt.day_name()).groupby('weekday', as_index=False).agg(**weekday_agg)",
                "weekday_profile['weekday'] = pd.Categorical(weekday_profile['weekday'], categories=weekday_order, ordered=True)",
                "weekday_profile = weekday_profile.sort_values('weekday')",
                "weekday_profile['weekday_cn'] = weekday_profile['weekday'].map(weekday_label_map)",
                "year_month_totals = (trend_df.assign(year=trend_df['order_day'].dt.year, month=trend_df['order_day'].dt.month).groupby(['year', 'month'], as_index=False)[['{0}']].sum())".format(sales_col),
                "year_month_pivot = year_month_totals.pivot(index='year', columns='month', values='{0}')".format(sales_col),
                "year_month_pivot = year_month_pivot.reindex(columns=list(range(1, 13)))",
                f"year_month_pivot.columns = {month_columns_literal}",
                "daily_sales.head(), monthly_sales.head(), trend_strategy_table",
            ]
        ),
        "\n".join(
            [
                "# chart_id: sales_trends_monthly_line",
                "if 'daily_rolling_line' in trend_view_map or 'sales_trends_monthly_line' in selected_chart_ids or 'rolling_volatility_line' in selected_chart_ids:",
                "    from plotly.subplots import make_subplots",
                "    import plotly.graph_objects as go",
                "    fig = make_subplots(rows=1, cols=1)",
                "",
                f"    fig.add_trace(go.Scatter(x=daily_sales['order_day'], y=daily_sales['{sales_col}'], mode='lines', name={daily_sales_label}, line=dict(color='#4E79A7', width=1.4), opacity=0.45), row=1, col=1)",
                f"    fig.add_trace(go.Scatter(x=daily_sales['order_day'], y=daily_sales['sales_rolling_mean'], mode='lines', name={rolling_average_label}, line=dict(color='#E15759', width=2.4)), row=1, col=1)",
                "",
                "",
                f"    fig.update_layout(title={sales_trend_title}, height=700, legend_orientation='h')",
                "    fig.show()",
                "else:",
                "    pass",
            ]
        ),
        "\n".join(
            [
                "# chart_id: rolling_volatility_line",
                "if 'rolling_volatility_line' in selected_chart_ids:",
                "    import plotly.graph_objects as go",
                "    volatility = daily_sales[['order_day', '{0}']].copy()".format(sales_col),
                "    volatility['daily_change'] = volatility['{0}'].pct_change()".format(sales_col),
                "    volatility['rolling_volatility'] = volatility['daily_change'].rolling(window=30, min_periods=7).std()",
                "    volatility = volatility.dropna(subset=['rolling_volatility'])",
                "    if not volatility.empty:",
                "        fig = go.Figure()",
                f"        fig.add_trace(go.Scatter(x=volatility['order_day'], y=volatility['rolling_volatility'], mode='lines', name={rolling_volatility_name}, line=dict(color='#B07AA1', width=2.2)))",
                f"        fig.update_layout(title={rolling_volatility_title}, height=420, legend_orientation='h')",
                "        fig.update_yaxes(tickformat='.0%')",
                "        fig.show()",
                "    else:",
                f"        print({rolling_volatility_skip_message})",
                "else:",
                "    pass",
            ]
        ),
        "\n".join(
            [
                "if 'monthly_line' in trend_view_map:",
                "    monthly_y_columns = [col for col in trend_value_columns if col in monthly_sales.columns]",
                "    fig = px.line(",
                "        monthly_sales,",
                "        x='order_month_dt',",
                "        y=monthly_y_columns,",
                "        markers=True,",
                "",
                "    )",
                "",
                "    fig.show()",
                "else:",
                "    pass",
            ]
        ),
        "\n".join(
            [
                "if 'weekday_bar' in trend_view_map and not weekday_profile.empty:",
                "    plt.figure(figsize=(9, 4.5))",
                "    sns.barplot(data=weekday_profile, x='weekday_cn', y='avg_sales', color='#F28E2B')",
                "",
                f"    plt.xlabel({weekday_label})",
                f"    plt.ylabel({average_daily_sales_label})",
                "    plt.tight_layout()",
                "    plt.show()",
                "    weekday_display_columns = [col for col in ['weekday_cn', 'avg_sales', 'avg_quantity'] if col in weekday_profile.columns]",
                "    weekday_profile[weekday_display_columns]",
                "else:",
                "    pass",
            ]
        ),
        "\n".join(
            [
                "# chart_id: sales_trends_month_heatmap",
                "if ('year_month_heatmap' in trend_view_map or 'sales_trends_month_heatmap' in selected_chart_ids) and year_month_pivot.notna().any().any():",
                "    plt.figure(figsize=(10, max(3.5, 1.6 * len(year_month_pivot))))",
                "    sns.heatmap(year_month_pivot, annot=True, fmt='.0f', cmap='YlGnBu')",
                f"    plt.title({year_month_heatmap_title})",
                f"    plt.xlabel({month_label})",
                f"    plt.ylabel({year_label})",
                "    plt.tight_layout()",
                "    plt.show()",
                "    year_month_pivot",
                "else:",
                "    pass",
            ]
        ),
        "\n".join(
            [
                "if 'recent_growth_bar' in selected_chart_ids and len(monthly_sales) >= 2:",
                "    recent_growth = monthly_sales.tail(6).copy()",
                "    recent_growth['growth_rate'] = recent_growth['{0}'].pct_change()".format(sales_col),
                f"    fig = px.bar(recent_growth, x='order_month_dt', y='growth_rate', title={recent_growth_title}, text='growth_rate')",
                "    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')",
                "    fig.update_yaxes(tickformat='.0%')",
                "    fig.show()",
                "else:",
                "    pass",
            ]
        ),
    ]
    if (
        (not isinstance(sales_trend_strategy, dict) or sales_trend_strategy.get("source") != "llm_sanitized")
        and not any(
            isinstance(item, dict) and item.get("section_id") == "sales_trends"
            for item in (chart_selection_plan or {}).get("selected_charts", [])
        )
    ):
        # Keep fallback trend charts hidden when the LLM did not select them.
        code_cells = code_cells[:1]
    chart_analysis = _chart_analysis_markdown(section_id, report, output_language=output_language)
    if chart_analysis:
        markdown_blocks.append(chart_analysis)
    selection_markdown = _chart_selection_markdown(chart_selection_plan, section_id, output_language=output_language)
    if selection_markdown:
        markdown_blocks.append(selection_markdown)
    return _apply_chart_selection_to_section(
        NotebookSectionContent(section_id=section_id, markdown_blocks=markdown_blocks, code_cells=code_cells),
        chart_selection_plan,
    )


def _build_product_and_category_fallback_section(
    *,
    section_id,
    report,
    schema_mapping,
    product_category_strategy,
    chart_selection_plan,
    output_language,
    _original_column,
    _nb_text,
    _nb_quote,
    _chart_selection_markdown,
    _apply_chart_selection_to_section,
    _product_category_section,
    markdown_blocks,
    order_date_col,
    sales_col,
    quantity_col,
    category_col,
    sub_category_col,
    product_col,
    profit_col,
    deal_size_col,
) -> NotebookSectionContent:
    section = _product_category_section(
        markdown_blocks=markdown_blocks,
        report=report,
        schema_mapping=schema_mapping,
        sales_col=sales_col,
        category_col=category_col,
        sub_category_col=sub_category_col,
        product_col=product_col,
        profit_col=profit_col,
        product_category_strategy=product_category_strategy,
        chart_selection_plan=chart_selection_plan,
        output_language=output_language,
    )
    selected_productline_charts = {
        str(item.get("chart_id"))
        for item in (chart_selection_plan or {}).get("selected_charts", [])
        if isinstance(item, dict)
    }
    productline_chart_ids = {
        "productline_sales_bar",
        "productline_sales_donut",
        "productline_monthly_trend",
        "productline_quantity_sales_combo",
        "productline_quantity_bar",
        "productline_deal_size_stacked_bar",
        "productline_deal_size_heatmap",
    }
    if productline_chart_ids & selected_productline_charts:
        productline_col = _original_column(schema_mapping, "productline", "PRODUCTLINE")
        productline_intro = _nb_text(
            output_language,
            "This section focuses on PRODUCTLINE and compares product-line sales contribution, quantity contribution, and deal-size structure.",
            "本节围绕 PRODUCTLINE 展开，优先比较产品线销售贡献、数量贡献和交易结构，而不是套用普通商品排行榜。",
        )
        productline_sales_title = _nb_quote(output_language, "Product Line Sales Contribution", "PRODUCTLINE 销售额贡献")
        productline_share_title = _nb_quote(output_language, "Product Line Sales Share", "PRODUCTLINE 销售额占比")
        productline_monthly_title = _nb_quote(output_language, "Product Line Monthly Sales Trend", "PRODUCTLINE 月度销售趋势")
        productline_combo_title = _nb_quote(
            output_language,
            "Product Line Sales and Quantity Comparison",
            "PRODUCTLINE 销售额与销量对比",
        )
        productline_quantity_title = _nb_quote(output_language, "Product Line Quantity Contribution", "PRODUCTLINE 数量贡献")
        productline_deal_title = _nb_quote(
            output_language,
            "Product Line x Deal Size Sales Structure",
            "PRODUCTLINE x 交易规模销售结构",
        )
        productline_deal_heatmap_title = _nb_quote(
            output_language,
            "Product Line x Deal Size Sales Heatmap",
            "PRODUCTLINE x 交易规模销售热力图",
        )
        productline_axis_label = _nb_quote(output_language, "Product Line", "产品线")
        deal_size_axis_label = _nb_quote(output_language, "Deal Size", "交易规模")
        sales_axis_label = _nb_quote(output_language, "Sales", "销售额")
        quantity_axis_label = _nb_quote(output_language, "Quantity", "销量")
        section.markdown_blocks.insert(
            0,
            productline_intro,
        )
        productline_cells = [
            "\n".join(
                [
                    f"productline_col = {productline_col!r}",
                    f"sales_col = {sales_col!r}",
                    f"quantity_col = {quantity_col!r}",
                    f"order_date_col = {order_date_col!r}",
                    f"deal_size_col = {deal_size_col!r}",
                    "productline_sales = clean_df.groupby(productline_col, as_index=False).agg(sales_amount=(sales_col, 'sum'))",
                    "if quantity_col in clean_df.columns:",
                    "    productline_quantity = clean_df.groupby(productline_col, as_index=False).agg(quantity=(quantity_col, 'sum'))",
                    "    productline_sales = productline_sales.merge(productline_quantity, on=productline_col, how='left')",
                    "productline_sales['sales_share'] = productline_sales['sales_amount'] / productline_sales['sales_amount'].sum()",
                    "productline_sales = productline_sales.sort_values('sales_amount', ascending=False)",
                    "productline_sales",
                ]
            )
        ]
        if "productline_sales_bar" in selected_productline_charts:
            productline_cells.append(
                "\n".join(
                    [
                        "# chart_id: productline_sales_bar",
                        "fig = px.bar(",
                        "    productline_sales,",
                        "    x='sales_amount',",
                        "    y=productline_col,",
                        "    orientation='h',",
                        f"    title={productline_sales_title}",
                        ")",
                        f"fig.update_layout(xaxis_title={sales_axis_label}, yaxis_title={productline_axis_label})",
                        "fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                        "fig.show()",
                    ]
                )
            )
        if "productline_sales_donut" in selected_productline_charts:
            productline_cells.append(
                "\n".join(
                    [
                        "# chart_id: productline_sales_donut",
                        f"fig = px.pie(productline_sales.head(8), names=productline_col, values='sales_amount', hole=0.45, title={productline_share_title})",
                        "fig.update_traces(textposition='inside', textinfo='percent+label')",
                        "fig.show()",
                    ]
                )
            )
        if "productline_monthly_trend" in selected_productline_charts:
            productline_cells.append(
                "\n".join(
                    [
                        "# chart_id: productline_monthly_trend",
                        "if order_date_col in clean_df.columns:",
                        "    productline_monthly = clean_df.copy()",
                        "    productline_monthly['order_month'] = productline_monthly[order_date_col].dt.to_period('M').astype(str)",
                        "    top_productlines = productline_sales.head(6)[productline_col].tolist()",
                        "    productline_monthly = productline_monthly[productline_monthly[productline_col].isin(top_productlines)]",
                        "    productline_monthly = productline_monthly.groupby([productline_col, 'order_month'], as_index=False)[sales_col].sum()",
                        f"    fig = px.line(productline_monthly, x='order_month', y=sales_col, color=productline_col, title={productline_monthly_title})",
                        f"    fig.update_layout(xaxis_title={_nb_quote(output_language, 'Month', '月份')}, yaxis_title={sales_axis_label})",
                        "    fig.show()",
                    ]
                )
            )
        if "productline_quantity_sales_combo" in selected_productline_charts:
            productline_cells.append(
                "\n".join(
                    [
                        "# chart_id: productline_quantity_sales_combo",
                        "if 'quantity' in productline_sales.columns:",
                        "    from plotly.subplots import make_subplots",
                        "    import plotly.graph_objects as go",
                        "    combo_data = productline_sales.head(10)",
                        "    fig = make_subplots(specs=[[{'secondary_y': True}]])",
                        f"    fig.add_trace(go.Bar(x=combo_data[productline_col], y=combo_data['sales_amount'], name={sales_axis_label}), secondary_y=False)",
                        f"    fig.add_trace(go.Scatter(x=combo_data[productline_col], y=combo_data['quantity'], name={quantity_axis_label}, mode='lines+markers'), secondary_y=True)",
                        f"    fig.update_layout(title={productline_combo_title})",
                        f"    fig.update_xaxes(title_text={productline_axis_label})",
                        f"    fig.update_yaxes(title_text={sales_axis_label}, secondary_y=False)",
                        f"    fig.update_yaxes(title_text={quantity_axis_label}, secondary_y=True)",
                        "    fig.show()",
                    ]
                )
            )
        if "productline_quantity_bar" in selected_productline_charts:
            productline_cells.append(
                "\n".join(
                    [
                        "# chart_id: productline_quantity_bar",
                        "if 'quantity' in productline_sales.columns:",
                        "    fig = px.bar(",
                        "        productline_sales,",
                        "        x='quantity',",
                        "        y=productline_col,",
                        "        orientation='h',",
                        f"        title={productline_quantity_title}",
                        "    )",
                        f"    fig.update_layout(xaxis_title={quantity_axis_label}, yaxis_title={productline_axis_label})",
                        "    fig.update_layout(yaxis={'categoryorder': 'total ascending'})",
                        "    fig.show()",
                    ]
                )
            )
        if "productline_deal_size_stacked_bar" in selected_productline_charts:
            productline_cells.append(
                "\n".join(
                    [
                        "# chart_id: productline_deal_size_stacked_bar",
                        "if deal_size_col in clean_df.columns:",
                        "    productline_deal_size = clean_df.groupby([productline_col, deal_size_col], as_index=False)[sales_col].sum()",
                        "    top_productlines = productline_sales.head(8)[productline_col].tolist()",
                        "    productline_deal_size = productline_deal_size[productline_deal_size[productline_col].isin(top_productlines)]",
                        f"    fig = px.bar(productline_deal_size, x=productline_col, y=sales_col, color=deal_size_col, title={productline_deal_title})",
                        f"    fig.update_layout(xaxis_title={productline_axis_label}, yaxis_title={sales_axis_label}, legend_title={deal_size_axis_label})",
                        "    fig.show()",
                    ]
                )
            )
        if "productline_deal_size_heatmap" in selected_productline_charts:
            productline_cells.append(
                "\n".join(
                    [
                        "# chart_id: productline_deal_size_heatmap",
                        "if deal_size_col in clean_df.columns:",
                        "    productline_deal_size = clean_df.groupby([productline_col, deal_size_col], as_index=False)[sales_col].sum()",
                        "    top_productlines = productline_sales.head(8)[productline_col].tolist()",
                        "    heatmap_source = productline_deal_size[productline_deal_size[productline_col].isin(top_productlines)]",
                        "    heatmap_df = heatmap_source.pivot(index=productline_col, columns=deal_size_col, values=sales_col).fillna(0)",
                        "    plt.figure(figsize=(10, max(4, len(heatmap_df) * 0.6)))",
                        "    sns.heatmap(heatmap_df, annot=True, fmt='.0f', cmap='YlGnBu')",
                        f"    plt.title({productline_deal_heatmap_title})",
                        "    plt.tight_layout()",
                        "    plt.show()",
                    ]
                )
            )
        section.code_cells.extend(productline_cells)
    selection_markdown = _chart_selection_markdown(chart_selection_plan, section_id, output_language=output_language)
    if selection_markdown:
        section.markdown_blocks.append(selection_markdown)
    return _apply_chart_selection_to_section(section, chart_selection_plan)


def _build_segment_and_region_fallback_section(
    *,
    section_id,
    report,
    schema_mapping,
    segment_region_strategy,
    chart_selection_plan,
    output_language,
    _chart_selection_markdown,
    _apply_chart_selection_to_section,
    _segment_region_section,
    markdown_blocks,
    sales_col,
    quantity_col,
    profit_col,
) -> NotebookSectionContent:
    section = _segment_region_section(
        markdown_blocks=markdown_blocks,
        report=report,
        schema_mapping=schema_mapping,
        sales_col=sales_col,
        quantity_col=quantity_col,
        profit_col=profit_col,
        segment_region_strategy=segment_region_strategy,
        chart_selection_plan=chart_selection_plan,
        output_language=output_language,
    )
    selection_markdown = _chart_selection_markdown(chart_selection_plan, section_id, output_language=output_language)
    if selection_markdown:
        section.markdown_blocks.append(selection_markdown)
    return _apply_chart_selection_to_section(section, chart_selection_plan)


def _build_country_market_fallback_section(
    *,
    section_id,
    schema_mapping,
    chart_selection_plan,
    output_language,
    _original_column,
    _nb_text,
    _nb_quote,
    _chart_selection_markdown,
    _apply_chart_selection_to_section,
    _selection_includes,
    markdown_blocks,
    order_date_col,
    sales_col,
    country_col,
    customer_col,
    order_id_col,
) -> NotebookSectionContent:
    country_label = _nb_quote(output_language, "Country", "国家")
    country_sales_title = _nb_quote(output_language, "Sales by Country", "国家销售额对比")
    country_sales_share_title = _nb_quote(output_language, "Country Sales Share", "国家销售额占比")
    country_aov_title = _nb_quote(output_language, "Average Order Value by Country", "国家平均订单金额对比")
    country_productline_title = _nb_quote(output_language, "Country x Product Line Sales Heatmap", "国家 x PRODUCTLINE 销售热力图")
    country_customer_title = _nb_quote(output_language, "Customer Count by Country", "国家客户数对比")
    country_order_title = _nb_quote(output_language, "Order Count by Country", "国家订单数对比")
    country_pareto_title = _nb_quote(output_language, "Country Sales Concentration Pareto", "国家销售集中度 Pareto")
    country_monthly_title = _nb_quote(output_language, "Monthly Sales Trend of Top Countries", "头部国家月度销售趋势")
    sales_axis_label = _nb_quote(output_language, "Sales", "销售额")
    customer_count_label = _nb_quote(output_language, "Customer Count", "客户数")
    order_count_label = _nb_quote(output_language, "Order Count", "订单数")
    avg_order_value_label = _nb_quote(output_language, "Average Order Value", "平均订单金额")
    country_takeaway = _nb_text(
        output_language,
        "### Section Takeaway\n\nThis section treats country markets as operating slices and compares sales, average order value, customer count, and order count. Follow-up review should combine top-country sales trends with order value to confirm whether growth comes from customer expansion or higher order value.",
        "### 本节结论\n\n本节把国家市场作为独立经营切片，优先比较销售额、客单价、客户数和订单数。头部国家与尾部国家的差距决定了市场集中度，也会影响库存、投放和履约资源分配。后续应把头部国家的销售趋势和订单价值放在一起复盘，确认增长来自客户扩张还是客单价提升。",
    )
    code_cells = [
        "\n".join(
            [
                f"country_col = {country_col!r}",
                f"sales_col = {sales_col!r}",
                f"order_id_col = {order_id_col!r}",
                f"customer_col = {customer_col!r}",
                f"order_date_col = {order_date_col!r}",
                f"productline_col = {_original_column(schema_mapping, 'productline', 'PRODUCTLINE')!r}",
                "country_market = clean_df.groupby(country_col, as_index=False).agg(sales_amount=(sales_col, 'sum'))",
                "if order_id_col in clean_df.columns:",
                "    order_counts = clean_df.groupby(country_col)[order_id_col].nunique().reset_index(name='order_count')",
                "else:",
                "    order_counts = clean_df.groupby(country_col).size().reset_index(name='order_count')",
                "country_market = country_market.merge(order_counts, on=country_col, how='left')",
                "if customer_col in clean_df.columns:",
                "    customer_counts = clean_df.groupby(country_col)[customer_col].nunique().reset_index(name='customer_count')",
                "    country_market = country_market.merge(customer_counts, on=country_col, how='left')",
                "country_market['avg_order_value'] = country_market['sales_amount'] / country_market['order_count'].mask(country_market['order_count'] == 0)",
                "country_market = country_market.sort_values('sales_amount', ascending=False)",
                "country_market.head(15)",
            ]
        ),
        f"# chart_id: country_sales_bar\nfig = px.bar(country_market.head(12), x=country_col, y='sales_amount', title={country_sales_title})\nfig.update_layout(xaxis_title={country_label}, yaxis_title={sales_axis_label})\nfig.show()",
        f"# chart_id: country_sales_donut\nfig = px.pie(country_market.head(8), names=country_col, values='sales_amount', hole=0.45, title={country_sales_share_title})\nfig.update_traces(textposition='inside', textinfo='percent+label')\nfig.show()",
        f"# chart_id: country_avg_order_value_bar\nfig = px.bar(country_market.head(12), x=country_col, y='avg_order_value', title={country_aov_title})\nfig.update_layout(xaxis_title={country_label}, yaxis_title={avg_order_value_label})\nfig.show()",
        "\n".join(
            [
                "# chart_id: country_productline_heatmap",
                "if productline_col in clean_df.columns:",
                "    country_productline = clean_df.groupby([country_col, productline_col], as_index=False)[sales_col].sum()",
                "    top_countries = country_market.head(8)[country_col].tolist()",
                "    country_productline = country_productline[country_productline[country_col].isin(top_countries)]",
                "    country_productline_matrix = country_productline.pivot(index=country_col, columns=productline_col, values=sales_col).fillna(0)",
                "    plt.figure(figsize=(10, max(4, len(country_productline_matrix) * 0.55)))",
                "    sns.heatmap(country_productline_matrix, annot=True, fmt='.0f', cmap='YlGnBu')",
                f"    plt.title({country_productline_title})",
                "    plt.tight_layout()",
                "    plt.show()",
            ]
        ),
        f"# chart_id: country_customer_count_bar\nif 'customer_count' in country_market.columns:\n    fig = px.bar(country_market.head(12), x=country_col, y='customer_count', title={country_customer_title})\n    fig.update_layout(xaxis_title={country_label}, yaxis_title={customer_count_label})\n    fig.show()",
        f"# chart_id: country_order_count_bar\nfig = px.bar(country_market.head(12), x=country_col, y='order_count', title={country_order_title})\nfig.update_layout(xaxis_title={country_label}, yaxis_title={order_count_label})\nfig.show()",
        "\n".join(
            [
                "# chart_id: country_concentration_pareto",
                "country_pareto = country_market[[country_col, 'sales_amount']].copy()",
                "country_pareto['cumulative_share'] = country_pareto['sales_amount'].cumsum() / country_pareto['sales_amount'].sum()",
                "fig, ax1 = plt.subplots(figsize=(10, 5))",
                "ax1.bar(country_pareto[country_col], country_pareto['sales_amount'], color='#4E79A7')",
                "ax1.tick_params(axis='x', rotation=60)",
                "ax2 = ax1.twinx()",
                "ax2.plot(country_pareto[country_col], country_pareto['cumulative_share'], color='#E15759', marker='o')",
                "ax2.axhline(0.8, color='#59A14F', linestyle='--', linewidth=1)",
                "ax2.set_ylim(0, 1.05)",
                f"plt.title({country_pareto_title})",
                "plt.tight_layout()",
                "plt.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: country_monthly_trend",
                "country_monthly = clean_df.copy()",
                "country_monthly['order_month'] = country_monthly[order_date_col].dt.to_period('M').astype(str)",
                "top_countries = country_market.head(5)[country_col].tolist()",
                "country_monthly = country_monthly[country_monthly[country_col].isin(top_countries)]",
                "country_monthly = country_monthly.groupby([country_col, 'order_month'], as_index=False)[sales_col].sum()",
                f"fig = px.line(country_monthly, x='order_month', y=sales_col, color=country_col, title={country_monthly_title})",
                "fig.show()",
            ]
        ),
    ]
    if isinstance(chart_selection_plan, dict):
        gated_country_cells = [code_cells[0]]
        if _selection_includes(chart_selection_plan, "country_sales_bar"):
            gated_country_cells.append(code_cells[1])
        if _selection_includes(chart_selection_plan, "country_sales_donut"):
            gated_country_cells.append(code_cells[2])
        if _selection_includes(chart_selection_plan, "country_avg_order_value_bar"):
            gated_country_cells.append(code_cells[3])
        if _selection_includes(chart_selection_plan, "country_productline_heatmap"):
            gated_country_cells.append(code_cells[4])
        if _selection_includes(chart_selection_plan, "country_customer_count_bar"):
            gated_country_cells.append(code_cells[5])
        if _selection_includes(chart_selection_plan, "country_order_count_bar"):
            gated_country_cells.append(code_cells[6])
        if _selection_includes(chart_selection_plan, "country_concentration_pareto"):
            gated_country_cells.append(code_cells[7])
        if _selection_includes(chart_selection_plan, "country_monthly_trend"):
            gated_country_cells.append(code_cells[8])
        code_cells = gated_country_cells
    markdown_blocks.append(country_takeaway)
    selection_markdown = _chart_selection_markdown(chart_selection_plan, section_id, output_language=output_language)
    if selection_markdown:
        markdown_blocks.append(selection_markdown)
    return _apply_chart_selection_to_section(
        NotebookSectionContent(section_id=section_id, markdown_blocks=markdown_blocks, code_cells=code_cells),
        chart_selection_plan,
    )


def _build_order_structure_fallback_section(
    *,
    section_id,
    chart_selection_plan,
    output_language,
    _nb_text,
    _nb_quote,
    _chart_selection_markdown,
    _apply_chart_selection_to_section,
    markdown_blocks,
    sales_col,
    quantity_col,
    customer_col,
    order_id_col,
    unit_price_col,
    deal_size_col,
    order_status_col,
) -> NotebookSectionContent:
    basket_distribution_title = _nb_quote(output_language, "Basket Size Distribution (P99 Clipped)", "订单篮子大小分布（P99截尾）")
    basket_distribution_raw_title = _nb_quote(output_language, "Basket Size Distribution", "订单篮子大小分布")
    basket_long_tail_message = _nb_quote(
        output_language,
        "Basket size is long-tailed, so the chart clips at P99 for readability; clipped records={distribution_readability_trace['clipped_count']}.",
        "订单篮子大小为长尾分布，图中按 P99={p99:.2f} 截尾展示主体区间；截尾记录数={distribution_readability_trace['clipped_count']}。",
    )
    invoice_distribution_title = _nb_quote(output_language, "Invoice Value Distribution (P99 Clipped)", "发票金额分布（P99截尾）")
    invoice_distribution_raw_title = _nb_quote(output_language, "Invoice Value Distribution", "发票金额分布")
    invoice_long_tail_message = _nb_quote(
        output_language,
        "Invoice value is long-tailed, so the chart clips at P99 for readability; clipped records={distribution_readability_trace['clipped_count']}.",
        "发票金额为长尾分布，图中按 P99={p99:.2f} 截尾展示主体区间；截尾记录数={distribution_readability_trace['clipped_count']}。",
    )
    customer_order_distribution_title = _nb_quote(output_language, "Customer Order Count Distribution", "客户订单数分布")
    deal_size_sales_title = _nb_quote(output_language, "Deal Size Sales Comparison", "DEALSIZE 销售额对比")
    deal_size_share_title = _nb_quote(output_language, "Deal Size Sales Share", "DEALSIZE 销售额占比")
    deal_size_order_count_title = _nb_quote(output_language, "Deal Size Order Count Comparison", "DEALSIZE 订单数对比")
    deal_size_aov_title = _nb_quote(output_language, "Deal Size Average Order Value Comparison", "DEALSIZE 平均订单金额对比")
    status_line_count_title = _nb_quote(output_language, "Order Line Count by Status", "STATUS 订单行数对比")
    deal_size_invoice_title = _nb_quote(output_language, "Deal Size Invoice Value Distribution", "DEALSIZE 订单金额分布")
    status_sales_title = _nb_quote(output_language, "Sales Structure by Status", "STATUS 销售额结构")
    status_sales_share_title = _nb_quote(output_language, "Sales Share by Status", "STATUS 销售额占比")
    status_order_count_title = _nb_quote(output_language, "Order Count by Status", "STATUS 订单数对比")
    status_deal_title = _nb_quote(output_language, "Status x Deal Size Sales Structure", "STATUS x 交易规模销售结构")
    unit_price_quantity_title = _nb_quote(output_language, "Unit Price vs. Quantity", "单价与数量关系")
    deal_size_label = _nb_quote(output_language, "Deal Size", "交易规模")
    sales_axis_label = _nb_quote(output_language, "Sales", "销售额")
    order_count_label = _nb_quote(output_language, "Order Count", "订单数")
    avg_order_value_label = _nb_quote(output_language, "Average Order Value", "平均订单金额")
    order_status_label = _nb_quote(output_language, "Order Status", "订单状态")
    order_structure_takeaway = _nb_text(
        output_language,
        "### Section Takeaway\n\nThis section breaks order structure into invoice value, basket size, repeat behavior, deal size, and order status. Follow-up review should focus on unusually large orders, canceled orders, and low-frequency high-value customers.",
        "### 本节结论\n\n本节把订单结构拆成发票金额、篮子大小、客户复购、交易规模和订单状态。高金额订单与普通订单的差距越大，少数大单对销售判断的影响越强。后续应把订单金额、购买件数和状态放回明细表，优先复盘异常大单、取消订单和低频高额客户。",
    )
    code_cells = [
        "\n".join(
            [
                f"order_id_col = {order_id_col!r}",
                f"customer_col = {customer_col!r}",
                f"sales_col = {sales_col!r}",
                f"quantity_col = {quantity_col!r}",
                f"unit_price_col = {unit_price_col!r}",
                f"deal_size_col = {deal_size_col!r}",
                f"order_status_col = {order_status_col!r}",
                "invoice_values = clean_df.groupby(order_id_col, as_index=False)[sales_col].sum().rename(columns={sales_col: 'invoice_value'}) if order_id_col in clean_df.columns else pd.DataFrame()",
                "basket_sizes = clean_df.groupby(order_id_col).size().reset_index(name='basket_size') if order_id_col in clean_df.columns else pd.DataFrame()",
                "customer_order_counts = clean_df.groupby(customer_col)[order_id_col].nunique().reset_index(name='order_count') if customer_col in clean_df.columns and order_id_col in clean_df.columns else pd.DataFrame()",
                "invoice_values.head(), basket_sizes.head(), customer_order_counts.head()",
            ]
        ),
        f"# chart_id: basket_size_distribution\nif not basket_sizes.empty:\n    basket_values = pd.to_numeric(basket_sizes['basket_size'], errors='coerce').dropna()\n    p99 = basket_values.quantile(0.99) if not basket_values.empty else None\n    median = basket_values.median() if not basket_values.empty else None\n    max_value = basket_values.max() if not basket_values.empty else None\n    long_tail = bool(p99 and ((max_value > p99 * 2) or (median and median > 0 and p99 / median > 10)))\n    basket_plot = basket_sizes[basket_sizes['basket_size'] <= p99].copy() if long_tail else basket_sizes.copy()\n    distribution_readability_trace = {{'chart_id': 'basket_size_distribution', 'distribution_readability_mode': 'p99_clipped' if long_tail else 'raw', 'p99': float(p99) if p99 is not None else None, 'max': float(max_value) if max_value is not None else None, 'median': float(median) if median is not None else None, 'clipped_count': int((basket_values > p99).sum()) if long_tail else 0, 'clipped_ratio': float((basket_values > p99).mean()) if long_tail else 0.0, 'reason': 'P99 clipped for readability' if long_tail else 'raw distribution'}}\n    fig = px.histogram(basket_plot, x='basket_size', nbins=30, title={basket_distribution_title} if long_tail else {basket_distribution_raw_title})\n    fig.show()\n    if long_tail:\n        print(f{basket_long_tail_message})",
        f"# chart_id: invoice_value_distribution\nif not invoice_values.empty:\n    invoice_value_series = pd.to_numeric(invoice_values['invoice_value'], errors='coerce').dropna()\n    p99 = invoice_value_series.quantile(0.99) if not invoice_value_series.empty else None\n    median = invoice_value_series.median() if not invoice_value_series.empty else None\n    max_value = invoice_value_series.max() if not invoice_value_series.empty else None\n    long_tail = bool(p99 and ((max_value > p99 * 2) or (median and median > 0 and p99 / median > 10)))\n    invoice_plot = invoice_values[invoice_values['invoice_value'] <= p99].copy() if long_tail else invoice_values.copy()\n    distribution_readability_trace = {{'chart_id': 'invoice_value_distribution', 'distribution_readability_mode': 'p99_clipped' if long_tail else 'raw', 'p99': float(p99) if p99 is not None else None, 'max': float(max_value) if max_value is not None else None, 'median': float(median) if median is not None else None, 'clipped_count': int((invoice_value_series > p99).sum()) if long_tail else 0, 'clipped_ratio': float((invoice_value_series > p99).mean()) if long_tail else 0.0, 'reason': 'P99 clipped for readability' if long_tail else 'raw distribution'}}\n    fig = px.histogram(invoice_plot, x='invoice_value', nbins=40, title={invoice_distribution_title} if long_tail else {invoice_distribution_raw_title})\n    fig.show()\n    if long_tail:\n        print(f{invoice_long_tail_message})",
        f"if not customer_order_counts.empty:\n    fig = px.histogram(customer_order_counts, x='order_count', nbins=30, title={customer_order_distribution_title})\n    fig.show()",
        "\n".join(
            [
                "# chart_id: deal_size_sales_bar",
                "if deal_size_col in clean_df.columns:",
                "    deal_size_sales = clean_df.groupby(deal_size_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)",
                f"    fig = px.bar(deal_size_sales, x=deal_size_col, y=sales_col, title={deal_size_sales_title})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: deal_size_sales_donut",
                "if deal_size_col in clean_df.columns:",
                "    deal_size_sales = clean_df.groupby(deal_size_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)",
                f"    fig = px.pie(deal_size_sales, names=deal_size_col, values=sales_col, hole=0.45, title={deal_size_share_title})",
                "    fig.update_traces(textposition='inside', textinfo='percent+label')",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: deal_size_order_count_bar",
                "if deal_size_col in clean_df.columns and order_id_col in clean_df.columns:",
                "    deal_size_order_counts = clean_df.groupby(deal_size_col)[order_id_col].nunique().reset_index(name='order_count').sort_values('order_count', ascending=False)",
                f"    fig = px.bar(deal_size_order_counts, x=deal_size_col, y='order_count', title={deal_size_order_count_title})",
                f"    fig.update_layout(xaxis_title={deal_size_label}, yaxis_title={order_count_label})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: deal_size_avg_order_value_bar",
                "if deal_size_col in clean_df.columns and order_id_col in clean_df.columns:",
                "    deal_order_values = clean_df.groupby([deal_size_col, order_id_col], as_index=False)[sales_col].sum().rename(columns={sales_col: 'order_sales'})",
                "    deal_size_aov = deal_order_values.groupby(deal_size_col, as_index=False)['order_sales'].mean().rename(columns={'order_sales': 'avg_order_value'}).sort_values('avg_order_value', ascending=False)",
                f"    fig = px.bar(deal_size_aov, x=deal_size_col, y='avg_order_value', title={deal_size_aov_title})",
                f"    fig.update_layout(xaxis_title={deal_size_label}, yaxis_title={avg_order_value_label})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: status_order_line_count_bar",
                "if order_status_col in clean_df.columns and order_id_col in clean_df.columns:",
                "    status_line_counts = clean_df.groupby(order_status_col).size().reset_index(name='order_line_count').sort_values('order_line_count', ascending=False)",
                f"    fig = px.bar(status_line_counts, x=order_status_col, y='order_line_count', title={status_line_count_title})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: deal_size_order_value_boxplot",
                "if deal_size_col in clean_df.columns and order_id_col in clean_df.columns:",
                "    deal_invoice_values = clean_df.groupby([deal_size_col, order_id_col], as_index=False)[sales_col].sum().rename(columns={sales_col: 'invoice_value'})",
                f"    fig = px.box(deal_invoice_values, x=deal_size_col, y='invoice_value', title={deal_size_invoice_title})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: order_status_breakdown",
                "if order_status_col in clean_df.columns:",
                "    order_status_breakdown = clean_df.groupby(order_status_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)",
                f"    fig = px.bar(order_status_breakdown, x=order_status_col, y=sales_col, title={status_sales_title})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: order_status_sales_donut",
                "if order_status_col in clean_df.columns:",
                "    order_status_breakdown = clean_df.groupby(order_status_col, as_index=False)[sales_col].sum().sort_values(sales_col, ascending=False)",
                f"    fig = px.pie(order_status_breakdown, names=order_status_col, values=sales_col, hole=0.45, title={status_sales_share_title})",
                "    fig.update_traces(textposition='inside', textinfo='percent+label')",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: status_order_count_bar",
                "if order_status_col in clean_df.columns and order_id_col in clean_df.columns:",
                "    status_order_counts = clean_df.groupby(order_status_col)[order_id_col].nunique().reset_index(name='order_count').sort_values('order_count', ascending=False)",
                f"    fig = px.bar(status_order_counts, x=order_status_col, y='order_count', title={status_order_count_title})",
                f"    fig.update_layout(xaxis_title={order_status_label}, yaxis_title={order_count_label})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: status_deal_size_stacked_bar",
                "if order_status_col in clean_df.columns and deal_size_col in clean_df.columns:",
                "    status_deal_size = clean_df.groupby([order_status_col, deal_size_col], as_index=False)[sales_col].sum()",
                f"    fig = px.bar(status_deal_size, x=order_status_col, y=sales_col, color=deal_size_col, title={status_deal_title})",
                f"    fig.update_layout(xaxis_title={order_status_label}, yaxis_title={sales_axis_label}, legend_title={deal_size_label})",
                "    fig.show()",
            ]
        ),
        "\n".join(
            [
                "# chart_id: unit_price_quantity_scatter",
                "if unit_price_col in clean_df.columns and quantity_col in clean_df.columns:",
                "    price_quantity_df = clean_df[[unit_price_col, quantity_col, sales_col]].copy()",
                "    price_quantity_df['_abs_sales_amount'] = pd.to_numeric(price_quantity_df[sales_col], errors='coerce').abs().fillna(0)",
                f"    fig = px.scatter(price_quantity_df, x=unit_price_col, y=quantity_col, size='_abs_sales_amount', title={unit_price_quantity_title})",
                "    fig.show()",
            ]
        ),
    ]
    markdown_blocks.append(order_structure_takeaway)
    selection_markdown = _chart_selection_markdown(chart_selection_plan, section_id, output_language=output_language)
    if selection_markdown:
        markdown_blocks.append(selection_markdown)
    return _apply_chart_selection_to_section(
        NotebookSectionContent(section_id=section_id, markdown_blocks=markdown_blocks, code_cells=code_cells),
        chart_selection_plan,
    )


def _build_discount_and_profit_fallback_section(
    *,
    section_id,
    report,
    schema_mapping,
    discount_profit_strategy,
    chart_selection_plan,
    output_language,
    _chart_selection_markdown,
    _apply_chart_selection_to_section,
    _chart_analysis_markdown,
    _discount_profit_section,
    _without_non_llm_chart_display_cells,
    markdown_blocks,
    sales_col,
    category_col,
    product_col,
    discount_col,
    profit_col,
) -> NotebookSectionContent:
    section = _discount_profit_section(
        markdown_blocks=markdown_blocks,
        schema_mapping=schema_mapping,
        sales_col=sales_col,
        category_col=category_col,
        product_col=product_col,
        discount_col=discount_col,
        profit_col=profit_col,
        discount_profit_strategy=discount_profit_strategy,
        chart_selection_plan=chart_selection_plan,
        output_language=output_language,
    )
    if (
        (not isinstance(discount_profit_strategy, dict) or discount_profit_strategy.get("source") != "llm_sanitized")
        and not any(
            isinstance(item, dict) and item.get("section_id") == "discount_and_profit"
            for item in (chart_selection_plan or {}).get("selected_charts", [])
        )
    ):
        section = NotebookSectionContent(
            section_id=section.section_id,
            markdown_blocks=section.markdown_blocks,
            code_cells=_without_non_llm_chart_display_cells(section.code_cells),
        )
    chart_analysis = _chart_analysis_markdown(section_id, report, output_language=output_language)
    if chart_analysis:
        section.markdown_blocks.append(chart_analysis)
    selection_markdown = _chart_selection_markdown(chart_selection_plan, section_id, output_language=output_language)
    if selection_markdown:
        section.markdown_blocks.append(selection_markdown)
    return _apply_chart_selection_to_section(section, chart_selection_plan)


def _build_modeling_fallback_section(
    *,
    section_id,
    report,
    _module_for_section,
    markdown_blocks,
) -> NotebookSectionContent:
    module = _module_for_section("modeling", report)
    warnings = list(module.warnings if module is not None else ["loss_risk_modeling output is unavailable."])
    markdown_blocks.extend(
        [
            "建模目标：识别更可能发生负利润的订单或记录，用于人工复核优先级排序。",
            "Target 定义：`is_loss = profit < 0`。",
            "该模型用于亏损风险预警和人工复核优先级排序，不代表特征与亏损之间存在因果关系。简言之，不代表因果关系。",
        ]
    )
    if warnings and (module is None or not module.tables.get("model_comparison")):
        markdown_blocks.append("当前数据暂不适合进行亏损风险建模。\n\n" + "\n".join(f"- {item}" for item in warnings))
    return NotebookSectionContent(section_id=section_id, markdown_blocks=markdown_blocks, code_cells=[])


def _build_conclusions_fallback_section(
    *,
    report,
    output_language,
    _conclusions_section_content,
) -> NotebookSectionContent:
    return _conclusions_section_content(report, output_language=output_language)


def _build_forecast_fallback_section(
    *,
    section_id,
    report,
    output_language,
    _chart_analysis_markdown,
    markdown_blocks,
    sales_col,
) -> NotebookSectionContent:
    code_cells = [
        "\n".join(
            [
                "forecast_source = monthly_sales.copy()",
                f"baseline_value = float(forecast_source['{sales_col}'].tail(3).mean())",
                "forecast_periods = pd.period_range(forecast_source['order_month'].iloc[-1], periods=4, freq='M')[1:]",
                "forecast_df = pd.DataFrame({'order_month': forecast_periods.astype(str)})",
                "forecast_df['forecast_sales'] = baseline_value",
                "forecast_df",
            ]
        ),
        "\n".join(
            [
                "forecast_plot_df = monthly_sales[['order_month', '{0}']].rename(columns={{'{0}': 'value'}})".format(
                    sales_col
                ),
                "forecast_plot_df['series'] = 'actual'",
                "future_plot_df = forecast_df.rename(columns={'forecast_sales': 'value'})",
                "future_plot_df['series'] = 'forecast'",
                "combined_plot_df = pd.concat([forecast_plot_df, future_plot_df], ignore_index=True)",
                "fig = px.line(combined_plot_df, x='order_month', y='value', color='series', title='实际与预测销售额对比')",
                "fig.show()",
            ]
        ),
    ]
    # Forecast charts are deterministic fallback, so keep them hidden.
    code_cells = code_cells[:1]
    chart_analysis = _chart_analysis_markdown(section_id, report, output_language=output_language)
    if chart_analysis:
        markdown_blocks.append(chart_analysis)
    return NotebookSectionContent(section_id=section_id, markdown_blocks=markdown_blocks, code_cells=code_cells)
