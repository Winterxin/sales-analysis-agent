from __future__ import annotations

from nbformat.v4 import new_code_cell, new_markdown_cell

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_narrative import NotebookSectionNarrative
from app.schemas.notebook_outline import NotebookSection
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping

from app.services.notebook.field_utils import mapped_fields
from app.services.notebook.markdown_sanitizer import (
    clean_business_text,
    paragraph_text,
    sanitize_final_markdown_text,
    trim_markdown_paragraph,
)
from app.services.notebook import modeling_renderer
from app.services.notebook.report_utils import CORE_SECTION_IDS, module_for_section
from app.services.output_language import (
    is_english_output,
    safe_english_sentence,
)


ENGLISH_SECTION_INTROS = {
    "metric_distributions": "Review sales, profit, discount, and other metric distributions before interpreting downstream findings.",
    "sales_trends": "Review monthly sales peaks, troughs, and volatility to understand business rhythm.",
    "product_and_category": "Compare product and category contribution with profit quality to identify mismatched growth.",
    "segment_and_region": "Break sales and profit down by customer and geography slices to locate weak segments.",
    "order_structure": "Review basket size, repeat behavior, and order structure to understand transaction patterns.",
    "discount_and_profit": "Assess whether higher discounts are eroding profit quality and increasing loss risk.",
    "modeling": "Use the model as a review-priority aid and explain why it must not drive automatic decisions.",
    "forecast": "Use the forecast baseline as a monitoring reference, not as a production forecast.",
}

ENGLISH_SECTION_OBJECT_FOCUS = {
    "metric_distributions": "long-tail metrics, outliers, and mean distortion",
    "sales_trends": "peaks, troughs, and monthly volatility",
    "product_and_category": "high-sales low-profit products and category margin differences",
    "segment_and_region": "segment / region slices",
    "order_structure": "basket size, repeat behavior, and unusual order patterns",
    "discount_and_profit": "high discounts, loss rate, and margin erosion",
    "forecast": "forecast baseline, deviation range, and monitoring threshold",
}


GENERIC_ENGLISH_SECTION_TAKEAWAY_FALLBACKS = {
    "mapped fields and module outputs define the current review scope.",
    "This section identifies the records and business objects that need follow-up review.",
}


def _explicit_english(output_language: str | None) -> bool:
    return output_language is not None and is_english_output(output_language)


def _is_generic_english_section_takeaway(text: str) -> bool:
    normalized = " ".join(str(text or "").split())
    lowered = normalized.lower()
    if not normalized or normalized in GENERIC_ENGLISH_SECTION_TAKEAWAY_FALLBACKS:
        return True
    if lowered.startswith("this section reviews"):
        return True
    if lowered.startswith("section purpose:"):
        return True
    if "mapped sales dataset" in lowered or "available module evidence" in lowered:
        return True
    if "summarize key findings, risks, and next actions" in lowered:
        return True
    return False


def section_intro_markdown(
    section: NotebookSection,
    output_language: str | None = None,
) -> str | None:
    if _explicit_english(output_language):
        return ENGLISH_SECTION_INTROS.get(section.section_id)
    intros = {
        "metric_distributions": "先查看销售、利润、折扣等核心指标的分布，判断长尾、异常值或均值失真是否会影响后续解读。",
        "sales_trends": "本节聚焦销售额的峰值、低谷和月度波动，用趋势视角复核经营节奏是否稳定。",
        "product_and_category": "本节比较商品和类目的销售贡献与利润质量，识别规模表现和利润表现不同步的对象。",
        "segment_and_region": "本节从客群、区域及其组合切片观察销售与利润差异，定位需要下钻的弱势切片。",
        "order_structure": "本节查看订单行数、复购和异常订单结构，判断订单形态是否影响规模与利润质量。",
        "discount_and_profit": "本节围绕折扣与利润关系展开，观察高折扣是否已经侵蚀利润并推高亏损率。",
        "modeling": "本节使用亏损风险分类模型识别需要人工复核的高风险订单，并明确模型只用于预警，不用于自动决策。",
        "forecast": "本节用基线预测建立后续监控参照，帮助识别销售趋势偏离和波动区间。",
    }
    return intros.get(section.section_id)


def section_conclusion_markdown(
    section: NotebookSection,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    output_language: str | None = None,
) -> str | None:
    if section.section_id not in CORE_SECTION_IDS:
        return None
    module = module_for_section(section.section_id, report)
    evidence = "；".join(module.findings[:2]) if module and module.findings else section.purpose
    evidence = clean_business_text(evidence)
    mapped = mapped_fields(schema_mapping)
    if _explicit_english(output_language):
        if not (module and module.findings):
            return None
        evidence = safe_english_sentence(
            evidence,
            fallback="mapped fields and module outputs define the current review scope.",
            require_complete_sentence=True,
        )
        if _is_generic_english_section_takeaway(evidence):
            return None
        object_focus = ENGLISH_SECTION_OBJECT_FOCUS.get(section.section_id, "the key business objects in this section")
        data_limit = ""
        if section.section_id == "discount_and_profit" and not {"discount", "profit"} <= mapped:
            data_limit = (
                "Data limitation: the current fields cannot fully validate discount and profit together, "
                "so this section keeps the review at profit-quality or risk-object level."
            )
        elif section.section_id == "segment_and_region" and not {"segment", "region"} <= mapped:
            data_limit = (
                "Data limitation: the current fields cannot form a stable segment x region slice, "
                "so this section only explains mapped dimensions."
            )
        return "\n".join(
            line
            for line in [
                "### Section Takeaway",
                f"The core evidence in `{section.title}` is: {evidence}",
                f"This section focuses on {object_focus}, so follow-up review should return to the detailed records behind these business objects.",
                data_limit,
            ]
            if line
        )
    if not evidence:
        evidence = "数据限制说明：当前章节只能基于已映射字段和现有模块结果解释。"
    object_focus = {
        "metric_distributions": "长尾、异常值和均值失真",
        "sales_trends": "峰值、低谷和月度波动",
        "product_and_category": "高销售低利润商品和类目利润率差异",
        "segment_and_region": "Segment / Region 等组合切片",
        "order_structure": "订单行数、复购和异常订单",
        "discount_and_profit": "高折扣、亏损率和利润侵蚀",
        "forecast": "预测基线、偏离区间和监控阈值",
    }.get(section.section_id, "本节关键对象")
    data_limit = ""
    if section.section_id == "discount_and_profit" and not {"discount", "profit"} <= mapped:
        data_limit = "数据限制说明：当前字段不足以同时验证折扣和利润，只能保留利润质量或风险对象层面的判断。"
    elif section.section_id == "segment_and_region" and not {"segment", "region"} <= mapped:
        data_limit = "数据限制说明：当前字段不足以形成客群 x 区域组合，只能解释已映射维度。"
    return "\n".join(
        line
        for line in [
            "### 本节结论",
            f"`{section.title}` 的核心证据是：{evidence}。",
            f"本节重点落在{object_focus}，因此后续复盘应优先回到这些业务对象的明细记录。",
            data_limit,
        ]
        if line
    )


def narrative_conclusion_markdown(
    section: NotebookSection,
    section_narrative: NotebookSectionNarrative | None,
    schema_mapping: SchemaMapping | None = None,
) -> str | None:
    if section_narrative is None:
        return None
    blocks: list[str] = []
    intro = clean_business_text(section_narrative.intro)
    if intro:
        blocks.append(intro)
    observations = [
        clean_business_text(item)
        for item in section_narrative.key_observations
        if clean_business_text(item)
    ]
    if observations:
        blocks.append("### 核心判断\n\n" + "\n".join(f"- {item}" for item in observations))
    takeaway = clean_business_text(section_narrative.business_takeaway)
    if takeaway:
        blocks.append("### 行动方向\n\n" + takeaway)
    if not blocks:
        return None
    text = f"## {section.title}\n\n" + "\n\n".join(blocks)
    return sanitize_final_markdown_text(text, schema_mapping)


def is_template_chart_analysis_block(block: str) -> bool:
    heading = block.strip().splitlines()[0] if block.strip() else ""
    legacy_mojibake_chart = "\u9365\u6350\u3003"
    legacy_matplotlib_mojibake = "\u95f8\u5d2a\u9286"
    return heading.startswith("### ") and (
        "\u56fe\u8868\u5206\u6790" in heading
        or legacy_mojibake_chart in heading
        or legacy_matplotlib_mojibake in heading
        or "\u037c" in heading
    )


def is_chart_selection_basis_block(block: str) -> bool:
    heading = block.strip().splitlines()[0] if block.strip() else ""
    return heading.startswith("### 图表选择依据")


def is_section_conclusion_block(block: str) -> bool:
    heading = block.strip().splitlines()[0] if block.strip() else ""
    return heading.startswith("### 本节结论")


def is_business_takeaway_block(block: str) -> bool:
    heading = block.strip().splitlines()[0].strip().lower() if block.strip() else ""
    return heading in {"### business takeaway", "### 业务含义", "### 业务结论"}


def split_section_markdown(markdown_blocks: list[str]) -> tuple[list[str], list[str]]:
    intro_blocks: list[str] = []
    analysis_blocks: list[str] = []
    for block in markdown_blocks:
        if is_template_chart_analysis_block(block):
            analysis_blocks.append(block)
        else:
            intro_blocks.append(block)
    return intro_blocks, analysis_blocks


def section_has_llm_chart_decision(code_cells: list[str]) -> bool:
    return any("llm_sanitized" in str(code_cell) for code_cell in code_cells)


def visible_final_markdown_blocks(markdown_blocks: list[str]) -> list[str]:
    # 最终 notebook 只展示 LLM 参与决策后的分析，不展示默认模板块。
    return [
        sanitize_final_markdown_text(block)
        for block in markdown_blocks
        if not is_template_chart_analysis_block(block)
        and not is_chart_selection_basis_block(block)
    ]


def prepare_final_section_markdown_blocks(
    markdown_blocks: list[str],
    section: NotebookSection,
    schema_mapping: SchemaMapping,
    output_language: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    intro_candidates: list[str] = []
    analysis_blocks: list[str] = []
    conclusion_blocks: list[str] = []
    for block in markdown_blocks:
        if is_chart_selection_basis_block(block):
            continue
        if is_section_conclusion_block(block) or is_business_takeaway_block(block):
            conclusion = sanitize_final_markdown_text(block, schema_mapping)
            if conclusion:
                body = paragraph_text(conclusion)
                if _explicit_english(output_language):
                    body = safe_english_sentence(
                        body,
                        fallback="This section identifies the records and business objects that need follow-up review.",
                        require_complete_sentence=True,
                    )
                    if not _is_generic_english_section_takeaway(body):
                        conclusion_blocks.append(f"### Section Takeaway\n\n{body}")
                else:
                    conclusion_blocks.append(f"### 本节结论\n\n{body}" if body else "### 本节结论")
            continue
        if is_template_chart_analysis_block(block):
            analysis = sanitize_final_markdown_text(block, schema_mapping)
            if analysis:
                analysis_blocks.append(analysis)
            continue
        sanitized = sanitize_final_markdown_text(block, schema_mapping)
        if not sanitized:
            continue
        intro_text = paragraph_text(sanitized)
        if intro_text:
            if not (_explicit_english(output_language) and _is_generic_english_section_takeaway(intro_text)):
                intro_candidates.append(intro_text)

    intro = intro_candidates[0] if intro_candidates else (section_intro_markdown(section, output_language=output_language) or section.purpose)
    if _explicit_english(output_language):
        intro = safe_english_sentence(
            intro,
            fallback=section_intro_markdown(section, output_language=output_language) or section.purpose,
            require_complete_sentence=True,
        )
        if _is_generic_english_section_takeaway(intro):
            intro = section_intro_markdown(section, output_language=output_language) or ""
    intro_blocks = [
        trim_markdown_paragraph(
            intro,
            require_complete_sentence=_explicit_english(output_language),
        )
    ] if intro else []
    if conclusion_blocks:
        return intro_blocks, analysis_blocks, [conclusion_blocks[0]]
    return intro_blocks, analysis_blocks, []


def code_step_intro(
    section_id: str,
    code_cell: str,
    output_language: str | None = None,
) -> str | None:
    english = _explicit_english(output_language)
    if section_id == "dataset_and_schema":
        if "pd.read_csv" in code_cell:
            if english:
                return "Load the raw dataset and inspect the first rows."
            return "先导入分析所需库并读取原始数据，确认这份销售表的字段和样例记录。"
        if "df.info()" in code_cell:
            if english:
                return "Check field types and missing values before analysis."
            return "接着检查字段类型和缺失情况，确认哪些列可以直接进入分析。"
        if "describe(include='all')" in code_cell:
            if english:
                return "Review descriptive statistics to understand numeric ranges and category fields."
            return "再看描述统计，先对数值分布和类别字段形成整体印象。"

    if section_id == "data_cleaning":
        if "missing_summary" in code_cell:
            if english:
                return "Check missing values and duplicate records before cleaning."
            return "先检查缺失值和重复记录，确认原始数据质量。"
        if "clean_df = df.copy()" in code_cell:
            if english:
                return "Clean date and numeric fields to build the analysis-ready `clean_df` table."
            return "然后完成日期和数值字段清洗，得到后续分析使用的 `clean_df`。"

    if section_id == "sales_trends":
        if "monthly_sales =" in code_cell:
            if english:
                return "Aggregate monthly sales and quantity to build the trend base table."
            return "先按月汇总销售额和销量，建立趋势分析底表。"
        if "px.line" in code_cell:
            if english:
                return "Plot monthly sales to review peaks, troughs, and volatility."
            return "再看按月销售额趋势图，判断高点、低点和波动节奏。"

    if section_id == "product_and_category":
        if "top_products =" in code_cell and "pareto_df" not in code_cell:
            if english:
                return "Inspect the top product sales table before comparing product contribution."
            return "先查看 Top 商品销售额表，确认头部商品及对应销售额。"
        if "category_sales =" in code_cell:
            if english:
                return "Review category and subcategory summary tables to locate major sales sources."
            return "再看类目与子类目销售汇总表，确认主要销售来源。"
        if "px.treemap" in code_cell:
            if english:
                return "Use the category structure chart to review where sales are concentrated."
            return "接着看类目与子类目销售结构图，判断销售额集中在哪些板块。"
        if "category_profit_quality" in code_cell and "orientation='h'" in code_cell:
            if english:
                return "Compare category profit quality to find sales and margin mismatches."
            return "再看类目利润率质量对比图，定位销售额和利润率不同步的类目。"
        if "high_sales_low_profit_products" in code_cell and "orientation='h'" in code_cell:
            if english:
                return "Review high-sales low-profit products to prioritize pricing and promotion checks."
            return "再看高销售低利润商品图，优先识别需要复盘定价和促销的商品。"
        if "product_profit_bridge" in code_cell and "orientation='h'" in code_cell:
            if english:
                return "Compare top product sales and profit to check whether leading products also contribute margin."
            return "再看头部商品销售额与利润桥接图，确认头部商品是否同步贡献利润。"
        if "orientation='h'" in code_cell:
            if english:
                return "Display the selected chart and inspect the strongest and weakest slices."
            return "再看 Top 商品销售额条形图，比较头部商品之间的差距。"
        if "pareto_df" in code_cell:
            if english:
                return "Review the product Pareto chart to test whether sales are concentrated in a small set of products."
            return "最后看头部商品帕累托图，判断销售是否过度集中在少数商品。"

    if section_id == "segment_and_region":
        if "segment_region =" in code_cell or "segment_view =" in code_cell:
            if english:
                return "Build the segment and region comparison table before reading slice performance."
            return "先汇总不同切片的销售与利润数据，明确后面要比较的对象。"
        if "px.bar" in code_cell:
            if english:
                return "Compare sales across slices to identify stronger and weaker groups."
            return "先看各切片的销售额对比图，确认谁强谁弱。"
        if "sns.heatmap" in code_cell:
            if english:
                return "Use the heatmap to locate where high and low values cluster."
            return "再看热力图，识别高值和低值集中在哪些切片上。"
        if "weak_segment_region" in code_cell or "top_segment_region" in code_cell or "weak_segments" in code_cell:
            if english:
                return "List the key slices so follow-up review can target concrete records."
            return "最后列出关键切片明细，方便定位后续重点复盘对象。"

    if section_id == "discount_and_profit":
        if "discount_profit =" in code_cell or "profit_overview =" in code_cell:
            if english:
                return "Prepare the discount and profit base table and confirm the comparable record scope."
            return "先准备折扣与利润分析底表，确认可用于比较的记录范围。"
        if "px.scatter" in code_cell:
            if english:
                return "Review the discount-profit scatter plot to see whether higher discounts coincide with weaker profit."
            return "先看折扣与利润散点图，判断高折扣是否伴随利润恶化。"
        if "discount_buckets =" in code_cell:
            if english:
                return "Compare average profit across discount tiers."
            return "再看折扣分桶汇总表，比较不同折扣区间的平均利润表现。"
        if "discount_cap_what_if" in code_cell:
            if english:
                return "Review the discount-tightening what-if estimate as a recovery scenario, not as a demand forecast."
            return "以下为折扣收紧情景估算，基于折扣回收金额计算，不代表真实需求、销量或客户行为变化。"
        if "loss_making_products" in code_cell or "loss_making_categories" in code_cell or "worst_profit_rows" in code_cell:
            if english:
                return "List high-risk products, categories, or records to locate loss sources."
            return "然后列出高风险商品、类目或记录，定位亏损来源。"
        if "sns.boxplot" in code_cell:
            if english:
                return "Use the boxplot to check whether profit risk concentrates in high-discount tiers."
            return "最后看不同折扣区间的利润分布箱线图，确认风险是否集中在高折扣区间。"

    if section_id == "forecast":
        if "forecast_df =" in code_cell:
            if english:
                return "Build the next 28-day baseline forecast table and confirm the forecast range."
            return "先生成未来 28 天的基线预测表，明确预测区间和基线水平。"
        if "Actual vs Forecast" in code_cell or "combined_plot_df" in code_cell:
            if english:
                return "Compare actual and forecast sales to clarify how the baseline should be used."
            return "再看实际与预测销售额对比图，判断这份基线预测更适合拿来做什么。"
    return None


def render_section_cells(
    section: NotebookSection,
    module_lookup: dict[str, object],
    narrative_lookup: dict[str, NotebookSectionNarrative],
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    plan: AnalysisPlan,
    modeling_interpretation: dict[str, str] | None = None,
    modeling_outcome: dict[str, object] | None = None,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    output_language: str | None = None,
) -> list:
    section_id = section.section_id
    section_narrative = narrative_lookup.get(section_id)
    narrative_blocks: list[str] = []
    if section_narrative:
        narrative_blocks.append(section_narrative.intro)
        if section_narrative.key_observations:
            narrative_blocks.append(
                "### Key Observations\n\n"
                + "\n".join(f"- {item}" for item in section_narrative.key_observations)
            )
        if section_narrative.business_takeaway:
            narrative_blocks.append(
                "### Business Takeaway\n\n" + section_narrative.business_takeaway
            )
        if section_narrative.followup_question:
            narrative_blocks.append(
                "### Follow-up Question\n\n" + section_narrative.followup_question
            )
    else:
        narrative_blocks.append(section.purpose)

    if section_id == "title_and_goal":
        return [
            new_markdown_cell(
                f"## {section.title}\n\n"
                + "\n\n".join(narrative_blocks)
            )
        ]

    if section_id == "dataset_and_schema":
        return [
            new_markdown_cell(f"## {section.title}\n\n" + "\n\n".join(narrative_blocks)),
            new_markdown_cell(
                "### Schema Mapping\n\n```json\n"
                + schema_mapping.model_dump_json(indent=2)
                + "\n```"
            ),
            new_markdown_cell(
                "### Analysis Plan\n\n```json\n" + plan.model_dump_json(indent=2) + "\n```"
            ),
            new_code_cell("import pandas as pd\ndf = pd.read_csv('raw.csv')\ndf.head()"),
        ]

    if section_id == "data_cleaning":
        quality_module = module_lookup.get("data_quality_check")
        findings = quality_module.findings if quality_module else ["No data quality module output available."]
        return [
            new_markdown_cell(
                f"## {section.title}\n\n"
                + "\n\n".join(narrative_blocks)
                + "\n\n### Findings\n\n"
                + "\n".join(f"- {item}" for item in findings)
            )
        ]

    section_to_module = {
        "sales_trends": "sales_trend_analysis",
        "product_and_category": "product_contribution_analysis",
        "segment_and_region": "dimension_breakdown_analysis",
        "forecast": "forecast_analysis",
    }
    if section_id in section_to_module:
        module = module_lookup.get(section_to_module[section_id])
        if module is None:
            return []
        return [
            new_markdown_cell(f"## {section.title}\n\n" + "\n\n".join(narrative_blocks)),
            new_markdown_cell("### Findings\n\n" + "\n".join(f"- {item}" for item in module.findings)),
            new_code_cell(
                "# Structured module output\n"
                f"module_output = {module.model_dump()!r}\n"
                "module_output"
            ),
        ]

    if section_id == "discount_and_profit":
        return [
            new_markdown_cell(
                f"## {section.title}\n\n"
                + "\n\n".join(narrative_blocks)
                + "\n\n"
                "This section should explain how discounting interacts with profitability "
                "once the dedicated analysis module is implemented."
            )
        ]

    if section_id == "modeling":
        return modeling_renderer.render_modeling_section_cells(
            module_lookup.get("loss_risk_modeling"),
            modeling_interpretation=modeling_interpretation,
            modeling_outcome=modeling_outcome,
            forecast_module=module_lookup.get("forecast_analysis"),
            modeling_outcome_interpretation=modeling_outcome_interpretation,
            output_language=output_language,
        )

    if section_id == "conclusions":
        conclusion = narrative_conclusion_markdown(section, section_narrative, schema_mapping)
        if conclusion:
            return [new_markdown_cell(conclusion)]
        return [
            new_markdown_cell(
                f"## {section.title}\n\n"
                + "\n\n".join(narrative_blocks)
                + "\n\n"
                + "\n".join(f"- {item}" for item in report.summary)
            )
        ]

    return []


_section_intro_markdown = section_intro_markdown
_section_conclusion_markdown = section_conclusion_markdown
_is_template_chart_analysis_block = is_template_chart_analysis_block
_is_chart_selection_basis_block = is_chart_selection_basis_block
_is_section_conclusion_block = is_section_conclusion_block
_is_business_takeaway_block = is_business_takeaway_block
_split_section_markdown = split_section_markdown
_section_has_llm_chart_decision = section_has_llm_chart_decision
_visible_final_markdown_blocks = visible_final_markdown_blocks
_prepare_final_section_markdown_blocks = prepare_final_section_markdown_blocks
_code_step_intro = code_step_intro
_render_section_cells = render_section_cells
