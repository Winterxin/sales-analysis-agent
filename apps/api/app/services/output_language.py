from __future__ import annotations

from typing import Any
import re

DEFAULT_OUTPUT_LANGUAGE = "en"
SUPPORTED_OUTPUT_LANGUAGES = {"en", "zh-CN"}

ENGLISH_USER_FACING_INSTRUCTION = (
    "Write all user-facing analysis, conclusions, chart explanations, and recommendations in English. "
    "Do not include Chinese narrative in artifact-facing sections. "
    "Keep raw dataset column names unchanged."
)


def normalize_output_language(value: str | None) -> str:
    language = (value or DEFAULT_OUTPUT_LANGUAGE).strip()
    if language in {"zh", "zh_cn", "zh-CN", "zh_CN"}:
        return "zh-CN"
    if language.lower() in {"en", "en-us", "english"}:
        return "en"
    return DEFAULT_OUTPUT_LANGUAGE


def is_english_output(output_language: str | None) -> bool:
    return normalize_output_language(output_language) == "en"


def user_facing_language_instruction(output_language: str | None) -> str:
    if is_english_output(output_language):
        return ENGLISH_USER_FACING_INSTRUCTION
    return "面向用户的分析、结论、图表解释和建议使用中文输出；保留原始数据字段名。"


ENGLISH_SECTION_TITLES = {
    "title_and_goal": ("Analysis Goal", "Clarify the business questions, analysis objective, and intended use of this notebook."),
    "dataset_and_schema": ("Dataset and Field Mapping", "Explain the dataset shape, source fields, and standardized business field mapping."),
    "data_cleaning": ("Data Cleaning and Preparation", "Show type fixes, missing values, duplicates, and preprocessing choices."),
    "metric_distributions": ("Metric Distributions and Outliers", "Profile sales, quantity, discount, profit, and other numeric fields."),
    "sales_trends": ("Sales Trend Analysis", "Analyze sales and quantity trends over time and identify peaks, troughs, and volatility."),
    "product_and_category": ("Product and Category Analysis", "Identify leading products, long-tail items, category structure, and concentration."),
    "segment_and_region": ("Segment and Region Analysis", "Break performance down by customer and geography dimensions."),
    "country_market": ("Country Market Analysis", "Compare country-level sales, orders, order value, customers, and trends."),
    "order_structure": ("Order Structure Analysis", "Analyze invoice value, basket size, repeat behavior, deal size, order status, and price-quantity patterns."),
    "discount_and_profit": ("Discount and Profit Analysis", "Assess discount, sales, and profit relationships and identify margin erosion risks."),
    "modeling": ("Modeling Analysis", "Use modeling output as a review-priority aid and explain its limits."),
    "forecast": ("Forecast and Outlook", "Provide a short-term baseline forecast and explain assumptions and limits."),
    "conclusions": ("Conclusions and Recommended Actions", "Summarize key findings, risks, and next actions."),
}


COMMON_ENGLISH_TEXT_REPLACEMENTS = {
    "销售数据分析 Notebook": "Sales Data Analysis Notebook",
    "销售经营复盘报告": "Business Review",
    "客户经营分析简报": "Client-ready Sales Analysis Report",
    "客户报告": "Client Report",
    "Notebook 分析摘要": "Notebook Analysis Summary",
    "数据概览": "Data Overview",
    "数据质量": "Data Quality",
    "经营复盘": "Business Review",
    "关键发现": "Key Findings",
    "核心发现": "Key Findings",
    "主要结论": "Main Conclusions",
    "行动建议": "Recommendations",
    "建议行动": "Recommended Actions",
    "后续行动建议": "Recommended Next Actions",
    "建议": "Recommendations",
    "销售趋势": "Sales Trends",
    "本节结论": "Section Takeaway",
    "图表解读": "Chart Commentary",
    "图表分析": "Chart Analysis",
    "图表选择依据": "Chart Selection Rationale",
    "总销售额": "Total Sales",
    "总利润": "Total Profit",
    "负利润记录占比": "Negative Profit Record Rate",
    "最高风险折扣阈值": "Highest-Risk Discount Threshold",
    "样本量": "Sample Size",
    "字段数": "Field Count",
    "数据类型": "Dataset Type",
    "核心摘要": "Executive Summary",
    "主要经营观察": "Key Business Observation",
    "预测边界": "Forecast Boundary",
    "模型边界": "Model Boundary",
    "模型复核": "Model Review",
    "亏损风险模型": "loss-risk model",
    "销售额预测建模": "sales forecast modeling",
    "结论与行动建议": "Conclusions and Recommendations",
    "结论": "Conclusion",
    "预测与展望": "Forecast and Outlook",
    "分析目标": "Analysis Goal",
    "数据集与字段说明": "Dataset and Field Mapping",
    "数据清洗与预处理": "Data Cleaning and Preparation",
    "指标分布与异常值分析": "Metric Distributions and Outliers",
    "商品与类目分析": "Product and Category Analysis",
    "客群与区域分析": "Segment and Region Analysis",
    "国家市场分析": "Country Market Analysis",
    "订单结构分析": "Order Structure Analysis",
    "折扣与利润分析": "Discount and Profit Analysis",
    "建模分析：亏损风险识别": "Modeling Analysis: Loss Risk Review",
    "建模分析": "Modeling Analysis",
    "目录": "Table of Contents",
    "业务含义": "Business Meaning",
    "业务结论": "Business Takeaway",
    "核心判断": "Key Judgment",
    "行动方向": "Action Direction",
    "使用边界": "Usage Limits",
    "依据：": "Evidence:",
    "暂无": "Not available",
    "记录数": "Record Count",
    "销售额": "Sales",
    "利润": "Profit",
    "折扣": "Discount",
    "数量": "Quantity",
    "月份": "Month",
    "国家": "Country",
    "类目": "Category",
    "产品线": "Product Line",
    "订单数": "Order Count",
    "客户数": "Customer Count",
    "平均订单金额": "Average Order Value",
    "交易规模": "Deal Size",
    "订单状态": "Order Status",
    "真实类别": "Actual Class",
    "预测类别": "Predicted Class",
    "指标值": "Metric Value",
    "复核量占比": "Review Load Rate",
    "召回率": "Recall",
    "精确率": "Precision",
}


def contains_cjk(value: object) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(value or "")))


def englishize_common_artifact_text(value: object) -> str:
    text = str(value or "")
    for source, replacement in COMMON_ENGLISH_TEXT_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text


NOTEBOOK_ENGLISH_TEXT_REPLACEMENTS = {
    "销售数据分析 Notebook": "Sales Data Analysis Notebook",
    "数据概览": "Data Overview",
    "数据质量": "Data Quality",
    "销售趋势分析": "Sales Trend Analysis",
    "销售趋势": "Sales Trends",
    "商品与类目分析": "Product and Category Analysis",
    "商品与类目": "Product and Category",
    "客群与区域分析": "Segment and Region Analysis",
    "客群与区域": "Segment and Region",
    "折扣与利润分析": "Discount and Profit Analysis",
    "折扣与利润": "Discount and Profit",
    "订单结构分析": "Order Structure Analysis",
    "图表解读": "Chart Commentary",
    "本节结论": "Section Takeaway",
    "建模分析：亏损风险识别": "Modeling Analysis: Loss Risk Review",
    "建模分析：亏损风险探索": "Modeling Analysis: Loss Risk Exploration",
    "建模分析：建模可行性判断": "Modeling Analysis: Feasibility Review",
    "建模分析：销售额预测 baseline": "Modeling Analysis: Sales Forecast Baseline",
    "建模分析：销售额预测回归": "Modeling Analysis: Sales Regression",
    "建模分析：销售额预测尝试": "Modeling Analysis: Sales Forecast Attempt",
    "建模分析：销售额预测": "Modeling Analysis: Sales Forecast",
    "建模分析": "Modeling Analysis",
    "建模可行性判断": "Feasibility Review",
    "当前建模状态": "Current Modeling Status",
    "判断结论": "Review Outcome",
    "判断Conclusion": "Review Outcome",
    "判断依据": "Review Evidence",
    "使用限制": "Usage Limits",
    "建模目标、数据条件与特征工程": "Modeling Goal, Data Conditions, and Feature Engineering",
    "建模目标、特征工程与时间切分": "Modeling Goal, Feature Engineering, and Time Split",
    "建模目标与数据条件": "Modeling Goal and Data Conditions",
    "建模目标": "Modeling Goal",
    "模型表现与阈值取舍": "Model Performance and Threshold Trade-offs",
    "模型表现与选择": "Model Performance and Selection",
    "模型表现": "Model Performance",
    "混淆矩阵分析与复核样例": "Confusion Matrix and Review Samples",
    "混淆矩阵分析": "Confusion Matrix",
    "风险信号解释": "Risk Signal Interpretation",
    "建模综合Conclusion": "Modeling Summary",
    "建模综合结论": "Modeling Summary",
    "建模分析小结": "Modeling Summary",
    "建模结论与限制": "Modeling Summary and Limits",
    "高风险样例（Top 3）": "High-risk Examples Top 3",
    "高风险样例 Top 3": "High-risk Examples Top 3",
    "漏判亏损样例 Top 3": "False Negative Loss Examples Top 3",
    "误报亏损数": "False positives",
    "漏判亏损数": "False negatives",
    "亏损复核召回": "Loss review recall",
    "行动建议": "Recommended Actions",
    "建议行动": "Recommended Actions",
    "使用边界": "Usage Limits",
    "后续应": "Next,",
    "建议优先": "Prioritize",
    "亏损率": "Loss Rate",
    "利润率": "Profit Margin",
    "平均Profit": "Average Profit",
    "平均利润": "Average Profit",
    "Discount区间": "Discount Tier",
    "折扣区间": "Discount Tier",
    "影响记录": "Affected Records",
    "利润分布概览": "Profit Distribution Overview",
    "各折扣区间利润质量：平均利润与亏损率": "Profit Quality by Discount Tier: Average Profit and Loss Rate",
    "各折扣区间利润质量": "Profit Quality by Discount Tier",
    "各类目利润率质量对比": "Category Profit Margin Comparison",
    "头部商品Profit缺口": "Top Product Profit Gap",
    "头部商品利润缺口": "Top Product Profit Gap",
    "高销售低Profit客群区域切片": "Low-Margin Segment and Region Slice",
    "高销售低利润客群区域切片": "Low-Margin Segment and Region Slice",
    "区域销售与Profit对比": "Sales and Profit by Region",
    "区域销售与利润对比": "Sales and Profit by Region",
    "订单篮子大小分布": "Basket Size Distribution",
    "发票金额分布": "Invoice Value Distribution",
    "Discount与Profit散点关系": "Discount vs. Profit Relationship",
    "折扣与利润散点关系": "Discount vs. Profit Relationship",
    "高折扣低利润商品": "High-Discount Low-Profit Products",
    "类目 x 折扣区间亏损率热力图": "Category x Discount Tier Loss Rate Heatmap",
    "折扣区间利润率分布": "Profit Distribution by Discount Tier",
    "实际与预测销售额对比": "Actual vs Forecast Sales",
    "Actual vs Best Model：销售额回测对比": "Actual vs Best Model Sales Backtest",
    "销售额回归特征重要性 Top 8": "Sales Regression Feature Importance Top 8",
    "销售趋势与滚动均线": "Monthly Sales Trend with Rolling Average",
    "Sales Trends与滚动均线": "Monthly Sales Trend with Rolling Average",
    "滚动波动": "Rolling Volatility",
    "双维切片用于确认Sales是否集中在特定Segment and Region组合。": "Two-dimensional slices test whether sales are concentrated in specific segment and region combinations.",
    "本节关注Segment和Region切片之间的强弱差异。": "This section compares stronger and weaker segment and region slices.",
    "缺少模块结果时，只保留结构拆解方向，不把某个切片写成事实Conclusion.": "When module outputs are unavailable, this section keeps the review structural and does not state any slice as a confirmed fact.",
    "后续如补齐Profit字段，应补齐Sales、Profit和Profit Margin对照，再定位Sales强但Profit弱的组合。": "After profit becomes available, compare sales, profit, and profit margin before identifying high-sales low-profit combinations.",
    "头部商品累计贡献 Pareto 分布": "Top Product Cumulative Contribution Pareto",
    "高销售低利润商品清单": "High-Sales Low-Profit Product List",
    "客群/区域销售额对比": "Sales by Segment and Region",
    "头部商品销售额对比": "Top Product Sales Comparison",
    "不同切片销售额对比": "Sales by Slice",
    "单维切片销售额对比": "Sales by Dimension",
    "订单数对比": "Order Count Comparison",
    "销售额对比": "Sales Comparison",
    "销售额占比": "Sales Share",
    "订单行数对比": "Order Line Count Comparison",
    "平均订单金额对比": "Average Order Value Comparison",
    "订单金额分布": "Order Value Distribution",
    "单价与数量关系": "Unit Price vs Quantity",
    "订单篮子大小为长尾分布": "Basket size has a long-tail distribution",
    "发票金额为长尾分布": "Invoice value has a long-tail distribution",
    "截尾记录数": "clipped record count",
    "高风险折扣区间没有可收紧到目标阈值以上的记录，已跳过 what-if 情景估算。": "No high-risk discount records exceed the target threshold, so the what-if estimate was skipped.",
    "这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。": "This what-if estimate is based on discount recovery only. It does not represent real demand, volume, or customer behavior changes.",
    "数据不足：缺少可用于Top Product Profit Gap图的商品、Sales或Profit记录。": "Insufficient data: product, sales, or profit records are missing for the top product profit-gap chart.",
    "数据不足：头部商品没有有效Sales，跳过Profit缺口图。": "Insufficient data: top products do not have valid sales, so the profit-gap chart was skipped.",
    "未识别到高风险Discount Tier，已跳过Discount收紧 what-if 情景估算。": "No high-risk discount tier was identified, so the discount-tightening what-if estimate was skipped.",
    "高风险Discount Tier没有可收紧到目标阈值以上的记录，已跳过 what-if 情景估算。": "No high-risk discount tier records exceed the target threshold, so the what-if estimate was skipped.",
    "高风险Discount收紧情景": "High-risk discount tightening scenario",
    "Profit率": "Profit Margin",
    "策略层跳过折扣区间利润质量图": "The strategy layer skipped the discount tier profit quality chart.",
    "策略层跳过类目折扣风险热力图": "The strategy layer skipped the category discount risk heatmap.",
    "策略层跳过折扣区间箱线图": "The strategy layer skipped the discount tier boxplot.",
    "暂无阈值分析数据，跳过阈值取舍图。": "No threshold analysis data is available, so the threshold trade-off chart was skipped.",
    "暂无混淆矩阵数据，跳过混淆矩阵图。": "No confusion matrix data is available, so the confusion matrix chart was skipped.",
    "暂无特征组重要性数据，跳过特征组重要性图。": "No feature-group importance data is available, so the feature-group importance chart was skipped.",
    "暂无 TimeSeriesSplit CV 汇总，跳过 CV 稳定性对比图。": "No TimeSeriesSplit CV summary is available, so the CV stability chart was skipped.",
    "当前 best model 是 baseline，跳过特征重要性图。": "The current best model is the baseline, so the feature importance chart was skipped.",
    "暂无 baseline 回测数据，跳过实际值与基线预测图。": "No baseline backtest data is available, so the actual-vs-baseline chart was skipped.",
    "暂无销售额回归回测数据，跳过实际值与预测值对比图。": "No sales regression backtest data is available, so the actual-vs-predicted chart was skipped.",
    "暂无销售额回归特征重要性，跳过特征重要性图。": "No sales regression feature importance is available, so the feature importance chart was skipped.",
    "高风险样例": "High-risk Examples",
    "漏判亏损样例": "False Negative Loss Examples",
    "设为Discount审批红线": "Set this as a discount approval review threshold.",
    "不代表真实需求": "does not represent real demand",
    "销量或客户行为变化": "volume, or customer behavior changes",
    "不代表因果关系": "does not imply causality",
    "不用于自动决策": "must not be used for automatic decisions",
    "预测类别": "Predicted Class",
    "真实类别": "Actual Class",
    "指标值": "Metric Value",
    "折扣": "Discount",
    "利润": "Profit",
    "销售额": "Sales",
    "订单数": "Order Count",
    "交易规模": "Deal Size",
    "订单状态": "Order Status",
    "类目": "Category",
    "客群": "Segment",
    "区域": "Region",
    "平均订单金额": "Average Order Value",
    "Section Takeaway：### Section Takeaway": "### Section Takeaway",
    "noting any unc。": "noting any uncertainty.",
    "revenue dy。": "revenue dynamics.",
    "guiding the focus of dee。": "guiding the focus of deeper review.",
    "assessing concentration and diversification w。": "assessing concentration and diversification.",
}


def englishize_notebook_text(value: object) -> str:
    text = englishize_common_artifact_text(value)
    for source, replacement in NOTEBOOK_ENGLISH_TEXT_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = text.replace("：### Section Takeaway", "\n\n### Section Takeaway")
    text = text.replace("Section Takeaway：", "Section Takeaway:")
    text = text.replace("Modeling Analysis：", "Modeling Analysis: ")
    text = text.replace("Conclusion：", "Conclusion: ")
    text = re.sub(r"(#{1,6}\s+[^\n。]*)。", r"\1", text)
    text = re.sub(r"([A-Za-z])。", r"\1.", text)
    return text


def safe_english_sentence(
    value: object,
    *,
    max_words: int = 32,
    fallback: str = "",
    require_complete_sentence: bool = False,
) -> str:
    text = englishize_notebook_text(value).strip()
    text = re.sub(r"[\u3002\uff0c\uff1a\uff1b]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or contains_cjk(text):
        text = fallback.strip()
    if not text:
        return ""
    if require_complete_sentence:
        for sentence in _split_complete_english_sentences(text):
            sentence = sentence.strip(" ,;:")
            if sentence and not contains_cjk(sentence) and not _has_hanging_tail_sentence(sentence):
                return sentence if sentence.endswith((".", "!", "?")) else f"{sentence}."
        fallback_text = fallback.strip()
        if fallback_text and fallback_text != text:
            return safe_english_sentence(fallback_text, fallback="", require_complete_sentence=True)
        return ""
    sentence_match = re.search(r"^(.+?[.!?])(?:\s|$)", text)
    if sentence_match:
        text = sentence_match.group(1).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(" ,;:")
    text = text.rstrip(" ,;:.!?")
    return f"{text}." if text else ""


_HANGING_TAIL_WORDS = {
    "while",
    "and",
    "or",
    "with",
    "to",
    "of",
    "for",
    "in",
    "at",
    "by",
    "from",
    "the",
    "a",
    "an",
}


_COMPLETE_SENTENCE_ABBREVIATIONS = (
    "vs.",
    "e.g.",
    "i.e.",
    "etc.",
    "mr.",
    "ms.",
    "dr.",
)


def _is_complete_sentence_abbreviation_period(value: str, index: int) -> bool:
    prefix = value[: index + 1].lower()
    return any(prefix.endswith(abbreviation) for abbreviation in _COMPLETE_SENTENCE_ABBREVIATIONS)


def _split_complete_english_sentences(value: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    length = len(value)
    for index, char in enumerate(value):
        if char not in ".!?":
            continue
        prev_char = value[index - 1] if index > 0 else ""
        next_char = value[index + 1] if index + 1 < length else ""
        if char == "." and prev_char.isdigit() and next_char.isdigit():
            continue
        if char == "." and _is_complete_sentence_abbreviation_period(value, index):
            continue
        if index + 1 < length and next_char and not next_char.isspace():
            continue
        candidate = value[start : index + 1].strip()
        if candidate:
            sentences.append(candidate)
        start = index + 1
    return sentences


def _has_hanging_tail_sentence(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    tail = re.sub(r"[^A-Za-z]+$", "", text).lower()
    return tail in _HANGING_TAIL_WORDS


def safe_complete_english_sentence(
    value: object,
    *,
    max_words: int = 32,
    fallback: str = "",
) -> str:
    return safe_english_sentence(
        value,
        max_words=max_words,
        fallback=fallback,
        require_complete_sentence=True,
    )


def safe_english_heading(value: object, *, fallback: str = "Business observation") -> str:
    text = englishize_notebook_text(value).strip()
    text = re.sub(r"[\u3002\uff0c\uff1a\uff1b]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;-")
    if not text or contains_cjk(text):
        text = fallback
    words = text.split()
    if len(words) > 14 or len(words) < 2:
        text = fallback
    return text.strip(" .,!?:;-") or fallback


def english_safe_bullets(
    items: list[object] | tuple[object, ...] | None,
    fallback: list[str],
    *,
    limit: int = 5,
) -> list[str]:
    bullets: list[str] = []
    for item in items or []:
        text = englishize_common_artifact_text(item).strip()
        if not text or contains_cjk(text):
            continue
        bullets.append(text)
        if len(bullets) >= limit:
            break
    return bullets or fallback[:limit]


def englishize_payload_strings(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: englishize_payload_strings(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [englishize_payload_strings(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(englishize_payload_strings(item) for item in payload)
    if isinstance(payload, str):
        return englishize_common_artifact_text(payload)
    return payload
