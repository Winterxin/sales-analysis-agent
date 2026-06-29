from __future__ import annotations

import json
import re
from nbformat.v4 import new_code_cell, new_markdown_cell

from app.services.output_language import englishize_notebook_text, is_english_output


def _markdown_table(rows: list[dict[str, object]], columns: list[str] | None = None, limit: int = 10) -> str:
    if not rows:
        return "暂无可展示记录。"
    selected_columns = columns or list(rows[0].keys())
    header = "| " + " | ".join(selected_columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected_columns) + " |"
    body = []
    for row in rows[:limit]:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in selected_columns) + " |")
    return "\n".join([header, separator, *body])


def _compact_table(rows: list[dict[str, object]], columns: list[str] | None = None, limit: int = 10) -> str:
    return _markdown_table(rows, columns=columns, limit=limit)


def _metric_value(summary_metrics: dict[str, object], key: str, default: object = "unknown") -> object:
    value = summary_metrics.get(key)
    return default if value is None else value


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _format_metric(value: object, digits: int = 4) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return str(value)
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def _format_percent(value: object, digits: int = 2) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return str(value)
    if abs(numeric) <= 1:
        numeric *= 100
    return f"{numeric:.{digits}f}%"


def _selected_columns_table(rows: list[dict[str, object]], columns: list[str], limit: int) -> str:
    if not rows:
        return "暂无可展示记录。"
    available_columns = [column for column in columns if any(column in row for row in rows)]
    return _compact_table(rows, columns=available_columns, limit=limit)


def _module_tables(module: object | None) -> dict[str, list[dict[str, object]]]:
    return getattr(module, "tables", {}) or {}


def _module_summary(module: object | None) -> dict[str, object]:
    return getattr(module, "summary_metrics", {}) or {}


def _optional_interpretation_block(
    modeling_interpretation: dict[str, str] | None,
    keys: list[str],
) -> str:
    if not modeling_interpretation:
        return ""
    values = [
        str(modeling_interpretation.get(key) or "").strip()
        for key in keys
        if str(modeling_interpretation.get(key) or "").strip()
    ]
    if not values:
        return ""
    return "\n\n### 分析师解释\n\n" + "\n\n".join(values)


def _split_display_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in str(text or "").strip():
        current.append(char)
        if char in "。！？.!?":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _forecast_interpretation_block(
    modeling_interpretation: dict[str, str] | None,
    keys: list[str],
    *,
    best_baseline: object = "",
) -> str:
    if not modeling_interpretation:
        return ""
    system_terms = ["状态为", "价值水平", "价值级别", "建模价值", "基线已评估", "弱基线"]
    values: list[str] = []
    for key in keys:
        raw_text = str(modeling_interpretation.get(key) or "").strip()
        if not raw_text:
            continue
        kept_sentences = [
            sentence
            for sentence in _split_display_sentences(raw_text)
            if not any(term in sentence for term in system_terms)
        ]
        cleaned = "".join(kept_sentences).strip()
        if best_baseline and "简单历史均值" in cleaned:
            cleaned = cleaned.replace("简单历史均值的基线模型", f"{best_baseline} baseline")
            cleaned = cleaned.replace("简单历史均值", str(best_baseline))
        cleaned = cleaned.replace("预测包装", "业务决策")
        if cleaned:
            values.append(cleaned)
    if not values:
        return ""
    return "\n\n### 分析师解释\n\n" + "\n\n".join(values)


def _modeling_feature_group_summary(tables: dict[str, list[dict[str, object]]]) -> str:
    groups = tables.get("feature_importance_grouped", [])
    if not groups:
        return "暂无可展示的特征组重要性。"
    top_groups = ", ".join(str(row.get("feature_group")) for row in groups[:3])
    return (
        _compact_table(groups, limit=8)
        + "\n\n这些字段是预测信号，不代表因果关系。"
        + f" 当前较突出的特征组是：{top_groups}；当折扣、子类目或类目等特征组重要性较高时，应回到折扣策略和商品结构明细复盘。"
    )


def _modeling_top_raw_features(tables: dict[str, list[dict[str, object]]]) -> str:
    features = []
    for row in tables.get("feature_importance", [])[:5]:
        if not isinstance(row, dict):
            continue
        feature = row.get("feature") or row.get("name")
        if feature:
            features.append(str(feature))
    return ", ".join(features) if features else "暂无可展示原始特征"


def _interpretation_value(
    modeling_interpretation: dict[str, str] | None,
    key: str,
    fallback: str,
    output_language: str | None = None,
) -> str:
    if not modeling_interpretation:
        return fallback
    value = str(modeling_interpretation.get(key, "")).strip()
    if value and output_language is not None:
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in value)
        if is_english_output(output_language) and has_cjk:
            return fallback
        if not is_english_output(output_language) and not has_cjk and re.search(r"[A-Za-z]{4,}", value):
            return fallback
    return value or fallback


def _modeling_business_warning(
    modeling_interpretation: dict[str, str] | None,
    fallback: str,
    output_language: str | None = None,
) -> str:
    warning = _interpretation_value(
        modeling_interpretation,
        "business_use_warning",
        fallback,
        output_language=output_language,
    )
    if "不代表因果关系" not in warning or "不用于自动决策" not in warning:
        warning = f"{warning} 不代表因果关系，不用于自动决策。"
    return warning


def _modeling_summary_fallback(
    summary_metrics: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
) -> str:
    groups = tables.get("feature_importance_grouped", [])
    top_groups = ", ".join(str(row.get("feature_group")) for row in groups[:3]) or "暂无稳定特征组"
    status = str(summary_metrics.get("model_quality_status") or "usable")
    weak_reasons = summary_metrics.get("weak_reasons") or []
    reason_text = "、".join(str(item) for item in weak_reasons) if isinstance(weak_reasons, list) else str(weak_reasons)
    if status == "weak":
        quality_sentence = (
            "当前模型质量状态为 weak，只适合作为探索性参考，不建议直接用于复核排序。"
            + (f"主要原因：{reason_text}。" if reason_text else "")
        )
        limitation_sentence = "该模型只适合作为探索性风险线索和人工复核参考，不建议直接用于复核排序，不代表因果关系，不用于自动决策。"
    else:
        quality_sentence = ""
        limitation_sentence = "模型只用于亏损风险预警和人工复核优先级排序，不代表因果关系，不用于自动决策。"
    return (
        f"本节以 is_loss = profit < 0 定义亏损风险，当前亏损样本占比为 {_metric_value(summary_metrics, 'target_positive_rate')}。"
        f"{quality_sentence}"
        f"最佳模型为 {_metric_value(summary_metrics, 'best_model')}，Recall={_metric_value(summary_metrics, 'best_recall')}，"
        f"F1={_metric_value(summary_metrics, 'best_f1')}，ROC AUC={_metric_value(summary_metrics, 'best_roc_auc')}；"
        f"默认阈值下 FP={_metric_value(summary_metrics, 'false_positive_count', 0)}、"
        f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)}。"
        f"主要预测信号组包括 {top_groups}。{limitation_sentence}"
    )


def _model_quality_notice(summary_metrics: dict[str, object]) -> str:
    status = str(summary_metrics.get("model_quality_status") or "usable")
    weak_reasons = summary_metrics.get("weak_reasons") or []
    reason_text = "、".join(str(item) for item in weak_reasons) if isinstance(weak_reasons, list) else str(weak_reasons)
    if status == "weak":
        return (
            "模型质量状态：weak。当前结果只适合作为探索性参考，不建议直接用于复核排序。"
            + (f" 主要原因：{reason_text}。" if reason_text else "")
        )
    if status == "strong":
        return "模型质量状态：strong。当前模型可作为人工复核优先级参考，但仍不代表因果关系，不用于自动决策。"
    return "模型质量状态：usable。当前模型可谨慎作为人工复核优先级参考，并需要结合业务规则复核。"


def _modeling_conclusion_bullets(
    summary_metrics: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
    findings: list[str],
    warnings: list[str],
) -> list[str]:
    groups = tables.get("feature_importance_grouped", [])
    top_groups = ", ".join(str(row.get("feature_group")) for row in groups[:3]) or "暂无稳定特征组"
    status = str(summary_metrics.get("model_quality_status") or "usable")
    quality_notice = _model_quality_notice(summary_metrics)
    usage_bullet = (
        "当前模型只适合作为探索性风险线索和人工复核参考，不建议直接用于复核排序。"
        if status == "weak"
        else "当前模型适合亏损风险预警和人工复核优先级排序，具体使用应结合阈值表和错误样本结构判断。"
    )
    bullets = [
        (
            f"当前最佳模型为 {_metric_value(summary_metrics, 'best_model')}，"
            f"best_recall={_metric_value(summary_metrics, 'best_recall')}。"
        ),
        "阈值选择需要在召回率和人工复核量之间取舍，不能只追求单一准确率。",
        f"主要风险特征组包括：{top_groups}，应结合业务明细复盘。",
        quality_notice,
        usage_bullet,
    ]
    if warnings:
        bullets.append("模型运行提示：" + "；".join(warnings[:3]))
    return bullets[:5]


def render_modeling_section_markdown(
    module: object | None,
    modeling_interpretation: dict[str, str] | None = None,
    modeling_opportunity_decision: dict[str, str] | None = None,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    if english and module is None:
        if modeling_opportunity_decision:
            decision_summary = str(modeling_opportunity_decision.get("decision_summary") or "").strip()
            notebook_message = str(modeling_opportunity_decision.get("notebook_message") or "").strip()
            risk_warning = str(modeling_opportunity_decision.get("risk_warning") or "").strip()
            return "\n\n".join(
                [
                    "## Modeling Analysis: Feasibility Review",
                    "### Review Outcome\n\n"
                    + (
                        englishize_notebook_text(notebook_message)
                        if notebook_message and not any("\u4e00" <= ch <= "\u9fff" for ch in notebook_message)
                        else "The current data does not provide a stable modeling target, so this run keeps descriptive analysis."
                    ),
                    "### Review Evidence\n\n"
                    + (
                        englishize_notebook_text(decision_summary)
                        if decision_summary and not any("\u4e00" <= ch <= "\u9fff" for ch in decision_summary)
                        else "Only the modeling opportunity is recorded; no new model is trained."
                    ),
                    "### Usage Limits\n\n"
                    + (
                        englishize_notebook_text(risk_warning)
                        if risk_warning and not any("\u4e00" <= ch <= "\u9fff" for ch in risk_warning)
                        else "No new model is trained, and the result must not be used for automatic decisions."
                    ),
                ]
            )
        return "\n\n".join(
            [
                "## Modeling Analysis: Loss Risk Review",
                "### Modeling Goal\n\nThe report planned to attempt loss-risk modeling, but no `loss_risk_modeling` module output was available.",
                "### Target Definition\n\n`is_loss = profit < 0`",
                "### Model Limits\n\nThe current data is not suitable for loss-risk modeling because the modeling module output is missing.",
                "This model output must not be used as a causal claim or an automatic decision rule.",
            ]
        )
    disclaimer = "该模型用于亏损风险预警和人工复核优先级排序，不代表特征与亏损之间存在因果关系，不用于自动决策。简言之，不代表因果关系，不用于自动决策。"
    weak_disclaimer = "该模型只适合作为探索性风险线索和人工复核参考，不建议直接用于复核排序，不代表因果关系，不用于自动决策。"
    skipped_disclaimer = "当前未生成可用模型，因此不用于复核排序，不代表因果关系，不用于自动决策。"
    if module is None:
        if modeling_opportunity_decision:
            decision_summary = str(modeling_opportunity_decision.get("decision_summary") or "").strip()
            notebook_message = str(modeling_opportunity_decision.get("notebook_message") or "").strip()
            risk_warning = str(modeling_opportunity_decision.get("risk_warning") or "").strip()
            return "\n\n".join(
                [
                    "## 建模分析：建模可行性判断",
                    "### 判断结论\n\n" + (notebook_message or "当前数据未形成稳定建模目标，本轮保留描述性分析。"),
                    "### 判断依据\n\n" + (decision_summary or "当前仅记录建模机会，不训练新模型。"),
                    "### 使用限制\n\n" + (risk_warning or "G6-2 不训练新模型，不用于自动决策。"),
                ]
            )
        return "\n\n".join(
            [
                "## 建模分析：亏损风险识别",
                "### 建模目标\n\n当前报告计划尝试亏损风险建模，但没有找到 `loss_risk_modeling` 模块输出。",
                "### Target 定义\n\n`is_loss = profit < 0`",
                "### 模型限制\n\n当前数据暂不适合进行亏损风险建模，原因是建模模块结果缺失。",
                disclaimer,
            ]
        )

    warnings = list(getattr(module, "warnings", []) or [])
    tables = getattr(module, "tables", {}) or {}
    summary_metrics = getattr(module, "summary_metrics", {}) or {}
    findings = list(getattr(module, "findings", []) or [])
    skipped = bool(warnings) and not tables.get("model_comparison")
    model_choice_fallback = (
        "当前最佳模型按 recall > f1 > roc_auc 选择。对于亏损风险预警，recall 更重要，"
        "因为漏判亏损订单的成本通常高于多复核一部分正常订单。"
    )
    threshold_fallback = (
        f"默认阈值 {_metric_value(summary_metrics, 'threshold_default', 0.5)} 下："
        f"TP={_metric_value(summary_metrics, 'true_positive_count', 0)}、"
        f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)}、"
        f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)}、"
        f"TN={_metric_value(summary_metrics, 'true_negative_count', 0)}。"
        "降低阈值通常提高 recall，但会增加复核量；提高阈值会减少复核量，但可能漏掉更多亏损订单。"
    )
    feature_fallback = (
        "这些字段是预测信号，不代表因果关系；当折扣、子类目或类目等特征组重要性较高时，"
        "应回到折扣策略和商品结构明细复盘。"
    )
    error_fallback = (
        "高风险样例用于人工复核，不是直接执行依据。False Negative 是被漏判的亏损样本，"
        "比 False Positive 更需要优先复盘；False Positive 主要代表额外复核成本。"
    )
    review_guidance_fallback = (
        "建议将模型输出作为人工复核优先级，优先查看高风险和漏判亏损样本，并结合折扣、类目和订单明细判断是否需要调整业务规则。"
    )
    model_quality_status = str(summary_metrics.get("model_quality_status") or "usable")
    if model_quality_status == "weak":
        review_guidance_fallback = (
            "建议先补充正例样本或调整建模口径；当前结果只适合探索性参考，不建议直接用于复核排序。"
        )

    blocks = [
        "## 建模分析：亏损风险识别",
    ]

    if skipped:
        if english:
            reason_text = "\n".join(f"- {item}" for item in warnings) or "- No usable modeling output."
            if modeling_opportunity_decision:
                decision_summary = str(modeling_opportunity_decision.get("decision_summary") or "").strip()
                notebook_message = str(modeling_opportunity_decision.get("notebook_message") or "").strip()
                risk_warning = str(modeling_opportunity_decision.get("risk_warning") or "").strip()
                return "\n\n".join(
                    [
                        "## Modeling Analysis: Feasibility Review",
                        "### Current Modeling Status\n\nThe current data is not suitable for loss-risk modeling.\n\nSkipped reasons:\n" + reason_text,
                        "### Follow-up Modeling Opportunity\n\n"
                        + (
                            englishize_notebook_text(notebook_message)
                            if notebook_message and not any("\u4e00" <= ch <= "\u9fff" for ch in notebook_message)
                            else "Only the modeling opportunity is recorded; no new model is trained."
                        ),
                        "### Review Evidence\n\n"
                        + (
                            englishize_notebook_text(decision_summary)
                            if decision_summary and not any("\u4e00" <= ch <= "\u9fff" for ch in decision_summary)
                            else "Only the modeling opportunity is recorded; no new model is trained."
                        ),
                        "### Usage Limits\n\n"
                        + (
                            englishize_notebook_text(risk_warning)
                            if risk_warning and not any("\u4e00" <= ch <= "\u9fff" for ch in risk_warning)
                            else "No new model is trained, and the result must not be used for automatic decisions."
                        ),
                    ]
                )
            return "\n\n".join(
                [
                    "## Modeling Analysis: Loss Risk Review",
                    "### Modeling Status\n\nThe current data is not suitable for loss-risk modeling.",
                    "### Skipped Reasons\n\n" + reason_text,
                    "### Model Limits\n\nNo usable model is generated, so this section must not be used for review ranking, causal claims, or automatic decisions.",
                ]
            )
        if modeling_opportunity_decision:
            decision_summary = str(modeling_opportunity_decision.get("decision_summary") or "").strip()
            notebook_message = str(modeling_opportunity_decision.get("notebook_message") or "").strip()
            risk_warning = str(modeling_opportunity_decision.get("risk_warning") or "").strip()
            return "\n\n".join(
                [
                    "## 建模分析：建模可行性判断",
                    "### 当前建模状态\n\n当前数据暂不适合进行亏损风险建模。\n\n跳过原因：\n" + "\n".join(f"- {item}" for item in warnings),
                    "### 后续建模机会\n\n" + (notebook_message or "当前仅记录建模机会，不训练新模型。"),
                    "### 判断依据\n\n" + (decision_summary or "当前仅记录建模机会，不训练新模型。"),
                    "### 使用限制\n\n" + (risk_warning or "G6-2 不训练新模型，不用于自动决策。"),
                ]
            )
        blocks.extend(
            [
                "### 建模状态\n\n当前数据暂不适合进行亏损风险建模。",
                "### 跳过原因\n\n" + "\n".join(f"- {item}" for item in warnings),
                "### 模型限制\n\n" + skipped_disclaimer,
            ]
        )
        return "\n\n".join(blocks)

    modeling_summary = _interpretation_value(
        modeling_interpretation,
        "modeling_summary",
        _modeling_summary_fallback(summary_metrics, tables),
        output_language=output_language,
    )
    blocks.append("### 建模分析小结\n\n" + modeling_summary)

    summary_columns = [
        "target_positive_count",
        "target_negative_count",
        "target_positive_rate",
        "train_rows",
        "test_rows",
        "train_positive_count",
        "test_positive_count",
        "class_imbalance_level",
        "model_quality_status",
        "business_priority_metric",
        "threshold_default",
    ]
    blocks.append(
        "### 建模任务与验证设计\n\n"
        + (
            "模型目标：尝试识别更可能发生负利润的订单；当前模型质量较弱时，只作为探索性风险线索和人工复核参考。\n\n"
            if model_quality_status == "weak"
            else "模型目标：识别更可能发生负利润的订单，作为亏损风险预警和人工复核优先级排序依据；模型输出只作为人工复核参考，不作为直接执行依据。\n\n"
        )
        + "Target：`is_loss = profit < 0`\n\n"
        + _compact_table([dict(summary_metrics)], columns=summary_columns, limit=1)
        + "\n\n特征工程说明：\n"
        "- 用于训练的字段：折扣、销售额、数量、单价等数值特征，以及类目、区域、客群等低基数类别特征。\n"
        "- 排除 profit、is_loss、profit_margin、loss_flag、negative_profit 等泄露字段。\n"
        "- product_name、customer_id、order_id 不进入训练，只用于样例定位。\n"
        "- 日期字段拆成 year/month/quarter/weekday 作为辅助信号。\n\n"
        "安全特征样例：\n"
        + _compact_table(tables.get("safe_features", []), limit=8)
        + "\n\n被排除字段：\n"
        + _compact_table(tables.get("blocked_features", []), limit=8)
        + "\n\n"
        + _model_quality_notice(summary_metrics)
    )
    blocks.append(
        "### 模型表现与选择\n\n"
        + _compact_table(tables.get("model_comparison", []))
        + "\n\n"
        + _interpretation_value(modeling_interpretation, "model_choice_takeaway", model_choice_fallback, output_language=output_language)
        + "\n\n"
        "选择依据："
        f"best_model={_metric_value(summary_metrics, 'best_model')}，"
        f"best_recall={_metric_value(summary_metrics, 'best_recall')}，"
        f"best_f1={_metric_value(summary_metrics, 'best_f1')}，"
        f"best_roc_auc={_metric_value(summary_metrics, 'best_roc_auc')}。"
    )
    blocks.append(
        "### 阈值取舍与混淆矩阵\n\n"
        + _compact_table(tables.get("threshold_analysis", []))
        + "\n\n"
        + _compact_table(tables.get("confusion_matrix", []))
        + "\n\nTP=正确识别亏损，FN=漏判亏损，FP=误报亏损，TN=正确识别非亏损。"
        + "\n\n"
        + _interpretation_value(modeling_interpretation, "threshold_takeaway", threshold_fallback, output_language=output_language)
    )
    blocks.append(
        "### 风险特征解释\n\n"
        + _compact_table(tables.get("feature_importance_grouped", []), limit=8)
        + "\n\n"
        + _interpretation_value(modeling_interpretation, "feature_takeaway", feature_fallback, output_language=output_language)
        + "\n\nTop 原始特征："
        + _modeling_top_raw_features(tables)
    )
    example_columns = [
        "row_index",
        "loss_probability",
        "actual_is_loss",
        "predicted_is_loss",
        "discount",
        "sales_amount",
        "category",
        "sub_category",
        "segment",
        "region",
    ]
    blocks.append(
        "### 复核样例摘要\n\n"
        f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)}，FN={_metric_value(summary_metrics, 'false_negative_count', 0)}。"
        + _interpretation_value(modeling_interpretation, "error_takeaway", error_fallback, output_language=output_language)
        + "\n\n"
        + _interpretation_value(modeling_interpretation, "review_guidance", review_guidance_fallback, output_language=output_language)
        + " 样例只用于人工复核，不作为直接执行依据。"
        + "\n\n"
        "#### 高风险样例（Top 3）\n\n"
        + _selected_columns_table(tables.get("high_risk_examples", []), example_columns, limit=3)
        + "\n\n#### False Negative 样例（Top 3）\n\n"
        + (
            _selected_columns_table(tables.get("false_negative_examples", []), example_columns, limit=3)
            if tables.get("false_negative_examples")
            else "暂无该类错误样本。"
        )
    )
    effective_disclaimer = weak_disclaimer if model_quality_status == "weak" else disclaimer
    business_warning = _modeling_business_warning(
        modeling_interpretation,
        effective_disclaimer,
        output_language=output_language,
    )
    if model_quality_status == "weak" and "不建议直接用于复核排序" not in business_warning:
        business_warning = weak_disclaimer
    review_guidance = _interpretation_value(
        modeling_interpretation,
        "review_guidance",
        review_guidance_fallback,
        output_language=output_language,
    )
    blocks.append(
        "### 建模结论与限制\n\n"
        + "\n".join(
            [
                "- "
                + (
                    "当前模型只适合作为探索性风险线索和人工复核参考，不建议直接用于复核排序。"
                    if model_quality_status == "weak"
                    else "当前模型适合亏损风险预警和人工复核优先级排序，具体使用应结合阈值表和错误样本结构判断。"
                ),
                f"- {review_guidance}",
                f"- {business_warning}",
            ]
        )
    )
    return "\n\n".join(blocks)


def _loss_risk_key_metrics_table(summary_metrics: dict[str, object]) -> str:
    columns = [
        "model_quality_status",
        "best_model",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_roc_auc",
        "target_positive_count",
        "target_positive_rate",
        "test_positive_count",
        "false_positive_count",
        "false_negative_count",
    ]
    return _compact_table([dict(summary_metrics)], columns=columns, limit=1)


def _feature_signal_table(tables: dict[str, list[dict[str, object]]], *, limit: int = 5) -> str:
    return _selected_columns_table(
        tables.get("feature_importance_grouped", []),
        ["feature_group", "total_importance", "top_features"],
        limit=limit,
    )


def _feature_names_from_table(rows: list[dict[str, object]], *, limit: int = 8) -> list[str]:
    names: list[str] = []
    for row in rows:
        name = str(row.get("feature") or row.get("field") or row.get("column") or "").strip()
        if not name or name in names:
            continue
        names.append(name)
        if len(names) >= limit:
            break
    return names


def _loss_risk_feature_engineering_block(
    summary_metrics: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
    output_language: str | None = None,
) -> str:
    safe_features = _feature_names_from_table(tables.get("safe_features", []), limit=8)
    blocked_features = _feature_names_from_table(tables.get("blocked_features", []), limit=6)
    english = is_english_output(output_language)
    safe_features_text = (
        ", ".join(safe_features)
        if english and safe_features
        else (
            "discount, sales, quantity, unit price, category, region, segment, and similar fields"
            if english
            else ("、".join(safe_features) if safe_features else "折扣、销售额、数量、单价、类目、区域、客群等字段")
        )
    )
    blocked_features_text = (
        ", ".join(blocked_features)
        if english and blocked_features
        else (
            "profit, is_loss, profit_margin, loss_flag, customer_id, order_id, product_name"
            if english
            else ("、".join(blocked_features) if blocked_features else "profit、is_loss、profit_margin、loss_flag、customer_id、order_id、product_name")
        )
    )
    if english:
        return (
            "## Modeling Analysis: Loss Risk Review\n\n"
            "### Modeling Goal, Data Conditions, and Feature Engineering\n\n"
            "The goal is to identify orders that are more likely to have negative profit, so they can be prioritized for manual review.\n\n"
            "Target: `is_loss = profit < 0`\n\n"
            "**Feature engineering and leakage guard**\n\n"
            "- Training fields include numeric features such as discount, sales, quantity, and unit price, plus low-cardinality categorical features such as category, region, and segment.\n"
            "- Leakage fields such as profit, is_loss, profit_margin, loss_flag, and negative_profit are excluded.\n"
            "- product_name, customer_id, and order_id are not used for training; they are retained only for locating examples.\n"
            "- Date fields are split into year, month, quarter, and weekday signals.\n\n"
            f"Main fields used: {safe_features_text}.\n\n"
            f"Excluded fields: {blocked_features_text}. These fields are excluded because they leak the target, are high-cardinality identifiers, or are useful only for example lookup.\n\n"
            "The code below rebuilds the target, features, models, and evaluation results from the current notebook data. You can edit `MODEL_CONFIG` to adjust model parameters and thresholds."
        )
    return (
        "## 建模分析：亏损风险识别\n\n"
        "### 建模目标、数据条件与特征工程\n\n"
        "目标是识别更可能发生负利润的订单，作为人工复核优先级参考。\n\n"
        "Target：`is_loss = profit < 0`\n\n"
        "**特征工程与防泄漏**\n\n"
        "- 用于训练的字段：折扣、销售额、数量、单价等数值特征，以及类目、区域、客群等低基数类别特征。\n"
        "- 排除 profit、is_loss、profit_margin、loss_flag、negative_profit 等泄漏字段。\n"
        "- product_name、customer_id、order_id 不进入训练，只用于样例定位。\n"
        "- 日期字段拆成 year/month/quarter/weekday 作为辅助信号。\n\n"
        f"主要使用字段：{safe_features_text}。\n\n"
        f"排除字段：{blocked_features_text}。这些字段主要因为目标泄漏、高基数标识或仅适合样例定位而不进入训练。\n\n"
        "下面的代码会从当前 notebook 数据构造 target、特征、模型和评估结果；可以直接修改 `MODEL_CONFIG` 中的模型参数和阈值后运行。"
    )


def _loss_risk_model_threshold_block(
    summary_metrics: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
    modeling_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    if is_english_output(output_language):
        return (
            "### Model Performance and Threshold Trade-offs\n\n"
            "The code below trains DummyClassifier, LogisticRegression, and RandomForestClassifier on the current notebook data, then uses the current best model to calculate recall, precision, F1, and review volume across thresholds."
        )
    return (
        "### 模型表现与阈值取舍\n\n"
        "下面基于 notebook 当前数据训练 DummyClassifier、LogisticRegression 和 RandomForestClassifier 输出模型对比，"
        "并用当前 best model 计算不同阈值下的 recall、precision、F1 和复核量。"
    )


def _loss_risk_threshold_explanation(
    summary_metrics: dict[str, object],
    modeling_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    fallback = (
        (
            f"The current best model is {_metric_value(summary_metrics, 'best_model', 'the current model')}, "
            f"with recall={_format_metric(_metric_value(summary_metrics, 'best_recall', 'unknown'))}, "
            f"precision={_format_metric(_metric_value(summary_metrics, 'best_precision', 'unknown'))}, and "
            f"F1={_format_metric(_metric_value(summary_metrics, 'best_f1', 'unknown'))}. "
            "This selection prioritizes identifying loss-making orders and is suitable only as a manual-review priority signal."
        )
        if english
        else (
            f"当前最佳模型为 {_metric_value(summary_metrics, 'best_model', '当前模型')}，"
            f"recall={_format_metric(_metric_value(summary_metrics, 'best_recall', 'unknown'))}、"
            f"precision={_format_metric(_metric_value(summary_metrics, 'best_precision', 'unknown'))}、"
            f"F1={_format_metric(_metric_value(summary_metrics, 'best_f1', 'unknown'))}。"
            "该选择更偏向识别亏损订单，适合作为人工复核优先级参考。"
        )
    )
    raw_model_sentence = _interpretation_value(
        modeling_interpretation,
        "model_choice_takeaway",
        fallback,
        output_language=output_language,
    )
    model_sentence = fallback if english and any("\u4e00" <= ch <= "\u9fff" for ch in raw_model_sentence) else raw_model_sentence
    raw_threshold_sentence = _interpretation_value(
        modeling_interpretation,
        "threshold_takeaway",
        _loss_risk_threshold_guidance_from_summary(summary_metrics, output_language=output_language),
        output_language=output_language,
    )
    threshold_sentence = (
        _loss_risk_threshold_guidance_from_summary(summary_metrics, output_language=output_language)
        if english and any("\u4e00" <= ch <= "\u9fff" for ch in raw_threshold_sentence)
        else raw_threshold_sentence
    )
    return model_sentence + "\n\n" + threshold_sentence


def _loss_risk_threshold_guidance_from_summary(
    summary_metrics: dict[str, object],
    output_language: str | None = None,
) -> str:
    threshold = _format_metric(_metric_value(summary_metrics, "threshold_default", 0.5))
    tp = _metric_value(summary_metrics, "true_positive_count", 0)
    fp = _metric_value(summary_metrics, "false_positive_count", 0)
    fn = _metric_value(summary_metrics, "false_negative_count", 0)
    tn = _metric_value(summary_metrics, "true_negative_count", 0)
    if is_english_output(output_language):
        return (
            f"At the default threshold {threshold}, TP={tp}, FP={fp}, FN={fn}, and TN={tn}. "
            f"The error structure requires a trade-off between missed loss-making orders (FN={fn}) and extra manual-review effort (FP={fp}). "
            "If the business is more sensitive to missed losses, it can accept more review volume; if review capacity is limited, FP should be controlled first."
        )
    return (
        f"默认阈值 {threshold} 下，TP={tp}、FP={fp}、FN={fn}、TN={tn}。"
        f"当前错误结构需要在漏判亏损（FN={fn}）和额外复核成本（FP={fp}）之间取舍；"
        "若业务更怕漏判，可接受更多复核量，若复核人力有限，则应优先控制 FP。"
    )


def _loss_risk_confusion_explanation(
    summary_metrics: dict[str, object],
    modeling_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    fallback = (
        (
            f"At the default threshold, TP={_metric_value(summary_metrics, 'true_positive_count', 0)}, "
            f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)}, "
            f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)}, and "
            f"TN={_metric_value(summary_metrics, 'true_negative_count', 0)}. "
            f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)} represents missed loss-making orders, while "
            f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)} represents extra review cost. "
            "The result is better suited to a manual-review pool than to automatic classification."
        )
        if english
        else (
            f"默认阈值下 TP={_metric_value(summary_metrics, 'true_positive_count', 0)}、"
            f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)}、"
            f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)}、"
            f"TN={_metric_value(summary_metrics, 'true_negative_count', 0)}。"
            f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)} 代表漏判亏损订单，"
            f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)} 代表额外复核成本，"
            "当前结果更适合做人工复核池，而不是自动判定。"
        )
    )
    value = _interpretation_value(
        modeling_interpretation,
        "error_takeaway",
        fallback,
        output_language=output_language,
    )
    return fallback if english and any("\u4e00" <= ch <= "\u9fff" for ch in value) else value


def _without_repeated_safety(existing_text: str, warning: str) -> str:
    text = str(warning or "").strip()
    if not text:
        return ""
    original = text
    if (
        "不代表因果关系" in existing_text
        and "不用于自动决策" in existing_text
        and ("不代表因果关系" in text or "不用于自动决策" in text)
    ):
        return ""
    if "不代表因果关系" in existing_text and "不代表因果关系" in text:
        text = text.replace("，不代表因果关系", "").replace("不代表因果关系，", "").replace("不代表因果关系", "")
    if "不用于自动决策" in existing_text and "不用于自动决策" in text:
        text = text.replace("，不用于自动决策", "").replace("不用于自动决策，", "").replace("不用于自动决策", "")
    text = text.replace("模型输出仅为风险概率估计。", "").replace("模型输出仅为风险概率估计，", "")
    text = text.replace("模型输出仅反映数据关联。", "").replace("模型输出仅反映数据关联，", "")
    text = text.replace("模型输出结果。", "").replace("模型输出结果，", "")
    text = text.replace("模型输出。", "").replace("模型输出；", "").replace("模型输出，", "")
    cleaned = " ".join(text.split()).strip("，。；; ")
    if "复核策略仍需结合业务规则" in cleaned and "人工判断" in cleaned:
        return ""
    if cleaned.startswith(("模型结果；", "模型结果，", "模型输出；", "模型输出，")):
        cleaned = cleaned[5:].strip("，。；; ")
    if cleaned in {"模型输出", "该模型", "模型结果"}:
        return ""
    if cleaned in {
        "复核策略仍需结合业务规则和人工判断",
        "复核策略仍需结合业务规则与人工判断",
        "复核策略仍需结合业务规则和人工判断。",
        "复核策略仍需结合业务规则与人工判断。",
        "具体使用仍需结合业务规则和人工判断",
        "具体使用仍需结合业务规则与人工判断",
        "模型输出仅反映数据关联",
    }:
        return ""
    if not cleaned and any(term in original for term in ["业务规则", "人工判断", "复核"]):
        return ""
    return cleaned


def _feature_takeaway_for_display(value: str, *, final_safety_present: bool) -> str:
    text = str(value or "").strip()
    if not text or not final_safety_present:
        return text
    text = text.replace("，不代表因果关系", "").replace("不代表因果关系，", "").replace("不代表因果关系", "")
    text = text.replace("，不用于自动决策", "").replace("不用于自动决策，", "").replace("不用于自动决策", "")
    text = text.replace("该字段仅为预测信号。", "").replace("该字段仅为预测信号", "")
    text = text.replace("但。", "。").replace("但，", "，").replace("但,", ",")
    return " ".join(text.split()).strip("，。；; ") + ("。" if text.strip("，。；; ") else "")


def _default_threshold_row(tables: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    rows = tables.get("threshold_analysis", [])
    for row in rows:
        threshold = _as_float(row.get("threshold"))
        if threshold is not None and abs(threshold - 0.5) < 0.000001:
            return row
    return rows[0] if rows else {}


def _top_feature_group_names(tables: dict[str, list[dict[str, object]]], limit: int = 3) -> list[str]:
    return [
        str(row.get("feature_group"))
        for row in tables.get("feature_importance_grouped", [])[:limit]
        if str(row.get("feature_group") or "").strip()
    ]


def _top_feature_signal_examples(tables: dict[str, list[dict[str, object]]], limit: int = 2) -> str:
    examples: list[str] = []
    for row in tables.get("feature_importance_grouped", [])[:3]:
        for raw in str(row.get("top_features") or "").split(","):
            item = raw.strip()
            if not item:
                continue
            for prefix in ("sub_category_", "category_", "productline_", "product_line_"):
                if item.startswith(prefix):
                    item = item[len(prefix) :]
                    break
            if item.lower() in {"discount", "quantity", "sales_amount", "unit_price"}:
                continue
            if item not in examples:
                examples.append(item)
            if len(examples) >= limit:
                return "、".join(examples)
    return "、".join(examples)


def _loss_risk_threshold_guidance(
    tables: dict[str, list[dict[str, object]]],
    output_language: str | None = None,
) -> str:
    threshold_row = _default_threshold_row(tables)
    threshold = _format_metric(threshold_row.get("threshold", 0.5))
    review_load_rate = _format_percent(threshold_row.get("review_load_rate", "unknown"))
    predicted_loss_count = _metric_value(threshold_row, "predicted_loss_count", "unknown")
    if is_english_output(output_language):
        return (
            f"At the default threshold {threshold}, about {review_load_rate} of orders enter the review pool "
            f"({predicted_loss_count} records). If review capacity is limited, raise the threshold between 0.5 and 0.7 to reduce review volume, while accepting lower recall."
        )
    return (
        f"按默认阈值 {threshold}，本轮约有 {review_load_rate} 的订单会进入复核池"
        f"（{predicted_loss_count} 条）。如果复核人力有限，可以在 0.5 到 0.7 之间调高阈值以降低复核量，"
        "但需要接受召回率下降。"
    )


def _review_guidance_for_display(
    guidance: str,
    tables: dict[str, list[dict[str, object]]],
    output_language: str | None = None,
) -> str:
    text = str(guidance or "").strip()
    if not text:
        return ""
    if "前25%" in text or "排名前25%" in text or "约25%" in text or "约 25%" in text:
        return _loss_risk_threshold_guidance(tables, output_language=output_language)
    return text


def _clean_modeling_summary_for_display(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    replacements = [
        ("，不代表因果关系，不用于自动决策", ""),
        ("不代表因果关系，不用于自动决策。", ""),
        ("不代表因果关系，不用于自动决策", ""),
        ("整体模型质量为strong", "整体表现较稳定"),
        ("整体建模质量良好（strong）", "整体表现较稳定"),
        ("整体建模质量为strong", "整体表现较稳定"),
        ("整体质量为strong", "整体表现较稳定"),
        ("模型质量为strong", "模型表现较稳定"),
        ("建模状态为强模型，建模价值水平高。", ""),
        ("建模状态为strong_model，建模价值等级高。", ""),
        ("该模型只用于风险预警和人工复核优先级排序。", ""),
    ]
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    return " ".join(cleaned.split()).strip("，。；; ") + ("。" if cleaned.strip("，。；; ") else "")


def _loss_risk_integrated_conclusion(
    summary_metrics: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
    modeling_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    llm_summary = ""
    if modeling_interpretation:
        llm_summary = _clean_modeling_summary_for_display(
            str(modeling_interpretation.get("modeling_summary") or "").strip()
        )
        if english and any("\u4e00" <= ch <= "\u9fff" for ch in llm_summary):
            llm_summary = ""
    top_groups = _top_feature_group_names(tables)
    top_groups_text = _loss_risk_feature_group_summary(top_groups, output_language=output_language)
    if english:
        modeling_summary = (
            "Overall, the current loss-risk model can be used as a manual-review priority reference. "
            f"{_metric_value(summary_metrics, 'best_model', 'the current model')} has "
            f"recall={_format_metric(_metric_value(summary_metrics, 'best_recall', 'unknown'))}, "
            f"precision={_format_metric(_metric_value(summary_metrics, 'best_precision', 'unknown'))}, and "
            f"F1={_format_metric(_metric_value(summary_metrics, 'best_f1', 'unknown'))}, "
            "which makes it more suitable for review prioritization than for automatic decisions."
        )
        threshold_sentence = (
            f"At the default threshold {_format_metric(_metric_value(summary_metrics, 'threshold_default', 0.5))}, "
            f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)} and "
            f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)}, so the threshold trade-off should be set according to review capacity."
        )
        signal_paragraph = (
            f"The main signals are concentrated in {top_groups_text}. More precisely, the model uses these fields to narrow the review pool, while business review should still return to order, product, discount, and profit details. "
            "The model is for risk warning and manual-review prioritization only; it does not imply causality and must not be used for automatic decisions."
        )
        if llm_summary:
            return llm_summary + " " + threshold_sentence + "\n\n" + signal_paragraph
        return modeling_summary + " " + threshold_sentence + "\n\n" + signal_paragraph
    modeling_summary = (
        f"综合来看，当前亏损风险模型可以作为人工复核优先级参考。"
        f"{_metric_value(summary_metrics, 'best_model', '当前模型')} 的 "
        f"recall={_format_metric(_metric_value(summary_metrics, 'best_recall', 'unknown'))}、"
        f"precision={_format_metric(_metric_value(summary_metrics, 'best_precision', 'unknown'))}、"
        f"F1={_format_metric(_metric_value(summary_metrics, 'best_f1', 'unknown'))}，"
        "说明它更适合作为人工复核优先级排序工具，而不是自动判定规则。"
    )
    threshold_sentence = (
        f"默认阈值 {_format_metric(_metric_value(summary_metrics, 'threshold_default', 0.5))} 下，"
        f"FP={_metric_value(summary_metrics, 'false_positive_count', 0)}、"
        f"FN={_metric_value(summary_metrics, 'false_negative_count', 0)}，"
        "阈值取舍需要结合复核人力决定。"
    )
    signal_paragraph = (
        f"主要信号集中在 {top_groups_text}等字段。这里更准确的理解是："
        "模型把这些字段用于缩小复核范围，业务复盘仍应回到订单、商品、折扣和利润明细。"
        "该模型只用于风险预警和人工复核优先级排序，不代表因果关系，不用于自动决策。"
    )
    if llm_summary:
        return llm_summary + " " + threshold_sentence + "\n\n" + signal_paragraph
    return modeling_summary + " " + threshold_sentence + "\n\n" + signal_paragraph


def _loss_risk_feature_group_summary(
    groups: list[str],
    output_language: str | None = None,
) -> str:
    labels = (
        {
            "sub_category": "subcategory",
            "category": "category",
            "discount": "discount",
            "sales_amount": "sales",
            "quantity": "quantity",
            "segment": "segment",
            "region": "region",
            "country": "country",
        }
        if is_english_output(output_language)
        else {
            "sub_category": "子类目",
            "category": "类目",
            "discount": "折扣",
            "sales_amount": "销售额",
            "quantity": "数量",
            "segment": "客群",
            "region": "区域",
            "country": "国家",
        }
    )
    selected = [labels.get(group, group) for group in groups[:3] if group]
    if is_english_output(output_language):
        return ", ".join(selected) if selected else "the main predictive signals"
    return "、".join(selected) if selected else "主要预测信号"


def _loss_risk_review_examples_block(
    summary_metrics: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
    modeling_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    review_guidance = ""
    if modeling_interpretation:
        review_guidance = _review_guidance_for_display(
            str(modeling_interpretation.get("review_guidance") or "").strip(),
            tables,
            output_language=output_language,
        )
        if is_english_output(output_language) and any("\u4e00" <= ch <= "\u9fff" for ch in review_guidance):
            review_guidance = ""
        if not is_english_output(output_language) and not any("\u4e00" <= ch <= "\u9fff" for ch in review_guidance) and re.search(r"[A-Za-z]{4,}", review_guidance):
            review_guidance = ""
    if is_english_output(output_language):
        return (
            _loss_risk_confusion_explanation(
                summary_metrics,
                modeling_interpretation,
                output_language=output_language,
            )
            + "\n\nThe code above generates high-risk examples and false-negative loss examples from the current predictions. These examples are for manual review only and must not be used as direct execution rules."
            + ("\n\n" + review_guidance if review_guidance else "")
        )
    return (
        _loss_risk_confusion_explanation(
            summary_metrics,
            modeling_interpretation,
            output_language=output_language,
        )
        + "\n\n上方代码会从当前预测结果生成高风险样例和漏判亏损样例；样例只用于人工复核，不作为直接执行依据。"
        + ("\n\n" + review_guidance if review_guidance else "")
    )


def _loss_risk_feature_synthesis_block(
    summary_metrics: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
    modeling_interpretation: dict[str, str] | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    feature_takeaway = ""
    business_warning = ""
    if modeling_interpretation:
        feature_takeaway = str(modeling_interpretation.get("feature_takeaway") or "").strip()
        business_warning = str(modeling_interpretation.get("business_use_warning") or "").strip()

    risk_warning = ""
    if modeling_outcome_interpretation:
        risk_warning = str(modeling_outcome_interpretation.get("risk_warning") or "").strip()
    final_safety_present = any(
        term in "\n".join(
            [
                str((modeling_interpretation or {}).get("modeling_summary") or ""),
                business_warning,
                risk_warning,
            ]
        )
        for term in ["不代表因果关系", "不用于自动决策"]
    )
    feature_takeaway = _feature_takeaway_for_display(feature_takeaway, final_safety_present=final_safety_present)

    if english and any("\u4e00" <= ch <= "\u9fff" for ch in feature_takeaway):
        feature_takeaway = ""
    feature_parts = [
        ("Top raw features: " if english else "Top 原始特征：") + _modeling_top_raw_features(tables),
        "",
    ]
    if feature_takeaway:
        feature_parts.extend([feature_takeaway, ""])
    elif english:
        feature_parts.extend(["These fields are predictive signals only. They need to be reviewed alongside order, product, discount, and profit details before any business rule is changed.", ""])
    else:
        feature_parts.extend(["这些字段是预测信号，不代表因果关系，需要结合订单、商品和折扣明细复盘。", ""])

    conclusion = _loss_risk_integrated_conclusion(
        summary_metrics,
        tables,
        modeling_interpretation,
        output_language=output_language,
    )

    return (
        "\n".join(feature_parts).rstrip()
        + ("\n\n### Modeling Summary\n\n" if english else "\n\n### 建模综合结论\n\n")
        + conclusion
    )


def _render_strong_loss_risk_outcome_cells(
    module: object,
    *,
    modeling_interpretation: dict[str, str] | None = None,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    output_language: str | None = None,
) -> list:
    tables = _module_tables(module)
    summary_metrics = _module_summary(module)
    english = is_english_output(output_language)
    return [
        new_markdown_cell(_loss_risk_feature_engineering_block(summary_metrics, tables, output_language=output_language)),
        new_code_cell(_loss_risk_code_first_training_code(module)),
        new_markdown_cell(_loss_risk_model_threshold_block(summary_metrics, tables, modeling_interpretation, output_language=output_language)),
        new_code_cell(_loss_risk_model_threshold_code(output_language=output_language)),
        new_markdown_cell(_loss_risk_threshold_explanation(summary_metrics, modeling_interpretation, output_language=output_language)),
        new_markdown_cell(
            "### Confusion Matrix and Review Samples\n\nThe confusion matrix at the default threshold shows hits, misses, and extra review load."
            if english
            else "### 混淆矩阵分析与复核样例\n\n默认阈值下的混淆矩阵用于查看识别命中、漏判和额外复核成本。"
        ),
        new_code_cell(_loss_risk_confusion_examples_code(output_language=output_language)),
        new_markdown_cell(_loss_risk_review_examples_block(summary_metrics, tables, modeling_interpretation, output_language=output_language)),
        new_markdown_cell(
            "### Risk Signal Interpretation\n\nThe feature importance chart shows which signal groups the model relies on most."
            if english
            else "### 风险信号解释\n\n下面的特征重要性图展示模型主要依赖哪些预测信号组。"
        ),
        new_code_cell(_loss_risk_feature_importance_code(output_language=output_language)),
        new_markdown_cell(
            _loss_risk_feature_synthesis_block(
                summary_metrics,
                tables,
                modeling_interpretation,
                modeling_outcome_interpretation,
                output_language=output_language,
            )
        ),
    ]


def _render_weak_loss_risk_code_first_cells(
    module: object,
    *,
    modeling_outcome: dict[str, object],
    modeling_interpretation: dict[str, str] | None = None,
    output_language: str | None = None,
) -> list:
    tables = _module_tables(module)
    summary_metrics = _module_summary(module)
    english = is_english_output(output_language)
    return [
        new_markdown_cell(_weak_loss_risk_intro_en(summary_metrics) if english else _weak_loss_risk_intro(summary_metrics)),
        new_code_cell(_loss_risk_code_first_training_code(module, weak=True)),
        new_markdown_cell(_weak_loss_risk_model_explanation(summary_metrics, output_language=output_language)),
        new_markdown_cell(
            "### Exploratory Risk Signals\n\nThe feature importance chart only reviews candidate signals captured by the weak model."
            if english
            else "### 探索性风险信号\n\n下面的特征重要性图只用于查看弱模型捕捉到的候选信号，不展开强模型式解释。"
        ),
        new_code_cell(_loss_risk_feature_importance_code(output_language=output_language)),
        new_markdown_cell(_weak_loss_risk_feature_explanation(tables, summary_metrics, modeling_interpretation, output_language=output_language)),
    ]


def _weak_loss_risk_intro(summary_metrics: dict[str, object]) -> str:
    weak_reasons = summary_metrics.get("weak_reasons") or []
    reason_text = "、".join(str(item) for item in weak_reasons) if isinstance(weak_reasons, list) else str(weak_reasons)
    return "\n\n".join(
        block
        for block in [
            "## 建模分析：亏损风险探索",
            (
                "### 为什么只是探索性参考\n\n"
                "当前亏损风险模型已跑通，但模型质量为 weak，亏损样本或测试集正例不足会让 precision / F1 波动很大。"
                + (f" 主要原因：{reason_text}。" if reason_text else "")
            ),
            (
                "### 可复现建模代码\n\n"
                "下面的代码会从当前 notebook 数据构造 `is_loss = profit < 0`、训练轻量模型并输出模型摘要与模型对比。"
                "由于当前是 weak model，后续只保留特征重要性图和必要解释，不展开阈值曲线与混淆矩阵样例。"
            ),
        ]
        if block
    )


def _weak_loss_risk_intro_en(summary_metrics: dict[str, object]) -> str:
    weak_reasons = summary_metrics.get("weak_reasons") or []
    reason_text = ", ".join(str(item) for item in weak_reasons) if isinstance(weak_reasons, list) else str(weak_reasons)
    return "\n\n".join(
        block
        for block in [
            "## Modeling Analysis: Loss Risk Exploration",
            (
                "### Why This Is Only Exploratory\n\n"
                "The loss-risk model can run, but model quality is weak. A small number of loss examples or test positives can make precision and F1 unstable."
                + (f" Main reasons: {reason_text}." if reason_text else "")
            ),
            (
                "### Reproducible Modeling Code\n\n"
                "The code below rebuilds `is_loss = profit < 0`, trains lightweight models, and outputs the model summary and comparison table. "
                "Because this is a weak model, the notebook keeps only feature importance and essential interpretation instead of full threshold curves and confusion-matrix examples."
            ),
        ]
        if block
    )


def _weak_loss_risk_model_explanation(
    summary_metrics: dict[str, object],
    output_language: str | None = None,
) -> str:
    false_positive = _metric_value(summary_metrics, "false_positive_count", 0)
    false_negative = _metric_value(summary_metrics, "false_negative_count", 0)
    positive_rate = _format_percent(_metric_value(summary_metrics, "target_positive_rate", "unknown"))
    test_positive = _metric_value(summary_metrics, "test_positive_count", "unknown")
    if is_english_output(output_language):
        return (
            "The model comparison table above shows that modeling can run, but the positive class share is low "
            f"(about {positive_rate}) and the test set has few positive examples ({test_positive}). "
            "In this setting, precision and F1 are very sensitive to a small number of records, so a single backtest is not stable enough for review ranking. "
            f"The current run has about {false_positive} false positives and {false_negative} false negatives: false positives add unnecessary review effort, while false negatives mean some loss-making orders may be missed. "
            "Therefore, the result should only be treated as exploratory risk evidence, not as a review-ranking or automatic-decision model."
        )
    return (
        "上方模型对比表说明，本轮建模虽然可以运行，但正例占比很低"
        f"（约 {positive_rate}），测试集正例数也很少（{test_positive}）。"
        "在这种情况下，precision / F1 对少量样本非常敏感，单次回测结果不适合作为稳定排序依据。"
        f"当前误报约 {false_positive} 条、漏报约 {false_negative} 条：误报会增加不必要复核，漏报则意味着真实亏损订单可能被放过。"
        "因此当前结果只适合作为探索性风险线索，不建议直接用于复核排序或自动决策。"
    )


def _weak_loss_risk_feature_explanation(
    tables: dict[str, list[dict[str, object]]],
    summary_metrics: dict[str, object],
    modeling_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    top_features = _modeling_top_raw_features(tables)
    llm_feature_takeaway = ""
    if modeling_interpretation:
        llm_feature_takeaway = str(modeling_interpretation.get("feature_takeaway") or "").strip()
        llm_feature_takeaway = llm_feature_takeaway.replace("该字段仅为预测信号", "这些字段只是预测信号")
    positive_rate = _format_percent(_metric_value(summary_metrics, "target_positive_rate", "unknown"))
    test_positive = _metric_value(summary_metrics, "test_positive_count", "unknown")
    if is_english_output(output_language):
        if any("\u4e00" <= ch <= "\u9fff" for ch in llm_feature_takeaway):
            llm_feature_takeaway = ""
        weak_note = (
            f"These features only show candidate signals captured by a weak model with very few positive examples (loss share about {positive_rate}, test positives {test_positive}). "
            "They do not explain why losses happen. Next, add more loss examples or adjust the profit definition before deciding whether country, sales, subcategory, or similar signals are stable."
        )
        if llm_feature_takeaway:
            return f"Top raw features: {top_features}\n\n{llm_feature_takeaway}\n\n{weak_note}"
        return f"Top raw features: {top_features}\n\n{weak_note}"
    weak_note = (
        f"这些特征只能说明弱模型在极少量正例下捕捉到的候选信号（亏损占比约 {positive_rate}，测试集正例 {test_positive}）。"
        "它们不能直接解释亏损原因；下一步应先补充亏损样本或调整 profit 口径，再判断国家、销售额、子类目等信号是否稳定。"
    )
    if llm_feature_takeaway:
        return f"Top 原始特征：{top_features}\n\n{llm_feature_takeaway}\n\n{weak_note}"
    return (
        f"Top 原始特征：{top_features}\n\n"
        + weak_note
    )


def _render_loss_risk_outcome_markdown(
    module: object,
    *,
    modeling_outcome: dict[str, object],
    modeling_interpretation: dict[str, str] | None = None,
) -> str:
    tables = _module_tables(module)
    summary_metrics = _module_summary(module)
    status = str(modeling_outcome.get("modeling_status") or summary_metrics.get("model_quality_status") or "usable")
    weak = status == "weak_model" or str(summary_metrics.get("model_quality_status") or "") == "weak"
    metric_table = _loss_risk_key_metrics_table(summary_metrics)
    feature_table = _feature_signal_table(tables, limit=3 if weak else 6)
    top_raw_features = _modeling_top_raw_features(tables)
    optional_explanation = _optional_interpretation_block(
        modeling_interpretation,
        ["modeling_summary", "feature_takeaway", "error_takeaway", "review_guidance"],
    )
    if weak:
        weak_reasons = summary_metrics.get("weak_reasons") or []
        reason_text = "、".join(str(item) for item in weak_reasons) if isinstance(weak_reasons, list) else str(weak_reasons)
        return "\n\n".join(
            block
            for block in [
                "## 建模分析：亏损风险探索",
                (
                    "### 为什么只是探索性参考\n\n"
                    "当前亏损风险模型已跑通，但模型质量为 weak，亏损样本或测试集正例不足会让 precision / F1 波动很大。"
                    + (f" 主要原因：{reason_text}。" if reason_text else "")
                ),
                (
                    "### 可复现建模代码\n\n"
                    "下面的代码会重新构造 `is_loss = profit < 0`、训练轻量模型并输出模型对比。"
                    "由于当前是 weak model，后续只保留特征重要性图和必要解释，不展开阈值曲线与混淆矩阵样例。"
                ),
                optional_explanation,
                (
                    "### 使用限制\n\n"
                    "当前结果只适合作为探索性风险线索和人工复核参考，不建议直接用于复核排序。"
                    "模型输出不代表因果关系，不用于自动决策。"
                ),
            ]
            if block
        )

    return "\n\n".join(
        block
        for block in [
            "## 建模分析：亏损风险识别",
            (
                "### 建模任务与验证设计\n\n"
                "目标是识别更可能发生负利润的订单，作为人工复核优先级参考。"
                "模型输出不代表因果关系，不用于自动决策。\n\n"
                + "Target：`is_loss = profit < 0`\n\n"
                + metric_table
            ),
            (
                "### 模型表现与阈值取舍\n\n"
                + _compact_table(tables.get("model_comparison", []), limit=6)
                + "\n\n"
                + _compact_table(tables.get("threshold_analysis", []), limit=8)
            ),
            (
                "### 混淆矩阵解释\n\n"
                f"TP=正确识别亏损（{_metric_value(summary_metrics, 'true_positive_count', 0)}），"
                f"FN=漏判亏损（{_metric_value(summary_metrics, 'false_negative_count', 0)}），"
                f"FP=额外复核成本（{_metric_value(summary_metrics, 'false_positive_count', 0)}），"
                f"TN=正确识别非亏损（{_metric_value(summary_metrics, 'true_negative_count', 0)}）。"
                "后面的混淆矩阵图用于查看默认阈值下的错误结构。"
            ),
            (
                "### 风险特征解释\n\n"
                + feature_table
                + "\n\nTop 原始特征："
                + top_raw_features
                + "\n\n这些字段是预测信号，不代表因果关系，需要结合订单、商品和折扣明细复盘。"
            ),
            optional_explanation,
            (
                "### 建模结论与限制\n\n"
                "当前模型可以作为亏损风险预警和人工复核优先级参考，但具体使用仍需结合阈值取舍、错误样本和业务规则。"
                "不代表因果关系，不用于自动决策。"
            ),
        ]
        if block
    )


def _forecast_metric_table(module: object | None) -> str:
    summary_metrics = _module_summary(module)
    rows = [
        {
            "time_split": summary_metrics.get("time_split"),
            "series_granularity": summary_metrics.get("series_granularity"),
            "best_baseline": summary_metrics.get("best_baseline"),
            "MAE": summary_metrics.get("best_baseline_mae"),
            "RMSE": summary_metrics.get("best_baseline_rmse"),
            "MAPE": summary_metrics.get("best_baseline_mape"),
            "train_period_count": summary_metrics.get("train_period_count"),
            "backtest_period_count": summary_metrics.get("backtest_period_count"),
        }
    ]
    return _compact_table(rows, limit=1)


def _render_forecast_outcome_markdown_parts(
    forecast_module: object,
    *,
    modeling_outcome: dict[str, object],
    modeling_outcome_interpretation: dict[str, str] | None = None,
) -> tuple[str, str]:
    summary_metrics = _module_summary(forecast_module)
    status = str(modeling_outcome.get("modeling_status") or "")
    high_mape = status == "weak_baseline"
    baseline = _metric_value(summary_metrics, "best_baseline")
    mape = _metric_value(summary_metrics, "best_baseline_mape")
    explanation = _forecast_interpretation_block(
        modeling_outcome_interpretation,
        ["outcome_summary", "business_interpretation", "recommended_action"],
        best_baseline=baseline,
    )
    if high_mape:
        value_sentence = (
            f"当前 best baseline 为 {baseline}，MAPE={mape}，MAPE 偏高，说明序列波动较大、预测难度高。"
            "这类结果更适合作为监控参照和后续模型对照。"
        )
    else:
        value_sentence = (
            f"当前 best baseline 为 {baseline}，MAPE={mape}。"
            "这个 baseline 可以作为后续预测模型或销售监控的对照基线。"
        )
    before_chart = "\n\n".join(
        block
        for block in [
            "## 建模分析：销售额预测 baseline",
            (
                "### 为什么选择这个任务\n\n"
                "当前数据的主要可建模方向是销售额预测 baseline。这里使用已有时间字段和销售额字段做时间顺序切分，"
                "不使用随机切分，也不加入复杂模型。"
            ),
            "### 回测设计与指标\n\n" + _forecast_metric_table(forecast_module),
            "下图展示回测窗口内实际销售额与 baseline 预测值的对比。",
        ]
        if block
    )
    after_chart = "\n\n".join(
        block
        for block in [
            "### 结果解释\n\n" + value_sentence,
            explanation,
            "### 使用限制\n\n这是 baseline，不是生产级预测，不用于自动决策。",
        ]
        if block
    )
    return before_chart, after_chart


def _render_forecast_outcome_markdown(
    forecast_module: object,
    *,
    modeling_outcome: dict[str, object],
    modeling_outcome_interpretation: dict[str, str] | None = None,
) -> str:
    before_chart, after_chart = _render_forecast_outcome_markdown_parts(
        forecast_module,
        modeling_outcome=modeling_outcome,
        modeling_outcome_interpretation=modeling_outcome_interpretation,
    )
    return before_chart + "\n\n" + after_chart


def _modeling_table_json(module: object | None, table_name: str) -> str:
    tables = getattr(module, "tables", {}) or {}
    rows = tables.get(table_name, [])
    return json.dumps(rows if isinstance(rows, list) else [], ensure_ascii=False, default=str)


def _modeling_plot_setup_code() -> str:
    return (
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n\n"
        "import matplotlib.font_manager as fm\n\n"
        "preferred_fonts = [\"Microsoft YaHei\", \"SimHei\", \"Noto Sans CJK JP\", \"Arial Unicode MS\"]\n"
        "available_fonts = {font.name for font in fm.fontManager.ttflist}\n"
        "for font_name in preferred_fonts:\n"
        "    if font_name in available_fonts:\n"
        "        plt.rcParams[\"font.family\"] = font_name\n"
        "        break\n"
        "plt.rcParams[\"axes.unicode_minus\"] = False\n"
    )


def _loss_risk_feature_plan_config(module: object | None) -> str:
    tables = getattr(module, "tables", {}) or {}
    return json.dumps(
        {
            "safe_features": tables.get("safe_features", []),
            "blocked_features": tables.get("blocked_features", []),
        },
        ensure_ascii=False,
        default=str,
    )


def _loss_risk_code_first_training_code(module: object | None = None, *, weak: bool = False) -> str:
    display_lines = (
        "display(pd.DataFrame({'used_features': feature_cols[:20]}))\n"
        "display(pd.DataFrame({'blocked_fields': blocked_cols[:20]}))\n"
    )
    if weak:
        display_lines = (
            "display(loss_risk_model_summary)\n"
            "display(model_comparison)\n"
        )
    return (
        "import numpy as np\n"
        "import pandas as pd\n"
        "from sklearn.compose import ColumnTransformer\n"
        "from sklearn.dummy import DummyClassifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.impute import SimpleImputer\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n\n"
        f"LOSS_RISK_FEATURE_PLAN = {_loss_risk_feature_plan_config(module)}\n\n"
        "MODEL_CONFIG = {\n"
        "    'test_size': 0.25,\n"
        "    'random_state': 42,\n"
        "    'threshold': 0.5,\n"
        "    'max_category_levels': 40,\n"
        "    'logistic_regression': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42},\n"
        "    'random_forest': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42},\n"
        "}\n\n"
        "source_df = df.copy()\n"
        "def _norm(name):\n"
        "    return ''.join(ch for ch in str(name).lower() if ch.isalnum())\n\n"
        "def _find_col(candidates):\n"
        "    lookup = {_norm(col): col for col in source_df.columns}\n"
        "    for item in candidates:\n"
        "        if _norm(item) in lookup:\n"
        "            return lookup[_norm(item)]\n"
        "    return None\n\n"
        "profit_col = _find_col(['profit', '利润'])\n"
        "if profit_col is None:\n"
        "    raise ValueError('亏损风险建模需要 profit 字段。')\n\n"
        "model_df = source_df.copy()\n"
        "model_df['is_loss'] = pd.to_numeric(model_df[profit_col], errors='coerce') < 0\n"
        "model_df = model_df[model_df[profit_col].notna()].copy()\n"
        "blocked_tokens = {'profit', 'isloss', 'profitmargin', 'lossflag', 'negativeprofit', 'customerid', 'orderid', 'productname', 'rowid', 'recordid', 'index'}\n"
        "planned_blocked = {str(row.get('feature')) for row in LOSS_RISK_FEATURE_PLAN.get('blocked_features', []) if row.get('feature')}\n"
        "blocked_cols = [\n"
        "    col for col in model_df.columns\n"
        "    if _norm(col) in blocked_tokens or _norm(col) in {_norm(name) for name in planned_blocked} or _norm(col).startswith('unnamed')\n"
        "]\n\n"
        "CANONICAL_ALIASES = {\n"
        "    'sales_amount': ['sales_amount', 'sales', 'amount', '销售额'],\n"
        "    'order_datetime': ['order_datetime', 'order_date', 'date', 'Order Date', '订单日期'],\n"
        "    'unit_price': ['unit_price', 'price', '单价'],\n"
        "    'sub_category': ['sub_category', 'sub-category', 'Sub-Category'],\n"
        "    'ship_mode': ['ship_mode', 'Ship Mode'],\n"
        "}\n"
        "def _actual_col(planned_name):\n"
        "    lookup = {_norm(col): col for col in model_df.columns}\n"
        "    for candidate in [planned_name] + CANONICAL_ALIASES.get(str(planned_name), []):\n"
        "        if _norm(candidate) in lookup:\n"
        "            return lookup[_norm(candidate)]\n"
        "    return None\n\n"
        "def _planned_feature_type(row, actual_col):\n"
        "    value = str(row.get('feature_type') or row.get('type') or '').lower()\n"
        "    reason = str(row.get('reason') or '').lower()\n"
        "    if value in {'numeric', 'categorical', 'date'}:\n"
        "        return value\n"
        "    if 'date' in value or 'date' in reason or 'date' in _norm(actual_col):\n"
        "        return 'date'\n"
        "    if pd.api.types.is_numeric_dtype(model_df[actual_col]):\n"
        "        return 'numeric'\n"
        "    return 'categorical'\n\n"
        "X = pd.DataFrame(index=model_df.index)\n"
        "numeric_features = []\n"
        "categorical_features = []\n"
        "safe_rows = LOSS_RISK_FEATURE_PLAN.get('safe_features', [])\n"
        "if not safe_rows:\n"
        "    raise ValueError('没有可用的安全特征，请检查 LOSS_RISK_FEATURE_PLAN。')\n"
        "for row in safe_rows:\n"
        "    planned_name = str(row.get('feature') or '').strip()\n"
        "    actual_col = _actual_col(planned_name)\n"
        "    if not planned_name or actual_col is None or actual_col in blocked_cols:\n"
        "        continue\n"
        "    feature_type = _planned_feature_type(row, actual_col)\n"
        "    if feature_type == 'date':\n"
        "        parsed = pd.to_datetime(model_df[actual_col], errors='coerce')\n"
        "        for part, values in {\n"
        "            'year': parsed.dt.year,\n"
        "            'month': parsed.dt.month,\n"
        "            'quarter': parsed.dt.quarter,\n"
        "            'weekday': parsed.dt.weekday,\n"
        "        }.items():\n"
        "            column = f'{planned_name}_{part}'\n"
        "            X[column] = values\n"
        "            numeric_features.append(column)\n"
        "    elif feature_type == 'numeric':\n"
        "        X[planned_name] = pd.to_numeric(model_df[actual_col], errors='coerce')\n"
        "        numeric_features.append(planned_name)\n"
        "    elif 1 < model_df[actual_col].nunique(dropna=True) <= MODEL_CONFIG['max_category_levels']:\n"
        "        X[planned_name] = model_df[actual_col]\n"
        "        categorical_features.append(planned_name)\n"
        "feature_cols = numeric_features + categorical_features\n"
        "if not feature_cols:\n"
        "    raise ValueError('没有可用的安全特征，请检查 LOSS_RISK_FEATURE_PLAN。')\n"
        "X = X[feature_cols]\n"
        "y = model_df['is_loss'].astype(bool)\n"
        "if y.nunique() < 2:\n"
        "    raise ValueError('亏损风险 target 只有单一类别，无法训练分类模型。')\n\n"
        "numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])\n"
        "try:\n"
        "    one_hot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)\n"
        "except TypeError:\n"
        "    one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)\n"
        "categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', one_hot)])\n"
        "preprocess = ColumnTransformer([\n"
        "    ('num', numeric_pipe, numeric_features),\n"
        "    ('cat', categorical_pipe, categorical_features),\n"
        "])\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=MODEL_CONFIG['test_size'], random_state=MODEL_CONFIG['random_state'], stratify=y\n"
        ")\n"
        "models = {\n"
        "    'DummyClassifier': DummyClassifier(strategy='most_frequent', random_state=MODEL_CONFIG['random_state']),\n"
        "    'LogisticRegression': LogisticRegression(**MODEL_CONFIG['logistic_regression']),\n"
        "    'RandomForestClassifier': RandomForestClassifier(**MODEL_CONFIG['random_forest']),\n"
        "}\n\n"
        "fitted_models = {}\n"
        "score_by_model = {}\n"
        "model_rows = []\n"
        "for name, estimator in models.items():\n"
        "    pipe = Pipeline([('preprocess', preprocess), ('model', estimator)])\n"
        "    pipe.fit(X_train, y_train)\n"
        "    fitted_models[name] = pipe\n"
        "    if hasattr(pipe.named_steps['model'], 'predict_proba'):\n"
        "        score = pipe.predict_proba(X_test)[:, 1]\n"
        "    else:\n"
        "        score = pipe.predict(X_test).astype(float)\n"
        "    score_by_model[name] = score\n"
        "    pred = pipe.predict(X_test)\n"
        "    try:\n"
        "        auc = roc_auc_score(y_test, score)\n"
        "    except ValueError:\n"
        "        auc = np.nan\n"
        "    model_rows.append({\n"
        "        'model': name,\n"
        "        'accuracy': round(accuracy_score(y_test, pred), 4),\n"
        "        'precision': round(precision_score(y_test, pred, zero_division=0), 4),\n"
        "        'recall': round(recall_score(y_test, pred, zero_division=0), 4),\n"
        "        'f1': round(f1_score(y_test, pred, zero_division=0), 4),\n"
        "        'roc_auc': round(float(auc), 4) if pd.notna(auc) else np.nan,\n"
        "        'is_baseline': name == 'DummyClassifier',\n"
        "    })\n"
        "model_comparison = pd.DataFrame(model_rows)\n"
        "best_model_name = model_comparison[~model_comparison['is_baseline']].sort_values(\n"
        "    ['recall', 'f1', 'roc_auc'], ascending=False\n"
        ").iloc[0]['model']\n"
        "best_model = fitted_models[best_model_name]\n"
        "best_scores = score_by_model[best_model_name]\n"
        "best_pred = best_scores >= MODEL_CONFIG['threshold']\n\n"
        "threshold_rows = []\n"
        "for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:\n"
        "    pred_t = best_scores >= threshold\n"
        "    threshold_rows.append({\n"
        "        'threshold': threshold,\n"
        "        'precision': round(precision_score(y_test, pred_t, zero_division=0), 4),\n"
        "        'recall': round(recall_score(y_test, pred_t, zero_division=0), 4),\n"
        "        'f1': round(f1_score(y_test, pred_t, zero_division=0), 4),\n"
        "        'predicted_loss_count': int(pred_t.sum()),\n"
        "        'review_load_rate': round(float(pred_t.mean()), 4),\n"
        "    })\n"
        "threshold_analysis = pd.DataFrame(threshold_rows)\n"
        "loss_risk_model_summary = pd.DataFrame([{\n"
        "    'best_model': best_model_name,\n"
        "    'target_positive_count': int(y.sum()),\n"
        "    'target_positive_rate': round(float(y.mean()), 6),\n"
        "    'test_positive_count': int(y_test.sum()),\n"
        "    'threshold': MODEL_CONFIG['threshold'],\n"
        "}])\n"
        + display_lines
    )


def _loss_risk_model_threshold_code(output_language: str | None = None) -> str:
    english = is_english_output(output_language)
    recall_label = "Recall" if english else "Recall 召回率"
    precision_label = "Precision" if english else "Precision 精确率"
    review_load_label = "Review Load Rate" if english else "复核量占比"
    title = "Threshold Trade-off Curve: Recall and Review Load" if english else "阈值取舍曲线：召回率与复核量"
    ylabel = "Metric Value" if english else "指标值"
    return (
        _modeling_plot_setup_code()
        + "\ndisplay(loss_risk_model_summary)\n"
        "display(model_comparison)\n"
        "display(threshold_analysis)\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "for column, label in [\n"
        f"    ('recall', {recall_label!r}),\n"
        f"    ('precision', {precision_label!r}),\n"
        "    ('f1', 'F1'),\n"
        f"    ('review_load_rate', {review_load_label!r}),\n"
        "]:\n"
        "    ax.plot(threshold_analysis['threshold'], threshold_analysis[column], marker='o', label=label)\n"
        f"ax.set_title({title!r})\n"
        "ax.set_xlabel('threshold')\n"
        f"ax.set_ylabel({ylabel!r})\n"
        "ax.set_ylim(0, 1.05)\n"
        "ax.grid(True, alpha=0.25)\n"
        "ax.legend(loc='best')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )


def _loss_risk_confusion_examples_code(output_language: str | None = None) -> str:
    english = is_english_output(output_language)
    title = "Confusion Matrix at Default Threshold" if english else "默认阈值下的混淆矩阵"
    x_label = "Predicted Class" if english else "预测类别"
    y_label = "Actual Class" if english else "真实类别"
    high_risk = "High-risk Examples Top 3" if english else "高风险样例 Top 3"
    false_negative = "False Negative Loss Examples Top 3" if english else "漏判亏损样例 Top 3"
    return (
        _modeling_plot_setup_code()
        + "\nresult_df = model_df.loc[X_test.index].copy()\n"
        "result_df['loss_probability'] = best_scores\n"
        "result_df['actual_is_loss'] = y_test.values\n"
        "result_df['predicted_is_loss'] = best_pred\n"
        "confusion_matrix_df = pd.crosstab(\n"
        "    result_df['actual_is_loss'].map({False: 'non_loss', True: 'loss'}),\n"
        "    result_df['predicted_is_loss'].map({False: 'non_loss', True: 'loss'}),\n"
        "    rownames=['actual'], colnames=['predicted'], dropna=False\n"
        ").reindex(index=['non_loss', 'loss'], columns=['non_loss', 'loss'], fill_value=0)\n"
        "fig, ax = plt.subplots(figsize=(5.8, 4.8))\n"
        "image = ax.imshow(confusion_matrix_df.values, cmap='Blues')\n"
        f"ax.set_title({title!r})\n"
        f"ax.set_xlabel({x_label!r})\n"
        f"ax.set_ylabel({y_label!r})\n"
        "ax.set_xticks(range(len(confusion_matrix_df.columns)), labels=list(confusion_matrix_df.columns))\n"
        "ax.set_yticks(range(len(confusion_matrix_df.index)), labels=list(confusion_matrix_df.index))\n"
        "for row_idx in range(confusion_matrix_df.shape[0]):\n"
        "    for col_idx in range(confusion_matrix_df.shape[1]):\n"
        "        ax.text(col_idx, row_idx, int(confusion_matrix_df.iloc[row_idx, col_idx]), ha='center', va='center')\n"
        "fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "example_cols = [col for col in ['loss_probability', 'actual_is_loss', 'predicted_is_loss', 'discount', 'sales_amount', 'category', 'sub_category', 'segment', 'region'] if col in result_df.columns]\n"
        f"print({high_risk!r})\n"
        "display(result_df.sort_values('loss_probability', ascending=False)[example_cols].head(3))\n"
        f"print({false_negative!r})\n"
        "display(result_df[(result_df['actual_is_loss']) & (~result_df['predicted_is_loss'])].sort_values('loss_probability', ascending=False)[example_cols].head(3))\n"
    )


def _loss_risk_feature_importance_code(output_language: str | None = None) -> str:
    title = "Loss-risk Feature Group Importance" if is_english_output(output_language) else "亏损风险预测信号组重要性"
    return (
        _modeling_plot_setup_code()
        + "\nfeature_names = best_model.named_steps['preprocess'].get_feature_names_out()\n"
        "model_step = best_model.named_steps['model']\n"
        "if hasattr(model_step, 'coef_'):\n"
        "    importances = np.abs(model_step.coef_[0])\n"
        "else:\n"
        "    importances = getattr(model_step, 'feature_importances_', np.zeros(len(feature_names)))\n"
        "feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})\n"
        "def _feature_group(name):\n"
        "    raw = str(name).split('__', 1)[-1]\n"
        "    for original in feature_cols:\n"
        "        if raw == original or raw.startswith(str(original) + '_'):\n"
        "            return original\n"
        "    return raw.split('_')[0]\n"
        "feature_importance['feature_group'] = feature_importance['feature'].map(_feature_group)\n"
        "feature_importance_grouped = (\n"
        "    feature_importance.groupby('feature_group', as_index=False)['importance'].sum()\n"
        "    .rename(columns={'importance': 'total_importance'})\n"
        "    .sort_values('total_importance', ascending=False)\n"
        ")\n"
        "plot_df = feature_importance_grouped.sort_values('total_importance', ascending=True).tail(10)\n"
        "fig, ax = plt.subplots(figsize=(8, 4.8))\n"
        "ax.barh(plot_df['feature_group'], plot_df['total_importance'], color='#3B82F6')\n"
        f"ax.set_title({title!r})\n"
        "ax.set_xlabel('total_importance')\n"
        "ax.set_ylabel('feature_group')\n"
        "ax.grid(axis='x', alpha=0.2)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )


def _threshold_tradeoff_chart_code(module: object | None, output_language: str | None = None) -> str:
    rows_json = _modeling_table_json(module, "threshold_analysis")
    english = is_english_output(output_language)
    skip_message = "No threshold analysis data is available, so the threshold trade-off chart was skipped." if english else "暂无阈值分析数据，跳过阈值取舍图。"
    recall_label = "Recall" if english else "Recall 召回率"
    precision_label = "Precision" if english else "Precision 精确率"
    review_load_label = "Review Load Rate" if english else "复核量占比"
    title = "Threshold Trade-off Curve: Recall and Review Load" if english else "阈值取舍曲线：召回率与复核量"
    ylabel = "Metric Value" if english else "指标值"
    return (
        _modeling_plot_setup_code()
        + f"\nthreshold_analysis = pd.DataFrame({rows_json})\n"
        "if threshold_analysis.empty:\n"
        f"    print({skip_message!r})\n"
        "else:\n"
        "    numeric_columns = [\"threshold\", \"precision\", \"recall\", \"f1\", \"review_load_rate\"]\n"
        "    for column in numeric_columns:\n"
        "        if column in threshold_analysis.columns:\n"
        "            threshold_analysis[column] = pd.to_numeric(threshold_analysis[column], errors=\"coerce\")\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "    for column, label in [\n"
        f"        (\"recall\", {recall_label!r}),\n"
        f"        (\"precision\", {precision_label!r}),\n"
        "        (\"f1\", \"F1\"),\n"
        f"        (\"review_load_rate\", {review_load_label!r}),\n"
        "    ]:\n"
        "        if column in threshold_analysis.columns:\n"
        "            ax.plot(threshold_analysis[\"threshold\"], threshold_analysis[column], marker=\"o\", label=label)\n"
        f"    ax.set_title({title!r})\n"
        "    ax.set_xlabel(\"threshold\")\n"
        f"    ax.set_ylabel({ylabel!r})\n"
        "    ax.set_ylim(0, 1.05)\n"
        "    ax.grid(True, alpha=0.25)\n"
        "    ax.legend(loc=\"best\")\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _confusion_matrix_chart_code(module: object | None, output_language: str | None = None) -> str:
    rows_json = _modeling_table_json(module, "confusion_matrix")
    english = is_english_output(output_language)
    skip_message = "No confusion matrix data is available, so the confusion matrix chart was skipped." if english else "暂无混淆矩阵数据，跳过混淆矩阵图。"
    title = "Confusion Matrix at Default Threshold" if english else "默认阈值下的混淆矩阵"
    x_label = "Predicted Class" if english else "预测类别"
    y_label = "Actual Class" if english else "真实类别"
    return (
        _modeling_plot_setup_code()
        + f"\nconfusion_matrix = pd.DataFrame({rows_json})\n"
        "if confusion_matrix.empty:\n"
        f"    print({skip_message!r})\n"
        "else:\n"
        "    confusion_matrix[\"count\"] = pd.to_numeric(confusion_matrix[\"count\"], errors=\"coerce\").fillna(0)\n"
        "    matrix = confusion_matrix.pivot_table(\n"
        "        index=\"actual\",\n"
        "        columns=\"predicted\",\n"
        "        values=\"count\",\n"
        "        aggfunc=\"sum\",\n"
        "        fill_value=0,\n"
        "    )\n"
        "    fig, ax = plt.subplots(figsize=(5.8, 4.8))\n"
        "    image = ax.imshow(matrix.values, cmap=\"Blues\")\n"
        f"    ax.set_title({title!r})\n"
        f"    ax.set_xlabel({x_label!r})\n"
        f"    ax.set_ylabel({y_label!r})\n"
        "    ax.set_xticks(range(len(matrix.columns)), labels=list(matrix.columns))\n"
        "    ax.set_yticks(range(len(matrix.index)), labels=list(matrix.index))\n"
        "    for row_idx in range(matrix.shape[0]):\n"
        "        for col_idx in range(matrix.shape[1]):\n"
        "            value = int(matrix.iloc[row_idx, col_idx])\n"
        "            ax.text(col_idx, row_idx, value, ha=\"center\", va=\"center\", color=\"black\")\n"
        "    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _feature_group_importance_chart_code(module: object | None, output_language: str | None = None) -> str:
    rows = [
        {
            "feature_group": row.get("feature_group"),
            "total_importance": row.get("total_importance"),
        }
        for row in _module_tables(module).get("feature_importance_grouped", [])
    ]
    rows_json = json.dumps(rows, ensure_ascii=False, default=str)
    english = is_english_output(output_language)
    skip_message = "No feature-group importance data is available, so the feature-group importance chart was skipped." if english else "暂无特征组重要性数据，跳过特征组重要性图。"
    title = "Loss-risk Feature Group Importance" if english else "亏损风险预测信号组重要性"
    return (
        _modeling_plot_setup_code()
        + f"\nfeature_importance_grouped = pd.DataFrame({rows_json})\n"
        "if feature_importance_grouped.empty:\n"
        f"    print({skip_message!r})\n"
        "else:\n"
        "    feature_importance_grouped[\"total_importance\"] = pd.to_numeric(\n"
        "        feature_importance_grouped[\"total_importance\"], errors=\"coerce\"\n"
        "    ).fillna(0)\n"
        "    plot_df = feature_importance_grouped.sort_values(\"total_importance\", ascending=True).tail(10)\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.8))\n"
        "    ax.barh(plot_df[\"feature_group\"], plot_df[\"total_importance\"], color=\"#3B82F6\")\n"
        f"    ax.set_title({title!r})\n"
        "    ax.set_xlabel(\"total_importance\")\n"
        "    ax.set_ylabel(\"feature_group\")\n"
        "    ax.grid(axis=\"x\", alpha=0.2)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _forecast_table_json(module: object | None, table_name: str) -> str:
    tables = _module_tables(module)
    rows = tables.get(table_name, [])
    return json.dumps(rows if isinstance(rows, list) else [], ensure_ascii=False, default=str)


def _sales_regression_metric_table(module: object | None) -> str:
    summary_metrics = _module_summary(module)
    rows = [
        {
            "time_split": summary_metrics.get("regression_time_split") or summary_metrics.get("time_split"),
            "series_granularity": summary_metrics.get("series_granularity"),
            "best_model": summary_metrics.get("best_model"),
            "baseline_model": summary_metrics.get("baseline_model"),
            "MAE": summary_metrics.get("best_model_mae"),
            "RMSE": summary_metrics.get("best_model_rmse"),
            "MAPE": summary_metrics.get("best_model_mape"),
            "R2": summary_metrics.get("best_model_r2"),
            "improvement_vs_baseline": summary_metrics.get("improvement_vs_baseline"),
        }
    ]
    return _compact_table(rows, limit=1)


def _sales_regression_condition_table(module: object | None) -> str:
    summary_metrics = _module_summary(module)
    rows = [
        {
            "granularity": summary_metrics.get("series_granularity"),
            "day_span": summary_metrics.get("day_span"),
            "observed_periods": summary_metrics.get("observed_period_count"),
            "train_periods": summary_metrics.get("train_period_count"),
            "backtest_periods": summary_metrics.get("backtest_period_count"),
            "effective_samples": summary_metrics.get("regression_effective_sample_count"),
            "cv_strategy": summary_metrics.get("regression_cv_strategy"),
        }
    ]
    return _compact_table(rows, limit=1)


def _sales_regression_model_comparison(module: object | None) -> str:
    rows = _module_tables(module).get("model_comparison", [])
    columns = [
        "model",
        "model_type",
        "mae",
        "rmse",
        "mape",
        "r2",
        "improvement_vs_baseline",
    ]
    return _selected_columns_table(rows, columns, limit=6)


def _sales_regression_cv_fold_table(module: object | None) -> str:
    rows = _module_tables(module).get("regression_cv_folds", [])
    return _selected_columns_table(
        rows,
        ["fold", "train_start", "train_end", "valid_start", "valid_end", "model", "mae", "rmse", "mape", "r2"],
        limit=12,
    )


def _sales_regression_cv_summary_table(module: object | None) -> str:
    rows = _module_tables(module).get("regression_cv_summary", [])
    return _selected_columns_table(
        rows,
        ["model", "cv_mae_mean", "cv_mae_std", "cv_mape_mean", "cv_mape_std", "cv_r2_mean", "cv_r2_std"],
        limit=6,
    )


def _sales_regression_feature_table(module: object | None) -> str:
    rows = _module_tables(module).get("feature_importance", [])
    return _selected_columns_table(rows, ["feature", "importance", "importance_type"], limit=8)


def _sales_regression_worst_errors(module: object | None) -> str:
    rows = _module_tables(module).get("worst_error_periods", [])
    return _selected_columns_table(
        rows,
        ["period", "actual", "predicted", "absolute_error", "percentage_error"],
        limit=5,
    )


def _sales_regression_intro(
    module: object | None,
    *,
    weak: bool = False,
    output_language: str | None = None,
) -> str:
    if is_english_output(output_language):
        status_note = (
            "When no usable loss-risk model is available, this section attempts a lightweight sales regression; the current regression does not yet provide a stable forecast, so the result is used to assess forecasting difficulty and monitoring value."
            if weak
            else "This section is only used when loss-risk modeling is not suitable. It provides a sales-monitoring baseline, a comparison point for future forecasting models, and a way to inspect abnormal sales movement."
        )
        return (
            "### Modeling Goal, Feature Engineering, and Time Split\n\n"
            "The target is the mapped sales field, and the task is sales forecasting or time-series regression."
            f"{status_note}\n\n"
            + _sales_regression_condition_table(module)
            + "\n\n"
            "Feature engineering uses only lightweight generic signals: calendar features include year, month, quarter, week, and weekday; "
            "lag features include lag_1, lag_2, lag_4, or lag_7; rolling features include rolling_mean and rolling_std. "
            "Lag and rolling features are built after shifting the target, so current or future sales are not leaked into the feature set. "
            "Backtesting uses a chronological split, not a random split. This first version avoids category-share, payment-share, customer, store, or other high-expansion business features that may leak information."
        )
    status_note = (
        "本节在没有可用亏损风险模型时，尝试轻量销售额回归；当前销售额回归模型没有形成稳定预测能力，结果用于判断预测难度和监控参考。"
        if weak
        else "本节只在亏损风险模型不适用时启用，用作销售监控基准、预测模型对照和异常波动参考。"
    )
    return (
        "### 建模目标、特征工程与时间切分\n\n"
        "预测目标是销售额字段，任务类型是销售额预测 / 时间序列回归。"
        f"{status_note}\n\n"
        + _sales_regression_condition_table(module)
        + "\n\n"
        "特征工程只使用通用轻量特征：calendar 特征包含 year / month / quarter / week / weekday，"
        "lag 特征包含 lag_1 / lag_2 / lag_4 或 lag_7，rolling 特征包含 rolling_mean 和 rolling_std。"
        "lag / rolling 均先 shift 后生成，不包含当前期或未来销售额；回测使用时间顺序切分，不使用 random split。"
        "第一版不使用类目占比、支付方式占比、客户特征、门店分组或其它容易膨胀、容易泄漏的业务特征。"
    )


def _sales_regression_top_features(module: object | None, limit: int = 3) -> list[str]:
    features: list[str] = []
    for row in _module_tables(module).get("feature_importance", [])[:limit]:
        feature = str(row.get("feature") or "").strip()
        if feature:
            features.append(feature)
    return features


def _sales_regression_largest_error(module: object | None) -> dict[str, object]:
    rows = _module_tables(module).get("worst_error_periods", [])
    return rows[0] if rows else {}


def _sales_regression_fit_profile(module: object | None) -> tuple[int, int]:
    over_predicted = 0
    under_predicted = 0
    for row in _module_tables(module).get("worst_error_periods", []):
        actual = _as_float(row.get("actual"))
        predicted = _as_float(row.get("predicted"))
        if actual is None or predicted is None:
            continue
        if predicted > actual:
            over_predicted += 1
        elif predicted < actual:
            under_predicted += 1
    return over_predicted, under_predicted


def _sales_regression_takeaway_value(
    modeling_outcome_interpretation: dict[str, str] | None,
    key: str,
) -> str:
    value = _interpretation_value(modeling_outcome_interpretation, key, "")
    prefixes = {
        "model_selection_takeaway": ["模型选择解释：", "模型选择解释:"],
        "cv_stability_takeaway": ["交叉验证稳定性解释：", "交叉验证稳定性解释:", "CV 稳定性解释：", "CV 稳定性解释:"],
        "prediction_fit_takeaway": ["预测效果解释：", "预测效果解释:"],
        "feature_importance_takeaway": ["特征重要性解释：", "特征重要性解释:"],
        "error_analysis_takeaway": ["误差分析解释：", "误差分析解释:"],
        "final_regression_synthesis": ["建模综合结论：", "建模综合结论:"],
    }
    for prefix in prefixes.get(key, []):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
            break
    value = (
        value.replace("月份（week）", "周次（week）")
        .replace("月份 (week)", "周次 (week)")
        .replace("“week”", "'week'")
        .replace("‘week’", "'week'")
    )
    if key != "final_regression_synthesis" and value:
        sentence_count = sum(value.count(mark) for mark in ("。", "！", "？", ".", "!", "?"))
        if len(value) < 60 or sentence_count < 2:
            return ""
    if key == "final_regression_synthesis" and value:
        sentence_count = sum(value.count(mark) for mark in ("。", "！", "？", ".", "!", "?"))
        if len(value) < 120 or sentence_count < 3:
            return ""
    return value


def _sales_regression_cv_stability_takeaway(
    module: object | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    *,
    weak: bool,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    supplied = _sales_regression_takeaway_value(modeling_outcome_interpretation, "cv_stability_takeaway")
    if supplied and not (english and any("\u4e00" <= ch <= "\u9fff" for ch in supplied)):
        return supplied
    summary = _module_summary(module)
    strategy = str(summary.get("regression_cv_strategy") or "")
    best_model = _metric_value(summary, "best_model")
    cv_mae_mean = _metric_value(summary, "cv_mae_mean")
    cv_mae_std = _metric_value(summary, "cv_mae_std")
    cv_mape_mean = _metric_value(summary, "cv_mape_mean")
    cv_r2_mean = _metric_value(summary, "cv_r2_mean")
    if strategy != "TimeSeriesSplit_3" or not _module_tables(module).get("regression_cv_summary"):
        if english:
            return "There are not enough effective periods for a formal TimeSeriesSplit, so this section uses the final chronological holdout window as the main stability reference."
        return "当前有效周期不足以展示正式 TimeSeriesSplit，本节以最后时间窗口 holdout 作为主要稳定性参考。"
    if english:
        if weak:
            return (
                f"TimeSeriesSplit shows {best_model} with cv_mae_mean={cv_mae_mean}, cv_mae_std={cv_mae_std}, "
                f"cv_mape_mean={cv_mape_mean}, and cv_r2_mean={cv_r2_mean}. Under weak regression, these metrics are better interpreted as evidence of forecasting difficulty and cross-window instability, not as proof that the model can be used for planning."
            )
        return (
            f"TimeSeriesSplit shows {best_model} with cv_mae_mean={cv_mae_mean}, cv_mae_std={cv_mae_std}, "
            f"cv_mape_mean={cv_mape_mean}, and cv_r2_mean={cv_r2_mean}. This helps test whether performance only looks good in the final holdout window; if the standard deviation remains high, the model should be treated as a monitoring baseline rather than a stable forecasting tool."
        )
    if weak:
        return (
            f"TimeSeriesSplit 显示 {best_model} 的 cv_mae_mean={cv_mae_mean}、cv_mae_std={cv_mae_std}、"
            f"cv_mape_mean={cv_mape_mean}、cv_r2_mean={cv_r2_mean}。在 weak regression 下，"
            "这些指标更适合说明预测难度和跨窗口不稳定，而不是证明模型可直接用于计划。"
        )
    return (
        f"TimeSeriesSplit 显示 {best_model} 的 cv_mae_mean={cv_mae_mean}、cv_mae_std={cv_mae_std}、"
        f"cv_mape_mean={cv_mape_mean}、cv_r2_mean={cv_r2_mean}。这可以用来判断模型是否只是最后一段回测表现较好；"
        "如果 std 仍偏大，业务使用时仍应把它当作监控基线而不是稳定预测工具。"
    )


def _sales_regression_model_selection_takeaway(
    module: object | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    *,
    weak: bool,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    summary = _module_summary(module)
    best_model = _metric_value(summary, "best_model")
    baseline_model = _metric_value(summary, "baseline_model")
    improvement = _metric_value(summary, "improvement_vs_baseline")
    best_mae = _metric_value(summary, "best_model_mae")
    baseline_mae = _metric_value(summary, "baseline_mae")
    mape = _metric_value(summary, "best_model_mape")
    r2 = _metric_value(summary, "best_model_r2")
    selection_basis = summary.get("selection_basis") if isinstance(summary.get("selection_basis"), dict) else {}
    cv_link_text = _sales_regression_selection_basis_sentence(selection_basis, output_language=output_language)
    supplied = _sales_regression_takeaway_value(modeling_outcome_interpretation, "model_selection_takeaway")
    if supplied and not (english and any("\u4e00" <= ch <= "\u9fff" for ch in supplied)):
        if cv_link_text and not any(term in supplied for term in ["CV", "TimeSeriesSplit", "交叉验证", "cv_mae"]):
            return f"{supplied}{cv_link_text}"
        return supplied
    r2_value = _as_float(r2)
    if english:
        r2_note = (
            "R2 is low, which means the current calendar, lag, and rolling features explain only a small share of sales movement."
            if r2_value is not None and r2_value <= 0.1
            else "R2 is positive, which means the model explains part of the movement in the backtest window."
        )
        if weak:
            return (
                f"The current best model is {best_model}, with {baseline_model} as the baseline, MAE={best_mae}, MAPE={mape}, and R2={r2}. "
                "The more complex models do not consistently outperform the simple baseline, so this regression is closer to a forecasting-difficulty assessment than a ready-to-use forecasting model."
                f"{cv_link_text}"
            )
        return (
            f"The model is selected by holdout MAE: {best_model}, compared with baseline {baseline_model}. "
            f"MAE changes from {baseline_mae} to {best_mae}, with a {improvement}% lift versus baseline. "
            f"{r2_note} {cv_link_text}It can be used as a lightweight monitoring comparison, but the practical value still depends on the business context."
        )
    r2_note = (
        "R2 较低，说明当前 calendar / lag / rolling 特征只能解释一小部分销售波动。"
        if r2_value is not None and r2_value <= 0.1
        else "R2 为正，说明模型对回测窗口中的一部分波动有解释力。"
    )
    if weak:
        return (
            f"当前最佳模型为 {best_model}，baseline 为 {baseline_model}，MAE={best_mae}，MAPE={mape}，R2={r2}。"
            "复杂模型没有稳定超过简单 baseline，说明本轮回归更像预测难度评估，而不是可直接使用的预测模型。"
            f"{cv_link_text}"
        )
    return (
        f"当前按回测 MAE 选择 {best_model}，baseline 为 {baseline_model}；"
        f"MAE 从 {baseline_mae} 变为 {best_mae}，相对 baseline 改善 {improvement}%。"
        f"{r2_note} {cv_link_text}因此它可以作为轻量监控对照，但提升幅度仍需要结合业务场景判断。"
    )


def _sales_regression_selection_basis_sentence(
    selection_basis: object,
    *,
    output_language: str | None = None,
) -> str:
    if not isinstance(selection_basis, dict) or not selection_basis:
        return ""
    holdout_rank = selection_basis.get("holdout_mae_rank")
    cv_rank = selection_basis.get("cv_mae_rank")
    cv_best_model = selection_basis.get("cv_best_model")
    stability_note = str(selection_basis.get("cv_stability_note") or "")
    if is_english_output(output_language):
        if cv_rank is None:
            return " CV evidence is insufficient, so the selection mainly comes from the final holdout window."
        if stability_note in {"holdout_and_cv_aligned", "cv_mean_best_but_variance_noticeable"}:
            return (
                f" CV is also part of the selection: holdout MAE rank is {holdout_rank}, "
                f"CV MAE rank is {cv_rank}, which suggests the candidate model is broadly aligned with cross-window averages."
            )
        if stability_note == "holdout_best_cv_near_top":
            return (
                f" CV is also part of the selection: holdout MAE rank is {holdout_rank}, "
                f"CV MAE rank is {cv_rank}; the model is not the most stable option but remains close to the best."
            )
        if stability_note == "holdout_best_cv_not_stable":
            return (
                f" CV is also part of the selection: holdout MAE rank is {holdout_rank}, "
                f"but CV MAE rank is {cv_rank} and the CV-best model is {cv_best_model}, so the conclusion should be downgraded."
            )
        return (
            f" CV is also part of the selection: holdout MAE rank is {holdout_rank}, "
            f"CV MAE rank is {cv_rank}, so stability should still be interpreted conservatively."
        )
    if cv_rank is None:
        return " CV 证据不足，当前选择主要来自最后 holdout 窗口。"
    if stability_note in {"holdout_and_cv_aligned", "cv_mean_best_but_variance_noticeable"}:
        return (
            f" CV 也参与了选择判断：holdout MAE 排名第 {holdout_rank}，"
            f"CV MAE 排名第 {cv_rank}，说明候选模型与跨窗口均值基本一致。"
        )
    if stability_note == "holdout_best_cv_near_top":
        return (
            f" CV 也参与了选择判断：holdout MAE 排名第 {holdout_rank}，"
            f"CV MAE 排名第 {cv_rank}，不是最稳但仍接近最优。"
        )
    if stability_note == "holdout_best_cv_not_stable":
        return (
            f" CV 也参与了选择判断：holdout MAE 排名第 {holdout_rank}，"
            f"但 CV MAE 排名第 {cv_rank}，CV 最优模型为 {cv_best_model}，因此结论需要降级。"
        )
    return (
        f" CV 也参与了选择判断：holdout MAE 排名第 {holdout_rank}，"
        f"CV MAE 排名第 {cv_rank}，稳定性仍需保守解读。"
    )


def _sales_regression_prediction_fit_takeaway(
    module: object | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    *,
    weak: bool,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    supplied = _sales_regression_takeaway_value(modeling_outcome_interpretation, "prediction_fit_takeaway")
    if supplied and not (english and any("\u4e00" <= ch <= "\u9fff" for ch in supplied)):
        return supplied
    over_predicted, under_predicted = _sales_regression_fit_profile(module)
    if english:
        if weak:
            return (
                "Actual vs Best Model is mainly used to inspect forecasting difficulty: when the fitted line cannot follow sudden peaks or dips, historical sales inertia alone is not enough for stable forecasting."
            )
        if over_predicted > under_predicted:
            bias = "The largest-error periods show more over-prediction, so low-demand or pullback periods may be missing explanatory variables."
        elif under_predicted > over_predicted:
            bias = "The largest-error periods show more under-prediction, so peaks, promotions, or event periods may be missing explanatory variables."
        else:
            bias = "The error direction is not one-sided, so the review should focus on how well peaks and troughs are captured."
        return (
            "Actual vs Best Model checks whether the model follows the overall backtest-window movement. "
            f"{bias} This chart is not evidence of production-grade forecasting; it is a monitoring reference and a baseline for future model improvements."
        )
    if weak:
        return (
            "Actual vs Best Model 主要用于观察预测难度：当曲线难以跟上突发峰值或低谷时，"
            "说明仅靠历史销售惯性不足以支撑稳定预测。"
        )
    if over_predicted > under_predicted:
        bias = "误差较大的周期里高估更多，需要关注低谷或回落期是否缺少解释变量。"
    elif under_predicted > over_predicted:
        bias = "误差较大的周期里低估更多，需要关注峰值、促销或活动期是否缺少解释变量。"
    else:
        bias = "误差方向没有明显单边偏差，重点应放在峰值和低谷的捕捉能力。"
    return (
        "Actual vs Best Model 图用于判断模型是否跟得上回测窗口的整体走势。"
        f"{bias} 这张图不用于证明生产级预测，只用于监控参照和后续模型增强对照。"
    )


def _sales_regression_feature_takeaway(
    module: object | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    supplied = _sales_regression_takeaway_value(modeling_outcome_interpretation, "feature_importance_takeaway")
    if supplied and not (english and any("\u4e00" <= ch <= "\u9fff" for ch in supplied)):
        return supplied
    top_features = _sales_regression_top_features(module)
    if english:
        top_text = ", ".join(top_features) if top_features else "no stable top features"
        return (
            f"The current top features are {top_text}. Calendar features represent seasonality, lag features represent historical sales inertia, and rolling features represent short-term averages and volatility. These signals show that the model mainly relies on historical continuation and seasonal movement. They are not business causes such as promotions, holidays, stores, or product mix, so they should not be framed as causal explanations."
        )
    top_text = "、".join(top_features) if top_features else "暂无稳定 Top features"
    return (
        f"当前 Top features 为 {top_text}。calendar 特征代表时间周期，lag 特征代表历史销售惯性，"
        "rolling 特征代表短期均值和波动；这些信号说明模型主要在利用历史走势延续和周期性变化。"
        "它们不是促销、节假日、门店或商品结构等业务原因本身，因此不应包装成强因果解释。"
    )


def _sales_regression_error_takeaway(
    module: object | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    *,
    weak: bool,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    supplied = _sales_regression_takeaway_value(modeling_outcome_interpretation, "error_analysis_takeaway")
    if supplied and not (english and any("\u4e00" <= ch <= "\u9fff" for ch in supplied)):
        return supplied
    largest = _sales_regression_largest_error(module)
    period = largest.get("period", "within the backtest window" if english else "回测窗口内")
    actual = largest.get("actual", "unknown")
    predicted = largest.get("predicted", "unknown")
    absolute_error = largest.get("absolute_error", "unknown")
    percentage_error = largest.get("percentage_error", "unknown")
    largest_actual = _as_float(actual)
    largest_predicted = _as_float(predicted)
    if english:
        if largest_actual is not None and largest_predicted is not None and largest_predicted > largest_actual:
            largest_direction = "This period is over-predicted, which means the model estimated sales too high."
        elif largest_actual is not None and largest_predicted is not None and largest_predicted < largest_actual:
            largest_direction = "This period is under-predicted, which means the model did not keep up with a sales peak."
        else:
            largest_direction = "This period has no clear over- or under-prediction direction; the main issue is high absolute error."
        over_predicted, under_predicted = _sales_regression_fit_profile(module)
        if over_predicted > under_predicted:
            direction = "The top-error periods lean toward over-prediction, suggesting weak recognition of pullbacks or troughs."
        elif under_predicted > over_predicted:
            direction = "The top-error periods lean toward under-prediction, suggesting weak capture of spikes or sudden growth."
        else:
            direction = "The top-error periods do not have a clear one-sided direction, so both peaks and troughs may need additional explanatory variables."
        weak_note = (
            "Under weak regression, these errors should be treated as evidence of forecasting difficulty."
            if weak
            else "These errors show that business variables are still needed before the model can support detailed sales planning."
        )
        return (
            f"The largest-error period is {period}, with actual={actual}, predicted={predicted}, "
            f"absolute_error={absolute_error}, and percentage_error={percentage_error}. "
            f"{largest_direction} {direction} If these periods correspond to promotions, holidays, replenishment, campaigns, or abnormal orders, lag and rolling features alone are unlikely to explain the movement. {weak_note}"
        )
    if largest_actual is not None and largest_predicted is not None and largest_predicted > largest_actual:
        largest_direction = "该单期属于高估，说明模型把这一期销售额预测得偏高。"
    elif largest_actual is not None and largest_predicted is not None and largest_predicted < largest_actual:
        largest_direction = "该单期属于低估，说明模型没有跟上这一期销售峰值。"
    else:
        largest_direction = "该单期没有明显高估或低估方向，主要体现为绝对误差较大。"
    over_predicted, under_predicted = _sales_regression_fit_profile(module)
    if over_predicted > under_predicted:
        direction = "误差 Top 5 中高估偏多，说明模型对销售回落或低谷的识别不足。"
    elif under_predicted > over_predicted:
        direction = "误差 Top 5 中低估偏多，说明模型对销售尖峰或突发增长的捕捉不足。"
    else:
        direction = "误差 Top 5 没有明显单边方向，说明峰值和低谷都可能需要额外解释变量。"
    weak_note = (
        "在 weak regression 下，这些误差更应被看作预测难度证据。"
        if weak
        else "这些误差说明模型仍需要业务变量增强，不能直接用于精细销售计划。"
    )
    return (
        f"误差最大的周期是 {period}，actual={actual}，predicted={predicted}，"
        f"absolute_error={absolute_error}，percentage_error={percentage_error}。"
        f"{largest_direction}{direction} 如果这些周期对应促销、节假日、补货、活动或异常订单，"
        f"仅靠 lag / rolling 特征很难解释这些变化。{weak_note}"
    )


def _sales_regression_final_synthesis(
    module: object | None,
    modeling_outcome_interpretation: dict[str, str] | None,
    *,
    weak: bool,
    output_language: str | None = None,
) -> str:
    english = is_english_output(output_language)
    summary = _module_summary(module)
    best_model = _metric_value(summary, "best_model")
    baseline_model = _metric_value(summary, "baseline_model")
    improvement = _metric_value(summary, "improvement_vs_baseline")
    mape = _metric_value(summary, "best_model_mape")
    r2 = _metric_value(summary, "best_model_r2")
    top_features = _sales_regression_top_features(module, limit=2)
    feature_text = ", ".join(top_features) if english and top_features else ("historical sales features" if english else ("、".join(top_features) if top_features else "历史销售特征"))
    selection_basis = summary.get("selection_basis") if isinstance(summary.get("selection_basis"), dict) else {}
    cv_link_text = _sales_regression_selection_basis_sentence(selection_basis, output_language=output_language)
    supplied = _sales_regression_takeaway_value(modeling_outcome_interpretation, "final_regression_synthesis")
    if supplied and not (english and any("\u4e00" <= ch <= "\u9fff" for ch in supplied)):
        text = supplied
    elif english and weak:
        text = (
            f"Overall, the current best model is {best_model}, with MAPE={mape} and R2={r2}; it has not produced stable forecasting capability. "
            "The value of this run is to show that the current data is hard to forecast and is better used as a monitoring reference and a basis for future data enrichment. "
            "Next, add promotion, holiday, replenishment, campaign, store, or product-mix variables before reassessing forecasting value."
        )
    elif english:
        text = (
            f"Overall, {best_model} improves MAE versus {baseline_model} by {improvement}% and mainly relies on {feature_text}. "
            "The model has some monitoring value and can serve as a comparison baseline for future improvements. "
            f"However, current R2={r2}, and errors may still concentrate around peaks or troughs, so the model does not yet explain major business shocks. "
            "Next, add promotion, holiday, replenishment, campaign, store, or product-mix variables."
        )
    elif weak:
        text = (
            f"综合来看，当前最佳模型为 {best_model}，MAPE={mape}，R2={r2}，"
            "模型没有形成稳定预测能力。本轮结果的价值在于说明当前数据预测难度高，"
            "更适合作为监控参照和后续数据补强依据。"
            "下一步应优先补充促销、节假日、补货、活动、门店或商品结构等业务变量。"
        )
    else:
        text = (
            f"综合来看，{best_model} 相对 {baseline_model} 的 MAE 改善为 {improvement}%，"
            f"主要依赖 {feature_text} 等历史走势信号。该模型有一定参考价值，"
            "适合作为销售监控和后续模型增强的对照基线。"
            f"但当前 R2={r2}，且误差仍可能集中在销售峰值或低谷，说明它还不能解释关键业务冲击。"
            "下一步应补充促销、节假日、补货、活动、门店或商品结构等变量。"
        )
    if english:
        if not any(term in text.lower() for term in ["promotion", "holiday", "replenishment", "product", "store"]):
            text = f"{text} Next, add promotion, holiday, replenishment, campaign, store, or product-mix variables and test whether they reduce extreme-period errors."
        if cv_link_text and not any(term in text for term in ["CV", "TimeSeriesSplit", "cv_mae"]):
            text = f"{text}{cv_link_text}"
        if not weak and not any(term in text.lower() for term in ["error", "peak", "trough"]):
            text = f"{text} Based on the top-error periods, the next review should focus on whether peaks or troughs come from campaigns, abnormal orders, or replenishment timing."
        if weak and "sales planning" not in text:
            text = f"{text} It should not be used for sales planning, inventory, replenishment, or business target setting."
        if "production-grade forecast" not in text and "automatic decisions" not in text:
            text = f"{text} Treat it as an analytical baseline, not a production-grade forecast, and do not use it for automatic decisions."
        return text
    if not any(term in text for term in ["促销", "节假日", "补货", "商品结构", "门店"]):
        text = f"{text}下一步应补充促销、节假日、补货、活动、门店或商品结构等业务变量，再评估预测能力是否真正提升。"
    if cv_link_text and not any(term in text for term in ["CV", "TimeSeriesSplit", "交叉验证", "跨窗口", "cv_mae"]):
        text = f"{text}{cv_link_text}"
    if not weak and "误差" not in text and "峰值" not in text and "低谷" not in text:
        text = f"{text}结合误差 Top 5，后续尤其需要关注销售峰值或低谷是否来自活动、异常订单或补货节奏。"
    text = text.replace("预测误差较大", "极端周期误差明显、跨窗口稳定性一般")
    text = text.replace(
        "并尝试更稳定的模型架构以提升预测价值",
        "先验证这些外部变量是否能降低极端周期误差，再考虑是否需要更复杂模型",
    )
    if weak:
        required = "不建议用于销售计划、库存、补货或经营目标制定"
        if required not in text:
            mentions_sales_plan_limit = any(
                term in text
                for term in [
                    "不建议用于销售计划",
                    "不适合直接用于销售计划",
                    "不适合用于销售计划",
                    "不具有业务计划",
                    "业务计划",
                ]
            )
            if not mentions_sales_plan_limit:
                text = f"{text}{required}。"
    if "生产级预测" not in text and "不用于自动决策" not in text:
        text = f"{text}使用上应视为分析基线，不是生产级预测，也不用于自动决策。"
    elif "生产级预测" not in text:
        text = f"{text}使用上应视为分析基线，不是生产级预测。"
    elif "不用于自动决策" not in text:
        text = f"{text}使用上仍需人工判断，不用于自动决策。"
    return text


def _sales_regression_code_config(module: object | None) -> dict[str, object]:
    summary = _module_summary(module)
    granularity = str(summary.get("series_granularity") or "daily")
    default_lags = [1, 2, 7] if granularity == "daily" else [1, 2, 4]
    default_rolling = 7 if granularity == "daily" else 4
    return {
        "date_column": summary.get("regression_date_column") or summary.get("date_column") or "",
        "sales_amount_column": summary.get("regression_sales_amount_column") or summary.get("sales_amount_column") or "",
        "series_granularity": granularity,
        "backtest_periods": summary.get("regression_backtest_period_count")
        or summary.get("backtest_period_count")
        or 7,
        "lag_periods": summary.get("regression_lag_periods") or default_lags,
        "rolling_window": summary.get("regression_rolling_window") or default_rolling,
        "min_train_periods": 20,
        "rf_params": summary.get("regression_random_forest_params")
        or {
            "n_estimators": 60,
            "max_depth": 8,
            "min_samples_leaf": 3,
            "random_state": 42,
        },
    }


def _sales_regression_config_code(module: object | None) -> str:
    return "sales_regression_config = " + json.dumps(
        _sales_regression_code_config(module),
        ensure_ascii=False,
        indent=2,
    )


def _sales_regression_feature_engineering_code(module: object | None) -> str:
    return (
        "import math\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from IPython.display import display\n\n"
        + _sales_regression_config_code(module)
        + "\n\n"
        "def _g6_parse_datetime(values):\n"
        "    try:\n"
        "        return pd.to_datetime(values, errors=\"coerce\", format=\"mixed\")\n"
        "    except TypeError:\n"
        "        return pd.to_datetime(values, errors=\"coerce\")\n\n"
        "def _g6_pick_column(frame, preferred, candidates):\n"
        "    if preferred and preferred in frame.columns:\n"
        "        return preferred\n"
        "    normalized = {str(column).strip().lower(): column for column in frame.columns}\n"
        "    for candidate in candidates:\n"
        "        key = str(candidate).strip().lower()\n"
        "        if key in normalized:\n"
        "            return normalized[key]\n"
        "    return None\n\n"
        "candidate_frames = [globals().get(\"clean_df\"), globals().get(\"df\")]\n"
        "source_df = next((frame for frame in candidate_frames if isinstance(frame, pd.DataFrame)), None)\n"
        "if source_df is None:\n"
        "    source_df = pd.read_csv(\"raw.csv\")\n\n"
        "date_col = _g6_pick_column(\n"
        "    source_df,\n"
        "    sales_regression_config.get(\"date_column\"),\n"
        "    [\"Date\", \"Order Date\", \"order_date\", \"order_datetime\", \"InvoiceDate\", \"date\"],\n"
        ")\n"
        "sales_col = _g6_pick_column(\n"
        "    source_df,\n"
        "    sales_regression_config.get(\"sales_amount_column\"),\n"
        "    [\"Weekly_Sales\", \"Sales\", \"sales\", \"sales_amount\", \"Amount\", \"amount\", \"Revenue\", \"Total\"],\n"
        ")\n"
        "if date_col is None or sales_col is None:\n"
        "    raise ValueError(\"销售额回归需要日期字段和销售额字段；当前 notebook 未找到可用列。\")\n\n"
        "working = source_df[[date_col, sales_col]].copy()\n"
        "working[\"period\"] = _g6_parse_datetime(working[date_col]).dt.normalize()\n"
        "working[\"sales_amount\"] = pd.to_numeric(working[sales_col], errors=\"coerce\")\n"
        "working = working.dropna(subset=[\"period\", \"sales_amount\"])\n"
        "sales_series = (\n"
        "    working.groupby(\"period\", as_index=False)[\"sales_amount\"].sum().sort_values(\"period\")\n"
        ")\n\n"
        "feature_df = sales_series.copy()\n"
        "feature_df = feature_df.rename(columns={\"sales_amount\": \"target\"})\n"
        "periods = pd.to_datetime(feature_df[\"period\"], errors=\"coerce\")\n"
        "feature_df[\"year\"] = periods.dt.year\n"
        "feature_df[\"month\"] = periods.dt.month\n"
        "feature_df[\"quarter\"] = periods.dt.quarter\n"
        "feature_df[\"week\"] = periods.dt.isocalendar().week.astype(int)\n"
        "feature_df[\"weekday\"] = periods.dt.weekday\n\n"
        "target = feature_df[\"target\"].astype(float)\n"
        "lag_periods = [int(lag) for lag in sales_regression_config.get(\"lag_periods\", [1, 2, 4])]\n"
        "rolling_window = int(sales_regression_config.get(\"rolling_window\", 4))\n"
        "for lag in lag_periods:\n"
        "    feature_df[f\"lag_{lag}\"] = target.shift(lag)\n"
        "rolling_source = target.shift(1)\n"
        "feature_df[f\"rolling_mean_{rolling_window}\"] = rolling_source.rolling(rolling_window).mean()\n"
        "feature_df[f\"rolling_std_{rolling_window}\"] = rolling_source.rolling(rolling_window).std()\n\n"
        "feature_df = feature_df.dropna().reset_index(drop=True)\n"
        "feature_cols = [column for column in feature_df.columns if column not in {\"period\", \"target\"}]\n"
        "feature_summary = pd.DataFrame([\n"
        "    {\n"
        "        \"date_column\": date_col,\n"
        "        \"sales_amount_column\": sales_col,\n"
        "        \"granularity\": sales_regression_config.get(\"series_granularity\"),\n"
        "        \"raw_period_count\": len(sales_series),\n"
        "        \"effective_sample_count\": len(feature_df),\n"
        "        \"feature_count\": len(feature_cols),\n"
        "        \"lag_features\": \", \".join(f\"lag_{lag}\" for lag in lag_periods),\n"
        "        \"rolling_features\": f\"rolling_mean_{rolling_window}, rolling_std_{rolling_window}\",\n"
        "        \"leakage_guard\": \"lag / rolling features use target.shift(...) before calculation\",\n"
        "    }\n"
        "])\n"
        "display(feature_df.head())\n"
        "display(feature_summary)\n"
    )


def _sales_regression_time_split_code(
    module: object | None,
    output_language: str | None = None,
) -> str:
    insufficient_message = (
        "Not enough effective samples for a chronological holdout."
        if is_english_output(output_language)
        else "有效样本不足，无法完成 chronological holdout。"
    )
    return (
        _sales_regression_config_code(module)
        + "\n\n"
        "backtest_periods = int(sales_regression_config.get(\"backtest_periods\", 7))\n"
        "backtest_periods = max(1, min(backtest_periods, len(feature_df) - 1))\n"
        "min_train_periods = int(sales_regression_config.get(\"min_train_periods\", 20))\n"
        "if len(feature_df) - backtest_periods < min_train_periods:\n"
        "    backtest_periods = max(1, len(feature_df) - min_train_periods)\n"
        "if backtest_periods <= 0:\n"
        f"    raise ValueError({insufficient_message!r})\n\n"
        "train_df = feature_df.iloc[:-backtest_periods].copy()\n"
        "test_df = feature_df.iloc[-backtest_periods:].copy()\n"
        "split_summary = pd.DataFrame([\n"
        "    {\n"
        "        \"train_start\": train_df[\"period\"].iloc[0],\n"
        "        \"train_end\": train_df[\"period\"].iloc[-1],\n"
        "        \"test_start\": test_df[\"period\"].iloc[0],\n"
        "        \"test_end\": test_df[\"period\"].iloc[-1],\n"
        "        \"train_period_count\": len(train_df),\n"
        "        \"test_period_count\": len(test_df),\n"
        "        \"split_method\": \"chronological_holdout\",\n"
        "    }\n"
        "])\n"
        "display(split_summary)\n"
    )


def _sales_regression_cv_code(
    module: object | None,
    output_language: str | None = None,
) -> str:
    skip_message = (
        "Effective samples are below 80, so formal TimeSeriesSplit is skipped and only the chronological holdout is kept."
        if is_english_output(output_language)
        else "有效样本少于 80，跳过正式 TimeSeriesSplit，仅保留 chronological holdout。"
    )
    return (
        "from sklearn.base import clone\n"
        "from sklearn.ensemble import RandomForestRegressor\n"
        "from sklearn.linear_model import Ridge\n"
        "from sklearn.metrics import r2_score\n"
        "from sklearn.model_selection import TimeSeriesSplit\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.preprocessing import StandardScaler\n\n"
        + _sales_regression_config_code(module)
        + "\n\n"
        "def _g6_mape(actual, predicted):\n"
        "    actual = pd.Series(actual).astype(float)\n"
        "    predicted = pd.Series(predicted, index=actual.index).astype(float)\n"
        "    non_zero = actual[actual.abs() > 1e-9]\n"
        "    if non_zero.empty:\n"
        "        return None\n"
        "    return float(((non_zero - predicted.loc[non_zero.index]).abs() / non_zero.abs()).mean() * 100)\n\n"
        "def _g6_r2(actual, predicted):\n"
        "    actual = pd.Series(actual).astype(float)\n"
        "    predicted = pd.Series(predicted, index=actual.index).astype(float)\n"
        "    if len(actual) < 2 or float(actual.var()) <= 1e-12:\n"
        "        return None\n"
        "    score = float(r2_score(actual, predicted))\n"
        "    return score if math.isfinite(score) else None\n\n"
        "def _g6_regression_metric_row(model, actual, predicted, baseline_mae):\n"
        "    actual = pd.Series(actual).astype(float)\n"
        "    predicted = pd.Series(predicted, index=actual.index).astype(float)\n"
        "    errors = actual - predicted\n"
        "    mae = float(errors.abs().mean())\n"
        "    rmse = float(math.sqrt((errors ** 2).mean()))\n"
        "    mape = _g6_mape(actual, predicted)\n"
        "    r2 = _g6_r2(actual, predicted)\n"
        "    improvement = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 1e-9 else 0.0\n"
        "    return {\n"
        "        \"model\": model,\n"
        "        \"mae\": round(mae, 2),\n"
        "        \"rmse\": round(rmse, 2),\n"
        "        \"mape\": round(mape, 2) if mape is not None else None,\n"
        "        \"r2\": round(r2, 4) if r2 is not None else None,\n"
        "        \"improvement_vs_baseline\": round(improvement, 2),\n"
        "    }\n\n"
        "rf_params = dict(sales_regression_config.get(\"rf_params\", {}))\n"
        "estimators = {\n"
        "    \"Ridge Regression\": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),\n"
        "    \"RandomForestRegressor-small\": RandomForestRegressor(\n"
        "        n_estimators=int(rf_params.get(\"n_estimators\", 60)),\n"
        "        max_depth=rf_params.get(\"max_depth\", 8),\n"
        "        min_samples_leaf=int(rf_params.get(\"min_samples_leaf\", 3)),\n"
        "        random_state=int(rf_params.get(\"random_state\", 42)),\n"
        "        n_jobs=1,\n"
        "    ),\n"
        "}\n\n"
        "cv_rows = []\n"
        "if len(feature_df) >= 80:\n"
        "    splitter = TimeSeriesSplit(n_splits=3)\n"
        "    for fold, (train_index, valid_index) in enumerate(splitter.split(feature_df), start=1):\n"
        "        train_fold = feature_df.iloc[train_index]\n"
        "        valid_fold = feature_df.iloc[valid_index]\n"
        "        actual = valid_fold[\"target\"].astype(float)\n"
        "        train_target = train_fold[\"target\"].astype(float)\n"
        "        fold_predictions = {\n"
        "            \"naive_last_value\": pd.Series(float(train_target.iloc[-1]), index=actual.index),\n"
        "            \"moving_average_7\": pd.Series(float(train_target.tail(7).mean()), index=actual.index),\n"
        "        }\n"
        "        for model_name, estimator in estimators.items():\n"
        "            fitted = clone(estimator)\n"
        "            fitted.fit(train_fold[feature_cols], train_fold[\"target\"])\n"
        "            fold_predictions[model_name] = pd.Series(fitted.predict(valid_fold[feature_cols]), index=actual.index)\n"
        "        for model_name, predicted in fold_predictions.items():\n"
        "            metrics = _g6_regression_metric_row(model_name, actual, predicted, baseline_mae=0)\n"
        "            cv_rows.append({\n"
        "                \"fold\": fold,\n"
        "                \"train_start\": train_fold[\"period\"].iloc[0],\n"
        "                \"train_end\": train_fold[\"period\"].iloc[-1],\n"
        "                \"valid_start\": valid_fold[\"period\"].iloc[0],\n"
        "                \"valid_end\": valid_fold[\"period\"].iloc[-1],\n"
        "                \"model\": model_name,\n"
        "                \"mae\": metrics[\"mae\"],\n"
        "                \"rmse\": metrics[\"rmse\"],\n"
        "                \"mape\": metrics[\"mape\"],\n"
        "                \"r2\": metrics[\"r2\"],\n"
        "            })\n"
        "else:\n"
        f"    print({skip_message!r})\n\n"
        "cv_folds = pd.DataFrame(cv_rows, columns=[\n"
        "    \"fold\", \"train_start\", \"train_end\", \"valid_start\", \"valid_end\", \"model\", \"mae\", \"rmse\", \"mape\", \"r2\"\n"
        "])\n"
        "summary_rows = []\n"
        "for model_name, group in cv_folds.groupby(\"model\") if not cv_folds.empty else []:\n"
        "    row = {\"model\": model_name}\n"
        "    for metric in [\"mae\", \"mape\", \"r2\"]:\n"
        "        values = pd.to_numeric(group[metric], errors=\"coerce\").dropna()\n"
        "        row[f\"cv_{metric}_mean\"] = round(float(values.mean()), 4 if metric == \"r2\" else 2) if not values.empty else None\n"
        "        row[f\"cv_{metric}_std\"] = round(float(values.std(ddof=0)), 4 if metric == \"r2\" else 2) if not values.empty else None\n"
        "    summary_rows.append(row)\n"
        "cv_summary = pd.DataFrame(summary_rows, columns=[\n"
        "    \"model\", \"cv_mae_mean\", \"cv_mae_std\", \"cv_mape_mean\", \"cv_mape_std\", \"cv_r2_mean\", \"cv_r2_std\"\n"
        "])\n"
        "display(cv_summary)\n"
    )


def _sales_regression_cv_chart_code(output_language: str | None = None) -> str:
    english = is_english_output(output_language)
    skip_message = (
        "No TimeSeriesSplit CV summary is available, so the CV stability chart was skipped."
        if english
        else "暂无 TimeSeriesSplit CV 汇总，跳过 CV 稳定性对比图。"
    )
    title = (
        "TimeSeriesSplit CV Model Stability Comparison"
        if english
        else "TimeSeriesSplit CV 模型稳定性对比"
    )
    return (
        _modeling_plot_setup_code()
        + "\nif cv_summary.empty:\n"
        f"    print({skip_message!r})\n"
        "else:\n"
        "    plot_df = cv_summary.copy()\n"
        "    y_col = \"cv_mape_mean\" if plot_df[\"cv_mape_mean\"].notna().any() else \"cv_mae_mean\"\n"
        "    err_col = \"cv_mape_std\" if y_col == \"cv_mape_mean\" else \"cv_mae_std\"\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.4))\n"
        "    ax.bar(plot_df[\"model\"], plot_df[y_col], color=\"#2563EB\", alpha=0.82)\n"
        "    if plot_df[err_col].notna().any():\n"
        "        ax.errorbar(plot_df[\"model\"], plot_df[y_col], yerr=plot_df[err_col].fillna(0), fmt=\"none\", ecolor=\"#111827\", capsize=4)\n"
        f"    ax.set_title({title!r})\n"
        "    ax.set_xlabel(\"model\")\n"
        "    ax.set_ylabel(y_col)\n"
        "    ax.tick_params(axis=\"x\", rotation=20)\n"
        "    ax.grid(axis=\"y\", alpha=0.25)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _sales_regression_model_comparison_code(output_language: str | None = None) -> str:
    english = is_english_output(output_language)
    reason_aligned = (
        "{best_model} ranks first by holdout MAE and is also near the top by CV MAE; CV variance should determine how strongly this result is used."
        if english
        else "{best_model} 的 holdout MAE 排名第 1，CV MAE 排名也靠前；CV 波动决定结论强弱。"
    )
    reason_near_top = (
        "{best_model} has the lowest holdout MAE and a CV MAE rank close to the best; use it as a monitoring model while keeping stability caveats."
        if english
        else "{best_model} 的 holdout MAE 最低，CV MAE 排名接近最优；可作为监控模型但稳定性需保守解读。"
    )
    reason_not_stable = (
        "{best_model} has the lowest holdout MAE, but CV does not confirm the same advantage; downgrade the conclusion to a monitoring reference."
        if english
        else "{best_model} 的 holdout MAE 最低，但 CV 未确认同等优势；结论需要降级为监控参照。"
    )
    reason_unavailable = (
        "{best_model} is selected by holdout MAE, and there is not enough CV evidence to confirm stability."
        if english
        else "{best_model} 由 holdout MAE 选择，当前没有足够 CV 证据确认稳定性。"
    )
    return (
        "actual = test_df[\"target\"].astype(float)\n"
        "baseline_predictions = {\n"
        "    \"naive_last_value\": pd.Series(float(train_df[\"target\"].iloc[-1]), index=actual.index),\n"
        "    \"moving_average_7\": pd.Series(float(train_df[\"target\"].tail(7).mean()), index=actual.index),\n"
        "}\n"
        "baseline_rows = [\n"
        "    _g6_regression_metric_row(model_name, actual, predicted, baseline_mae=0)\n"
        "    for model_name, predicted in baseline_predictions.items()\n"
        "]\n"
        "baseline_mae = min(row[\"mae\"] for row in baseline_rows)\n"
        "comparison_rows = []\n"
        "model_predictions = {}\n"
        "for row in baseline_rows:\n"
        "    row = {**row, \"model_type\": \"baseline\"}\n"
        "    row[\"improvement_vs_baseline\"] = round((baseline_mae - row[\"mae\"]) / baseline_mae * 100, 2) if baseline_mae > 1e-9 else 0.0\n"
        "    comparison_rows.append(row)\n"
        "for model_name, predicted in baseline_predictions.items():\n"
        "    model_predictions[model_name] = predicted\n\n"
        "fitted_estimators = {}\n"
        "for model_name, estimator in estimators.items():\n"
        "    fitted = clone(estimator)\n"
        "    fitted.fit(train_df[feature_cols], train_df[\"target\"])\n"
        "    predicted = pd.Series(fitted.predict(test_df[feature_cols]), index=actual.index)\n"
        "    row = _g6_regression_metric_row(model_name, actual, predicted, baseline_mae=baseline_mae)\n"
        "    row = {**row, \"model_type\": \"regression\"}\n"
        "    comparison_rows.append(row)\n"
        "    model_predictions[model_name] = predicted\n"
        "    fitted_estimators[model_name] = fitted\n\n"
        "model_comparison = pd.DataFrame(comparison_rows)\n"
        "model_comparison = model_comparison[[\"model\", \"model_type\", \"mae\", \"rmse\", \"mape\", \"r2\", \"improvement_vs_baseline\"]]\n"
        "model_comparison[\"holdout_mae_rank\"] = model_comparison[\"mae\"].rank(method=\"min\", ascending=True).astype(int)\n"
        "if not cv_summary.empty and \"cv_mae_mean\" in cv_summary.columns:\n"
        "    cv_rank_lookup = cv_summary.assign(\n"
        "        cv_mae_rank=pd.to_numeric(cv_summary[\"cv_mae_mean\"], errors=\"coerce\").rank(method=\"min\", ascending=True)\n"
        "    ).set_index(\"model\")[\"cv_mae_rank\"]\n"
        "    model_comparison[\"cv_mae_rank\"] = model_comparison[\"model\"].map(cv_rank_lookup)\n"
        "else:\n"
        "    model_comparison[\"cv_mae_rank\"] = np.nan\n"
        "best_row = model_comparison.sort_values(\"mae\", ascending=True).iloc[0]\n"
        "best_model = str(best_row[\"model\"])\n"
        "cv_best_model = None\n"
        "selected_cv_row = pd.DataFrame()\n"
        "if not cv_summary.empty and \"cv_mae_mean\" in cv_summary.columns:\n"
        "    valid_cv = cv_summary.dropna(subset=[\"cv_mae_mean\"]).sort_values(\"cv_mae_mean\", ascending=True)\n"
        "    if not valid_cv.empty:\n"
        "        cv_best_model = str(valid_cv.iloc[0][\"model\"])\n"
        "        selected_cv_row = cv_summary[cv_summary[\"model\"] == best_model]\n"
        "selected_cv_mean = selected_cv_row[\"cv_mae_mean\"].iloc[0] if not selected_cv_row.empty else np.nan\n"
        "selected_cv_std = selected_cv_row[\"cv_mae_std\"].iloc[0] if not selected_cv_row.empty else np.nan\n"
        "selected_cv_rank = best_row.get(\"cv_mae_rank\")\n"
        "if pd.notna(selected_cv_rank):\n"
        "    cv_ratio = selected_cv_std / selected_cv_mean if pd.notna(selected_cv_mean) and selected_cv_mean > 1e-9 and pd.notna(selected_cv_std) else np.nan\n"
        "    if int(selected_cv_rank) == 1 and (pd.isna(cv_ratio) or cv_ratio <= 0.2):\n"
        "        cv_stability_note = \"holdout_and_cv_aligned\"\n"
        "    elif int(selected_cv_rank) == 1:\n"
        "        cv_stability_note = \"cv_mean_best_but_variance_noticeable\"\n"
        "    elif int(selected_cv_rank) <= 2:\n"
        "        cv_stability_note = \"holdout_best_cv_near_top\"\n"
        "    else:\n"
        "        cv_stability_note = \"holdout_best_cv_not_stable\"\n"
        "else:\n"
        "    cv_stability_note = \"cv_unavailable\"\n"
        "if cv_stability_note in {\"holdout_and_cv_aligned\", \"cv_mean_best_but_variance_noticeable\"}:\n"
        f"    selected_reason = f{reason_aligned!r}\n"
        "elif cv_stability_note == \"holdout_best_cv_near_top\":\n"
        f"    selected_reason = f{reason_near_top!r}\n"
        "elif cv_stability_note == \"holdout_best_cv_not_stable\":\n"
        f"    selected_reason = f{reason_not_stable!r}\n"
        "else:\n"
        f"    selected_reason = f{reason_unavailable!r}\n"
        "selection_basis = pd.DataFrame([\n"
        "    {\n"
        "        \"selected_model\": best_model,\n"
        "        \"holdout_mae_rank\": int(best_row[\"holdout_mae_rank\"]),\n"
        "        \"cv_mae_rank\": int(selected_cv_rank) if pd.notna(selected_cv_rank) else None,\n"
        "        \"cv_best_model\": cv_best_model,\n"
        "        \"cv_stability_note\": cv_stability_note,\n"
        "        \"selected_reason\": selected_reason,\n"
        "    }\n"
        "])\n"
        "best_predictions = model_predictions[best_model]\n"
        "display(model_comparison)\n"
    )


def _sales_regression_backtest_code(output_language: str | None = None) -> str:
    title = (
        "Actual vs Best Model Sales Backtest"
        if is_english_output(output_language)
        else "Actual vs Best Model：销售额回测对比"
    )
    return (
        _modeling_plot_setup_code()
        + "\nbacktest_df = pd.DataFrame({\n"
        "    \"period\": test_df[\"period\"].values,\n"
        "    \"actual\": actual.values,\n"
        "    \"predicted\": pd.Series(best_predictions, index=actual.index).values,\n"
        "})\n"
        "backtest_df[\"absolute_error\"] = (backtest_df[\"actual\"] - backtest_df[\"predicted\"]).abs()\n"
        "backtest_df[\"percentage_error\"] = np.where(\n"
        "    backtest_df[\"actual\"].abs() > 1e-9,\n"
        "    backtest_df[\"absolute_error\"] / backtest_df[\"actual\"].abs() * 100,\n"
        "    np.nan,\n"
        ")\n"
        "backtest_df[\"model\"] = best_model\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
        "ax.plot(backtest_df[\"period\"], backtest_df[\"actual\"], marker=\"o\", label=\"Actual\")\n"
        "ax.plot(backtest_df[\"period\"], backtest_df[\"predicted\"], marker=\"o\", label=\"Best Model\")\n"
        f"ax.set_title({title!r})\n"
        "ax.set_xlabel(\"period\")\n"
        "ax.set_ylabel(\"sales_amount\")\n"
        "ax.grid(True, alpha=0.25)\n"
        "ax.legend(loc=\"best\")\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )


def _sales_regression_feature_importance_code(output_language: str | None = None) -> str:
    english = is_english_output(output_language)
    skip_message = "The current best model is the baseline, so the feature importance chart was skipped." if english else "当前 best model 是 baseline，跳过特征重要性图。"
    title = "Sales Regression Feature Importance Top 8" if english else "销售额回归特征重要性 Top 8"
    return (
        _modeling_plot_setup_code()
        + "\nif best_model not in fitted_estimators:\n"
        f"    print({skip_message!r})\n"
        "    feature_importance_df = pd.DataFrame(columns=[\"feature\", \"importance\", \"importance_type\"])\n"
        "else:\n"
        "    best_estimator = fitted_estimators[best_model]\n"
        "    if best_model == \"Ridge Regression\":\n"
        "        ridge = getattr(best_estimator, \"named_steps\", {}).get(\"ridge\")\n"
        "        values = np.abs(getattr(ridge, \"coef_\", []))\n"
        "        importance_type = \"absolute_coefficient\"\n"
        "    else:\n"
        "        values = getattr(best_estimator, \"feature_importances_\", [])\n"
        "        importance_type = \"feature_importance\"\n"
        "    feature_importance_df = pd.DataFrame({\n"
        "        \"feature\": feature_cols,\n"
        "        \"importance\": values,\n"
        "    })\n"
        "    feature_importance_df[\"importance_type\"] = importance_type\n"
        "    feature_importance_df = feature_importance_df.sort_values(\"importance\", ascending=False).head(8)\n"
        "    plot_df = feature_importance_df.sort_values(\"importance\", ascending=True)\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
        "    ax.barh(plot_df[\"feature\"], plot_df[\"importance\"], color=\"#2563EB\")\n"
        f"    ax.set_title({title!r})\n"
        "    ax.set_xlabel(\"importance\")\n"
        "    ax.set_ylabel(\"feature\")\n"
        "    ax.grid(axis=\"x\", alpha=0.2)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _sales_regression_error_analysis_code() -> str:
    return (
        "error_top5 = backtest_df.sort_values(\"absolute_error\", ascending=False).head(5).copy()\n"
        "error_top5[\"error_direction\"] = np.where(\n"
        "    error_top5[\"predicted\"] > error_top5[\"actual\"],\n"
        "    \"over\",\n"
        "    np.where(error_top5[\"predicted\"] < error_top5[\"actual\"], \"under\", \"exact\"),\n"
        ")\n"
        "error_top5 = error_top5[[\"period\", \"actual\", \"predicted\", \"absolute_error\", \"percentage_error\", \"error_direction\"]]\n"
        "display(error_top5)\n\n"
        "over_prediction_count = int((backtest_df[\"predicted\"] > backtest_df[\"actual\"]).sum())\n"
        "under_prediction_count = int((backtest_df[\"predicted\"] < backtest_df[\"actual\"]).sum())\n"
        "over_errors = backtest_df.loc[backtest_df[\"predicted\"] > backtest_df[\"actual\"], \"absolute_error\"]\n"
        "under_errors = backtest_df.loc[backtest_df[\"predicted\"] < backtest_df[\"actual\"], \"absolute_error\"]\n"
        "max_over_prediction = float(over_errors.max()) if not over_errors.empty else 0.0\n"
        "max_under_prediction = float(under_errors.max()) if not under_errors.empty else 0.0\n"
        "print({\n"
        "    \"over_prediction_count\": over_prediction_count,\n"
        "    \"under_prediction_count\": under_prediction_count,\n"
        "    \"max_over_prediction\": round(max_over_prediction, 2),\n"
        "    \"max_under_prediction\": round(max_under_prediction, 2),\n"
        "})\n"
    )


def _render_sales_regression_code_first_cells(
    forecast_module: object,
    *,
    weak: bool,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    output_language: str | None = None,
) -> list:
    english = is_english_output(output_language)
    heading = (
        ("## Modeling Analysis: Sales Forecast Attempt" if weak else "## Modeling Analysis: Sales Regression")
        if english
        else ("## 建模分析：销售额预测尝试" if weak else "## 建模分析：销售额预测回归")
    )
    feature_intro = (
        "### Feature Engineering\n\nThe code rebuilds a sales time series and regression features from the current notebook data, using only calendar, lag, and rolling features."
        if english
        else "### 特征工程\n\n下面的代码从 notebook 当前数据重新构造销售额时间序列和回归特征，只使用 calendar / lag / rolling 这类通用特征。"
    )
    cells = [
        new_markdown_cell(
            "\n\n".join(
                [
                    heading,
                    (
                        "### Modeling Goal and Data Conditions\n\n"
                        if english
                        else "### 建模目标与数据条件\n\n"
                    )
                    + _sales_regression_intro(
                        forecast_module,
                        weak=weak,
                        output_language=output_language,
                    ).split("\n\n", 1)[-1],
                    feature_intro,
                ]
            )
        ),
        new_code_cell(_sales_regression_feature_engineering_code(forecast_module)),
        new_markdown_cell(
            "Calendar features capture year, month, quarter, week, and weekday seasonality; lag features capture sales inertia; rolling features capture short-term averages and volatility. Lag and rolling features are built after `target.shift(...)`, so current or future sales are not leaked into features."
            if english
            else "calendar 特征用于表达年、月、季度、周和星期等周期信息；lag 特征表示历史销售惯性，rolling 特征表示短期均值和波动。lag / rolling 都基于 `target.shift(...)` 后的数据生成，因此不会把当前期或未来销售额放进特征里。"
        ),
        new_markdown_cell(
            "### Time Split\n\nSales forecasting must be backtested in chronological order. The last period is used as test/backtest data, and earlier periods are used for training."
            if english
            else "### 时间切分\n\n销售额预测必须按时间顺序回测，下面使用最后一段时间作为 test/backtest，前面的时间作为 train，不使用 random split。"
        ),
        new_code_cell(_sales_regression_time_split_code(forecast_module, output_language=output_language)),
        new_markdown_cell(
            "This split is closer to a real forecasting setting: the model learns from the past and predicts later periods."
            if english
            else "这种切分方式更接近真实预测场景：模型只能从过去学习，再预测后面的周期。"
        ),
        new_markdown_cell(
            "### Time Series Cross-validation\n\nTimeSeriesSplit checks whether model performance is stable across time windows and compares baseline, Ridge Regression, and RandomForestRegressor-small."
            if english
            else "### 时间序列交叉验证\n\n下面用 TimeSeriesSplit 检查模型跨时间窗口是否稳定，并对 baseline、Ridge Regression 和 RandomForestRegressor-small 做轻量对比。"
        ),
        new_code_cell(_sales_regression_cv_code(forecast_module, output_language=output_language)),
        new_code_cell(_sales_regression_cv_chart_code(output_language=output_language)),
        new_markdown_cell(
            _sales_regression_cv_stability_takeaway(
                forecast_module,
                modeling_outcome_interpretation,
                weak=weak,
                output_language=output_language,
            )
        ),
        new_markdown_cell(
            "### Model Comparison and Selection\n\nThe chronological holdout trains Ridge Regression and RandomForestRegressor-small while also calculating naive_last_value and moving_average_7 baselines."
            if english
            else "### 模型对比与选择\n\n下面在 chronological holdout 上训练 Ridge Regression 和 RandomForestRegressor-small，同时计算 naive_last_value 与 moving_average_7 baseline。"
        ),
        new_code_cell(_sales_regression_model_comparison_code(output_language=output_language)),
        new_markdown_cell(
            _sales_regression_model_selection_takeaway(
                forecast_module,
                modeling_outcome_interpretation,
                weak=weak,
                output_language=output_language,
            )
        ),
        new_markdown_cell(
            "### Sales Regression Backtest\n\nThe following cell builds backtest details for the best model and plots Actual vs Best Model."
            if english
            else "### 回测效果\n\n下面基于 best model 输出回测明细，并绘制 Actual vs Best Model。"
        ),
        new_code_cell(_sales_regression_backtest_code(output_language=output_language)),
        new_markdown_cell(
            _sales_regression_prediction_fit_takeaway(
                forecast_module,
                modeling_outcome_interpretation,
                weak=weak,
                output_language=output_language,
            )
        ),
    ]
    if not weak:
        cells.extend(
            [
                new_markdown_cell(
                    "### Feature Importance\n\nThe following chart explains the top signals used by the current best regression model."
                    if english
                    else "### 特征重要性\n\n下面只解释当前 best regression model 的 Top 特征信号。"
                ),
                new_code_cell(_sales_regression_feature_importance_code(output_language=output_language)),
                new_markdown_cell(
                    _sales_regression_feature_takeaway(
                        forecast_module,
                        modeling_outcome_interpretation,
                        output_language=output_language,
                    )
                ),
            ]
        )
    cells.extend(
        [
            new_markdown_cell(
                "### Error Analysis\n\nThe following table identifies periods with the largest absolute errors and marks over- or under-prediction."
                if english
                else "### 误差分析\n\n下面从回测明细中找出绝对误差最大的周期，并标记高估或低估方向。"
            ),
            new_code_cell(_sales_regression_error_analysis_code()),
            new_markdown_cell(
                _sales_regression_error_takeaway(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=weak,
                    output_language=output_language,
                )
            ),
            new_markdown_cell(
                ("### Overall Summary\n\n" if english else "### 综合结论\n\n")
                + _sales_regression_final_synthesis(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=weak,
                    output_language=output_language,
                )
            ),
        ]
    )
    return cells


def _render_sales_regression_usable_parts(
    forecast_module: object,
    *,
    modeling_outcome_interpretation: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    before_chart = "\n\n".join(
        [
            "## 建模分析：销售额预测",
            _sales_regression_intro(forecast_module),
            (
                "### 时间序列交叉验证\n\n"
                "下表展示 TimeSeriesSplit 的每折验证窗口和各模型表现，用于判断模型是否只在最后一段回测中表现较好。\n\n"
                + _sales_regression_cv_fold_table(forecast_module)
                + "\n\nCV 汇总：\n\n"
                + _sales_regression_cv_summary_table(forecast_module)
                + "\n\n"
                + _sales_regression_cv_stability_takeaway(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=False,
                )
            ),
            (
                "### 模型对比与选择\n\n"
                + _sales_regression_model_comparison(forecast_module)
                + "\n\n"
                + _sales_regression_model_selection_takeaway(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=False,
                )
            ),
            (
                "### 回测效果\n\n"
                "下图展示回测窗口内实际销售额与 best model 预测值的对比。"
            ),
        ]
    )
    after_chart_before_feature = (
        _sales_regression_prediction_fit_takeaway(
            forecast_module,
            modeling_outcome_interpretation,
            weak=False,
        )
        + "\n\n"
        "### 特征重要性\n\n"
        "下图展示当前 best regression model 的 Top 特征重要性。"
    )
    conclusion = (
        _sales_regression_feature_takeaway(forecast_module, modeling_outcome_interpretation)
        + "\n\n### 误差分析\n\n误差最大周期 Top 5：\n\n"
        + _sales_regression_worst_errors(forecast_module)
        + "\n\n"
        + _sales_regression_error_takeaway(
            forecast_module,
            modeling_outcome_interpretation,
            weak=False,
        )
        + "\n\n### 综合结论\n\n"
        + _sales_regression_final_synthesis(
            forecast_module,
            modeling_outcome_interpretation,
            weak=False,
        )
    )
    return before_chart, after_chart_before_feature, conclusion


def _render_sales_regression_weak_parts(
    forecast_module: object,
    *,
    modeling_outcome_interpretation: dict[str, str] | None = None,
) -> tuple[str, str]:
    before_chart = "\n\n".join(
        [
            "## 建模分析：销售额预测尝试",
            _sales_regression_intro(forecast_module, weak=True),
            (
                "### 时间序列交叉验证\n\n"
                "下表用于查看模型在多个时间窗口上的误差是否稳定；weak regression 下，这部分主要用于说明预测难度。\n\n"
                + _sales_regression_cv_fold_table(forecast_module)
                + "\n\nCV 汇总：\n\n"
                + _sales_regression_cv_summary_table(forecast_module)
                + "\n\n"
                + _sales_regression_cv_stability_takeaway(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=True,
                )
            ),
            (
                "### 模型对比与选择\n\n"
                "下表使用 MAE / RMSE / MAPE / R2 和 improvement_vs_baseline 比较 baseline 与轻量回归模型。\n\n"
                + _sales_regression_model_comparison(forecast_module)
                + "\n\n"
                + _sales_regression_model_selection_takeaway(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=True,
                )
            ),
            (
                "### 回测效果\n\n"
                "下图展示回测窗口内实际销售额与 best model 预测值的对比。"
            ),
        ]
    )
    after_chart = "\n\n".join(
        block
        for block in [
            (
                _sales_regression_prediction_fit_takeaway(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=True,
                )
                + "\n\n### 误差分析\n\n误差最大周期 Top 5：\n\n"
                + _sales_regression_worst_errors(forecast_module)
                + "\n\n"
                + _sales_regression_error_takeaway(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=True,
                )
            ),
            (
                "### 综合结论\n\n"
                + _sales_regression_final_synthesis(
                    forecast_module,
                    modeling_outcome_interpretation,
                    weak=True,
                )
            ),
        ]
        if block
    )
    return before_chart, after_chart


def _forecast_backtest_chart_code(module: object | None, output_language: str | None = None) -> str:
    rows_json = _forecast_table_json(module, "baseline_backtest")
    english = is_english_output(output_language)
    skip_message = "No baseline backtest data is available, so the actual-vs-baseline chart was skipped." if english else "暂无 baseline 回测数据，跳过实际值与基线预测图。"
    title = "Actual vs Baseline Sales Backtest" if english else "Actual vs Baseline：销售额回测对比"
    return (
        _modeling_plot_setup_code()
        + f"\nbaseline_backtest = pd.DataFrame({rows_json})\n"
        "if baseline_backtest.empty:\n"
        f"    print({skip_message!r})\n"
        "else:\n"
        "    baseline_backtest[\"period\"] = pd.to_datetime(baseline_backtest[\"period\"], errors=\"coerce\")\n"
        "    actual_col = \"actual_sales_amount\" if \"actual_sales_amount\" in baseline_backtest.columns else \"actual\"\n"
        "    forecast_col = \"best_baseline_forecast\" if \"best_baseline_forecast\" in baseline_backtest.columns else \"forecast\"\n"
        "    for column in [actual_col, forecast_col]:\n"
        "        if column in baseline_backtest.columns:\n"
        "            baseline_backtest[column] = pd.to_numeric(baseline_backtest[column], errors=\"coerce\")\n"
        "    plot_df = baseline_backtest.dropna(subset=[\"period\"])\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
        "    if actual_col in plot_df.columns:\n"
        "        ax.plot(plot_df[\"period\"], plot_df[actual_col], marker=\"o\", label=\"Actual\")\n"
        "    if forecast_col in plot_df.columns:\n"
        "        ax.plot(plot_df[\"period\"], plot_df[forecast_col], marker=\"o\", label=\"Baseline\")\n"
        f"    ax.set_title({title!r})\n"
        "    ax.set_xlabel(\"period\")\n"
        "    ax.set_ylabel(\"sales_amount\")\n"
        "    ax.grid(True, alpha=0.25)\n"
        "    ax.legend(loc=\"best\")\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _sales_regression_backtest_chart_code(module: object | None, output_language: str | None = None) -> str:
    rows_json = _forecast_table_json(module, "regression_backtest")
    english = is_english_output(output_language)
    skip_message = "No sales regression backtest data is available, so the actual-vs-predicted chart was skipped." if english else "暂无销售额回归回测数据，跳过实际值与预测值对比图。"
    title = "Actual vs Best Model Sales Backtest" if english else "Actual vs Best Model：销售额回测对比"
    return (
        _modeling_plot_setup_code()
        + f"\nregression_backtest = pd.DataFrame({rows_json})\n"
        "if regression_backtest.empty:\n"
        f"    print({skip_message!r})\n"
        "else:\n"
        "    regression_backtest[\"period\"] = pd.to_datetime(regression_backtest[\"period\"], errors=\"coerce\")\n"
        "    for column in [\"actual\", \"predicted\"]:\n"
        "        if column in regression_backtest.columns:\n"
        "            regression_backtest[column] = pd.to_numeric(regression_backtest[column], errors=\"coerce\")\n"
        "    plot_df = regression_backtest.dropna(subset=[\"period\"])\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
        "    ax.plot(plot_df[\"period\"], plot_df[\"actual\"], marker=\"o\", label=\"Actual\")\n"
        "    ax.plot(plot_df[\"period\"], plot_df[\"predicted\"], marker=\"o\", label=\"Best Model\")\n"
        f"    ax.set_title({title!r})\n"
        "    ax.set_xlabel(\"period\")\n"
        "    ax.set_ylabel(\"sales_amount\")\n"
        "    ax.grid(True, alpha=0.25)\n"
        "    ax.legend(loc=\"best\")\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _sales_regression_feature_chart_code(module: object | None, output_language: str | None = None) -> str:
    rows_json = _forecast_table_json(module, "feature_importance")
    english = is_english_output(output_language)
    skip_message = "No sales regression feature importance is available, so the feature importance chart was skipped." if english else "暂无销售额回归特征重要性，跳过特征重要性图。"
    title = "Sales Regression Feature Importance Top 8" if english else "销售额回归特征重要性 Top 8"
    return (
        _modeling_plot_setup_code()
        + f"\nfeature_importance = pd.DataFrame({rows_json})\n"
        "if feature_importance.empty:\n"
        f"    print({skip_message!r})\n"
        "else:\n"
        "    feature_importance[\"importance\"] = pd.to_numeric(feature_importance[\"importance\"], errors=\"coerce\").fillna(0)\n"
        "    plot_df = feature_importance.sort_values(\"importance\", ascending=True).tail(8)\n"
        "    fig, ax = plt.subplots(figsize=(8, 4.6))\n"
        "    ax.barh(plot_df[\"feature\"], plot_df[\"importance\"], color=\"#2563EB\")\n"
        f"    ax.set_title({title!r})\n"
        "    ax.set_xlabel(\"importance\")\n"
        "    ax.set_ylabel(\"feature\")\n"
        "    ax.grid(axis=\"x\", alpha=0.2)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
    )


def _split_modeling_markdown_for_chart_cells(markdown: str) -> tuple[str, str, str]:
    before_feature, feature_heading, after_feature = markdown.partition("### 风险特征解释")
    if not feature_heading:
        return markdown, "", ""
    feature_section, review_heading, review_section = after_feature.partition("### 复核样例摘要")
    middle = (feature_heading + feature_section).strip()
    tail = (review_heading + review_section).strip() if review_heading else ""
    return before_feature.strip(), middle, tail


def render_modeling_section_cells(
    module: object | None,
    modeling_interpretation: dict[str, str] | None = None,
    modeling_opportunity_decision: dict[str, str] | None = None,
    modeling_outcome: dict[str, object] | None = None,
    forecast_module: object | None = None,
    modeling_outcome_interpretation: dict[str, str] | None = None,
    output_language: str | None = None,
) -> list:
    def _finish(cells: list) -> list:
        if not is_english_output(output_language):
            return cells
        for cell in cells:
            if getattr(cell, "cell_type", None) in {"markdown", "code"}:
                cell.source = englishize_notebook_text(str(cell.source))
        return cells

    if modeling_outcome:
        primary_task = str(modeling_outcome.get("primary_modeling_task") or "")
        modeling_status = str(modeling_outcome.get("modeling_status") or "")
        if primary_task == "sales_amount_regression" and forecast_module is not None:
            if modeling_status == "regression_usable":
                return _finish(_render_sales_regression_code_first_cells(
                    forecast_module,
                    weak=False,
                    modeling_outcome_interpretation=modeling_outcome_interpretation,
                    output_language=output_language,
                ))
            return _finish(_render_sales_regression_code_first_cells(
                forecast_module,
                weak=True,
                modeling_outcome_interpretation=modeling_outcome_interpretation,
                output_language=output_language,
            ))
        if primary_task == "sales_amount_forecast_baseline" and forecast_module is not None:
            before_chart, after_chart = _render_forecast_outcome_markdown_parts(
                forecast_module,
                modeling_outcome=modeling_outcome,
                modeling_outcome_interpretation=modeling_outcome_interpretation,
            )
            return _finish([
                new_markdown_cell(before_chart),
                new_code_cell(_forecast_backtest_chart_code(forecast_module, output_language=output_language)),
                new_markdown_cell(after_chart),
            ])
        if primary_task == "loss_risk_classification" and module is not None:
            markdown = _render_loss_risk_outcome_markdown(
                module,
                modeling_outcome=modeling_outcome,
                modeling_interpretation=modeling_interpretation,
            )
            if modeling_status == "weak_model":
                return _finish(_render_weak_loss_risk_code_first_cells(
                    module,
                    modeling_outcome=modeling_outcome,
                    modeling_interpretation=modeling_interpretation,
                    output_language=output_language,
                ))
            return _finish(_render_strong_loss_risk_outcome_cells(
                module,
                modeling_interpretation=modeling_interpretation,
                modeling_outcome_interpretation=modeling_outcome_interpretation,
                output_language=output_language,
            ))

    markdown = render_modeling_section_markdown(
        module,
        modeling_interpretation=modeling_interpretation,
        modeling_opportunity_decision=modeling_opportunity_decision,
        output_language=output_language,
    )
    tables = getattr(module, "tables", {}) or {}
    warnings = list(getattr(module, "warnings", []) or [])
    skipped = module is None or (bool(warnings) and not tables.get("model_comparison"))
    if skipped:
        return _finish([new_markdown_cell(markdown)])

    first_markdown, feature_markdown, review_markdown = _split_modeling_markdown_for_chart_cells(markdown)
    cells = [
        new_markdown_cell(first_markdown),
        new_code_cell(_threshold_tradeoff_chart_code(module, output_language=output_language)),
        new_code_cell(_confusion_matrix_chart_code(module, output_language=output_language)),
    ]
    if feature_markdown:
        cells.append(new_markdown_cell(feature_markdown))
    cells.append(new_code_cell(_feature_group_importance_chart_code(module, output_language=output_language)))
    if review_markdown:
        cells.append(new_markdown_cell(review_markdown))
    return _finish(cells)
