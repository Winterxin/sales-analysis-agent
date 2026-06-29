from __future__ import annotations

import inspect

from app.schemas.llm_trace import LLMStageTrace
from app.schemas.notebook_narrative import NotebookNarrative, NotebookSectionNarrative
from app.schemas.notebook_outline import NotebookOutline
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.output_language import is_english_output, user_facing_language_instruction


def _module(report: AnalysisReport, module_id: str) -> ModuleReport | None:
    return next((module for module in report.modules if module.module_id == module_id), None)


def _format_number(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)


def _format_percent(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if abs(numeric) <= 1:
            numeric *= 100
        return f"{numeric:.2f}%"
    return str(value)


def _table_rows(module: ModuleReport | None, table_name: str) -> list[dict[str, object]]:
    if module is None:
        return []
    rows = module.tables.get(table_name, [])
    return [row for row in rows if isinstance(row, dict)]


def _dimension_value(row: dict[str, object], canonical_name: object) -> object | None:
    key = str(canonical_name or "").strip()
    candidates = {
        "segment": ("Segment", "segment"),
        "region": ("Region", "region"),
        "category": ("Category", "category"),
    }.get(key.lower(), (key, key.title()))
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    return None


def _dimension_label(
    row: dict[str, object],
    primary_dimension: object,
    secondary_dimension: object | None,
) -> str:
    parts = [
        str(value)
        for value in (
            _dimension_value(row, primary_dimension),
            _dimension_value(row, secondary_dimension) if secondary_dimension else None,
        )
        if value is not None and str(value).strip()
    ]
    return " / ".join(parts) if parts else "unknown"


def _fallback_narrative(
    outline: NotebookOutline,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    output_language: str | None = None,
) -> NotebookNarrative:
    if is_english_output(output_language):
        sections = [
            NotebookSectionNarrative(
                section_id=section.section_id,
                intro=f"This section reviews {section.title.lower()} using the mapped sales dataset and available module evidence.",
                key_observations=[
                    f"Section purpose: {section.purpose}",
                    f"Mapped field count: {len(schema_mapping.field_mapping)}",
                ],
                business_takeaway="Treat the result as decision support and validate it with business context before acting.",
                followup_question="Which evidence should the business team validate first?",
            )
            for section in outline.sections
        ]
        return NotebookNarrative(
            sections=sections,
            suggested_followups=["validate_priority_slices"],
        )
    mapped_fields = set(schema_mapping.field_mapping.values())
    sections: list[NotebookSectionNarrative] = []
    suggested_followups: list[str] = []

    trend_module = _module(report, "sales_trend_analysis")
    product_module = _module(report, "product_contribution_analysis")
    dimension_module = _module(report, "dimension_breakdown_analysis")
    discount_profit_module = _module(report, "discount_profit_analysis")
    forecast_module = _module(report, "forecast_analysis")

    for section in outline.sections:
        if section.section_id == "sales_trends":
            total_sales = trend_module.summary_metrics.get("total_sales_amount", "unknown") if trend_module else "unknown"
            peak_period = trend_module.summary_metrics.get("peak_period", "unknown") if trend_module else "unknown"
            trough_period = trend_module.summary_metrics.get("trough_period", "unknown") if trend_module else "unknown"
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro=f"本节聚焦销售趋势变化。当前累计销售额为 {total_sales}，需要识别时间维度上的波动和拐点。",
                    key_observations=[
                        f"销售峰值日期：{peak_period}",
                        f"销售低谷日期：{trough_period}",
                    ],
                    business_takeaway="时间趋势的变化能帮助我们判断促销时点、季节性影响以及阶段性经营表现。",
                    followup_question="哪些月份或季度对整体销售增长贡献最大？",
                )
            )
            continue

        if section.section_id == "product_and_category":
            top_product = product_module.summary_metrics.get("top_product", "unknown") if product_module else "unknown"
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro="本节用于识别商品和类目的销售贡献结构，判断是否存在头部集中或长尾拖累现象。",
                    key_observations=[f"Top 商品：{top_product}"],
                    business_takeaway="商品集中度既可能带来增长杠杆，也可能带来对单品依赖的经营风险。",
                    followup_question="哪些商品销售额高，但利润表现并不理想？",
                )
            )
            continue

        if section.section_id == "segment_and_region":
            primary_dimension = (
                dimension_module.summary_metrics.get("primary_dimension", "unknown")
                if dimension_module
                else "unknown"
            )
            secondary_dimension = (
                dimension_module.summary_metrics.get("secondary_dimension")
                if dimension_module
                else None
            )
            dimension_totals = (
                _table_rows(dimension_module, "segment_region_matrix")
                if secondary_dimension
                else []
            ) or _table_rows(dimension_module, "dimension_totals")
            strongest_cut = max(
                dimension_totals,
                key=lambda row: float(row.get("Sales", 0) or 0),
                default={},
            )
            weakest_cut = min(
                dimension_totals,
                key=lambda row: float(row.get("Profit", 0) or 0),
                default={},
            )
            if secondary_dimension:
                strongest_label = _dimension_label(strongest_cut, primary_dimension, secondary_dimension)
                weakest_label = _dimension_label(weakest_cut, primary_dimension, secondary_dimension)
                intro = (
                    "本节比较客群和区域两个维度下的表现差异，帮助识别结构性的强弱切片。"
                    f" 当前最强的 `{strongest_label}` 销售额约为 `{_format_number(strongest_cut.get('Sales'))}`，"
                    f"而 `{weakest_label}` 的利润约为 `{_format_number(weakest_cut.get('Profit'))}`，说明不同切片之间已经拉开稳定差距。"
                )
                key_observations = [
                    f"当前主维度：{primary_dimension}",
                    f"当前次维度：{secondary_dimension}",
                    f"{weakest_label} 是当前利润表现最弱的切片。",
                ]
                takeaway = "双维拆解的价值不只是分出高低，更是为后续继续下钻到区域、商品和策略层提供落点。"
                followup = "哪些区域里的某类客群销售不差，但利润持续偏弱？"
            else:
                intro = "本节先做单维客群拆解，因为当前数据缺少第二个可稳定使用的区域维度。"
                key_observations = [f"当前主维度：{primary_dimension}", "当前章节已降级为单维分析。"]
                takeaway = "即使只有单维视角，仍然可以先识别表现偏弱的客群切片。"
                followup = "哪些客群的销售贡献不低，但利润质量持续偏弱？"
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro=intro,
                    key_observations=key_observations,
                    business_takeaway=takeaway,
                    followup_question=followup,
                )
            )
            continue

        if section.section_id == "discount_and_profit":
            has_discount = "discount" in mapped_fields
            negative_profit_orders = (
                discount_profit_module.summary_metrics.get("negative_profit_order_count", "unknown")
                if discount_profit_module
                else "unknown"
            )
            high_discount_orders = (
                discount_profit_module.summary_metrics.get("high_discount_order_count", "unknown")
                if discount_profit_module
                else "unknown"
            )
            negative_profit_rate = (
                discount_profit_module.summary_metrics.get("negative_profit_rate", "unknown")
                if discount_profit_module
                else "unknown"
            )
            discount_buckets = _table_rows(discount_profit_module, "discount_buckets")
            worst_bucket = min(
                discount_buckets,
                key=lambda row: float(row.get("avg_profit", 0) or 0),
                default={},
            )
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro=(
                        "本节分析折扣策略与利润结果之间的关系，重点识别利润被折扣侵蚀的区间。"
                        f" 当前高折扣记录约 `{_format_number(high_discount_orders)}` 条，负利润记录约 `{_format_number(negative_profit_orders)}` 条，说明利润风险已经不只是零星异常。"
                        if has_discount
                        else "本节先做利润概览，定位亏损订单、亏损商品和利润承压类目。"
                    ),
                    key_observations=[
                        (
                            f"当前高折扣记录数：{_format_number(high_discount_orders)}"
                            if has_discount
                            else f"当前利润为负的记录数：{_format_number(negative_profit_orders)}"
                        ),
                        (
                            f"负利润占比约为 {_format_percent(negative_profit_rate)}，最弱区间是 {worst_bucket.get('discount_bucket', 'unknown')}。"
                            if has_discount
                            else "缺少 discount 字段，本章已降级为利润概览。"
                        ),
                    ],
                    business_takeaway=(
                        "健康的销售额并不等于健康的利润，高折扣一旦越过临界点，就会从促销工具变成利润损耗。"
                        if has_discount
                        else "即使没有折扣字段，利润概览仍能帮助我们定位亏损集中在哪些商品和类目。"
                    ),
                    followup_question=(
                        "折扣从什么水平开始明显侵蚀利润？"
                        if has_discount
                        else "哪些商品销售额不低，但利润持续偏弱？"
                    ),
                )
            )
            suggested_followups.append("discount_and_profit")
            continue

        if section.section_id == "forecast":
            baseline = forecast_module.summary_metrics.get("baseline_sales_amount", "unknown") if forecast_module else "unknown"
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro="本节给出一个短期的基线预测，用于判断近期销售走势是否延续当前水平。",
                    key_observations=[f"当前基线预测值：{baseline}"],
                    business_takeaway="预测结果的参考价值依赖于季节性、促销活动和聚合粒度是否处理得当。",
                    followup_question="为了提升稳定性，预测应该按周还是按月聚合？",
                )
            )
            continue

        if section.section_id == "modeling":
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro="本节基于亏损风险模型，把前面发现的利润风险转化为可复核的订单风险识别问题。",
                    key_observations=[],
                    business_takeaway="建模结果的价值不在于替代业务判断，而在于帮助确定人工复核优先级。",
                    followup_question="在当前阈值下，业务团队能承受多少复核量，且漏判亏损订单的风险是否可接受？",
                )
            )
            continue

        if section.section_id == "conclusions":
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro="本节汇总最重要的发现，并将它们转化成可执行的经营动作。",
                    key_observations=report.summary[:5],
                    business_takeaway="一份分析 notebook 的最终价值，不在于图表多少，而在于能否支持后续经营决策。",
                    followup_question="下一步还应该补充哪些分析，才能支持定价、品类或区域动作？",
                )
            )
            continue

        if section.section_id == "dataset_and_schema":
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro="本节用于说明这份销售数据包含什么，以及字段是如何被标准化到统一语义层的。",
                    key_observations=[
                        f"已识别字段数：{len(schema_mapping.field_mapping)}",
                        f"待确认字段数：{len(schema_mapping.uncertain_fields)}",
                    ],
                    business_takeaway="字段映射越可靠，后续分析越稳定，结论也越可信。",
                    followup_question="进入深挖分析之前，还有哪些字段需要人工确认？",
                )
            )
            continue

        if section.section_id == "data_cleaning":
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro="本节在正式解释业务结果前，先验证原始 CSV 的质量是否可靠。",
                    key_observations=report.summary[:2],
                    business_takeaway="清洗质量会直接影响每一张图和每一条结论的可信度。",
                    followup_question="是否仍存在异常值或解析问题需要进一步人工确认？",
                )
            )
            continue

        if section.section_id == "title_and_goal":
            sections.append(
                NotebookSectionNarrative(
                    section_id=section.section_id,
                    intro="这份笔记的目标不是简单堆叠模块结果，而是形成一条完整、连贯、可复查的销售分析叙事。",
                    key_observations=[],
                    business_takeaway="这份 Notebook 应同时满足业务阅读和分析师复用两类场景。",
                    followup_question="对于这份数据，最值得优先回答的业务问题是什么？",
                )
            )

    return NotebookNarrative(sections=sections, suggested_followups=suggested_followups)


def _sanitize_narrative(payload: dict, fallback: NotebookNarrative) -> NotebookNarrative:
    valid_section_ids = {section.section_id for section in fallback.sections}
    sections: list[NotebookSectionNarrative] = []
    seen: set[str] = set()

    for section in payload.get("sections", []):
        section_id = str(section.get("section_id", "")).strip()
        if section_id not in valid_section_ids or section_id in seen:
            continue
        seen.add(section_id)
        sections.append(
            NotebookSectionNarrative(
                section_id=section_id,
                intro=str(section.get("intro", "")),
                key_observations=[str(item) for item in section.get("key_observations", [])],
                business_takeaway=str(section.get("business_takeaway", "")),
                followup_question=str(section.get("followup_question", "")),
            )
        )

    if not sections:
        return fallback

    return NotebookNarrative(
        sections=sections,
        suggested_followups=[str(item) for item in payload.get("suggested_followups", [])],
    )


def _accepts_keyword(method, keyword: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def build_notebook_narrative(
    outline: NotebookOutline,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    llm_client=None,
    evidence_pack: dict[str, object] | None = None,
    output_language: str | None = None,
) -> NotebookNarrative:
    narrative, _ = build_notebook_narrative_with_trace(
        outline=outline,
        report=report,
        schema_mapping=schema_mapping,
        llm_client=llm_client,
        evidence_pack=evidence_pack,
        output_language=output_language,
    )
    return narrative


def build_notebook_narrative_with_trace(
    outline: NotebookOutline,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    llm_client=None,
    evidence_pack: dict[str, object] | None = None,
    output_language: str | None = None,
) -> tuple[NotebookNarrative, LLMStageTrace]:
    fallback = _fallback_narrative(outline, report, schema_mapping, output_language)

    if llm_client is None or not getattr(llm_client, "enabled", False):
        return fallback, build_llm_stage_trace(
            stage="notebook_narrative",
            llm_client=llm_client,
            status="disabled",
            reason="LLM is disabled, so notebook narratives used the deterministic fallback copy.",
        )

    suggest_narrative = getattr(llm_client, "suggest_notebook_narrative", None)
    if suggest_narrative is None:
        return fallback, build_llm_stage_trace(
            stage="notebook_narrative",
            llm_client=llm_client,
            status="skipped",
            reason="The configured LLM client does not provide notebook narrative generation.",
        )

    try:
        if _accepts_keyword(suggest_narrative, "evidence_pack"):
            kwargs = {"evidence_pack": evidence_pack}
            if _accepts_keyword(suggest_narrative, "language_instruction"):
                kwargs["language_instruction"] = user_facing_language_instruction(output_language)
            llm_narrative = suggest_narrative(outline, report, schema_mapping, fallback, **kwargs)
        elif _accepts_keyword(suggest_narrative, "language_instruction"):
            llm_narrative = suggest_narrative(
                outline,
                report,
                schema_mapping,
                fallback,
                language_instruction=user_facing_language_instruction(output_language),
            )
        else:
            llm_narrative = suggest_narrative(outline, report, schema_mapping, fallback)
    except Exception as exc:
        return fallback, build_llm_stage_trace(
            stage="notebook_narrative",
            llm_client=llm_client,
            status="fallback_on_error",
            reason=f"Notebook narrative generation failed; used deterministic narrative copy instead. {describe_llm_error(exc)}",
            attempted=True,
        )

    if isinstance(llm_narrative, NotebookNarrative):
        result = llm_narrative
    elif isinstance(llm_narrative, dict):
        result = _sanitize_narrative(llm_narrative, fallback)
    else:
        return fallback, build_llm_stage_trace(
            stage="notebook_narrative",
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason="Notebook narrative LLM returned an unsupported payload, so deterministic copy was kept.",
            attempted=True,
        )

    applied = result != fallback
    return result, build_llm_stage_trace(
        stage="notebook_narrative",
        llm_client=llm_client,
        status="llm_applied" if applied else "llm_no_change",
        reason=(
            "Notebook narratives were enriched by the LLM."
            if applied
            else "Notebook narrative LLM output was equivalent to the deterministic narrative."
        ),
        attempted=True,
        applied=applied,
    )
