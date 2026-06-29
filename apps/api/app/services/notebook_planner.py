from __future__ import annotations

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.llm_trace import LLMStageTrace
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.output_language import (
    ENGLISH_SECTION_TITLES,
    is_english_output,
    user_facing_language_instruction,
)

SECTION_REGISTRY: dict[str, NotebookSection] = {
    "title_and_goal": NotebookSection(
        section_id="title_and_goal",
        title="分析目标",
        purpose="说明这份 Notebook 要解决的业务问题、分析目标和使用场景。",
    ),
    "dataset_and_schema": NotebookSection(
        section_id="dataset_and_schema",
        title="数据集与字段说明",
        purpose="介绍数据集背景、字段结构，以及标准化后的业务字段映射。",
    ),
    "data_cleaning": NotebookSection(
        section_id="data_cleaning",
        title="数据清洗与预处理",
        purpose="展示类型修正、缺失值、重复值和预处理决策。",
    ),
    "metric_distributions": NotebookSection(
        section_id="metric_distributions",
        title="指标分布与异常值分析",
        purpose="对销售额、销量、折扣和利润等核心数值字段做分布画像，识别长尾、离群点和后续专题分析重点。",
    ),
    "sales_trends": NotebookSection(
        section_id="sales_trends",
        title="销售趋势分析",
        purpose="分析销售额与销量随时间的变化，并识别峰值、低谷和波动。",
    ),
    "product_and_category": NotebookSection(
        section_id="product_and_category",
        title="商品与类目分析",
        purpose="识别头部商品、长尾商品、类目结构与商品集中度。",
    ),
    "segment_and_region": NotebookSection(
        section_id="segment_and_region",
        title="客群与区域分析",
        purpose="按客群和地理维度拆解表现，识别优势和薄弱区域。",
    ),
    "country_market": NotebookSection(
        section_id="country_market",
        title="国家市场分析",
        purpose="按国家拆解销售额、订单数、客单价、客户数和月度趋势。",
    ),
    "order_structure": NotebookSection(
        section_id="order_structure",
        title="订单结构分析",
        purpose="分析发票金额、篮子大小、客户复购、交易规模、订单状态和价格数量关系。",
    ),
    "discount_and_profit": NotebookSection(
        section_id="discount_and_profit",
        title="折扣与利润分析",
        purpose="评估折扣、销售额和利润之间的关系，识别利润侵蚀风险。",
    ),
    "modeling": NotebookSection(
        section_id="modeling",
        title="建模分析：亏损风险识别",
        purpose="基于亏损风险分类模型识别需要人工复核的高风险订单，并说明模型限制。",
    ),
    "forecast": NotebookSection(
        section_id="forecast",
        title="预测与展望",
        purpose="给出近期趋势预测，并说明预测假设和限制。",
    ),
    "conclusions": NotebookSection(
        section_id="conclusions",
        title="结论与行动建议",
        purpose="总结关键发现、风险点和下一步行动建议。",
    ),
}


def _section_registry(output_language: str | None = None) -> dict[str, NotebookSection]:
    if not is_english_output(output_language):
        return SECTION_REGISTRY
    return {
        section_id: NotebookSection(
            section_id=section_id,
            title=ENGLISH_SECTION_TITLES.get(section_id, (section.title, section.purpose))[0],
            purpose=ENGLISH_SECTION_TITLES.get(section_id, (section.title, section.purpose))[1],
        )
        for section_id, section in SECTION_REGISTRY.items()
    }


def _outline_title(output_language: str | None = None) -> str:
    return "Sales Data Analysis Notebook" if is_english_output(output_language) else "销售数据分析 Notebook"


FOCUS_TO_SECTION = {
    "discount_erosion_focus": "discount_and_profit",
    "profit_quality_focus": "discount_and_profit",
    "segment_region_focus": "segment_and_region",
    "product_concentration_focus": "product_and_category",
    "productline_performance_focus": "product_and_category",
    "country_market_focus": "country_market",
    "customer_order_structure_focus": "order_structure",
    "deal_size_focus": "order_structure",
    "order_status_focus": "order_structure",
    "trend_volatility_focus": "sales_trends",
}

STABLE_SECTION_ORDER = [
    "metric_distributions",
    "sales_trends",
    "product_and_category",
    "segment_and_region",
    "country_market",
    "order_structure",
    "discount_and_profit",
    "modeling",
]

DYNAMIC_NARRATIVE_SECTION_ORDER = [
    "metric_distributions",
    "sales_trends",
    "discount_and_profit",
    "segment_and_region",
    "product_and_category",
    "order_structure",
    "country_market",
    "modeling",
]

NARRATIVE_SECTION_ORDER = STABLE_SECTION_ORDER


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _ordered_priority_sections(
    priority: dict[str, object],
    *,
    dynamic_section_reordering: bool = False,
) -> list[str]:
    core_sections = set(priority.get("core_sections", []))
    support_sections = set(priority.get("support_sections", []))
    skipped_sections = set(priority.get("skipped_sections", []))
    eligible_sections = (core_sections | support_sections) - skipped_sections
    section_order = DYNAMIC_NARRATIVE_SECTION_ORDER if dynamic_section_reordering else STABLE_SECTION_ORDER
    return [
        section_id
        for section_id in section_order
        if section_id in eligible_sections
    ]


def build_section_priority(
    schema_mapping: SchemaMapping,
    analysis_plan: AnalysisPlan,
    dataset_profile: dict | None = None,
    analysis_focus: dict | None = None,
) -> dict[str, object]:
    mapped_fields = set(schema_mapping.field_mapping.values())
    plan_modules = set(analysis_plan.analysis_plan)
    profile = dataset_profile or {}
    focus = analysis_focus or {}
    selected_focuses = list(focus.get("selected_focuses", []))
    support_focuses = list(focus.get("support_focuses", []))
    skipped_focuses = set(focus.get("skipped_focuses", []))

    core_sections: list[str] = []
    support_sections: list[str] = []
    skipped_sections: list[str] = []
    reasons: dict[str, str] = {}

    for focus_name in selected_focuses:
        section_id = FOCUS_TO_SECTION.get(focus_name)
        if section_id:
            _append_unique(core_sections, section_id)
            reasons[section_id] = f"core focus: {focus_name}"

    for focus_name in support_focuses:
        section_id = FOCUS_TO_SECTION.get(focus_name)
        if section_id and section_id not in core_sections:
            _append_unique(support_sections, section_id)
            reasons[section_id] = f"support focus: {focus_name}"

    if "metric_distribution_analysis" in plan_modules:
        _append_unique(support_sections, "metric_distributions")
        reasons.setdefault("metric_distributions", "support section for data shape and numeric ranges")
    if "forecast_analysis" in plan_modules and "trend_volatility_focus" in selected_focuses:
        _append_unique(support_sections, "forecast")
        reasons.setdefault("forecast", "support section for selected trend volatility focus")
    if "loss_risk_modeling" in plan_modules:
        _append_unique(support_sections, "modeling")
        reasons.setdefault("modeling", "support section for loss risk modeling")

    if profile.get("country_count") == 1 or "country_market_focus" in skipped_focuses:
        _append_unique(skipped_sections, "country_market")
        reasons["country_market"] = "skipped because country_count=1 or country focus was not eligible"
    if not {"profit", "discount"} <= mapped_fields:
        _append_unique(skipped_sections, "discount_and_profit")
        reasons["discount_and_profit"] = "skipped because profit/discount fields are incomplete"

    if "customer_order_structure_focus" in skipped_focuses and "deal_size_focus" in skipped_focuses and "order_status_focus" in skipped_focuses:
        _append_unique(skipped_sections, "order_structure")
        reasons.setdefault("order_structure", "skipped because order structure focuses were not eligible")

    all_analytic_sections = {
        "metric_distributions",
        "sales_trends",
        "product_and_category",
        "segment_and_region",
        "country_market",
        "order_structure",
        "discount_and_profit",
        "modeling",
        "forecast",
    }
    planned_sections = set(core_sections) | set(support_sections) | set(skipped_sections)
    for section_id in sorted(all_analytic_sections - planned_sections):
        _append_unique(skipped_sections, section_id)
        reasons.setdefault(section_id, "not selected by dataset-profile focus planning")

    core_sections = [section for section in core_sections if section not in skipped_sections]
    support_sections = [
        section for section in support_sections if section not in skipped_sections and section not in core_sections
    ]
    if "loss_risk_modeling" in plan_modules and "modeling" not in core_sections and "modeling" not in support_sections:
        _append_unique(support_sections, "modeling")

    return {
        "core_sections": core_sections,
        "support_sections": support_sections,
        "skipped_sections": skipped_sections,
        "section_reasons": reasons,
    }


def _fallback_outline(
    schema_mapping: SchemaMapping,
    analysis_plan: AnalysisPlan,
    dataset_profile: dict | None = None,
    analysis_focus: dict | None = None,
    *,
    dynamic_section_reordering: bool = False,
    modeling_opportunity_plan: dict | None = None,
    output_language: str | None = None,
) -> NotebookOutline:
    registry = _section_registry(output_language)
    if analysis_focus:
        priority = build_section_priority(
            schema_mapping=schema_mapping,
            analysis_plan=analysis_plan,
            dataset_profile=dataset_profile,
            analysis_focus=analysis_focus,
        )
        section_ids = [
            "title_and_goal",
            "dataset_and_schema",
            "data_cleaning",
            *_ordered_priority_sections(
                priority,
                dynamic_section_reordering=dynamic_section_reordering,
            ),
            "conclusions",
        ]
        if modeling_opportunity_plan and "modeling" not in section_ids:
            section_ids.insert(max(0, len(section_ids) - 1), "modeling")
        sections = [registry[section_id] for section_id in section_ids if section_id in registry]
        return NotebookOutline(title=_outline_title(output_language), sections=sections)

    mapped_fields = set(schema_mapping.field_mapping.values())
    eligible_sections: set[str] = set()
    sections = [
        registry["title_and_goal"],
        registry["dataset_and_schema"],
        registry["data_cleaning"],
    ]

    if "metric_distribution_analysis" in analysis_plan.analysis_plan:
        eligible_sections.add("metric_distributions")

    if "sales_trend_analysis" in analysis_plan.analysis_plan:
        eligible_sections.add("sales_trends")

    if "product_contribution_analysis" in analysis_plan.analysis_plan:
        eligible_sections.add("product_and_category")

    if {"segment", "region", "state", "city", "country"} & mapped_fields:
        eligible_sections.add("segment_and_region")

    if "country_market_analysis" in analysis_plan.analysis_plan:
        eligible_sections.add("country_market")

    if "order_structure_analysis" in analysis_plan.analysis_plan:
        eligible_sections.add("order_structure")

    if "profit" in mapped_fields:
        eligible_sections.add("discount_and_profit")

    if "loss_risk_modeling" in analysis_plan.analysis_plan or modeling_opportunity_plan:
        eligible_sections.add("modeling")

    for section_id in STABLE_SECTION_ORDER:
        if section_id in eligible_sections:
            sections.append(registry[section_id])
    sections.append(registry["conclusions"])
    return NotebookOutline(title=_outline_title(output_language), sections=sections)


def _fallback_section_order(fallback: NotebookOutline) -> list[str]:
    return [section.section_id for section in fallback.sections]


def _mandatory_section_ids(fallback: NotebookOutline) -> list[str]:
    fallback_ids = {section.section_id for section in fallback.sections}
    mandatory = [
        "title_and_goal",
        "dataset_and_schema",
        "data_cleaning",
    ]
    if "modeling" in fallback_ids:
        mandatory.append("modeling")
    mandatory.append("conclusions")
    return [section_id for section_id in mandatory if section_id in fallback_ids]


def _ensure_mandatory_sections(
    sections_by_id: dict[str, NotebookSection],
    fallback: NotebookOutline,
    output_language: str | None = None,
) -> dict[str, NotebookSection]:
    registry = _section_registry(output_language)
    fallback_lookup = {section.section_id: section for section in fallback.sections}
    result = dict(sections_by_id)
    for section_id in _mandatory_section_ids(fallback):
        if section_id in result:
            continue
        fallback_section = fallback_lookup[section_id]
        result[section_id] = NotebookSection(
            section_id=section_id,
            title=registry[section_id].title,
            purpose=fallback_section.purpose,
        )
    return result


def _normalize_outline_order(
    outline: NotebookOutline,
    fallback: NotebookOutline,
    *,
    dynamic_section_reordering: bool = False,
    output_language: str | None = None,
) -> NotebookOutline:
    registry = _section_registry(output_language)
    allowed_sections = {section.section_id for section in fallback.sections}
    fallback_lookup = {section.section_id: section for section in fallback.sections}
    sections_by_id: dict[str, NotebookSection] = {}
    for section in outline.sections:
        if section.section_id not in registry or section.section_id not in allowed_sections:
            continue
        sections_by_id.setdefault(
            section.section_id,
            NotebookSection(
                section_id=section.section_id,
                title=registry[section.section_id].title,
                purpose=section.purpose or fallback_lookup[section.section_id].purpose,
            ),
        )
    if not sections_by_id:
        return fallback
    sections_by_id = _ensure_mandatory_sections(sections_by_id, fallback, output_language)

    if dynamic_section_reordering:
        ordered_ids = []
        for section in outline.sections:
            if section.section_id in sections_by_id and section.section_id not in ordered_ids:
                ordered_ids.append(section.section_id)
        ordered_ids.extend(
            section_id
            for section_id in _fallback_section_order(fallback)
            if section_id in sections_by_id and section_id not in ordered_ids
        )
    else:
        ordered_ids = [
            section.section_id
            for section in fallback.sections
            if section.section_id in sections_by_id
        ]
    return NotebookOutline(
        title=fallback.title or _outline_title(output_language),
        sections=[sections_by_id[section_id] for section_id in ordered_ids],
    )


def _sanitize_outline(
    payload: dict,
    fallback: NotebookOutline,
    *,
    dynamic_section_reordering: bool = False,
    output_language: str | None = None,
) -> NotebookOutline:
    registry = _section_registry(output_language)
    requested_sections = payload.get("sections", [])
    valid_sections: list[NotebookSection] = []
    seen: set[str] = set()
    allowed_sections = {section.section_id for section in fallback.sections}

    for section in requested_sections:
        section_id = str(section.get("section_id", "")).strip()
        if section_id not in registry or section_id in seen or section_id not in allowed_sections:
            continue
        seen.add(section_id)
        valid_sections.append(
            NotebookSection(
                section_id=section_id,
                title=registry[section_id].title,
                purpose=str(section.get("purpose") or registry[section_id].purpose),
            )
        )

    if not valid_sections:
        return fallback

    return _normalize_outline_order(
        NotebookOutline(
            title=fallback.title or _outline_title(output_language),
            sections=valid_sections,
        ),
        fallback,
        dynamic_section_reordering=dynamic_section_reordering,
        output_language=output_language,
    )


def build_notebook_outline(
    schema_mapping: SchemaMapping,
    analysis_plan: AnalysisPlan,
    llm_client=None,
    dataset_profile: dict | None = None,
    analysis_focus: dict | None = None,
    dynamic_section_reordering: bool = False,
    modeling_opportunity_plan: dict | None = None,
    output_language: str | None = None,
) -> NotebookOutline:
    outline, _ = build_notebook_outline_with_trace(
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_client=llm_client,
        dataset_profile=dataset_profile,
        analysis_focus=analysis_focus,
        dynamic_section_reordering=dynamic_section_reordering,
        modeling_opportunity_plan=modeling_opportunity_plan,
        output_language=output_language,
    )
    return outline


def build_notebook_outline_with_trace(
    schema_mapping: SchemaMapping,
    analysis_plan: AnalysisPlan,
    llm_client=None,
    dataset_profile: dict | None = None,
    analysis_focus: dict | None = None,
    dynamic_section_reordering: bool = False,
    modeling_opportunity_plan: dict | None = None,
    output_language: str | None = None,
) -> tuple[NotebookOutline, LLMStageTrace]:
    fallback = _fallback_outline(
        schema_mapping,
        analysis_plan,
        dataset_profile,
        analysis_focus,
        dynamic_section_reordering=dynamic_section_reordering,
        modeling_opportunity_plan=modeling_opportunity_plan,
        output_language=output_language,
    )

    if llm_client is None or not getattr(llm_client, "enabled", False):
        return fallback, build_llm_stage_trace(
            stage="notebook_outline",
            llm_client=llm_client,
            status="disabled",
            reason="LLM is disabled, so the notebook outline used the deterministic fallback structure.",
        )

    suggest_outline = getattr(llm_client, "suggest_notebook_outline", None)
    if suggest_outline is None:
        return fallback, build_llm_stage_trace(
            stage="notebook_outline",
            llm_client=llm_client,
            status="skipped",
            reason="The configured LLM client does not provide notebook outline generation.",
        )

    try:
        llm_outline = suggest_outline(
            schema_mapping,
            analysis_plan,
            fallback,
            language_instruction=user_facing_language_instruction(output_language),
        )
    except Exception as exc:
        return fallback, build_llm_stage_trace(
            stage="notebook_outline",
            llm_client=llm_client,
            status="fallback_on_error",
            reason=f"Notebook outline generation failed; used the deterministic outline instead. {describe_llm_error(exc)}",
            attempted=True,
        )

    if isinstance(llm_outline, NotebookOutline):
        result = _normalize_outline_order(
            llm_outline,
            fallback,
            dynamic_section_reordering=dynamic_section_reordering,
            output_language=output_language,
        )
    elif isinstance(llm_outline, dict):
        result = _sanitize_outline(
            llm_outline,
            fallback,
            dynamic_section_reordering=dynamic_section_reordering,
            output_language=output_language,
        )
    else:
        return fallback, build_llm_stage_trace(
            stage="notebook_outline",
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason="Notebook outline LLM returned an unsupported payload, so the deterministic outline was kept.",
            attempted=True,
        )

    applied = result != fallback
    return result, build_llm_stage_trace(
        stage="notebook_outline",
        llm_client=llm_client,
        status="llm_applied" if applied else "llm_no_change",
        reason=(
            "Notebook outline was enriched by the LLM."
            if applied
            else "Notebook outline LLM output was equivalent to the deterministic outline."
        ),
        attempted=True,
        applied=applied,
    )
