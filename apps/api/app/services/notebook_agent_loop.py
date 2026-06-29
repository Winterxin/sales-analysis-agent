from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.schemas.notebook_agent_state import NotebookAgentLoopStage, NotebookAgentLoopState
from app.schemas.llm_trace import LLMStageTrace
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.notebook_revision import NotebookRevisionDecision, NotebookRevisionPlan
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_trace_utils import build_llm_stage_trace, describe_llm_error
from app.services.notebook.output_policy import is_compact_notebook_mode
from app.services.notebook_toolset import (
    append_code_cell,
    append_markdown_cell,
    find_markdown_heading_index,
    insert_code_cell_at_index,
    insert_markdown_cell_at_index,
    load_notebook,
    save_notebook,
)

MAX_REVISION_DECISIONS = 2

REVISION_TITLES = {
    "discount_bucket_loss_rate": "折扣区间亏损率",
    "top_product_profit_bridge": "头部商品销售额与利润桥接",
    "weak_segment_category_cross": "弱势客群与类目交叉",
}


def _original_column(schema_mapping: SchemaMapping, canonical_name: str, fallback: str) -> str:
    for original, canonical in schema_mapping.field_mapping.items():
        if canonical == canonical_name:
            return original
    return fallback


def _module_map(report: AnalysisReport) -> dict[str, Any]:
    return {module.module_id: module for module in report.modules}


def _table_head(rows: list[dict[str, object]], limit: int = 3) -> list[dict[str, object]]:
    return [row for row in rows[:limit] if isinstance(row, dict)]


def _discount_quality_table_has_loss_rate(discount_module: Any) -> bool:
    for table_name in ("discount_profit_risk_buckets", "discount_buckets"):
        rows = discount_module.tables.get(table_name, []) if discount_module is not None else []
        if not rows:
            continue
        first_row = next((row for row in rows if isinstance(row, dict)), {})
        if {"avg_profit", "negative_profit_rate"} <= set(first_row):
            return True
        if {"avg_profit", "loss_rate"} <= set(first_row):
            return True
    return False


REVISION_PARENT_SECTIONS = {
    "discount_bucket_loss_rate": "discount_and_profit",
    "top_product_profit_bridge": "product_and_category",
    "weak_segment_category_cross": "segment_and_region",
}


def revision_heading_title(revision_key: str) -> str:
    return f"追加分析：{REVISION_TITLES.get(revision_key, revision_key)}"


def _build_allowed_revisions(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
) -> dict[str, dict[str, Any]]:
    mapped_fields = set(schema_mapping.field_mapping.values())
    modules = _module_map(report)
    allowed: dict[str, dict[str, Any]] = {}

    discount_module = modules.get("discount_profit_analysis")
    if (
        {"discount", "profit", "sales_amount"} <= mapped_fields
        and discount_module is not None
        and not _discount_quality_table_has_loss_rate(discount_module)
    ):
        allowed["discount_bucket_loss_rate"] = {
            "revision_key": "discount_bucket_loss_rate",
            "title": REVISION_TITLES["discount_bucket_loss_rate"],
            "goal": "补充判断亏损是否已经集中在中高折扣区间，并把平均利润与亏损率放在同一张图里看。",
            "why_now": "当折扣与利润章节已经提示高折扣负利润明显时，这个补充分析能把风险区间拆得更细。",
            "evidence": {
                "summary_metrics": discount_module.summary_metrics,
                "discount_buckets": _table_head(discount_module.tables.get("discount_buckets", [])),
                "findings": discount_module.findings[:3],
            },
        }

    product_module = modules.get("product_contribution_analysis")
    if (
        {"product_name", "sales_amount", "profit"} <= mapped_fields
        and product_module is not None
    ):
        allowed["top_product_profit_bridge"] = {
            "revision_key": "top_product_profit_bridge",
            "title": REVISION_TITLES["top_product_profit_bridge"],
            "goal": "补充核查高销售额头部商品是否同步贡献利润，避免只看销售额排名。",
            "why_now": "当头部商品明显领先时，这个补充分析能判断头部结构究竟是优质优势还是规模幻觉。",
            "evidence": {
                "summary_metrics": product_module.summary_metrics,
                "top_products": _table_head(product_module.tables.get("top_products", [])),
                "findings": product_module.findings[:3],
            },
        }

    dimension_module = modules.get("dimension_breakdown_analysis")
    if (
        {"segment", "category", "sales_amount", "profit"} <= mapped_fields
        and dimension_module is not None
    ):
        allowed["weak_segment_category_cross"] = {
            "revision_key": "weak_segment_category_cross",
            "title": REVISION_TITLES["weak_segment_category_cross"],
            "goal": "把弱势客群继续拆到类目层，判断问题究竟集中在哪些经营切片。",
            "why_now": "当客群或区域章节已经识别出弱势切片时，这个补充分析能把问题进一步落到可执行的商品结构层。",
            "evidence": {
                "summary_metrics": dimension_module.summary_metrics,
                "weak_performance_cuts": _table_head(
                    dimension_module.tables.get("weak_performance_cuts", [])
                ),
                "findings": dimension_module.findings[:3],
            },
        }

    return allowed


def _report_context_for_revision(report: AnalysisReport) -> dict[str, Any]:
    return {
        "task_id": report.task_id,
        "dataset_type": report.dataset_type,
        "summary": report.summary[:8],
        "modules": [
            {
                "module_id": module.module_id,
                "title": module.title,
                "summary_metrics": module.summary_metrics,
                "tables": {
                    table_name: _table_head(rows)
                    for table_name, rows in module.tables.items()
                },
                "findings": module.findings[:4],
                "warnings": module.warnings[:3],
            }
            for module in report.modules
        ],
    }


def _compact_chart_contexts(chart_contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in chart_contexts[:8]:
        compacted.append(
            {
                "section_id": item.get("section_id"),
                "chart_title": item.get("chart_title"),
                "chart_kind": item.get("chart_kind"),
                "chart_summary": item.get("chart_summary"),
                "analysis_focus": item.get("analysis_focus"),
                "table_preview": str(item.get("table_preview", ""))[:420],
                "module_findings": list(item.get("module_findings", []))[:3],
            }
        )
    return compacted


def _sanitize_revision_plan(
    payload: dict[str, Any],
    allowed_revisions: dict[str, dict[str, Any]],
    max_revision_decisions: int = MAX_REVISION_DECISIONS,
) -> NotebookRevisionPlan:
    decisions: list[NotebookRevisionDecision] = []
    seen: set[str] = set()

    for item in payload.get("decisions", []):
        if not isinstance(item, dict):
            continue
        revision_key = str(item.get("revision_key", "")).strip()
        if revision_key not in allowed_revisions or revision_key in seen:
            continue
        seen.add(revision_key)
        reason = str(item.get("reason") or allowed_revisions[revision_key]["why_now"]).strip()
        if not reason:
            reason = str(allowed_revisions[revision_key]["why_now"])
        decisions.append(
            NotebookRevisionDecision(
                revision_key=revision_key,
                reason=reason,
            )
        )
        if len(decisions) >= max_revision_decisions:
            break

    return NotebookRevisionPlan(decisions=decisions)


def build_notebook_revision_plan_with_trace(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    chart_contexts: list[dict[str, Any]],
    llm_client=None,
    max_revision_decisions: int = MAX_REVISION_DECISIONS,
    evidence_pack: dict[str, Any] | None = None,
    llm_profile: str | None = None,
) -> tuple[NotebookRevisionPlan, LLMStageTrace]:
    empty_plan = NotebookRevisionPlan()
    if max_revision_decisions <= 0:
        profile = llm_profile or "selected"
        return empty_plan, build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=llm_client,
            status="skipped_by_profile",
            reason=f"{profile} profile disables notebook revision decision.",
            attempted=False,
            applied=False,
        )
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return empty_plan, build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=llm_client,
            status="disabled",
            reason="LLM is disabled, so the revise loop recorded no append decisions.",
        )

    allowed_revisions = _build_allowed_revisions(report, schema_mapping)
    if not allowed_revisions:
        return empty_plan, build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=llm_client,
            status="not_applicable",
            reason="No safe revision templates were eligible for this dataset; the revise loop had no append decisions to request.",
        )

    suggest_revisions = getattr(llm_client, "suggest_notebook_revision_decisions", None)
    if suggest_revisions is None:
        return empty_plan, build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=llm_client,
            status="skipped",
            reason="The configured LLM client does not provide notebook revision decisions.",
        )

    try:
        payload = suggest_revisions(
            report_context=(
                {
                    "task": "Decide safe notebook revision from evidence pack and revision playbook.",
                    "evidence_pack": evidence_pack,
                }
                if isinstance(evidence_pack, dict)
                else _report_context_for_revision(report)
            ),
            chart_contexts=_compact_chart_contexts(chart_contexts),
            allowed_revisions=list(allowed_revisions.values()),
        )
    except Exception as exc:
        return empty_plan, build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=llm_client,
            status="fallback_on_error",
            reason=(
                "Notebook revision decision failed, so the revise loop kept the initial notebook only. "
                f"{describe_llm_error(exc)}"
            ),
            attempted=True,
        )

    if isinstance(payload, NotebookRevisionPlan):
        plan = _sanitize_revision_plan(
            payload.model_dump(),
            allowed_revisions,
            max_revision_decisions=max_revision_decisions,
        )
    elif isinstance(payload, dict):
        plan = _sanitize_revision_plan(
            payload,
            allowed_revisions,
            max_revision_decisions=max_revision_decisions,
        )
    else:
        return empty_plan, build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=llm_client,
            status="fallback_invalid_payload",
            reason="Notebook revision decision LLM returned an unsupported payload, so no revision sections were appended.",
            attempted=True,
        )

    if not plan.decisions:
        return plan, build_llm_stage_trace(
            stage="notebook_revision_decision",
            llm_client=llm_client,
            status="llm_no_change",
            reason="LLM did not choose any whitelisted revision sections, so the initial notebook stayed unchanged.",
            attempted=True,
        )

    return plan, build_llm_stage_trace(
        stage="notebook_revision_decision",
        llm_client=llm_client,
        status="llm_applied",
        reason="LLM selected safe revision sections from the sales-analysis playbook.",
        attempted=True,
        applied=True,
    )


def _discount_bucket_loss_rate_cells(schema_mapping: SchemaMapping) -> list[str]:
    discount_col = _original_column(schema_mapping, "discount", "Discount")
    profit_col = _original_column(schema_mapping, "profit", "Profit")
    sales_col = _original_column(schema_mapping, "sales_amount", "Sales")
    return [
        "\n".join(
            [
                f"discount_col = {discount_col!r}",
                f"profit_col = {profit_col!r}",
                f"sales_col = {sales_col!r}",
                "",
                "discount_profit = clean_df[[discount_col, profit_col, sales_col]].dropna().copy()",
                "discount_profit['discount_bucket'] = pd.cut(",
                "    discount_profit[discount_col],",
                "    bins=[-0.001, 0.0, 0.1, 0.2, 0.3, 1.0],",
                "    labels=['0%', '0-10%', '10-20%', '20-30%', '30%+'],",
                "    include_lowest=True,",
                ")",
                "discount_loss_rate = (",
                "    discount_profit.groupby('discount_bucket', observed=False)",
                "    .agg(",
                "        order_count=(profit_col, 'size'),",
                "        loss_orders=(profit_col, lambda series: int((series < 0).sum())),",
                "        avg_profit=(profit_col, 'mean'),",
                "        avg_sales=(sales_col, 'mean'),",
                "    )",
                "    .reset_index()",
                ")",
                "discount_loss_rate['loss_rate'] = discount_loss_rate.apply(",
                "    lambda row: row['loss_orders'] / row['order_count'] if row['order_count'] else 0.0,",
                "    axis=1,",
                ")",
                "discount_loss_rate = discount_loss_rate.sort_values('loss_rate', ascending=False)",
                "discount_loss_rate",
            ]
        ),
        "\n".join(
            [
                "fig, ax1 = plt.subplots(figsize=(10, 5))",
                "sns.barplot(",
                "    data=discount_loss_rate,",
                "    x='discount_bucket',",
                "    y='avg_profit',",
                "    color='#4C72B0',",
                "    ax=ax1,",
                ")",
                "ax1.set_title('各折扣区间平均利润与亏损率')",
                "ax1.set_xlabel('折扣区间')",
                "ax1.set_ylabel('平均利润')",
                "",
                "ax2 = ax1.twinx()",
                "ax2.plot(",
                "    ax1.get_xticks(),",
                "    discount_loss_rate['loss_rate'],",
                "    color='#DD8452',",
                "    marker='o',",
                ")",
                "ax2.set_ylabel('亏损订单占比')",
                "plt.tight_layout()",
                "plt.show()",
            ]
        ),
    ]


def _top_product_profit_bridge_cells(schema_mapping: SchemaMapping) -> list[str]:
    product_col = _original_column(schema_mapping, "product_name", "Product Name")
    sales_col = _original_column(schema_mapping, "sales_amount", "Sales")
    profit_col = _original_column(schema_mapping, "profit", "Profit")
    return [
        "\n".join(
            [
                f"product_col = {product_col!r}",
                f"sales_col = {sales_col!r}",
                f"profit_col = {profit_col!r}",
                "",
                "top_product_profit_bridge = (",
                "    clean_df.groupby(product_col, as_index=False)",
                "    .agg({sales_col: 'sum', profit_col: 'sum'})",
                f"    .sort_values(sales_col, ascending=False)",
                "    .head(12)",
                ")",
                "top_product_profit_bridge['profit_margin'] = top_product_profit_bridge.apply(",
                "    lambda row: row[profit_col] / row[sales_col] if row[sales_col] else 0.0,",
                "    axis=1,",
                ")",
                "top_product_profit_bridge",
            ]
        ),
        "\n".join(
            [
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))",
                "sns.barplot(",
                "    data=top_product_profit_bridge,",
                "    x=sales_col,",
                "    y=product_col,",
                "    color='#4C72B0',",
                "    ax=axes[0],",
                ")",
                "axes[0].set_title('头部商品销售额对比')",
                "axes[0].set_xlabel('销售额')",
                "axes[0].set_ylabel('商品')",
                "",
                "sns.barplot(",
                "    data=top_product_profit_bridge.sort_values(profit_col, ascending=False),",
                "    x=profit_col,",
                "    y=product_col,",
                "    color='#55A868',",
                "    ax=axes[1],",
                ")",
                "axes[1].set_title('头部商品利润对比')",
                "axes[1].set_xlabel('利润')",
                "axes[1].set_ylabel('商品')",
                "plt.tight_layout()",
                "plt.show()",
            ]
        ),
    ]


def _weak_segment_category_cross_cells(schema_mapping: SchemaMapping) -> list[str]:
    segment_col = _original_column(schema_mapping, "segment", "Segment")
    region_col = _original_column(schema_mapping, "region", "Region")
    category_col = _original_column(schema_mapping, "category", "Category")
    sales_col = _original_column(schema_mapping, "sales_amount", "Sales")
    profit_col = _original_column(schema_mapping, "profit", "Profit")
    has_region = "region" in set(schema_mapping.field_mapping.values())
    group_cols = "[segment_col, region_col, category_col]" if has_region else "[segment_col, category_col]"
    slice_group_cols = "[segment_col, region_col]" if has_region else "[segment_col]"
    return [
        "\n".join(
            [
                f"segment_col = {segment_col!r}",
                f"region_col = {region_col!r}",
                f"category_col = {category_col!r}",
                f"sales_col = {sales_col!r}",
                f"profit_col = {profit_col!r}",
                "",
                "segment_category_profit = (",
                f"    clean_df.groupby({group_cols}, as_index=False)",
                "    .agg({sales_col: 'sum', profit_col: 'sum'})",
                ")",
                "segment_category_profit['profit_margin'] = segment_category_profit.apply(",
                "    lambda row: row[profit_col] / row[sales_col] if row[sales_col] else 0.0,",
                "    axis=1,",
                ")",
                "segment_totals = (",
                f"    segment_category_profit.groupby({slice_group_cols}, as_index=False)",
                "    .agg({sales_col: 'sum', profit_col: 'sum'})",
                ")",
                "segment_totals['profit_margin'] = segment_totals.apply(",
                "    lambda row: row[profit_col] / row[sales_col] if row[sales_col] else 0.0,",
                "    axis=1,",
                ")",
                "segment_totals = segment_totals.sort_values(['profit_margin', profit_col, sales_col], ascending=[True, True, False])",
                "weak_slice = segment_totals.iloc[0]",
                "weak_segment = weak_slice[segment_col]",
                ("weak_region = weak_slice[region_col]" if has_region else "weak_region = None"),
                (
                    "weak_slice_label = f'{weak_segment} / {weak_region}'"
                    if has_region
                    else "weak_slice_label = str(weak_segment)"
                ),
                "weak_segment_category = (",
                "    segment_category_profit[segment_category_profit[segment_col] == weak_segment]",
                ")",
                (
                    "weak_segment_category = weak_segment_category[weak_segment_category[region_col] == weak_region]"
                    if has_region
                    else "# 当前缺少区域字段，弱势对象保持为单一客群。"
                ),
                "weak_segment_category = weak_segment_category.sort_values(profit_col, ascending=True)",
                "weak_segment_category",
            ]
        ),
        "\n".join(
            [
                "plt.figure(figsize=(10, 5))",
                "sns.barplot(",
                "    data=weak_segment_category,",
                "    x=profit_col,",
                "    y=category_col,",
                "    color='#C44E52',",
                ")",
                "plt.title(f'弱势组合切片 {weak_slice_label} 的类目利润表现')",
                "plt.xlabel('利润')",
                "plt.ylabel('类目')",
                "plt.tight_layout()",
                "plt.show()",
            ]
        ),
    ]


def _revision_markdown(decision: NotebookRevisionDecision) -> str:
    title = REVISION_TITLES.get(decision.revision_key, decision.revision_key)
    reason = _business_revision_reason(decision)
    return "\n".join(
        [
            f"## 追加分析：{title}",
            "",
            "这一节由 Agent 在第一次执行后追加，沿着前面已经暴露出来但还没有拆透的经营问题继续展开。",
            "",
            f"**业务问题 / 分析问题：** {reason}",
        ]
    )


def _business_revision_reason(decision: NotebookRevisionDecision) -> str:
    reason = str(decision.reason or "").strip()
    if reason and not re.search(r"[A-Za-z]{4,}", reason):
        return reason
    fallback = {
        "discount_bucket_loss_rate": "折扣与利润章节已经暴露出利润承压，需要继续拆到折扣区间，确认平均利润和亏损率是否同步恶化。",
        "top_product_profit_bridge": "商品贡献章节已经显示头部对象，需要继续核查高销售商品是否同步贡献利润。",
        "weak_segment_category_cross": "客群区域章节已经识别出弱势组合切片，需要继续拆到类目层，确认问题集中在哪些商品结构上。",
    }
    return fallback.get(decision.revision_key, "第一次执行后仍有经营问题没有拆透，需要沿着已验证证据继续补充分析。")


def _revision_cells(decision: NotebookRevisionDecision, schema_mapping: SchemaMapping) -> list[str]:
    if decision.revision_key == "discount_bucket_loss_rate":
        return _discount_bucket_loss_rate_cells(schema_mapping)
    if decision.revision_key == "top_product_profit_bridge":
        return _top_product_profit_bridge_cells(schema_mapping)
    if decision.revision_key == "weak_segment_category_cross":
        return _weak_segment_category_cross_cells(schema_mapping)
    raise KeyError(f"Unsupported revision key: {decision.revision_key}")


def _revision_playbook_markdown(decision: NotebookRevisionDecision) -> str:
    playbook = {
        "discount_bucket_loss_rate": {
            "question": "高折扣订单是否已经集中产生亏损，亏损率从哪个折扣区间开始明显抬升？",
            "business_value": "把“折扣高不高”转成“哪些折扣区间正在侵蚀利润”，便于后续制定折扣红线和审批规则。",
            "reading_guide": "先看各折扣区间的亏损率，再对照平均利润。如果 20% 以上区间亏损率和平均利润同时恶化，就说明问题更像结构性让利，而不是少数异常订单。",
        },
        "top_product_profit_bridge": {
            "question": "头部高销售额商品是否也同步贡献利润，还是只贡献了规模？",
            "business_value": "把商品销售排名和利润表现放在一起看，避免把低利润甚至亏损的头部商品误判成优质商品。",
            "reading_guide": "先看销售额排名，再看利润排名和利润率。如果销售额靠前但利润靠后，需要继续检查折扣、成本或履约结构。",
        },
        "weak_segment_category_cross": {
            "question": "弱势客群的问题究竟落在哪些类目上，是整体偏弱还是少数类目拖累？",
            "business_value": "把客群问题继续拆到类目层，方便把后续动作落到选品、定价、区域经营或资源配置上。",
            "reading_guide": "先定位利润最弱的客群，再看该客群下哪些类目利润最低。如果弱势集中在少数类目，处理动作应优先针对类目，而不是粗暴调整整个客群策略。",
        },
    }.get(
        decision.revision_key,
        {
            "question": "当前结果里还有哪个经营问题没有拆透？",
            "business_value": "把已发现的问题继续拆细，避免结论停留在表层现象。",
            "reading_guide": "先看补充表格，再看图表中的集中区间、异常点和弱势切片。",
        },
    )
    return "\n".join(
        [
            "### 追加分析口径",
            "",
            f"**关键证据：** {playbook['question']}",
            "",
            f"**经营含义 / 业务价值：** {playbook['business_value']}",
            "",
            f"**建议动作 / 阅读方式：** {playbook['reading_guide']}",
            "",
            "**下一步需要的数据：** 订单级销售额、利润、折扣、商品和切片字段，用来验证追加分析发现是否来自结构性经营问题。",
        ]
    )


def augment_outline_with_revision_sections(
    outline: NotebookOutline,
    revision_plan: NotebookRevisionPlan,
) -> NotebookOutline:
    if not revision_plan.decisions:
        return outline

    sections = list(outline.sections)
    for decision in revision_plan.decisions:
        parent_section_id = REVISION_PARENT_SECTIONS.get(decision.revision_key)
        if parent_section_id is None:
            continue
        sections.append(
            NotebookSection(
                section_id=parent_section_id,
                title=revision_heading_title(decision.revision_key),
                purpose="Agent 追加的补充分析。",
            )
        )
    return NotebookOutline(title=outline.title, sections=sections)


def apply_notebook_revision_plan(
    notebook_path: Path,
    output_path: Path,
    revision_plan: NotebookRevisionPlan,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    notebook_output_mode: str = "full",
) -> Path:
    notebook = load_notebook(notebook_path)

    output_revision_plan = revision_plan_for_output(
        revision_plan,
        notebook_output_mode=notebook_output_mode,
    )

    if not output_revision_plan.decisions:
        return save_notebook(notebook, output_path)

    insertion_index = find_markdown_heading_index(notebook, "## 结论与行动建议")
    if insertion_index is None:
        insertion_index = len(notebook.cells)

    trace_lines = [
        "## Agent 追加分析记录",
        "",
        "本轮在第一次执行后，Agent 继续检查了图表和模块结果，并追加了以下补充分析：",
        "",
    ]
    for index, decision in enumerate(output_revision_plan.decisions, start=1):
        title = REVISION_TITLES.get(decision.revision_key, decision.revision_key)
        trace_lines.append(f"{index}. {title}：{decision.reason}")
    insert_markdown_cell_at_index(notebook, insertion_index, "\n".join(trace_lines))
    insertion_index += 1

    compact_mode = is_compact_notebook_mode(notebook_output_mode)
    for decision in output_revision_plan.decisions:
        insert_markdown_cell_at_index(notebook, insertion_index, _revision_markdown(decision))
        insertion_index += 1
        if not compact_mode:
            insert_markdown_cell_at_index(
                notebook,
                insertion_index,
                _revision_playbook_markdown(decision),
            )
            insertion_index += 1
        code_cells = _revision_cells(decision, schema_mapping)
        if compact_mode:
            code_cells = code_cells[:2]
        for code_cell in code_cells:
            insert_code_cell_at_index(notebook, insertion_index, code_cell)
            insertion_index += 1

    return save_notebook(notebook, output_path)


def revision_plan_for_output(
    revision_plan: NotebookRevisionPlan,
    *,
    notebook_output_mode: str,
) -> NotebookRevisionPlan:
    if not is_compact_notebook_mode(notebook_output_mode):
        return revision_plan
    return NotebookRevisionPlan(decisions=list(revision_plan.decisions[:1]))


def build_notebook_agent_loop_state(
    *,
    task_id: str,
    dataset_type: str,
    outline_section_ids: list[str],
    initial_chart_contexts: list[dict[str, Any]],
    final_chart_contexts: list[dict[str, Any]],
    revision_plan: NotebookRevisionPlan,
    llm_trace_payload: dict[str, Any],
) -> NotebookAgentLoopState:
    revision_keys = [decision.revision_key for decision in revision_plan.decisions]
    inspect_sections = sorted(
        {
            str(item.get("section_id"))
            for item in initial_chart_contexts
            if str(item.get("section_id", "")).strip()
        }
    )

    stages = [
        NotebookAgentLoopStage(
            stage="plan",
            status="completed",
            summary="识别销售数据类型并确定本轮分析结构。",
            metadata={
                "outline_section_ids": outline_section_ids,
                "section_count": len(outline_section_ids),
            },
        ),
        NotebookAgentLoopStage(
            stage="compose",
            status="completed",
            summary="生成 notebook 结构、叙事说明和初始分析代码。",
            metadata={
                "outline_status": str(llm_trace_payload.get("notebook_outline", {}).get("status", "")),
                "content_status": str(llm_trace_payload.get("notebook_content", {}).get("status", "")),
            },
        ),
        NotebookAgentLoopStage(
            stage="execute",
            status="completed",
            summary="执行首轮 notebook，得到真实表格和图表输出。",
            metadata={
                "initial_chart_count": len(initial_chart_contexts),
            },
        ),
        NotebookAgentLoopStage(
            stage="inspect",
            status="completed",
            summary="读取执行结果并识别值得继续追问的图表切片。",
            metadata={
                "chart_count": len(initial_chart_contexts),
                "inspected_section_ids": inspect_sections,
            },
        ),
        NotebookAgentLoopStage(
            stage="revise",
            status="applied" if revision_keys else "no_change",
            summary=(
                "基于执行结果追加了安全白名单分析。"
                if revision_keys
                else "本轮未追加新的分析 section。"
            ),
            metadata={
                "llm_revision_status": str(
                    llm_trace_payload.get("notebook_revision_decision", {}).get("status", "")
                ),
                "selected_revision_keys": revision_keys,
                "revision_count": len(revision_keys),
            },
        ),
        NotebookAgentLoopStage(
            stage="finalize",
            status="completed",
            summary="完成图后分析回填并输出最终 notebook 产物。",
            metadata={
                "final_chart_count": len(final_chart_contexts),
                "postrun_reflection_status": str(
                    llm_trace_payload.get("postrun_chart_reflection", {}).get("status", "")
                ),
            },
        ),
    ]

    return NotebookAgentLoopState(
        task_id=task_id,
        dataset_type=dataset_type,
        revision_count=len(revision_keys),
        stages=stages,
    )
