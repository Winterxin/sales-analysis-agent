from __future__ import annotations

from typing import Any

from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping

from app.services.notebook import action_evidence as action_evidence_selector
from app.services.notebook.capability_matrix import capability_rows
from app.services.notebook.evidence_formatter import (
    collect_evidence_items,
    collect_risk_items,
    evidence_pack_findings,
    remove_loss_evidence_when_no_negative_profit,
)
from app.services.notebook.field_utils import field_list, mapped_fields
from app.services.notebook.formatters import format_percent, format_scalar
from app.services.notebook.markdown_sanitizer import clean_business_text, sanitize_final_markdown_text


def dedupe_action_texts(actions: list[str], profile: dict[str, Any]) -> list[str]:
    deduped: list[str] = []
    seen_objects: set[str] = set()
    object_terms = [
        ("折扣", ("折扣", "审批")),
        ("类目", ("类目",)),
        ("商品", ("商品", "SKU")),
        ("组合切片", ("组合切片", "切片", "客群", "区域")),
        ("建模高风险订单", ("建模", "高风险订单", "训练标签")),
        ("客户订单", ("客户", "复购", "订单")),
        ("波动", ("波动", "月份", "趋势")),
    ]
    for action in actions:
        object_key = next(
            (key for key, terms in object_terms if any(term in action for term in terms)),
            action[:24],
        )
        if object_key in seen_objects:
            continue
        seen_objects.add(object_key)
        deduped.append(action)
    fallback_actions = [
        "按类目复盘销售额、利润率和折扣结构，优先修正利润率偏低的类目策略。",
        "下钻高销售低利润商品，核查折扣、成本和履约费用是否压低利润。",
        "锁定弱势组合切片，拆解客群、区域和类目之间的利润质量差异。",
        "沉淀高风险订单复核标签，供后续模型迭代使用。",
        f"围绕月度波动 {format_percent(profile.get('monthly_volatility'))} 建立经营节奏复盘。",
    ]
    for action in fallback_actions:
        if len(deduped) >= 3:
            break
        object_key = next(
            (key for key, terms in object_terms if any(term in action for term in terms)),
            action[:24],
        )
        if object_key in seen_objects:
            continue
        seen_objects.add(object_key)
        deduped.append(action)
    return deduped[:3]


def evidence_action_rows(
    evidence_pack: dict[str, Any] | None,
    selected_focuses: list[str],
    *,
    has_discount: bool = True,
) -> list[dict[str, str]]:
    if not isinstance(evidence_pack, dict):
        return []
    signature = evidence_pack.get("dataset_signature") or {}
    dominant_story = str(signature.get("dominant_story", ""))
    evidence_by_focus = evidence_pack.get("focus_evidence") or {}

    def first_summary(focus: str) -> str:
        for item in evidence_by_focus.get(focus, []) or []:
            if isinstance(item, dict):
                meaning = clean_business_text(item.get("business_meaning", ""))
                if meaning:
                    return meaning
        facts = evidence_pack_findings(evidence_pack)
        return facts[0] if facts else "数据限制说明：缺少可验证字段"

    rows: list[dict[str, str]] = []
    if has_discount and (
        dominant_story in {"discount_loss", "profit_quality"} or "discount_erosion_focus" in selected_focuses
    ):
        rows.append(
            {
                "priority": "P1",
                "issue": "折扣和利润质量需要先收敛风险口径",
                "evidence": first_summary("discount_erosion_focus"),
                "action": "按折扣区间设置审批阈值，优先处理证据中命中的高风险折扣或低利润切片。",
                "required_fields": field_list({"discount", "profit", "sales_amount"}),
            }
        )
    if "country_market_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P1",
                "issue": "国家市场贡献差异需要单独经营",
                "evidence": first_summary("country_market_focus"),
                "action": "按国家拆分销售额、订单数、客单价和客户数，优先复核头部市场集中度。",
                "required_fields": field_list({"country", "sales_amount"}),
            }
        )
    if "customer_order_structure_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P2",
                "issue": "客户复购和订单篮子结构影响增长质量",
                "evidence": first_summary("customer_order_structure_focus"),
                "action": "按复购层级和订单行数分组运营，分别提升复购客户占比和高价值订单占比。",
                "required_fields": field_list({"customer_id", "order_id", "quantity", "sales_amount"}),
            }
        )
    if "productline_performance_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P1",
                "issue": "PRODUCTLINE 组合决定销售主线",
                "evidence": first_summary("productline_performance_focus"),
                "action": "按 PRODUCTLINE 分配销售资源，保留高贡献产品线放量动作并修正低效率产品线。",
                "required_fields": field_list({"productline", "sales_amount"}),
            }
        )
    if "deal_size_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P1",
                "issue": "DEALSIZE 结构决定大单和小单策略",
                "evidence": first_summary("deal_size_focus"),
                "action": "按 DEALSIZE 拆分大中小单，分别设置大单跟进和小单效率目标。",
                "required_fields": field_list({"deal_size", "sales_amount"}),
            }
        )
    if "order_status_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P2",
                "issue": "STATUS 影响收入确认和履约质量",
                "evidence": first_summary("order_status_focus"),
                "action": "按 STATUS 追踪取消、争议或未完成订单，压降异常状态占比。",
                "required_fields": field_list({"order_status", "sales_amount"}),
            }
        )
    if "product_concentration_focus" in selected_focuses and len(rows) < 3:
        rows.append(
            {
                "priority": f"P{len(rows) + 1}",
                "issue": "商品或类目集中度需要转成组合动作",
                "evidence": first_summary("product_concentration_focus"),
                "action": "按头部商品和类目拆分增长责任，检查高销售对象是否同步贡献利润。",
                "required_fields": field_list({"product_name", "sales_amount"}),
            }
        )
    return rows


def profile_evidence_items(profile: dict[str, Any], selected_focuses: list[str]) -> list[str]:
    items: list[str] = []
    if profile.get("order_count") is not None or profile.get("customer_count") is not None:
        items.append(
            "数据结构显示订单数为 "
            f"{format_scalar(profile.get('order_count'))}，客户数为 {format_scalar(profile.get('customer_count'))}，"
            f"平均每单行数为 {format_scalar(profile.get('line_per_order_avg'))}。"
        )
    if profile.get("product_count") is not None:
        items.append(
            f"商品数为 {format_scalar(profile.get('product_count'))}，头部类目销售占比为 {format_percent(profile.get('top_category_sales_share'))}。"
        )
    if profile.get("negative_profit_rate") is not None:
        items.append(
            f"负利润记录占比为 {format_percent(profile.get('negative_profit_rate'))}，利润率分布跨度为 {format_percent(profile.get('profit_margin_spread'))}。"
        )
    if profile.get("monthly_volatility") is not None:
        items.append(
            f"月度销售波动系数为 {format_percent(profile.get('monthly_volatility'))}，最近一期增长率为 {format_percent(profile.get('recent_growth_rate'))}。"
        )
    if "productline_performance_focus" in selected_focuses:
        items.insert(
            0,
            f"PRODUCTLINE 数量为 {format_scalar(profile.get('productline_count'))}，DEALSIZE 数量为 {format_scalar(profile.get('distinct_deal_size_count'))}，STATUS 数量为 {format_scalar(profile.get('distinct_order_status_count'))}。",
        )
    return [clean_business_text(item) for item in items if clean_business_text(item)]


def focus_evidence_items(
    report: AnalysisReport,
    profile: dict[str, Any],
    selected_focuses: list[str],
) -> list[str]:
    evidence = collect_evidence_items(report)
    filtered: list[str] = []
    if "discount_erosion_focus" in selected_focuses and profile.get("has_discount") and profile.get("has_profit"):
        filtered.extend([item for item in evidence if any(term in item for term in ("折扣", "负利润", "亏损"))])
    if "profit_quality_focus" in selected_focuses and profile.get("has_profit"):
        profit_items = [
            item
            for item in evidence
            if any(term in item for term in ("利润", "利润率", "低利润"))
        ]
        if profile.get("negative_profit_rate") == 0:
            profit_items = [
                item
                for item in profit_items
                if "亏损来源" not in item
                and "亏损商品" not in item
                and "利润为负" not in item
                and "负利润" not in item
            ]
            profit_items.insert(0, "负利润记录占比为 0，本轮应关注利润率结构差异而不是亏损排查。")
        filtered.extend(profit_items)
    if "product_concentration_focus" in selected_focuses:
        filtered.extend([item for item in evidence if any(term in item for term in ("商品", "Top", "头部", "Product"))])
    if "country_market_focus" in selected_focuses:
        filtered.extend([item for item in evidence if any(term in item for term in ("国家", "Country", "市场"))])
    if "customer_order_structure_focus" in selected_focuses:
        filtered.append(
            f"复购客户占比为 {format_percent(profile.get('repeat_customer_rate'))}，平均每单商品行数为 {format_scalar(profile.get('line_per_order_avg') or profile.get('avg_basket_size'))}。"
        )
    if "productline_performance_focus" in selected_focuses:
        filtered.append("数据包含 PRODUCTLINE，可按产品线拆解销售贡献和产品组合效率。")
    if "deal_size_focus" in selected_focuses:
        filtered.append("数据包含 DEALSIZE，可按交易规模拆解销售贡献和订单结构。")
    if "order_status_focus" in selected_focuses:
        filtered.append("数据包含 STATUS，可识别已发货、取消或异常订单状态对销售结构的影响。")
    if "trend_volatility_focus" in selected_focuses:
        filtered.append(
            f"月度波动系数为 {format_percent(profile.get('monthly_volatility'))}，最近一期增长率为 {format_percent(profile.get('recent_growth_rate'))}。"
        )
    deduped: list[str] = []
    for item in filtered:
        text = clean_business_text(item)
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def focus_risk_items(
    report: AnalysisReport,
    profile: dict[str, Any],
    selected_focuses: list[str],
) -> list[str]:
    risks: list[str] = []
    if "discount_erosion_focus" in selected_focuses and profile.get("has_discount") and profile.get("has_profit"):
        risks.extend(collect_risk_items(report))
    if "profit_quality_focus" in selected_focuses and profile.get("has_profit"):
        negative_rate = profile.get("negative_profit_rate")
        if isinstance(negative_rate, (int, float)) and negative_rate > 0:
            risks.append(f"负利润记录占比为 {format_percent(negative_rate)}，需要优先定位亏损来源。")
        else:
            risks.append("负利润记录占比为 0，本轮不把亏损排查作为主线，应转向利润质量和结构差异。")
    if "country_market_focus" in selected_focuses:
        risks.append(f"最大国家销售占比为 {format_percent(profile.get('top_country_sales_share'))}，需关注市场集中度。")
    if "customer_order_structure_focus" in selected_focuses:
        risks.append("客户复购和订单篮子结构会影响真实增长质量，需避免只看销售总额。")
    if "deal_size_focus" in selected_focuses:
        risks.append("交易规模结构可能掩盖大单依赖或小单效率问题。")
    if "order_status_focus" in selected_focuses:
        risks.append("订单状态差异可能影响实际可确认收入。")
    deduped: list[str] = []
    for item in risks:
        text = clean_business_text(item)
        if text and text not in deduped:
            deduped.append(text)
    return deduped


def focus_actions(
    profile: dict[str, Any],
    selected_focuses: list[str],
    evidence: list[str],
) -> list[str]:
    sample = evidence or ["当前数据画像"]
    actions: list[str] = []
    if "discount_erosion_focus" in selected_focuses and profile.get("has_discount") and profile.get("has_profit"):
        actions.append(
            f"设定折扣审批阈值，并用负利润占比 {format_percent(profile.get('negative_profit_rate'))} 和利润率跨度 {format_percent(profile.get('profit_margin_spread'))} 作为复盘基线。"
        )
    if "profit_quality_focus" in selected_focuses and profile.get("has_profit"):
        actions.append(f"下钻利润质量异常切片，核查成本、定价和履约因素。")
    if "country_market_focus" in selected_focuses and profile.get("country_count") != 1:
        actions.append("按国家市场建立销售额、订单数和客单价看板，优先复盘高占比市场。")
    if "customer_order_structure_focus" in selected_focuses:
        actions.append(
            f"按复购率 {format_percent(profile.get('repeat_customer_rate'))} 和平均每单行数 {format_scalar(profile.get('line_per_order_avg'))} 分层运营，提升复购客户和高价值订单占比。"
        )
    if "productline_performance_focus" in selected_focuses:
        actions.append("按产品线调整资源投放，区分高销售产品线和低效率产品线。")
    if "deal_size_focus" in selected_focuses:
        actions.append("按 DEALSIZE 拆分大中小单，分别设置大单跟进和小单效率目标。")
    if "order_status_focus" in selected_focuses:
        actions.append("按 STATUS 追踪取消、争议或未完成订单，减少异常状态对收入确认的影响。")
    if "trend_volatility_focus" in selected_focuses:
        actions.append("围绕高波动月份建立补货和促销节奏复盘，降低月度波动。")
    while len(actions) < 3:
        if len(actions) == 0:
            actions.append(f"补齐缺失字段并复跑分析，验证 {sample[0]}。")
        else:
            actions.append(f"围绕 {sample[min(len(actions), len(sample)-1)]} 建立复盘责任人和验收口径。")
    return actions[:3]


def action_plan_rows(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    analysis_focus: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    evidence = collect_evidence_items(report)
    profile = dataset_profile or {}
    evidence = remove_loss_evidence_when_no_negative_profit(evidence, profile)
    if not evidence:
        evidence = ["数据限制说明：当前 notebook 仅能使用已映射字段和模块结果生成行动建议。"]
    rows: list[dict[str, str]] = []
    mapped = mapped_fields(schema_mapping)
    has_discount = "discount" in mapped
    selected_focuses = list((analysis_focus or {}).get("selected_focuses", []))
    used_action_evidence: set[str] = set()
    evidence_rows = evidence_action_rows(evidence_pack, selected_focuses, has_discount=has_discount)
    if evidence_rows:
        has_specific_product_category = {"product_name", "category", "sales_amount", "profit"} <= mapped
        for row in evidence_rows:
            if (
                has_specific_product_category
                and ("商品或类目" in row.get("issue", "") or "商品和类目" in row.get("action", ""))
            ):
                continue
            action_type = action_evidence_selector.action_type_for_row(row)
            if action_type:
                row = dict(row)
                row["evidence"] = action_evidence_selector.select_action_evidence(
                    action_type,
                    report,
                    profile,
                    evidence_pack,
                    used_action_evidence,
                )
            else:
                used_action_evidence.add(row.get("evidence", ""))
            rows.append(row)
    capability_status_rows = capability_rows(schema_mapping)
    enabled_names = {row["capability"] for row in capability_status_rows if row["status"] == "enabled"}

    has_discount_action = any("折扣" in row.get("issue", "") or "折扣" in row.get("action", "") for row in rows)
    if "discount_erosion_focus" in selected_focuses and "折扣侵蚀分析" in enabled_names and not has_discount_action:
        rows.append(
            {
                "priority": "P1",
                "issue": "折扣可能侵蚀利润",
                "evidence": action_evidence_selector.select_action_evidence("discount", report, profile, evidence_pack, used_action_evidence),
                "action": "设定高折扣审批阈值，并逐周复盘折扣区间的利润变化。",
                "required_fields": field_list({"discount", "profit"}),
            }
        )
    if {"category", "sales_amount", "profit"} <= mapped and (
        "profit_quality_focus" in selected_focuses or "product_concentration_focus" in selected_focuses
    ):
        rows.append(
            {
                "priority": "P1",
                "issue": "类目利润质量需要形成经营责任",
                "evidence": action_evidence_selector.select_action_evidence("category", report, profile, evidence_pack, used_action_evidence),
                "action": (
                    "按类目复盘销售额、利润率和折扣结构，优先修正利润率偏低的类目策略。"
                    if has_discount
                    else "按类目复盘销售额、利润率、成本和定价结构，优先修正利润率偏低的类目策略。"
                ),
                "required_fields": field_list({"category", "sales_amount", "profit"}),
            }
        )
    if {"product_name", "sales_amount", "profit"} <= mapped and (
        "profit_quality_focus" in selected_focuses or "product_concentration_focus" in selected_focuses
    ):
        rows.append(
            {
                "priority": "P1",
                "issue": "高销售低利润商品需要单独治理",
                "evidence": action_evidence_selector.select_action_evidence("product", report, profile, evidence_pack, used_action_evidence),
                "action": (
                    "下钻头部商品的折扣、成本和履约费用，调整低利润商品的定价或促销资源。"
                    if has_discount
                    else "下钻头部商品的成本、定价和履约费用，调整低利润商品的定价或资源配置。"
                ),
                "required_fields": field_list({"product_name", "sales_amount", "profit"}),
            }
        )
    if {"segment", "region", "category"} & mapped:
        rows.append(
            {
                "priority": "P2",
                "issue": "组合切片表现差异需要落到经营动作",
                "evidence": action_evidence_selector.select_action_evidence("slice", report, profile, evidence_pack, used_action_evidence),
                "action": "按客群、区域或类目建立责任清单，优先修正低利润切片的价格和商品组合。",
                "required_fields": field_list(({"segment", "region", "category"} & mapped) | {"sales_amount"}),
            }
        )
    if {"order_id", "sales_amount"} <= mapped and "profit_quality_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P3",
                "issue": "建模高风险订单需要先沉淀验收口径",
                "evidence": action_evidence_selector.select_action_evidence(
                    "discount" if has_discount else "product",
                    report,
                    profile,
                    evidence_pack,
                    used_action_evidence,
                ),
                "action": "使用当前模型结果建立人工复核优先级，并沉淀高风险订单复核标签，供后续模型迭代使用。",
                "required_fields": field_list({"order_id", "sales_amount", "profit"} | ({"discount"} if has_discount else set())),
            }
        )
    if "country_market_focus" in selected_focuses and profile.get("country_count") != 1:
        rows.append(
            {
                "priority": "P1",
                "issue": "国家市场贡献差异需要单独经营",
                "evidence": action_evidence_selector.select_action_evidence("country", report, profile, evidence_pack, used_action_evidence),
                "action": "按国家拆分销售额、订单数、客单价和客户数，优先复盘高占比市场。",
                "required_fields": field_list({"country", "sales_amount"}),
            }
        )
    if "customer_order_structure_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P2",
                "issue": "客户复购和订单篮子结构影响增长质量",
                "evidence": action_evidence_selector.select_action_evidence("customer_order", report, profile, evidence_pack, used_action_evidence),
                "action": "按客户复购层级和篮子大小设计运营动作，提高复购和高价值订单占比。",
                "required_fields": field_list({"customer_id", "order_id", "quantity", "sales_amount"}),
            }
        )
    if "productline_performance_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P1",
                "issue": "产品线表现决定销售主线",
                "evidence": action_evidence_selector.select_action_evidence("productline", report, profile, evidence_pack, used_action_evidence),
                "action": "按 PRODUCTLINE 分配销售资源，分别制定高贡献产品线放量和低效率产品线修正动作。",
                "required_fields": field_list({"productline", "sales_amount"}),
            }
        )
    if "deal_size_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P1",
                "issue": "交易规模结构可能决定销售主线",
                "evidence": action_evidence_selector.select_action_evidence("deal_size", report, profile, evidence_pack, used_action_evidence),
                "action": "分别制定大单跟进和小单效率策略，并监控不同 DEALSIZE 的销售贡献。",
                "required_fields": field_list({"deal_size", "sales_amount"}),
            }
        )
    if "order_status_focus" in selected_focuses:
        rows.append(
            {
                "priority": "P2",
                "issue": "STATUS 订单状态影响收入确认和履约质量",
                "evidence": action_evidence_selector.select_action_evidence("status", report, profile, evidence_pack, used_action_evidence),
                "action": "追踪取消、争议或未完成状态订单，压降异常状态占比。",
                "required_fields": field_list({"order_status", "sales_amount"}),
            }
        )
    deduped_rows: list[dict[str, str]] = []
    seen_objects: set[str] = set()
    object_terms = [
        ("建模高风险订单", ("建模高风险订单", "高风险订单", "训练标签")),
        ("组合切片", ("组合切片", "切片", "客群", "区域")),
        ("类目", ("类目",)),
        ("商品", ("商品", "SKU")),
        ("折扣", ("折扣", "审批")),
        ("国家", ("国家", "市场")),
        ("客户订单", ("客户", "复购", "订单篮子")),
        ("产品线", ("产品线", "PRODUCTLINE")),
        ("交易规模", ("交易规模", "DEALSIZE")),
        ("订单状态", ("订单状态", "STATUS")),
    ]
    for row in rows:
        issue_text = row.get("issue", "")
        text = f"{issue_text} {row.get('action', '')}"
        object_key = next(
            (key for key, terms in object_terms if any(term in issue_text for term in terms)),
            "",
        ) or next(
            (key for key, terms in object_terms if any(term in text for term in terms)),
            text[:24],
        )
        if object_key in seen_objects:
            continue
        seen_objects.add(object_key)
        deduped_rows.append(row)
    rows = deduped_rows

    if not rows:
        rows.append(
            {
                "priority": "P1",
                "issue": "字段不足导致部分经营问题不能直接验证",
                "evidence": evidence[0],
                "action": "补齐缺失字段后重新运行分析，并把新增字段纳入下次经营复盘。",
                "required_fields": "按 Capability Matrix 的数据限制说明补齐",
            }
        )
    return rows[:5]


def build_action_plan_markdown(
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    analysis_focus: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
) -> str:
    rows = action_plan_rows(report, schema_mapping, dataset_profile, analysis_focus, evidence_pack)
    lines = [
        "## Action Plan",
        "",
        "| priority | issue | evidence | action | required_fields |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        clean_row = {key: sanitize_final_markdown_text(value) for key, value in row.items()}
        lines.append(
            "| {priority} | {issue} | {evidence} | {action} | {required_fields} |".format(
                **clean_row
            )
        )
    return "\n".join(lines)


_dedupe_action_texts = dedupe_action_texts
_evidence_action_rows = evidence_action_rows
_profile_evidence_items = profile_evidence_items
_focus_evidence_items = focus_evidence_items
_focus_risk_items = focus_risk_items
_focus_actions = focus_actions
_action_plan_rows = action_plan_rows
_action_plan_markdown = build_action_plan_markdown
