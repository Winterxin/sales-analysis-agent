from __future__ import annotations

import json
import re
from types import SimpleNamespace

import app.application.analysis_run_service as analysis_run_service
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.client_report_builder import (
    build_client_report_view_model,
    build_client_report_payload_with_trace,
    render_client_report_html,
)
from app.services.notebook.summary_builder import build_final_conclusion_markdown


def _final_synthesis() -> dict[str, object]:
    return {
        "main_conclusions": [
            {
                "conclusion": "当前销售结构的核心矛盾是商品类目极度分散",
                "evidence": "79个类目中，头部类目销售额占比仅6.77%",
                "business_meaning": "资源分散会削弱采购、库存和营销效率，需要优先聚焦核心品类。",
            }
        ],
        "recommended_actions": [
            {
                "action": "对销售类目进行ABC分析",
                "linked_metric_or_segment": "头部类目销售占比6.77%，类目总数79个",
                "expected_use": "明确商品策略焦点",
                "display_text": "针对类目高度分散的问题，建议立即对类目进行ABC分析，识别贡献前80%销售额的核心品类。",
            },
            {
                "action": "补充促销和节假日字段",
                "linked_metric_or_segment": "R²为6.62%，MAPE为10.98%",
                "expected_use": "提升销售波动解释能力",
                "display_text": "预测模型解释力较弱，应补充促销、节假日、价格和库存字段，再评估模型是否值得升级。",
            },
        ],
    }


def _sales_regression_outcome() -> dict[str, object]:
    return {
        "primary_modeling_task": "sales_amount_regression",
        "modeling_status": "regression_usable",
        "modeling_value_level": "medium",
        "metrics_summary": {
            "best_model": "RandomForestRegressor-small",
            "best_model_r2": 0.0662,
            "best_model_mae": 241668.61,
            "best_model_mape": 10.98,
            "baseline_model": "naive_last_value",
            "baseline_mae": 253982.68,
            "improvement_vs_baseline": 4.85,
        },
        "limitations": [
            "这是轻量回归模型，不是生产级预测，不用于自动决策。",
            "尚未纳入促销、节假日、库存、门店活动等外部变量。",
        ],
    }


def _loss_risk_outcome() -> dict[str, object]:
    return {
        "primary_modeling_task": "loss_risk_classification",
        "modeling_status": "strong_model",
        "modeling_value_level": "high",
        "metrics_summary": {
            "best_model": "RandomForestClassifier-small",
            "best_recall": 0.88,
            "best_f1": 0.76,
            "best_roc_auc": 0.91,
            "false_positive_count": 12,
            "false_negative_count": 4,
        },
        "limitations": ["模型输出不代表因果关系，不用于自动决策。"],
    }


def _forecast_baseline_outcome() -> dict[str, object]:
    return {
        "primary_modeling_task": "sales_amount_forecast_baseline",
        "modeling_status": "baseline_evaluated",
        "metrics_summary": {
            "best_baseline": "naive_last_value",
            "best_baseline_mae": 435.97,
            "best_baseline_rmse": 580.8,
            "best_baseline_mape": 30.09,
            "forecast_status": "regression_evaluated",
            "series_granularity": "daily",
            "time_split": "chronological",
        },
        "limitations": ["这是时间顺序回测得到的 baseline 结果，不用于自动预测。"],
    }


def _weak_loss_risk_outcome() -> dict[str, object]:
    return {
        "primary_modeling_task": "loss_risk_classification",
        "modeling_status": "classification_evaluated",
        "modeling_value_level": "low",
        "metrics_summary": {
            "best_model": "LogisticRegression",
            "best_recall": 0.6667,
            "best_f1": 0.0976,
            "best_roc_auc": 0.997,
            "false_positive_count": 36,
            "false_negative_count": 1,
            "model_quality_status": "weak",
        },
    }


def _generic_report() -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=4,
        summary=[
            "总销售额为 1250000，负利润记录占比为 18.72%。",
            "高风险折扣区间的亏损率明显高于其他区间。",
        ],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1250000.0},
                findings=["销售额集中在少数高峰月份，需要结合促销节奏复盘。"],
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品与类目分析",
                chart_type="bar",
                findings=["头部类目贡献销售额，但利润质量存在分化。"],
            ),
            ModuleReport(
                module_id="dimension_breakdown_analysis",
                title="客群与区域分析",
                chart_type="bar",
                findings=["部分客群与区域组合的利润率偏低。"],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                summary_metrics={
                    "total_profit_amount": 180000.0,
                    "negative_profit_rate": 0.1872,
                },
                tables={
                    "discount_threshold_candidates": [
                        {
                            "discount_bucket": "20-30%",
                            "negative_profit_rate": 0.24,
                            "profit_margin": 0.08,
                            "avg_profit": 12.0,
                            "risk_level": "medium",
                        },
                        {
                            "discount_bucket": "30%+",
                            "negative_profit_rate": 0.72,
                            "profit_margin": -0.31,
                            "avg_profit": -44.5,
                            "risk_level": "high",
                        },
                    ],
                    "discount_cap_what_if": [
                        {
                            "scenario_name": "高风险折扣收紧情景估算",
                            "current_threshold": "30%+",
                            "high_risk_bucket": "30%+",
                            "target_discount_cap": 0.3,
                            "affected_row_count": 42,
                            "affected_sales_amount": 86000.0,
                            "current_profit": -12000.0,
                            "estimated_profit_after_cap": 9200.0,
                            "estimated_profit_delta": 21200.0,
                            "current_negative_profit_rate": 0.72,
                            "estimated_negative_profit_rate_after_cap": 0.38,
                        }
                    ],
                },
                findings=["折扣越高的区间利润质量越弱，需要优先设置审批红线。"],
            ),
            ModuleReport(
                module_id="loss_risk_modeling",
                title="亏损风险建模",
                chart_type="model",
                summary_metrics={
                    "best_model": "LogisticRegression",
                    "best_recall": 0.91,
                    "best_f1": 0.78,
                    "best_roc_auc": 0.86,
                    "false_negative_count": 8,
                    "false_positive_count": 31,
                },
                findings=["模型可用于亏损记录人工复核优先级排序。"],
            ),
        ],
    )


def _sales_only_report() -> AnalysisReport:
    return AnalysisReport(
        task_id="task-sales-only",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["Sales analysis is limited to order status, quantity, and time fields."],
        modules=[
            ModuleReport(
                module_id="data_quality_check",
                title="数据质量检查",
                chart_type="table",
                summary_metrics={
                    "missing_order_datetime": 12,
                    "duplicate_rows": 4,
                },
                findings=["存在部分时间缺失和重复记录。"],
            ),
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={
                    "total_sales_amount": 110875937.33,
                    "total_quantity": 14993,
                },
                findings=["销售额随时间波动明显。"],
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="SKU 分析",
                chart_type="bar",
                tables={"sku_sales": [{"sku": "SKU-001", "sales_amount": 120000.0}]},
                findings=["SKU 层级存在销售集中度。"],
            ),
        ],
    )


def _profit_no_discount_report() -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["总销售额为 1250000，负利润记录占比为 18.72%。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1250000.0},
                findings=["销售额集中在少数高峰月份。"],
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品与类目分析",
                chart_type="bar",
                findings=["已识别高销售低利润商品，需要进一步核查毛利、定价、成本和补货策略。"],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="利润质量分析",
                chart_type="table",
                summary_metrics={
                    "total_profit_amount": 180000.0,
                    "negative_profit_rate": 0.1872,
                },
                tables={},
                findings=["利润质量需要结合商品和成本继续复盘。"],
            ),
        ],
    )


def test_client_report_fallback_payload_includes_what_if_actions_and_modeling() -> None:
    payload, trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="zh-CN",
    )

    assert trace.status == "disabled"
    assert payload["discount_what_if"]["high_risk_bucket"] == "30%+"
    assert payload["discount_what_if"]["estimated_profit_delta"] == 21200.0
    assert payload["modeling_support"]["best_model"] == "LogisticRegression"
    assert payload["modeling_support"]["false_negative_count"] == 8
    assert payload["priority_actions"]
    assert any(action["priority"] == "P1" for action in payload["priority_actions"])


def test_english_client_report_profit_discount_summary_uses_structured_facts() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="en",
    )

    summary = payload["executive_summary"]
    combined = json.dumps(payload, ensure_ascii=False)

    assert not summary[0].startswith("The run completed")
    assert any("Total sales reached 1,250,000.00" in item for item in summary)
    assert any("Total profit was 180,000.00" in item for item in summary)
    assert any("30%+" in item and "discount tier" in item for item in summary)
    assert "discount risk" in combined.lower() or "discount tier" in combined.lower()


def test_english_client_report_without_profit_discount_removes_unsupported_claims() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_sales_only_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "created_at": "order_datetime",
                "sku": "sku",
                "qty_ordered": "quantity",
                "grand_total": "sales_amount",
                "status": "order_status",
            },
            confidence=0.5,
        ),
        dataset_profile={"row_count": 20000, "column_count": 8},
        analysis_focus={"selected_focuses": ["trend_volatility_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["sales_trend"]},
        llm_client=None,
        output_language="en",
    )

    html = render_client_report_html(payload, output_language="en")
    combined = (json.dumps(payload, ensure_ascii=False) + "\n" + html).lower()

    for blocked in (
        "margin risk",
        "low-margin",
        "profit quality",
        "high-discount",
        "discount risk",
        "highest-risk discount",
        "discount approval",
        "what-if profit lift",
        "estimated what-if profit lift",
    ):
        assert blocked not in combined
    labels = {card["label"] for card in payload["kpi_cards"]}
    assert "Total Profit" not in labels
    assert "Highest-Risk Discount Threshold" not in labels
    assert "Estimated What-if Profit Lift" not in labels
    assert any(
        "Profit and discount fields were not mapped, so profit-margin and discount-risk conclusions are out of scope."
        in item
        for item in payload["executive_summary"]
    )


def test_english_client_report_with_profit_without_discount_keeps_profit_quality_actions() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_profit_no_discount_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "Order Date": "order_datetime",
                "Product Name": "product_name",
                "Category": "category",
                "Region": "region",
                "Sales": "sales_amount",
                "Profit": "profit",
            },
            confidence=1.0,
        ),
        dataset_profile={"row_count": 240, "column_count": 12},
        analysis_focus={"selected_focuses": ["profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["profit_quality"]},
        llm_client=None,
        output_language="en",
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    labels = {card["label"] for card in payload["kpi_cards"]}
    action_issues = [action["issue"] for action in payload["priority_actions"]]

    assert "Total Profit" in labels
    assert action_issues[:3] == [
        "Review profit quality slices",
        "Review product or SKU concentration",
        "Compare operational dimensions",
    ]
    assert "Discount field was not mapped, so discount-risk conclusions are out of scope." in payload["limitations"]
    assert "Profit and discount fields were not mapped" not in serialized
    for blocked in (
        "Review the scoped discount tier",
        "discount tier requires review",
        "Discount-profit relationship requires scoped review",
        "what-if profit lift",
    ):
        assert blocked not in serialized


def test_english_client_report_without_profit_or_discount_uses_precise_missing_field_limitation() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_sales_only_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "created_at": "order_datetime",
                "sku": "sku",
                "qty_ordered": "quantity",
                "grand_total": "sales_amount",
                "status": "order_status",
            },
            confidence=0.5,
        ),
        dataset_profile={"row_count": 20000, "column_count": 8},
        analysis_focus={"selected_focuses": ["trend_volatility_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["sales_trend"]},
        llm_client=None,
        output_language="en",
    )

    serialized = json.dumps(payload, ensure_ascii=False)

    assert (
        "Profit and discount fields were not mapped, so profit-margin and discount-risk conclusions are out of scope."
        in serialized
    )


def test_english_client_report_with_profit_discount_evidence_keeps_discount_risk_p1() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="en",
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    action_issues = [action["issue"] for action in payload["priority_actions"]]

    assert action_issues[0] == "Review discount-profit risk tier"
    assert "The `30%+` discount tier had the weakest profit quality" in serialized
    assert "Discount field was not mapped" not in serialized


def test_english_client_report_without_discount_filters_nested_final_synthesis_discount_actions() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_profit_no_discount_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 12},
        analysis_focus={"selected_focuses": ["profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["profit_quality"]},
        llm_client=None,
        output_language="en",
        final_synthesis={
            "recommended_actions": [
                {
                    "action": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
                    "issue": "Discount-profit relationship requires scoped review.",
                    "display_text": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
                    "priority": "P3",
                },
                {
                    "action": "Review low-profit products against cost and pricing evidence.",
                    "display_text": "Review low-profit products against cost and pricing evidence.",
                    "priority": "P2",
                },
            ],
            "metadata": {
                "english_public_narrative_guard_signature": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
                "final_action_render_mode": "display_text",
            },
        },
    )

    serialized = json.dumps(payload, ensure_ascii=False)

    assert "Review the scoped discount tier" not in serialized
    assert "Discount-profit relationship requires scoped review" not in serialized
    assert "Review low-profit products against cost and pricing evidence" in serialized
    assert payload["final_synthesis"]["metadata"]["final_action_render_mode"] == "display_text"
    assert "english_public_narrative_guard_signature" not in payload["final_synthesis"]["metadata"]


def test_chinese_client_report_without_discount_keeps_profit_kpis_and_removes_discount_scenario() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_profit_no_discount_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 12},
        analysis_focus={"selected_focuses": ["profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="zh-CN",
    )

    labels = {card["label"]: card["value"] for card in payload["kpi_cards"]}
    serialized = json.dumps(payload, ensure_ascii=False)

    assert labels["总利润"] != "暂无"
    assert labels["负利润记录占比"] != "暂无"
    assert "最高风险折扣阈值" not in labels
    assert "what-if 估算利润改善" not in labels
    assert payload["discount_what_if"] == {}
    assert payload["business_theme"] == "围绕销售贡献和利润质量，识别优先复盘对象并沉淀行动清单。"
    assert "缺少 discount 字段，当前无法开展折扣区间和 what-if 情景分析。" in payload["limitations"]
    assert "折扣风险" not in payload["business_theme"]
    assert "折扣回收金额的情景估算" not in serialized


class NoDiscountWrongChineseLLMClient:
    enabled = True
    source = "test"
    configured_model = "test-model"
    last_completion_metrics = {}

    def complete_json(self, *, system_prompt, user_payload, preferred_models=None, cache_stage=None):
        return {
            "business_theme": "围绕销售贡献、利润质量和折扣风险，识别优先复盘对象并沉淀行动清单。",
            "executive_summary": [
                "高折扣订单需要立即复盘折扣策略。",
                "缺少 discount 字段，当前无法进行折扣分析。",
            ],
            "key_findings": {
                "discount_profit": [
                    "按折扣区间可以看到折扣力度过大。",
                    "缺少 discount 字段，当前无法进行折扣分析。",
                ],
                "product_category": ["利润质量需要结合成本和定价继续复盘。"],
            },
            "limitations": [
                "这是基于折扣回收金额的情景估算，不代表真实需求、销量或客户行为变化。",
                "缺少 discount 字段，当前无法开展折扣区间和 what-if 情景分析。",
            ],
        }


def test_chinese_client_report_no_discount_guard_drops_wrong_llm_discount_claims_but_keeps_limitations() -> None:
    payload, trace = build_client_report_payload_with_trace(
        report=_profit_no_discount_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 12},
        analysis_focus={"selected_focuses": ["profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=NoDiscountWrongChineseLLMClient(),
        output_language="zh-CN",
        final_synthesis={
            "brief_findings": ["高折扣订单需要立即复盘折扣策略。"],
            "brief_actions": ["缺少 discount 字段，当前无法进行折扣分析。"],
            "main_conclusions": [
                {"conclusion": "折扣阈值需要重新设置。"},
                {"conclusion": "缺少 discount 字段，当前无法进行折扣分析。"},
            ],
            "recommended_actions": [
                {"action": "对高折扣订单做专项分析。"},
                {"action": "围绕成本和定价复盘低利润商品。"},
            ],
        },
    )

    serialized = json.dumps(payload, ensure_ascii=False)

    assert trace.status == "llm_applied"
    assert "高折扣订单" not in serialized
    assert "折扣策略" not in serialized
    assert "折扣阈值" not in serialized
    assert "折扣回收金额的情景估算" not in serialized
    assert "缺少 discount 字段" in serialized
    assert "利润质量需要结合成本和定价继续复盘" in serialized
    assert payload["discount_what_if"] == {}
    assert payload["final_synthesis"]


def test_client_report_payload_falls_back_field_count_to_schema_mapping() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={
                "Sales": "sales_amount",
                "Profit": "profit",
                "Discount": "discount",
                "Category": "category",
            },
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="zh-CN",
    )

    assert payload["sample_size"] == 240
    assert payload["field_count"] == 4


def test_client_report_html_renders_poster_ui_without_legacy_placeholders() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="zh-CN",
    )

    html = render_client_report_html(payload)

    assert "销售经营分析报告" in html
    assert "主要风险 Spotlight" in html
    assert "核心洞察" in html
    assert "建议行动" in html
    assert "evidence-ribbon" in html
    assert "模型辅助判断" in html
    assert "真实预测" not in html
    assert "不代表真实需求、销量或客户行为变化" in html
    assert "受影响记录数" in html
    assert "受影响销售额" in html
    assert "估算利润改善" in html
    assert "<table" not in html
    assert "Priority Actions" not in html
    assert "Executive Summary" not in html
    assert "Charts" not in html
    assert "v1" not in html
    assert "Capability Matrix" not in html
    assert "Target discount cap" not in html
    assert "Affected rows" not in html
    assert "Affected sales" not in html
    assert "Estimated profit delta" not in html
    for blocked in ["unknown", "nan", "None", "%%", "{{", "}}"]:
        assert blocked not in html


class ProcessSummaryLLMClient:
    enabled = True
    source = "test"
    configured_model = "test-model"
    last_completion_metrics = {}

    def complete_json(self, *, system_prompt, user_payload):
        return {
            "executive_summary": [
                "发现 0 条时间缺失记录。",
                "完成 4 个指标画像。",
            ]
        }


def test_client_report_quality_guard_promotes_business_summary_before_process_logs() -> None:
    payload, trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=ProcessSummaryLLMClient(),
        output_language="zh-CN",
    )

    summary = payload["executive_summary"]

    assert trace.status == "llm_applied"
    assert len(summary) <= 4
    assert not summary[0].startswith("发现 0 条时间缺失记录")
    assert any("最高风险折扣阈值" in item and "30%+" in item for item in summary)
    assert any("what-if" in item and "21,200.00" in item for item in summary)


class StrongInferenceReportLLMClient:
    enabled = True
    source = "test"
    configured_model = "test-model"
    last_completion_metrics = {}

    def complete_json(self, *, system_prompt, user_payload, preferred_models=None, cache_stage=None):
        return {
            "business_theme": "折扣审批失控是利润下滑的核心原因。",
            "executive_summary": [
                "30%+ 折扣必然导致利润质量恶化，负利润率为 18.72%。",
            ],
            "key_findings": {
                "discount_profit": ["高风险折扣区间是唯一原因，what-if 估算利润改善为 21,200.00。"]
            },
            "limitations": ["当前结果充分证明审批失控。"],
        }


def test_client_report_payload_downgrades_strong_inference_without_touching_numbers() -> None:
    report = _generic_report().model_copy(
        update={
            "summary": ["折扣审批失控是核心原因，必然导致利润质量恶化。"],
        }
    )
    payload, trace = build_client_report_payload_with_trace(
        report=report,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack={
            "dataset_signature": {"dominant_story": "discount_loss"},
            "focus_evidence": {
                "discount_erosion_focus": [
                    {
                        "business_meaning": "折扣审批失控是核心原因，必然导致 30%+ 区间利润质量恶化。"
                    }
                ]
            },
        },
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=StrongInferenceReportLLMClient(),
        output_language="zh-CN",
    )

    serialized_text = str(
        {
            "business_theme": payload["business_theme"],
            "executive_summary": payload["executive_summary"],
            "key_findings": payload["key_findings"],
            "limitations": payload["limitations"],
            "priority_actions": payload["priority_actions"],
        }
    )
    assert trace.status == "llm_applied"
    assert "折扣审批失控" not in serialized_text
    assert "核心原因" not in serialized_text
    assert "必然导致" not in serialized_text
    assert "充分证明" not in serialized_text
    assert "可能的重要风险信号" in serialized_text
    assert "18.72%" in serialized_text
    assert "21,200.00" in serialized_text


class ParentScopeEnglishLLMClient:
    enabled = True
    source = "test"
    configured_model = "test-model"
    last_completion_metrics = {}

    def complete_json(self, *, system_prompt, user_payload, preferred_models=None, cache_stage=None):
        return {
            "business_theme": "Furniture category loss of -44,169.46 requires urgent review.",
            "executive_summary": [
                "Furniture category margin of -45.90% shows Furniture is unprofitable.",
                "Furniture aggregate profit is positive in this run.",
                "Furniture in the 30%+ discount tier had -44,169.46 profit and -45.90% margin.",
            ],
            "key_findings": {
                "discount_profit": [
                    "Furniture category loss of -44,169.46 should drive a full category stop-sell review.",
                    "Furniture in the 30%+ discount tier had -44,169.46 profit and -45.90% margin.",
                ]
            },
        }


def _scoped_discount_report() -> AnalysisReport:
    return AnalysisReport(
        task_id="task-scope-guard",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["Furniture has positive aggregate profit, while one high-discount slice is negative."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend Analysis",
                chart_type="line",
                summary_metrics={"total_sales_amount": 250000.0},
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="Product Contribution Analysis",
                chart_type="bar",
                summary_metrics={"top_category": "Furniture", "total_profit_amount": 18000.0},
                tables={
                    "category_summary": [
                        {"category": "Furniture", "profit": 18000.0, "profit_margin": 0.08},
                    ]
                },
                findings=["Furniture aggregate profit is positive in this run."],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="Discount Profit Analysis",
                chart_type="table",
                summary_metrics={"total_profit_amount": 18000.0, "negative_profit_rate": 0.18},
                tables={
                    "discount_threshold_candidates": [
                        {
                            "category": "Furniture",
                            "discount_bucket": "30%+",
                            "profit": -44169.46,
                            "profit_margin": -0.459,
                            "negative_profit_rate": 0.72,
                            "risk_level": "high",
                        }
                    ],
                    "discount_cap_what_if": [
                        {
                            "high_risk_bucket": "30%+",
                            "affected_row_count": 42,
                            "affected_sales_amount": 86000.0,
                            "current_profit": -44169.46,
                            "profit_margin": -0.459,
                            "current_negative_profit_rate": 0.72,
                        }
                    ],
                },
                findings=["Furniture in the 30%+ discount tier had negative profit."],
            ),
        ],
    )


def test_english_client_report_scope_guard_does_not_expand_slice_metrics_to_parent_category() -> None:
    payload, trace = build_client_report_payload_with_trace(
        report=_scoped_discount_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount", "Category": "category"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 200, "column_count": 8},
        analysis_focus=None,
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit", "product_and_category"]},
        llm_client=ParentScopeEnglishLLMClient(),
        output_language="en",
    )

    serialized = json.dumps(payload, ensure_ascii=False)

    assert trace.status == "llm_applied"
    assert "Furniture in the 30%+ discount tier had -44,169.46 profit and -45.90% margin." in serialized
    assert "Furniture aggregate profit is positive in this run." in serialized
    assert "Furniture category loss of -44,169.46" not in serialized
    assert "Furniture category margin of -45.90%" not in serialized
    assert "Furniture is unprofitable" not in serialized
    assert "full category stop-sell" not in serialized.lower()


def test_client_report_view_model_uses_dynamic_payload_and_hides_missing_model() -> None:
    report = _generic_report().model_copy(
        update={
            "modules": [
                module
                for module in _generic_report().modules
                if module.module_id != "loss_risk_modeling"
            ]
        }
    )
    payload, _trace = build_client_report_payload_with_trace(
        report=report,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="zh-CN",
    )

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload)

    assert vm["model_panel"] is None
    assert "模型辅助判断" not in html
    assert "LogisticRegression" not in html
    assert "Furniture" not in html
    assert len(vm["kpis"]) >= 3
    assert vm["spotlight"]["headline_metric"] != "暂无"


def test_client_report_payload_and_view_model_prefer_final_synthesis_actions() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Category": "category"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["product_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["product_category"]},
        llm_client=None,
        output_language="zh-CN",
        final_synthesis=_final_synthesis(),
        modeling_outcome=_sales_regression_outcome(),
        modeling_outcome_interpretation={"outcome_summary": "回归模型仅适合作为监控参考。"},
    )

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload)

    assert payload["final_synthesis"]["recommended_actions"][0]["display_text"].startswith("针对类目高度分散")
    assert payload["modeling_outcome"]["primary_modeling_task"] == "sales_amount_regression"
    assert vm["hero"]["headline"] == "类目分散削弱经营聚焦"
    assert vm["action_cards"][0]["title"] == "对销售类目进行ABC分析"
    assert "针对类目高度分散的问题" in vm["action_cards"][0]["body"]
    assert vm["action_cards"][0]["evidence"] == "头部类目销售占比6.77%，类目总数79个"
    assert "对销售类目进行ABC分析" in html
    assert "Priority Actions" not in html


def _payload_with_insight_conclusion(conclusion: str, *, output_language: str = "zh-CN") -> dict[str, object]:
    return {
        "title": "客户经营分析简报" if output_language == "zh-CN" else "Client-ready Sales Analysis Report",
        "output_language": output_language,
        "business_theme": "本次分析已形成关键经营观察。" if output_language == "zh-CN" else "This report summarizes business observations.",
        "executive_summary": ["本次分析已形成关键经营观察。" if output_language == "zh-CN" else "This report summarizes business observations."],
        "priority_actions": [],
        "kpi_cards": [],
        "final_synthesis": {
            "main_conclusions": [
                {
                    "conclusion": "本次分析已形成关键经营观察。" if output_language == "zh-CN" else "The report summarizes business observations.",
                    "evidence": "基础经营观察。",
                    "business_meaning": "用于稳定报告主标题。",
                },
                {
                    "conclusion": conclusion,
                    "evidence": "当前数据支持该经营观察。",
                    "business_meaning": "后续说明由正文和证据承担。",
                }
            ],
            "recommended_actions": [],
        },
    }


def test_chinese_insight_title_uses_natural_model_clause_without_ellipsis() -> None:
    source = "亏损风险已被高精度模型有效捕捉，但模型输出仅用于人工复核优先级排序，不用于自动决策。"
    payload = _payload_with_insight_conclusion(source)

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload, output_language="zh-CN")
    title = next(card["title"] for card in vm["insight_cards"] if "亏损风险" in card["title"])

    assert title == "亏损风险已被高精度模型有效捕捉"
    assert "…" not in title
    assert "亏损风险已被高精度模型有效捕捉" in html
    assert "亏损风险已被高精度模型有效捕捉…" not in html


def test_chinese_insight_title_keeps_enumeration_phrase_intact() -> None:
    source = "利润质量在类目、客群和区域维度存在明显分化，需要按切片制定经营策略。"
    payload = _payload_with_insight_conclusion(source)

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload, output_language="zh-CN")
    title = next(card["title"] for card in vm["insight_cards"] if "利润质量" in card["title"])

    assert title == "利润质量在类目、客群和区域维度存在明显分化"
    assert title != "利润质量在类目"
    assert "…" not in title
    assert "利润质量在类目、客群和区域维度存在明显分化" in html


def test_chinese_insight_title_without_natural_split_keeps_full_source() -> None:
    source = "这是一个没有自然切分符号但需要完整保留给浏览器自然换行的中文关键发现标题"
    payload = _payload_with_insight_conclusion(source)

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload, output_language="zh-CN")
    title = next(card["title"] for card in vm["insight_cards"] if card["title"] == source)

    assert title == source
    assert "…" not in title
    assert len(title) == len(source)
    assert source in html


def test_chinese_insight_title_preserves_existing_discount_profit_mapping() -> None:
    source = "折扣持续抬高后利润质量明显承压，需要围绕高风险折扣区间建立审批红线和复盘机制。"
    payload = _payload_with_insight_conclusion(source)

    vm = build_client_report_view_model(payload)
    title = next(card["title"] for card in vm["insight_cards"] if "折扣" in card["title"])

    assert title == "高折扣侵蚀利润质量"
    assert "…" not in title


def test_english_insight_title_path_is_unchanged_by_chinese_title_helper() -> None:
    source = "High discount tiers are driving margin risk and require review before pricing changes."
    payload = _payload_with_insight_conclusion(source, output_language="en")

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload, output_language="en")
    titles = [card["title"] for card in vm["insight_cards"]]

    assert "High-discount margin risk requires review" in titles
    assert "主要经营观察" not in json.dumps(vm["insight_cards"], ensure_ascii=False)
    assert "High-discount margin risk requires review" in html


def test_chinese_insight_evidence_uses_natural_clause_without_ellipsis_for_category_metrics() -> None:
    payload = {
        "title": "客户经营分析简报",
        "output_language": "zh-CN",
        "business_theme": "本次分析已形成关键经营观察。",
        "executive_summary": ["本次分析已形成关键经营观察。"],
        "priority_actions": [],
        "kpi_cards": [],
        "final_synthesis": {
            "main_conclusions": [
                {
                    "conclusion": "类目利润质量需要复盘",
                    "evidence": "top_category_sales_share = 72.23%，profit_margin_spread = 40.27%，Road Bikes利润率30.43%而部分商品利润率不足10%",
                    "business_meaning": "类目间利润质量差异较大。",
                }
            ],
            "recommended_actions": [],
        },
    }

    vm = build_client_report_view_model(payload)
    evidence = vm["insight_cards"][0]["evidence"]

    assert evidence == "top_category_sales_share = 72.23%，profit_margin_spread = 40.27%"
    assert "…" not in evidence
    assert "Road" not in evidence


def test_chinese_insight_evidence_uses_natural_clause_without_ellipsis_for_model_metrics() -> None:
    payload = {
        "title": "客户经营分析简报",
        "output_language": "zh-CN",
        "business_theme": "本次分析已形成关键经营观察。",
        "executive_summary": ["本次分析已形成关键经营观察。"],
        "priority_actions": [],
        "kpi_cards": [],
        "final_synthesis": {
            "main_conclusions": [
                {
                    "conclusion": "亏损模型仅作探索线索",
                    "evidence": "best_model = LogisticRegression，best_f1 = 9.76%，best_roc_auc = 99.7%，weak_reasons = extreme_class_imbalance/too_few_test_positives/low_precision/low_f1",
                    "business_meaning": "模型输出仅用于人工复核参考。",
                }
            ],
            "recommended_actions": [],
        },
    }

    vm = build_client_report_view_model(payload)
    evidence = vm["insight_cards"][0]["evidence"]

    assert evidence == "best_model = LogisticRegression，best_f1 = 9.76%，best_roc_auc = 99.7%"
    assert "…" not in evidence
    assert "weak_reasons" not in evidence


def test_english_insight_evidence_still_uses_existing_short_evidence_behavior() -> None:
    payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "business_theme": "This report summarizes business observations.",
        "executive_summary": ["This report summarizes business observations."],
        "priority_actions": [],
        "kpi_cards": [],
        "final_synthesis": {
            "main_conclusions": [
                {
                    "conclusion": "Product mix requires concentration review",
                    "evidence": "top category sales share was 72.23 percent and profit margin spread was 40.27 percent while Road Bikes had a materially stronger profit margin than tail products",
                    "business_meaning": "Review category mix before changing growth priorities.",
                }
            ],
            "recommended_actions": [],
        },
    }

    vm = build_client_report_view_model(payload)
    evidence = vm["insight_cards"][0]["evidence"]

    assert evidence.endswith(".")
    assert "…" not in evidence
    assert "Road Bikes" not in evidence


def test_client_report_view_model_renders_sales_regression_modeling() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report().model_copy(
            update={
                "modules": [
                    module
                    for module in _generic_report().modules
                    if module.module_id not in {"loss_risk_modeling", "discount_profit_analysis"}
                ]
            }
        ),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Date": "order_datetime"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 20000, "column_count": 12},
        analysis_focus={"selected_focuses": ["sales_forecast"]},
        evidence_pack=None,
        section_priority={"core_sections": ["sales_trend"]},
        llm_client=None,
        output_language="zh-CN",
        final_synthesis=_final_synthesis(),
        modeling_outcome=_sales_regression_outcome(),
    )

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload)

    assert vm["model_panel"]["kind"] == "regression"
    assert vm["model_panel"]["metric_label"] == "R²"
    assert vm["model_panel"]["headline_metric"] == "0.07"
    assert vm["spotlight"]["kind"] == "forecast_boundary"
    assert any(kpi["label"] == "R²" for kpi in vm["kpis"])
    assert any(kpi["label"] == "MAPE" for kpi in vm["kpis"])
    assert "RandomForestRegressor-small" in html
    assert "MAE" in html
    assert "MAPE" in html
    assert "漏判亏损数" not in html
    assert "误报亏损数" not in html
    assert "LogisticRegression" not in html


def test_client_report_view_model_renders_forecast_baseline_without_r2_placeholder() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report().model_copy(
            update={
                "modules": [
                    module
                    for module in _generic_report().modules
                    if module.module_id not in {"loss_risk_modeling", "discount_profit_analysis"}
                ]
            }
        ),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Date": "order_datetime"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 20000, "column_count": 12},
        analysis_focus={"selected_focuses": ["sales_forecast"]},
        evidence_pack=None,
        section_priority={"core_sections": ["sales_trend"]},
        llm_client=None,
        output_language="zh-CN",
        final_synthesis={"main_conclusions": [{"conclusion": "当前销售预测只能作为监控参照"}]},
        modeling_outcome=_forecast_baseline_outcome(),
    )

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload)
    model_panel = vm["model_panel"]

    assert model_panel["kind"] == "forecast_baseline"
    assert model_panel["best_model"] == "naive_last_value"
    assert model_panel["headline_metric"] == "30.09"
    assert model_panel["metric_label"] == "MAPE"
    assert {"label": "RMSE", "value": "580.8"} in model_panel["scores"]
    assert {"label": "baseline", "value": "naive_last_value"} in model_panel["scores"]
    assert vm["spotlight"]["kind"] == "forecast_boundary"
    assert vm["spotlight"]["metric_label"] == "MAPE"
    assert any(kpi["label"] == "RMSE" for kpi in vm["kpis"])
    assert not any(kpi["label"] == "R²" for kpi in vm["kpis"])
    assert "R²" not in html
    assert "暂无" not in model_panel["headline_metric"]
    assert "forecast baseline" not in model_panel["best_model"]


def test_client_report_view_model_uses_classification_modeling_outcome_without_legacy_support() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report().model_copy(
            update={
                "modules": [
                    module
                    for module in _generic_report().modules
                    if module.module_id not in {"loss_risk_modeling", "discount_profit_analysis"}
                ]
            }
        ),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["loss_risk"]},
        evidence_pack=None,
        section_priority={"core_sections": ["modeling"]},
        llm_client=None,
        output_language="zh-CN",
        modeling_outcome=_loss_risk_outcome(),
    )

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload)

    assert vm["model_panel"]["kind"] == "classification"
    assert vm["model_panel"]["best_model"] == "RandomForestClassifier-small"
    assert vm["spotlight"]["kind"] == "model_review"
    assert "Recall" in html
    assert "RandomForestClassifier-small" in html


def test_client_report_view_model_downgrades_weak_classification_copy() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report().model_copy(
            update={
                "modules": [
                    module
                    for module in _generic_report().modules
                    if module.module_id not in {"loss_risk_modeling", "discount_profit_analysis"}
                ]
            }
        ),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 20000, "column_count": 12},
        analysis_focus={"selected_focuses": ["loss_risk"]},
        evidence_pack=None,
        section_priority={"core_sections": ["modeling"]},
        llm_client=None,
        output_language="zh-CN",
        modeling_outcome=_weak_loss_risk_outcome(),
    )

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload)
    model_panel = vm["model_panel"]

    assert model_panel["kind"] == "classification"
    assert model_panel["title"] == "仅适合作为探索性风险线索，不建议直接用于复核排序"
    assert "质量状态偏弱" in model_panel["description"]
    assert vm["spotlight"]["pill"] == "模型边界"
    assert vm["spotlight"]["title"] == "亏损风险模型仅适合作为探索性线索"
    assert vm["spotlight"]["metric_label"] == "F1"
    assert any("质量偏弱" in item for item in vm["limitations"])
    assert "可以辅助复核排序" not in html
    assert "模型可辅助人工复核排序" not in html
    assert "探索性风险线索" in html
    assert "不建议直接用于复核排序" in html


def test_english_client_report_has_demo_ready_labels_titles_and_units() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="en",
        final_synthesis={
            "main_conclusions": [
                {
                    "conclusion": "Overall, 18.65% of records have negative profit and require review.",
                    "evidence": "30%+ Discount区间 has the highest 亏损率.",
                    "business_meaning": "The primary margin issue is concentrated in high-discount orders.",
                },
                {
                    "conclusion": "A strong model signal can support review prioritization.",
                    "evidence": "Recall is 91%, with 31 false positives and 8 false negatives.",
                    "business_meaning": "The model flags likely loss-making orders for manual review.",
                },
            ],
            "recommended_actions": [
                {
                    "priority": "P1",
                    "action": "Review the 30%+ Discount区间",
                    "linked_metric_or_segment": "高风险区间：30%+",
                    "expected_use": "Reduce Profit率 exposure.",
                    "display_text": "Review the 30%+ Discount区间 before approving similar records.",
                }
            ],
        },
        modeling_outcome=_loss_risk_outcome(),
    )

    html = render_client_report_html(payload, output_language="en")
    serialized = json.dumps(payload, ensure_ascii=False) + "\n" + html
    headings = re.findall(r"<h[13]>(.*?)</h[13]>", html)

    for blocked in (
        "The primary…",
        "A strong…",
        "Overall, 18.65% of…",
        "最终Conclusion",
        "最终结论",
        "经营观察",
        "核心摘要",
        "高风险区间",
        "亏损率",
        "核心指标",
        "亏损复核召回",
        "模型召回率仪表盘",
        "可以辅助复核排序",
        "误报亏损数",
        "漏判亏损数",
        "受影响",
        "Profit率",
        "平均Profit",
        "Discount区间",
        "折扣区间",
        "影响记录",
        "主要经营观察",
        "核心信号",
        "趋势波动",
        "集中度观察",
        "万",
        "亿",
        "catastrophic",
        "confirms",
        "systemic issue",
        "proves",
        "causal",
    ):
        assert blocked not in serialized
    assert all(not heading.endswith("…") for heading in headings)
    assert all(not heading.startswith(("The primary", "A strong", "Overall,")) for heading in headings)

    for expected in (
        "Client-ready Sales Analysis Report",
        "Dataset Overview",
        "Key Risk Spotlight",
        "Key Findings",
        "Recommended Actions",
        "Usage Limits",
        "High-risk discount tier",
        "Loss Rate",
        "Affected records",
        "Affected sales",
        "Profit margin",
        "Average profit",
        "False positives",
        "False negatives",
    ):
        assert expected in serialized


def test_english_client_report_general_spotlight_fallback_uses_english_labels() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["Business mix changed across recent orders and needs follow-up."],
        modules=[
            ModuleReport(
                module_id="generic_analysis",
                title="Generic analysis",
                chart_type="bar",
                findings=["Concentration changed across product groups and should be reviewed."],
            )
        ],
    )
    payload, _trace = build_client_report_payload_with_trace(
        report=report,
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["product_mix"]},
        evidence_pack=None,
        section_priority={"core_sections": ["product_category"]},
        llm_client=None,
        output_language="en",
    )

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload, output_language="en")
    serialized = json.dumps(vm, ensure_ascii=False) + "\n" + html

    assert vm["spotlight"]["pill"] in {"Business Observation", "Concentration Signal"}
    assert vm["spotlight"]["metric_label"] == "Core Signal"
    assert "主要经营观察" not in serialized
    assert "核心信号" not in serialized
    assert "趋势波动" not in serialized
    assert "集中度观察" not in serialized
    assert "html[lang=\"en\"] h1" in html
    assert "html[lang=\"en\"] .insight-card h3" in html
    assert "html[lang=\"en\"] .spotlight h3" in html


def test_english_client_report_avoids_generic_headline_and_regression_chinese_label() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["sales_prediction_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["sales_trend"]},
        llm_client=None,
        output_language="en",
        final_synthesis={
            "main_conclusions": [
                {
                    "conclusion": "This analysis shows the business has a measurable forecasting signal.",
                    "business_meaning": "The business should use regression output for monitoring only.",
                }
            ]
        },
        modeling_outcome=_sales_regression_outcome(),
    )

    html = render_client_report_html(payload, output_language="en")
    serialized = json.dumps(payload, ensure_ascii=False) + "\n" + html
    vm = build_client_report_view_model(payload)

    assert not vm["hero"]["headline"].startswith("This analysis")
    assert not vm["hero"]["headline"].startswith("The business")
    assert "基线改善" not in serialized
    assert "相对基线改善" not in serialized
    assert "Baseline lift" in serialized or "Baseline improvement" in serialized


def test_english_client_report_structured_presentation_avoids_generic_insight_placeholders() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="en",
        final_synthesis={
            "main_conclusions": [
                {
                    "conclusion": "Overall, the business shows a 12.56% overall profit margin.",
                    "evidence": "Total profit is measured from the mapped profit field.",
                    "business_meaning": "This is a profitability fact, not a high-discount risk fact.",
                },
                {
                    "conclusion": "The 30%+ discount tier has the weakest profit quality.",
                    "evidence": "Loss rate is 72.00% in the threshold analysis.",
                    "business_meaning": "This tier requires review before similar discount approvals.",
                },
            ],
            "recommended_actions": [
                {
                    "priority": "P3",
                    "action": "Review approval rules for the 30%+ discount tier",
                    "issue": "30%+ discount tier requires review",
                    "display_text": "Review approval rules for orders in the 30%+ discount tier.",
                    "linked_metric_or_segment": "Loss rate 72.00%",
                }
            ],
        },
    )

    vm = build_client_report_view_model(payload)
    insight_cards = vm["insight_cards"]
    serialized_cards = json.dumps(insight_cards, ensure_ascii=False)

    assert "Business observation" not in serialized_cards
    assert "Evidence: Business observation" not in serialized_cards
    assert "Key business finding" not in serialized_cards
    assert any(card["title"] == "Total Sales" for card in insight_cards)
    assert any(card["title"] == "Total Profit" for card in insight_cards)
    assert any("30%+" in card["title"] and "72.00%" in card["metric"] for card in insight_cards)
    assert not any(card["title"] == "High discount tiers are driving margin risk" and card["metric"] == "12.56%" for card in insight_cards)
    action_cards = vm["action_cards"]
    serialized_actions = json.dumps(action_cards, ensure_ascii=False)
    assert "Recommended action 3" not in serialized_actions
    assert [card["priority"] for card in action_cards] == ["P1", "P3"]
    assert "Review discount-profit risk tier" in serialized_actions
    assert "approval rules" not in serialized_actions.lower()


def test_english_client_report_regression_negative_zero_is_display_only() -> None:
    outcome = _sales_regression_outcome()
    outcome["metrics_summary"] = {
        **outcome["metrics_summary"],  # type: ignore[index]
        "improvement_vs_baseline": -0.0,
    }
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["sales_prediction_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["sales_trend"]},
        llm_client=None,
        output_language="en",
        modeling_outcome=outcome,
    )

    vm = build_client_report_view_model(payload)
    serialized = json.dumps(vm["model_panel"], ensure_ascii=False) + "\n" + json.dumps(vm["spotlight"], ensure_ascii=False)

    assert payload["modeling_outcome"]["metrics_summary"]["improvement_vs_baseline"] == -0.0
    assert '"value": "-0"' not in serialized
    assert '"value_label": "-0"' not in serialized
    assert "Baseline lift" in serialized


def test_english_client_report_removes_dynamic_business_observation_placeholders() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["loss_risk_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["modeling"]},
        llm_client=None,
        output_language="en",
        modeling_outcome={
            "primary_modeling_task": "loss_risk_classification",
            "modeling_status": "classification_evaluated",
            "main_findings": ["Business observation", " Business observation "],
            "limitations": ["The model is only for human review prioritization."],
            "metrics_summary": {"best_recall": 0.88},
        },
        modeling_outcome_interpretation={
            "outcome_summary": "Business observation",
            "main_findings": ["Business observation"],
            "limitations": ["The model requires manual review prioritization."],
        },
    )

    dynamic_payload = {
        "modeling_outcome": payload["modeling_outcome"],
        "modeling_outcome_interpretation": payload["modeling_outcome_interpretation"],
    }
    serialized_dynamic = json.dumps(dynamic_payload, ensure_ascii=False)
    html = render_client_report_html(payload, output_language="en")

    assert "Business observation" not in serialized_dynamic
    assert "main_findings" not in payload["modeling_outcome"]
    assert "main_findings" not in payload["modeling_outcome_interpretation"]
    assert "outcome_summary" not in payload["modeling_outcome_interpretation"]
    assert payload["modeling_outcome"]["limitations"] == ["The model is only for human review prioritization."]
    assert payload["modeling_outcome"]["metrics_summary"]["best_recall"] == 0.88
    assert "<li></li>" not in html
    assert "Business observation\nBusiness observation" not in html


def test_client_report_action_card_title_priority_is_language_specific() -> None:
    base_payload = {
        "title": "客户经营分析简报",
        "dataset_type": "sales_transaction",
        "sample_size": 10,
        "field_count": 4,
        "business_theme": "本次分析已形成行动建议。",
        "executive_summary": ["本次分析已形成行动建议。"],
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P1",
                    "issue": "这是 issue 标题",
                    "action": "这是原 action 标题",
                    "display_text": "这是展示文本。",
                    "linked_metric_or_segment": "字段映射",
                }
            ]
        },
    }

    zh_vm = build_client_report_view_model({**base_payload, "output_language": "zh-CN"})
    en_vm = build_client_report_view_model(
        {
            **base_payload,
            "output_language": "en",
            "final_synthesis": {
                "recommended_actions": [
                    {
                        "priority": "P1",
                        "issue": "Field coverage requires review",
                        "action": "Use the original action title",
                        "display_text": "Review field coverage before execution.",
                    }
                ]
            },
        }
    )

    assert zh_vm["action_cards"][0]["title"] == "这是原 action 标题"
    assert zh_vm["action_cards"][0]["title"] != "这是 issue 标题"
    assert en_vm["action_cards"][0]["title"] == "Field coverage requires review"
    assert en_vm["action_cards"][0]["title"] != "Use the original action title"


def test_english_action_cards_use_structured_priority_actions_as_canonical_source() -> None:
    payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "business_theme": "Recommended actions are available.",
        "executive_summary": ["Recommended actions are available."],
        "priority_actions": [
            {
                "priority": "P3",
                "issue": "Structured P3 action",
                "action": "Structured P3 body",
                "evidence": "Structured P3 evidence",
            },
            {
                "priority": "P1",
                "issue": "Structured P1 action",
                "action": "Structured P1 body",
                "evidence": "Structured P1 evidence",
            },
            {
                "priority": "P2",
                "issue": "Structured P2 action",
                "action": "Structured P2 body",
                "evidence": "Structured P2 evidence",
            },
        ],
        "final_synthesis": {
            "recommended_actions": [
                {"priority": "P3", "action": "Fallback P3 action", "display_text": "Fallback P3 body"},
                {"priority": "P3", "action": "Fallback P3 action two", "display_text": "Fallback P3 body two"},
                {"action": "Fallback missing-priority action", "display_text": "Fallback missing-priority body"},
                {"priority": "P4", "action": "Fallback P4 action", "display_text": "Fallback P4 body"},
            ]
        },
    }

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload, output_language="en")
    serialized_cards = json.dumps(vm["action_cards"], ensure_ascii=False)

    assert [card["priority"] for card in vm["action_cards"]] == ["P1", "P2", "P3"]
    assert "Structured P1 action" in serialized_cards
    assert "Structured P2 action" in serialized_cards
    assert "Structured P3 action" in serialized_cards
    assert "Structured P1 body" in serialized_cards
    assert "Structured P2 body" in serialized_cards
    assert "Structured P3 body" in serialized_cards
    assert "Structured P1 evidence" in serialized_cards
    assert "Structured P2 evidence" in serialized_cards
    assert "Structured P3 evidence" in serialized_cards
    for phrase in (
        "Fallback P3 action",
        "Fallback P3 action two",
        "Fallback missing-priority action",
        "Fallback P4 action",
    ):
        assert phrase not in serialized_cards
        assert phrase not in html
    assert html.count("class='action-card'") == 3
    assert "P1" in html and "P2" in html and "P3" in html
    assert "P4" not in html


def test_english_action_cards_do_not_mix_final_synthesis_when_structured_priorities_are_partial() -> None:
    payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "business_theme": "Recommended actions are available.",
        "executive_summary": ["Recommended actions are available."],
        "priority_actions": [
            {"priority": "P1", "issue": "Structured P1 action", "action": "Structured P1 body"},
            {"priority": "P3", "issue": "Structured P3 action", "action": "Structured P3 body"},
            {"priority": "P4", "issue": "Structured P4 action", "action": "Structured P4 body"},
            {"priority": "", "issue": "Structured missing priority", "action": "Structured missing body"},
        ],
        "final_synthesis": {
            "recommended_actions": [
                {"priority": "P2", "action": "Fallback P2 action", "display_text": "Fallback P2 body"},
            ]
        },
    }

    vm = build_client_report_view_model(payload)
    serialized_cards = json.dumps(vm["action_cards"], ensure_ascii=False)

    assert [card["priority"] for card in vm["action_cards"]] == ["P1", "P3"]
    assert "Structured P1 action" in serialized_cards
    assert "Structured P3 action" in serialized_cards
    assert "Fallback P2 action" not in serialized_cards
    assert "Structured P4 action" not in serialized_cards
    assert "Structured missing priority" not in serialized_cards


def test_english_action_cards_use_final_synthesis_only_as_fallback_source() -> None:
    payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "business_theme": "Recommended actions are available.",
        "executive_summary": ["Recommended actions are available."],
        "priority_actions": [],
        "final_synthesis": {
            "recommended_actions": [
                {"priority": "P2", "action": "Fallback P2 action", "display_text": "Fallback P2 body"},
                {"action": "Fallback missing-priority action", "display_text": "Fallback missing-priority body"},
                {"priority": "P1", "action": "Fallback P1 action", "display_text": "Fallback P1 body"},
                {"priority": "P4", "action": "Fallback P4 action", "display_text": "Fallback P4 body"},
            ]
        },
    }

    vm = build_client_report_view_model(payload)
    serialized_cards = json.dumps(vm["action_cards"], ensure_ascii=False)

    assert [card["priority"] for card in vm["action_cards"]] == ["P1", "P2", "P3"]
    assert len(vm["action_cards"]) == 3
    assert "Fallback P1 body" in serialized_cards
    assert "Fallback P2 body" in serialized_cards
    assert "Fallback missing-priority body" in serialized_cards
    assert "Fallback P4 action" not in serialized_cards
    assert "Fallback P4 body" not in serialized_cards
    assert "P4" not in serialized_cards


def test_chinese_action_cards_keep_final_synthesis_priority_over_structured_actions() -> None:
    payload = {
        "title": "客户经营分析简报",
        "output_language": "zh-CN",
        "business_theme": "本次分析已形成行动建议。",
        "executive_summary": ["本次分析已形成行动建议。"],
        "priority_actions": [
            {"priority": "P1", "issue": "结构化行动", "action": "结构化行动正文", "evidence": "结构化证据"},
        ],
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P2",
                    "action": "最终综合行动",
                    "display_text": "最终综合行动正文。",
                    "linked_metric_or_segment": "最终综合证据",
                }
            ]
        },
    }

    vm = build_client_report_view_model(payload)
    serialized_cards = json.dumps(vm["action_cards"], ensure_ascii=False)

    assert vm["action_cards"][0]["priority"] == "P2"
    assert "最终综合行动" in serialized_cards
    assert "结构化行动" not in serialized_cards


def test_english_action_card_prefers_model_title_when_body_is_model_review() -> None:
    payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "dataset_type": "sales_transaction",
        "sample_size": 100,
        "field_count": 6,
        "business_theme": "Review loss-risk model results.",
        "executive_summary": ["The model output supports human review prioritization."],
        "kpi_cards": [],
        "priority_actions": [],
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P3",
                    "issue": "High discount tiers are driving margin risk",
                    "action": "Use the LogisticRegression loss-risk model for review prioritization.",
                    "display_text": "Use the LogisticRegression loss-risk model with recall and score thresholds to support manual review prioritization.",
                    "linked_metric_or_segment": "Recall 66.67%",
                }
            ]
        },
        "modeling_outcome": _weak_loss_risk_outcome(),
    }

    vm = build_client_report_view_model(payload)

    assert vm["action_cards"][0]["title"] == "Loss-risk model supports review prioritization"
    assert vm["action_cards"][0]["title"] != "High discount tiers are driving margin risk"


def test_english_action_card_title_prefers_real_slice_before_discount_keywords() -> None:
    payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "dataset_type": "sales_transaction",
        "sample_size": 100,
        "field_count": 6,
        "business_theme": "Review margin slices.",
        "executive_summary": ["Consumer and Central require a scoped margin review."],
        "kpi_cards": [],
        "priority_actions": [],
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P3",
                    "issue": "High discount tiers are driving margin risk",
                    "action": "Review Consumer × Central order structure and discount patterns.",
                    "display_text": "Review Consumer × Central order structure and discount patterns before changing broader pricing rules.",
                    "linked_metric_or_segment": "Consumer × Central margin review.",
                }
            ]
        },
    }

    vm = build_client_report_view_model(payload)

    assert "Consumer × Central" in vm["action_cards"][0]["title"]
    assert vm["action_cards"][0]["title"] != "High discount tiers are driving margin risk"


def test_english_action_card_title_keeps_discount_and_product_scopes_when_primary() -> None:
    discount_payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "dataset_type": "sales_transaction",
        "sample_size": 100,
        "field_count": 6,
        "business_theme": "Review margin slices.",
        "executive_summary": ["The 30%+ discount tier requires review."],
        "kpi_cards": [],
        "priority_actions": [],
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P2",
                    "issue": "High discount tiers are driving margin risk",
                    "action": "Review the 30%+ discount tier before approving similar discount levels.",
                    "display_text": "Review the 30%+ discount tier against profit outcomes before broader pricing changes.",
                    "linked_metric_or_segment": "30%+ discount tier.",
                }
            ]
        },
    }
    product_payload = {
        **discount_payload,
        "final_synthesis": {
            "recommended_actions": [
                {
                    "priority": "P3",
                    "issue": "",
                    "action": "Review SKU ABC-123 contribution and margin before expanding replenishment.",
                    "display_text": "Review SKU ABC-123 contribution and margin before expanding replenishment.",
                    "linked_metric_or_segment": "SKU ABC-123.",
                }
            ]
        },
    }

    discount_vm = build_client_report_view_model(discount_payload)
    product_vm = build_client_report_view_model(product_payload)

    assert "30%+ discount tier" in discount_vm["action_cards"][0]["title"]
    assert "SKU ABC-123" in product_vm["action_cards"][0]["title"]
    assert "…" not in json.dumps(discount_vm["action_cards"] + product_vm["action_cards"], ensure_ascii=False)
    assert not re.search(r"[\u4e00-\u9fff]", json.dumps(discount_vm["action_cards"] + product_vm["action_cards"], ensure_ascii=False))


def test_english_client_report_visible_text_has_no_ellipsis_or_broken_truncation() -> None:
    payload = {
        "title": "Client-ready Sales Analysis Report",
        "output_language": "en",
        "dataset_type": "sales_transaction",
        "sample_size": 100,
        "field_count": 6,
        "business_theme": "The primary issue must be addr…",
        "executive_summary": ["The primary issue must be addr…"],
        "kpi_cards": [],
        "priority_actions": [],
        "final_synthesis": {
            "main_conclusions": [
                {
                    "conclusion": "The primary issue must be addr…",
                    "business_meaning": "The primary issue must be addr…",
                }
            ]
        },
    }

    vm = build_client_report_view_model(payload)
    html = render_client_report_html(payload, output_language="en")
    serialized = json.dumps(vm, ensure_ascii=False) + "\n" + html

    assert "…" not in serialized
    assert "addr." not in serialized


def test_english_client_report_consumes_guarded_final_synthesis() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["Total sales reached 2,297,200.86."],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend Analysis",
                chart_type="line",
                summary_metrics={"total_sales_amount": 2297200.86},
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="Discount and Profit Analysis",
                chart_type="bar",
                summary_metrics={"total_profit_amount": 286397.02, "negative_profit_rate": 0.1865},
                tables={
                    "discount_threshold_candidates": [
                        {
                            "Category": "Furniture",
                            "discount_bucket": "30%+",
                            "avg_profit": -44169.46,
                            "profit_margin": -0.4590,
                            "risk_level": "high",
                        },
                        {
                            "category": "Furniture",
                            "profit": 19730.00,
                            "profit_margin": 0.0261,
                        },
                    ]
                },
            ),
        ],
    )
    schema = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Category": "category",
            "Discount Bucket": "discount_bucket",
        },
        confidence=0.95,
    )
    payload, _trace = build_client_report_payload_with_trace(
        report=report,
        schema_mapping=schema,
        dataset_profile={"row_count": 999},
        analysis_focus=None,
        evidence_pack=None,
        section_priority=None,
        llm_client=None,
        final_synthesis={
            "metadata": {"final_synthesis_used_llm": True},
            "main_conclusions": [
                {
                    "conclusion": "Profit erosion is primarily driven by unsustainable discounting practices.",
                    "evidence": "The current data has sales, profit, and discount fields.",
                    "business_meaning": "Discounting is the root cause.",
                },
                {
                    "conclusion": "Furniture category has a -45.90% margin.",
                    "evidence": "The row also includes discount_bucket 30%+.",
                    "business_meaning": "Furniture is unprofitable.",
                },
                {
                    "conclusion": "Furniture in the 30%+ discount tier had a -45.90% margin.",
                    "evidence": "The evidence row includes Category and discount_bucket.",
                    "business_meaning": "Review the scoped discount tier before category-wide action.",
                },
            ],
            "recommended_actions": [
                {
                    "action": "Implement a mandatory approval workflow.",
                    "linked_metric_or_segment": "No approval records were mapped.",
                    "expected_use": "Treat approval controls as the cause.",
                    "display_text": "Implement a mandatory approval workflow.",
                },
                {
                    "action": "Supplier pricing caused losses.",
                    "linked_metric_or_segment": "No supplier field exists.",
                    "expected_use": "Treat supplier terms as the cause.",
                    "display_text": "Supplier pricing caused losses.",
                },
                {
                    "action": "Review Furniture in the 30%+ discount tier before changing category-wide pricing.",
                    "linked_metric_or_segment": "Furniture / 30%+ discount tier.",
                    "expected_use": "Keep the action scoped to evidence.",
                    "display_text": "Review Furniture in the 30%+ discount tier before changing category-wide pricing.",
                },
            ],
        },
        output_language="en",
    )

    html = render_client_report_html(payload, output_language="en")
    notebook_markdown = build_final_conclusion_markdown(
        report,
        schema,
        dataset_profile={"row_count": 999},
        final_synthesis=payload["final_synthesis"],
        output_language="en",
    )
    serialized = json.dumps(payload, ensure_ascii=False) + "\n" + html + "\n" + notebook_markdown

    assert "Furniture in the 30%+ discount tier had a -45.90% margin" in serialized
    assert "Review Furniture in the 30%+ discount tier before changing category-wide pricing" in serialized
    assert "primarily driven by" not in serialized
    assert "Discounting is the root cause" not in serialized
    assert "mandatory approval workflow" not in serialized
    assert "Furniture category has a -45.90% margin" not in serialized
    assert "Furniture is unprofitable" not in serialized
    assert "Supplier pricing caused losses" not in serialized
    assert "…" not in serialized
    assert "批准" not in serialized
    assert payload["final_synthesis"]["metadata"]["guarded_llm_final_synthesis_used"] is True


def test_chinese_client_report_keeps_existing_chinese_mode() -> None:
    payload, _trace = build_client_report_payload_with_trace(
        report=_generic_report(),
        schema_mapping=SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Sales": "sales_amount", "Profit": "profit", "Discount": "discount"},
            confidence=0.9,
        ),
        dataset_profile={"row_count": 240, "column_count": 16},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack=None,
        section_priority={"core_sections": ["discount_and_profit"]},
        llm_client=None,
        output_language="zh-CN",
    )

    html = render_client_report_html(payload, output_language="zh-CN")
    serialized = json.dumps(payload, ensure_ascii=False) + "\n" + html

    assert payload["title"] == "客户经营分析简报"
    assert "销售经营分析报告" in html
    assert "客户经营分析简报" in serialized
    assert "关键发现" in html
    assert "建议行动" in html
    assert "主要风险 Spotlight" in html
    assert "高风险区间" in html
    assert "亏损率" in html
    assert "受影响记录数" in html
    assert "模型召回率仪表盘" in html
    assert "Client-ready Sales Analysis Report" not in html
    assert "High-risk discount tier" not in html


def test_analysis_run_service_passes_final_synthesis_and_modeling_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_client_report_payload_with_trace(**kwargs):
        captured.update(kwargs)
        return {"title": "客户经营分析简报"}, SimpleNamespace(model_dump=lambda: {"status": "disabled"})

    class FakeRunBudget:
        def has_budget_for_stage(self, stage: str) -> bool:
            return True

    class FakeStore:
        def save_json(self, task_id, name, payload):
            return f"{task_id}/{name}"

        def save_text(self, task_id, name, text):
            return f"{task_id}/{name}"

    class FakeService:
        def _report(self, ctx):
            return "report"

        def _schema_mapping(self, ctx):
            return "schema"

        def _evidence_pack(self, ctx):
            return "evidence"

        def _section_priority(self, ctx):
            return "section"

        def _llm_evidence_pack(self, ctx):
            return "llm-evidence"

        def _client_report_payload(self, ctx):
            return ctx.client_report_payload

        def _save_llm_trace(self, ctx):
            ctx.saved_trace = True

    monkeypatch.setattr(
        analysis_run_service,
        "build_client_report_payload_with_trace",
        fake_build_client_report_payload_with_trace,
    )
    monkeypatch.setattr(
        analysis_run_service,
        "render_client_report_html",
        lambda payload, output_language=None: "<html></html>",
    )
    ctx = SimpleNamespace(
        task_id="task-1",
        store=FakeStore(),
        run_budget=FakeRunBudget(),
        dataset_profile={"row_count": 10},
        analysis_focus={},
        llm_client=None,
        llm_profile_policy={"client_report_enabled": True},
        llm_trace={},
        output_language="en",
        final_synthesis={"main_conclusions": []},
        modeling_outcome={"primary_modeling_task": "sales_amount_regression"},
        modeling_outcome_interpretation={"outcome_summary": "summary"},
        client_report_payload=None,
        client_report_json_path=None,
        client_report_html_path=None,
    )

    analysis_run_service.AnalysisRunService._build_client_report(FakeService(), ctx)

    assert captured["final_synthesis"] == ctx.final_synthesis
    assert captured["modeling_outcome"] == ctx.modeling_outcome
    assert captured["modeling_outcome_interpretation"] == ctx.modeling_outcome_interpretation
    assert ctx.client_report_json_path == "task-1/client_report.json"
    assert ctx.client_report_html_path == "task-1/client_report.html"
