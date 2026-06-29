from __future__ import annotations

import json
import re

from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
import app.services.notebook.final_synthesis as final_synthesis_module
from app.services.client_report_builder import (
    build_client_report_payload_with_trace,
    render_client_report_html,
)
from app.services.notebook.final_synthesis import build_final_synthesis_with_trace
from app.services.notebook.summary_builder import (
    build_final_conclusion_markdown,
    build_kaggle_analysis_brief_markdown,
)
from app.services.final_synthesis_guard import guard_english_final_synthesis
from app.services.notebook.markdown_sanitizer import clean_business_text


class FinalSynthesisLLM:
    enabled = True
    source = "fake"
    configured_model = "fake-final-synthesis"
    last_completion_metrics = {"prompt_chars": 1234, "response_chars": 567}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete_json(self, *, system_prompt, user_payload, preferred_models=None, cache_stage=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "preferred_models": preferred_models,
                "cache_stage": cache_stage,
            }
        )
        return {
            "brief_findings": [
                {
                    "finding": "Superstore 的亏损压力集中在高折扣和低利润切片，已经不是单纯销售规模问题。",
                    "evidence": "负利润记录占比 18.65%，亏损风险模型 recall 96%。",
                    "business_meaning": "需要把折扣和亏损复核放在同一条经营主线上看，避免只按销售额分配资源。",
                },
                {
                    "finding": "已完成 12 个指标画像。",
                    "evidence": "流程日志",
                    "business_meaning": "不可用于结论",
                },
            ],
            "brief_actions": [
                {
                    "action": "先抽取模型高风险订单形成复核池，再按折扣区间分层核查。",
                    "linked_evidence": "LogisticRegression recall 96%，误报 183、漏判 19。",
                    "priority_reason": "阈值调整会改变复核量和漏判风险。",
                },
                {
                    "action": "加强管理。",
                    "linked_evidence": "",
                    "priority_reason": "",
                },
            ],
            "main_conclusions": [
                {
                    "conclusion": "亏损风险模型已经能把高折扣、低利润订单转成可排序的人工复核线索。",
                    "evidence": "LogisticRegression precision 71%、recall 96%、F1 82%、ROC AUC 99%。",
                    "business_meaning": "模型适合做复核优先级，不适合自动决策；阈值要在复核量、误报和漏判之间取舍。",
                }
            ],
            "recommended_actions": [
                {
                    "action": "以模型高风险分数生成订单清单，先试跑 0.4/0.5/0.7 三档阈值。",
                    "linked_metric_or_segment": "误报 183、漏判 19、负利润记录占比 18.65%。",
                    "expected_use": "选择运营团队能承受的复核量，并记录人工复核标签供下一轮模型校准。",
                    "display_text": "由于误报 183、漏判 19 且负利润记录占比 18.65%，建议用模型高风险分数生成订单清单，并试跑 0.4/0.5/0.7 三档阈值，先确定运营团队能承受的复核量，再用人工标签校准下一轮模型。",
                }
            ],
        }


def _schema() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Order ID": "order_id",
            "Category": "category",
            "Region": "region",
        },
        confidence=0.95,
    )


def _report() -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["已完成 12 个指标画像。", "负利润记录占比为 18.65%，高折扣订单需要优先复核。"],
        modules=[
            ModuleReport(
                module_id="loss_risk_modeling",
                title="亏损风险建模",
                chart_type="model",
                findings=["LogisticRegression 对亏损样本的召回较高，但误报和漏判需要结合阈值复核。"],
                summary_metrics={
                    "best_model": "LogisticRegression",
                    "best_precision": 0.7136,
                    "best_recall": 0.96,
                    "best_f1": 0.8187,
                    "best_roc_auc": 0.9867,
                    "false_positive_count": 183,
                    "false_negative_count": 19,
                    "model_quality_status": "strong",
                },
            )
        ],
    )


def _scoped_report() -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=3,
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
                tables={
                    "discount_threshold_candidates": [
                        {
                            "Category": "Furniture",
                            "discount_bucket": "30%+",
                            "avg_profit": -44169.46,
                            "profit_margin": -0.4590,
                        },
                        {
                            "category": "Furniture",
                            "profit": 19730.00,
                            "profit_margin": 0.0261,
                        },
                    ],
                    "segment_region": [
                        {
                            "Segment": "Consumer",
                            "Region": "West",
                            "profit_margin": -0.184,
                        }
                    ],
                    "product_channel": [
                        {
                            "Product Name": "Copier",
                            "Sales Channel": "Online",
                            "profit_margin": -0.212,
                        }
                    ],
                },
            ),
        ],
    )


def _scoped_schema() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Category": "category",
            "Discount Bucket": "discount_bucket",
            "Segment": "segment",
            "Region": "region",
            "Product Name": "product_name",
            "Sales Channel": "channel",
        },
        confidence=0.95,
    )


def test_english_final_synthesis_guard_normalizes_scope_keys_and_preserves_aggregate() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "Furniture category has a -45.90% margin.",
                "evidence": "Category Furniture and discount_bucket 30%+ produced the weak margin.",
                "business_meaning": "Furniture is unprofitable.",
            },
            {
                "conclusion": "Furniture in the 30%+ discount tier had a -45.90% margin.",
                "evidence": "The scoped row combines Category and discount_bucket.",
                "business_meaning": "This should remain a scoped discount-tier review.",
            },
            {
                "conclusion": "Furniture overall profit was 19,730.00 with a 2.61% margin.",
                "evidence": "The aggregate row only uses category.",
                "business_meaning": "The aggregate category fact should remain separate from the scoped discount loss.",
            },
            {
                "conclusion": "Consumer segment had an -18.40% margin.",
                "evidence": "Segment and Region were grouped together.",
                "business_meaning": "This incorrectly drops Region.",
            },
            {
                "conclusion": "Copier in the Online channel had a -21.20% margin.",
                "evidence": "Product Name and Sales Channel were both present.",
                "business_meaning": "Custom raw field names should still preserve scope.",
            },
        ],
        "recommended_actions": [
            {
                "action": "Stop selling Furniture category-wide because it is unprofitable.",
                "linked_metric_or_segment": "Furniture -45.90% margin.",
                "expected_use": "Category-wide stop-sell.",
                "display_text": "Stop selling Furniture category-wide because it is unprofitable.",
            },
            {
                "action": "Review Furniture in the 30%+ discount tier before changing category-wide pricing.",
                "linked_metric_or_segment": "Furniture / 30%+ discount tier.",
                "expected_use": "Keep the review at the supported evidence scope.",
                "display_text": "Review Furniture in the 30%+ discount tier before changing category-wide pricing.",
            },
        ],
    }

    guarded = guard_english_final_synthesis(
        raw,
        report=_scoped_report(),
        schema_mapping=_scoped_schema(),
    )
    serialized = str(guarded)

    assert "Furniture category has a -45.90% margin" not in serialized
    assert "Furniture is unprofitable" not in serialized
    assert "Stop selling Furniture category-wide" not in serialized
    assert "Consumer segment had an -18.40% margin" not in serialized
    assert "Furniture in the 30%+ discount tier had a -45.90% margin" in serialized
    assert "Furniture overall profit was 19,730.00 with a 2.61% margin" in serialized
    assert "Copier in the Online channel had a -21.20% margin" in serialized
    assert guarded["metadata"]["dropped_for_scope"] >= 3
    assert guarded["metadata"]["guarded_llm_final_synthesis_used"] is True


def test_english_final_synthesis_guard_downgrades_unsupported_causal_actions() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "Supplier pricing caused losses.",
                "evidence": "No supplier field is mapped.",
                "business_meaning": "Procurement is responsible.",
            },
            {
                "conclusion": "Discount risk should be validated with supporting records.",
                "evidence": "The current data has sales, profit, and discount fields.",
                "business_meaning": "Root-cause claims need extra fields.",
            },
        ],
        "recommended_actions": [
            {
                "action": "Approval tiers are failing and caused the loss.",
                "linked_metric_or_segment": "No approval records were mapped.",
                "expected_use": "Treat approval controls as the cause.",
                "display_text": "Approval tiers are failing and caused the loss.",
            },
            {
                "action": "Review campaign, promotion, or inventory data if available before assigning a root cause.",
                "linked_metric_or_segment": "Those fields are not mapped.",
                "expected_use": "Validate rather than assert cause.",
                "display_text": "Review campaign, promotion, or inventory data if available before assigning a root cause.",
            },
        ],
    }

    guarded = guard_english_final_synthesis(
        raw,
        report=_scoped_report(),
        schema_mapping=_scoped_schema(),
    )
    serialized = str(guarded)

    assert "Supplier pricing caused losses" not in serialized
    assert "Procurement is responsible" not in serialized
    assert "Approval tiers are failing" not in serialized
    assert "Review approval records if available before changing approval controls." in serialized
    assert "Review campaign, promotion, or inventory data if available before assigning a root cause." in serialized
    assert guarded["metadata"]["dropped_for_unsupported_cause"] >= 1
    assert guarded["metadata"]["downgraded_for_missing_fields"] >= 1


def test_english_final_synthesis_guard_removes_causal_strength_and_missing_domain_claims() -> None:
    raw = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "Profit erosion is primarily driven by unsustainable discounting practices.",
                "evidence": "The current data has sales, profit, and discount fields.",
                "business_meaning": "Discounting is the root cause.",
            },
            {
                "conclusion": "Current promotional strategies are not value-accretive.",
                "evidence": "No promotion or campaign fields are mapped.",
                "business_meaning": "Campaign decisions caused the decline.",
            },
            {
                "conclusion": "Pricing does not cover their costs.",
                "evidence": "No cost field is mapped.",
                "business_meaning": "Cost structure is the root cause.",
            },
            {
                "conclusion": "Furniture in the 30%+ discount tier had a -45.90% margin.",
                "evidence": "The scoped row includes Category and discount_bucket.",
                "business_meaning": "Keep this risk observation scoped to the discount tier.",
            },
            {
                "conclusion": "The current discount tier is associated with elevated loss risk.",
                "evidence": "The threshold analysis shows elevated loss risk.",
                "business_meaning": "Use this as association evidence, not causality.",
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
                "action": "Renegotiate supplier terms.",
                "linked_metric_or_segment": "No supplier field exists.",
                "expected_use": "Treat supplier terms as the cause.",
                "display_text": "Renegotiate supplier terms.",
            },
            {
                "action": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
                "linked_metric_or_segment": "30%+ discount tier.",
                "expected_use": "Keep the action evidence-bound.",
                "display_text": "Review the scoped discount tier against profit outcomes before broader pricing changes.",
            },
        ],
    }

    guarded = guard_english_final_synthesis(
        raw,
        report=_scoped_report(),
        schema_mapping=_scoped_schema(),
    )
    guarded_again = guard_english_final_synthesis(
        guarded,
        report=_scoped_report(),
        schema_mapping=_scoped_schema(),
    )
    serialized = str(guarded)
    serialized_again = str(guarded_again)

    forbidden = (
        "primarily driven by",
        "Discounting is the root cause",
        "Cost structure is the root cause",
        "promotional strategies are not value-accretive",
        "pricing does not cover their costs",
        "mandatory approval workflow",
        "Renegotiate supplier terms",
    )
    for phrase in forbidden:
        assert phrase not in serialized
        assert phrase not in serialized_again
    assert "Furniture in the 30%+ discount tier had a -45.90% margin" in serialized
    assert "associated with elevated loss risk" in serialized
    assert "Review approval records if available before changing approval controls." in serialized
    assert "Review the scoped discount tier against profit outcomes before broader pricing changes." in serialized
    assert guarded["metadata"]["dropped_for_unsupported_cause"] >= 1
    assert guarded["metadata"]["downgraded_for_missing_fields"] >= 1
    assert any(
        "before" in item.get("action", "").lower()
        and (
            "approval records" in item.get("action", "").lower()
            or "scoped discount tier" in item.get("action", "").lower()
        )
        for item in guarded["recommended_actions"]
    )
    actions_after_second_guard = guarded_again["recommended_actions"]
    assert sum(1 for item in actions_after_second_guard if item.get("action") == "Review approval records if available before changing approval controls.") == 1
    assert guarded_again["metadata"]["fallback_item_count"] <= guarded["metadata"]["fallback_item_count"]


def test_english_final_conclusion_uses_guarded_llm_synthesis_with_local_fallbacks() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "Furniture in the 30%+ discount tier had a -45.90% margin.",
                "evidence": "The evidence row includes Category and discount_bucket.",
                "business_meaning": "Review the scoped discount tier before category-wide action.",
            },
            {
                "conclusion": "This section reviews conclusions using available module evidence.",
                "evidence": "Template prose.",
                "business_meaning": "Generic statement.",
            },
            {
                "conclusion": "Furniture category has a -45.90% margin.",
                "evidence": "The scoped row also had discount_bucket 30%+.",
                "business_meaning": "Furniture is unprofitable.",
            },
        ],
        "recommended_actions": [
            {
                "action": "复核高折扣订单。",
                "linked_metric_or_segment": "30%+ discount tier.",
                "expected_use": "中文 action should not remain.",
                "display_text": "复核高折扣订单。",
            },
            {
                "action": "Review Furniture in the 30%+ discount tier before changing category-wide pricing.",
                "linked_metric_or_segment": "Furniture / 30%+ discount tier.",
                "expected_use": "Keep the action scoped to evidence.",
                "display_text": "Review Furniture in the 30%+ discount tier before changing category-wide pricing.",
            },
            {
                "action": "Address margi…",
                "linked_metric_or_segment": "Truncated.",
                "expected_use": "Should fallback.",
                "display_text": "Address margi…",
            },
        ],
    }

    markdown = build_final_conclusion_markdown(
        _scoped_report(),
        _scoped_schema(),
        dataset_profile={"row_count": 999},
        final_synthesis=synthesis,
        output_language="en",
    )

    assert "Furniture in the 30%+ discount tier had a -45.90% margin" in markdown
    assert "Review Furniture in the 30%+ discount tier before changing category-wide pricing" in markdown
    assert "This section reviews" not in markdown
    assert "Furniture category has a -45.90% margin" not in markdown
    assert "Furniture is unprofitable" not in markdown
    assert "复核" not in markdown
    assert "…" not in markdown
    assert "Total sales reached 2,297,200.86" in markdown
    assert "## Conclusions and Recommendations" in markdown
    assert "### Usage Limits" in markdown


def test_english_final_conclusion_falls_back_when_guarded_synthesis_is_unavailable() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "Furniture category has a -45.90% margin.",
                "evidence": "Scoped row has discount_bucket.",
                "business_meaning": "Furniture is unprofitable.",
            }
        ],
        "recommended_actions": [
            {
                "action": "批准流程失效导致亏损。",
                "linked_metric_or_segment": "无审批字段。",
                "expected_use": "中文应被删除。",
                "display_text": "批准流程失效导致亏损。",
            }
        ],
    }

    markdown = build_final_conclusion_markdown(
        _scoped_report(),
        _scoped_schema(),
        dataset_profile={"row_count": 999},
        final_synthesis=synthesis,
        output_language="en",
    )

    assert "Total sales reached 2,297,200.86" in markdown
    assert "Furniture category has a -45.90% margin" not in markdown
    assert "批准流程" not in markdown
    assert "### Recommendations" in markdown


def test_final_synthesis_uses_llm_context_and_renders_llm_output() -> None:
    llm = FinalSynthesisLLM()
    report = _report()

    synthesis, trace = build_final_synthesis_with_trace(
        report=report,
        schema_mapping=_schema(),
        dataset_profile={
            "row_count": 10194,
            "total_sales_amount": 2297200.86,
            "total_profit_amount": 286397.02,
            "negative_profit_rate": 0.1865,
            "monthly_volatility": 0.5189,
        },
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack={
            "dataset_signature": {"dominant_story": "discount_loss"},
            "focus_evidence": {
                "discount_erosion_focus": [
                    {
                        "finding": "高折扣订单更容易侵蚀利润。",
                        "business_meaning": "折扣不是单纯促销工具，也会改变订单盈利质量。",
                    }
                ]
            },
        },
        modeling_outcome={
            "primary_modeling_task": "loss_risk_classification",
            "modeling_status": "strong_model",
            "metrics_summary": report.modules[0].summary_metrics,
        },
        action_plan_rows=[
            {
                "issue": "折扣和利润质量需要先收敛风险口径",
                "action": "按折扣区间设置审批阈值。",
            }
        ],
        chart_selection_plan={
            "selected_charts": [
                {"chart_id": "discount_profit_quality_bar", "business_question": "折扣是否侵蚀利润"}
            ]
        },
        llm_client=llm,
        output_language="zh-CN",
    )

    assert llm.calls, "final synthesis should call LLM as the normal path"
    call = llm.calls[0]
    assert call["cache_stage"] == "final_synthesis"
    assert "不要写流程日志" in call["system_prompt"]
    assert "display_text" in call["system_prompt"]
    assert "不要只是 action、linked_metric_or_segment、expected_use 的拼接" in call["system_prompt"]
    assert call["user_payload"]["final_synthesis_context"]["dataset_profile"]["negative_profit_rate"] == 0.1865
    assert call["user_payload"]["final_synthesis_context"]["modeling_outcome"]["primary_modeling_task"] == "loss_risk_classification"
    assert call["user_payload"]["final_synthesis_context"]["action_plan_rows"][0]["issue"] == "折扣和利润质量需要先收敛风险口径"

    assert trace.status == "llm_applied"
    assert synthesis["metadata"]["final_synthesis_used_llm"] is True
    assert synthesis["metadata"]["dropped_process_log_bullets_count"] == 1
    assert synthesis["metadata"]["dropped_generic_bullets_count"] == 1
    assert synthesis["metadata"]["final_action_display_text_missing_count"] == 0
    assert synthesis["recommended_actions"][0]["display_text"].startswith("由于误报 183")

    brief = build_kaggle_analysis_brief_markdown(
        report,
        {"total_sales_amount": 2297200.86, "negative_profit_rate": 0.1865},
        {"selected_focuses": ["discount_erosion_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )
    conclusion = build_final_conclusion_markdown(
        report,
        _schema(),
        {"negative_profit_rate": 0.1865},
        {"selected_focuses": ["discount_erosion_focus"]},
        {},
        modeling_outcome={"primary_modeling_task": "loss_risk_classification"},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "Superstore 的亏损压力集中在高折扣和低利润切片" in brief
    assert "### 行动建议" not in brief
    assert "先抽取模型高风险订单形成复核池" not in brief
    assert "亏损风险模型已经能把高折扣、低利润订单转成可排序的人工复核线索" in conclusion
    assert "建议用模型高风险分数生成订单清单" in conclusion
    assert "0.4/0.5/0.7 三档阈值" in conclusion
    assert "以模型高风险分数生成订单清单，先试跑 0.4/0.5/0.7 三档阈值；误报 183" not in conclusion
    assert "已完成 12 个指标画像" not in brief + conclusion
    assert "加强管理" not in brief + conclusion
    assert synthesis["metadata"]["final_action_display_text_used_count"] == 1
    assert synthesis["metadata"]["final_action_display_text_fallback_count"] == 0
    assert synthesis["metadata"]["final_action_render_mode"] == "display_text"


def _visible_html_text(html: str) -> str:
    text = re.sub(r"(?is)<style.*?</style>", " ", html)
    text = re.sub(r"(?is)<script.*?</script>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def test_english_final_synthesis_builder_applies_final_public_seal_after_modeling_backfill(monkeypatch) -> None:
    unsafe_display = (
        "Integrate risk scores into an order-processing workflow to automatically flag "
        "the top 7% highest-risk orders for mandatory human review before fulfillment."
    )

    def fake_complete_json_for_stage(*args, **kwargs):
        return {
            "metadata": {
                "final_synthesis_used_llm": True,
                "english_public_narrative_guard_applied": True,
            },
            "brief_findings": [
                {
                    "finding": "The model can support manual review prioritization.",
                    "evidence": "Recall and precision are available in the modeling metrics.",
                    "business_meaning": "The model must not be used for automatic decisions.",
                }
            ],
            "recommended_actions": [
                {
                    "priority": "P1",
                    "action": "Deploy the risk model for order auditing.",
                    "issue": "Model scores need operational workflow integration.",
                    "linked_metric_or_segment": "Loss-risk model scores.",
                    "expected_use": "Focus audit resources on the highest-risk orders.",
                    "display_text": unsafe_display,
                }
            ],
        }

    monkeypatch.setattr(final_synthesis_module, "complete_json_for_stage", fake_complete_json_for_stage)
    report = _report()
    synthesis, trace = build_final_synthesis_with_trace(
        report=report,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10194, "negative_profit_rate": 0.1865},
        analysis_focus={"selected_focuses": ["loss_risk_focus"]},
        evidence_pack={},
        modeling_outcome={
            "primary_modeling_task": "loss_risk_classification",
            "modeling_status": "strong_model",
            "metrics_summary": report.modules[0].summary_metrics,
        },
        chart_contexts=[],
        llm_client=FinalSynthesisLLM(),
        output_language="en",
    )

    final_markdown = build_final_conclusion_markdown(
        report,
        _schema(),
        {"negative_profit_rate": 0.1865},
        {"selected_focuses": ["loss_risk_focus"]},
        {},
        modeling_outcome={"primary_modeling_task": "loss_risk_classification"},
        final_synthesis=synthesis,
        output_language="en",
    )
    client_payload, _client_trace = build_client_report_payload_with_trace(
        report=report,
        schema_mapping=_schema(),
        dataset_profile={"row_count": 10194, "column_count": 8},
        analysis_focus={"selected_focuses": ["loss_risk_focus"]},
        evidence_pack=None,
        section_priority=None,
        llm_client=None,
        final_synthesis=synthesis,
        modeling_outcome={"primary_modeling_task": "loss_risk_classification"},
        output_language="en",
    )
    client_html = render_client_report_html(client_payload, output_language="en")
    public_text = "\n".join(
        [
            json.dumps(synthesis, ensure_ascii=False),
            final_markdown,
            json.dumps(client_payload, ensure_ascii=False),
            _visible_html_text(client_html),
        ]
    )

    assert trace.status == "llm_applied"
    assert synthesis["metadata"]["quality_gate_backfilled_required_modeling_coverage"] is True
    assert synthesis["metadata"]["english_public_narrative_guard_applied"] is True
    assert "automatically" not in public_text.lower()
    assert "order-processing workflow" not in public_text.lower()
    assert "order processing" not in public_text.lower()
    assert "top 7%" not in public_text.lower()
    assert "mandatory human review" not in public_text.lower()
    assert "before fulfillment" not in public_text.lower()
    assert "manual review prioritization" in public_text.lower() or "review capacity" in public_text.lower()
    assert "must not be used for automatic decisions" in public_text.lower()


def test_llm_final_action_display_text_is_rendered_instead_of_field_joining() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "亏损风险模型能提供人工复核线索。",
                "evidence": "recall 96%。",
                "business_meaning": "模型只能用于排序辅助。",
            }
        ],
        "recommended_actions": [
            {
                "action": "机械动作字段不应该直接展示。",
                "linked_metric_or_segment": "机械指标字段不应该直接展示。",
                "expected_use": "机械用途字段不应该直接展示。",
                "display_text": "由于模型召回率达到 96%，建议先把高风险订单整理成人工复核清单，并在阈值试跑后再决定复核量，避免把模型分数直接当成自动决策依据。",
            }
        ],
    }

    markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        {"negative_profit_rate": 0.1865},
        {"selected_focuses": ["discount_erosion_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "由于模型召回率达到 96%" in markdown
    assert "机械动作字段" not in markdown
    assert "机械指标字段" not in markdown
    assert "机械用途字段" not in markdown
    assert synthesis["metadata"]["final_action_display_text_used_count"] == 1
    assert synthesis["metadata"]["final_action_display_text_missing_count"] == 0
    assert synthesis["metadata"]["final_action_display_text_fallback_count"] == 0
    assert synthesis["metadata"]["final_action_display_text_rejected_count"] == 0


def test_llm_final_action_fallback_renders_natural_sentence_without_display_text() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "预测模型只能作为监控参考。",
                "evidence": "R²为6.62%。",
                "business_meaning": "不能用于生产级预测。",
            }
        ],
        "recommended_actions": [
            {
                "action": "回查预测误差最大的时间段。",
                "linked_metric_or_segment": "最大误差周期集中在节假日前后，R²为6.62%。",
                "expected_use": "判断误差来自促销、大单还是模型缺少解释变量。",
            }
        ],
    }

    markdown = build_final_conclusion_markdown(
        _report(),
        SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Date": "order_datetime", "Sales": "sales_amount"},
            confidence=0.9,
        ),
        {"has_profit": False},
        {"selected_focuses": ["trend_volatility_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "建议回查预测误差最大的时间段" in markdown
    assert "依据是最大误差周期集中在节假日前后，R²为6.62%" in markdown
    assert "用于判断误差来自促销、大单还是模型缺少解释变量" in markdown
    assert "回查预测误差最大的时间段。；最大误差周期" not in markdown
    assert synthesis["metadata"]["final_action_display_text_missing_count"] == 1
    assert synthesis["metadata"]["final_action_display_text_fallback_count"] == 1
    assert synthesis["metadata"]["final_action_render_mode"] == "fallback_natural"


def test_llm_final_action_rejects_mechanical_display_text_and_falls_back() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "折扣切片需要复盘。",
                "evidence": "折扣 30%+。",
                "business_meaning": "需要核查定价口径。",
            }
        ],
        "recommended_actions": [
            {
                "action": "复盘折扣 30%+ 且盈利为负的订单。",
                "linked_metric_or_segment": "折扣 30%+、盈利为负。",
                "expected_use": "核查审批、定价和成本口径。",
                "display_text": "复盘折扣 30%+ 且盈利为负的订单；折扣 30%+、盈利为负；核查审批、定价和成本口径。",
            }
        ],
    }

    markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        {"negative_profit_rate": 0.1865},
        {"selected_focuses": ["discount_erosion_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "复盘折扣 30%+ 且盈利为负的订单；折扣 30%+、盈利为负；核查审批" not in markdown
    assert "建议复盘折扣 30%+ 且盈利为负的订单" in markdown
    assert "依据是折扣 30%+、盈利为负" in markdown
    assert synthesis["metadata"]["final_action_display_text_rejected_count"] == 1
    assert synthesis["metadata"]["final_action_display_text_fallback_count"] == 1


def test_llm_final_conclusion_keeps_model_metric_key_value_evidence() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "亏损风险建模受限于正例极度不足，当前仅适合探索性参考。",
                "evidence": "target_positive_rate=0.055%，test_positive_count=3，F1=9.76%，模型状态weak。",
                "business_meaning": "不建议直接用于复核排序或自动决策，应依赖 EDA 和业务复盘。",
            }
        ],
        "recommended_actions": [
            {
                "action": "不要直接使用当前亏损风险模型做排序。",
                "linked_metric_or_segment": "测试集正例=3，模型质量=weak_model。",
                "expected_use": "等待正例标签更稳定后再评估模型。",
                "display_text": "由于测试集正例只有 3 个且模型质量为 weak_model，不要直接使用当前亏损风险模型做排序，应先回到 EDA 切片和业务复盘，待正例标签更稳定后再评估模型。",
            }
        ],
    }

    markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        {"negative_profit_rate": 0.00055},
        {"selected_focuses": ["profit_quality_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "当前仅适合探索性参考" in markdown
    assert "target_positive_rate 为 0.055%" in markdown
    assert "不建议直接用于复核排序" in markdown


def test_llm_final_actions_are_lightly_deduped_before_rendering() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "亏损风险模型已经能把折扣和利润压力转成可复核线索。",
                "evidence": "precision 71%、recall 96%。",
                "business_meaning": "适合人工复核排序，不适合自动决策。",
            }
        ],
        "recommended_actions": [
            {
                "action": "将模型预测为高风险的订单生成每日人工复核清单。",
                "linked_metric_or_segment": "高风险订单。",
                "expected_use": "让运营先处理最可能亏损的订单。",
            },
            {
                "action": "用模型高风险订单试跑 0.4/0.5/0.7 三档阈值，并记录误报、漏判和复核量。",
                "linked_metric_or_segment": "误报 183、漏判 19。",
                "expected_use": "选择运营团队能承受的复核量。",
            },
            {
                "action": "补充价格、库存、渠道字段后重跑预测误差分析。",
                "linked_metric_or_segment": "当前缺少价格、库存、渠道字段。",
                "expected_use": "解释模型误差来源。",
            },
            {
                "action": "补充促销、节假日、价格、库存、渠道字段。",
                "linked_metric_or_segment": "字段不足。",
                "expected_use": "提升后续建模可解释性。",
            },
            {
                "action": "复盘高折扣低利润订单。",
                "linked_metric_or_segment": "折扣和利润。",
                "expected_use": "控制利润侵蚀。",
            },
            {
                "action": "优先复盘折扣 30%+ 且利润为负的订单，核查审批、定价和成本口径。",
                "linked_metric_or_segment": "折扣 30%+、利润为负。",
                "expected_use": "把最明确的利润侵蚀场景先收敛。",
            },
        ],
    }

    markdown = build_final_conclusion_markdown(
        _report(),
        _schema(),
        {"negative_profit_rate": 0.1865},
        {"selected_focuses": ["discount_erosion_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "将模型预测为高风险的订单生成每日人工复核清单" not in markdown
    assert "0.4/0.5/0.7 三档阈值" in markdown
    assert "误报" in markdown and "漏判" in markdown and "复核量" in markdown
    assert markdown.count("补充") == 1
    assert "复盘高折扣低利润订单。" not in markdown
    assert "折扣 30%+ 且利润为负" in markdown


def test_llm_final_conclusion_avoids_profit_analysis_terms_without_profit_fields() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "main_conclusions": [
            {
                "conclusion": "当前数据存在 68 条负销售额记录，不能直接进入销售-利润交叉分析。",
                "evidence": "字段只覆盖销售额、时间和类目。",
                "business_meaning": "利润质量分析需要先补充利润、成本或毛利字段。",
            }
        ],
        "recommended_actions": [
            {
                "action": "将当前预测模型（RandomForestRegressor-small，MAE改善4.85%）作为内部销售监控的基线参照，但不用于实际业务决策。",
                "linked_metric_or_segment": "销售额预测，baseline MAE 253982.68，R²为6.62%。",
                "expected_use": "为后续引入更多变量（如促销、库存）的模型迭代提供一个可量化的比较基准。",
            },
            {
                "action": "基于 68 条负销售额记录，进行类目级销售-利润分析。",
                "linked_metric_or_segment": "负销售额记录。",
                "expected_use": "核查利润质量分析。",
            }
        ],
    }
    schema = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Weekly_Sales": "sales_amount", "Date": "order_datetime", "Dept": "category"},
        confidence=0.9,
    )

    markdown = build_final_conclusion_markdown(
        _report(),
        schema,
        {"has_profit": False},
        {"selected_focuses": ["trend_volatility_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "销售-利润交叉分析" not in markdown
    assert "类目级销售-利润分析" not in markdown
    assert "利润质量分析" not in markdown
    assert "先核查负销售额记录的业务含义" in markdown
    assert "退货、冲销、录入错误还是特殊交易" in markdown
    assert "盈利相关、成本、退货等字段" in markdown
    assert "profit" not in markdown.lower()
    assert "margin" not in markdown.lower()
    assert "RandomForestRegressor-small" in markdown
    assert "baseline MAE 253982.68" in markdown
    assert "R²为6.62%" in markdown


def test_llm_final_conclusion_reuses_brief_sales_regression_metrics_when_action_lacks_r2() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "brief_findings": [
            {
                "finding": "预测模型只能作为监控基线。",
                "evidence": "最佳模型R²为6.62%，MAPE为10.98%，相对baseline MAE改善4.85%。",
                "business_meaning": "解释力有限。",
            }
        ],
        "main_conclusions": [
            {
                "conclusion": "预测模型仅具备有限参照价值。",
                "evidence": "模型依赖历史波动信号。",
                "business_meaning": "不应用作生产级预测。",
            }
        ],
        "recommended_actions": [
            {
                "action": "将当前预测模型（RandomForestRegressor-small，MAE改善4.85%）作为内部销售监控的基线参照。",
                "linked_metric_or_segment": "销售额预测，baseline MAE 253982.68。",
                "expected_use": "为后续引入促销、库存变量提供比较基准。",
            }
        ],
    }
    schema = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Weekly_Sales": "sales_amount", "Date": "order_datetime"},
        confidence=0.9,
    )

    markdown = build_final_conclusion_markdown(
        _report(),
        schema,
        {"has_profit": False},
        {"selected_focuses": ["trend_volatility_focus"]},
        {},
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "R²为6.62%" in markdown
    assert "相对baseline MAE改善4.85%" in markdown


def test_llm_final_conclusion_uses_modeling_outcome_when_display_text_lacks_baseline_metrics() -> None:
    synthesis = {
        "metadata": {"final_synthesis_used_llm": True},
        "brief_findings": [
            {
                "finding": "预测模型解释力偏弱。",
                "evidence": "最佳模型R²为6.62%，相对基线改善有限。",
                "business_meaning": "只能做监控参考。",
            }
        ],
        "main_conclusions": [
            {
                "conclusion": "预测模型只能作为轻量监控参考。",
                "evidence": "R²为6.62%。",
                "business_meaning": "不适合作为生产级预测。",
            }
        ],
        "recommended_actions": [
            {
                "action": "将预测模型限定为销售趋势监控器。",
                "linked_metric_or_segment": "R²为6.62%。",
                "expected_use": "识别波动异常。",
                "display_text": "鉴于预测模型解释力很弱（R²仅6.62%），建议将其用途严格限定为销售趋势的轻量监控器，并优先补充促销、节假日等外部字段。",
            }
        ],
    }

    markdown = build_final_conclusion_markdown(
        _report(),
        SchemaMapping(
            dataset_type="sales_transaction",
            field_mapping={"Weekly_Sales": "sales_amount", "Date": "order_datetime"},
            confidence=0.9,
        ),
        {"has_profit": False},
        {"selected_focuses": ["trend_volatility_focus"]},
        {},
        modeling_outcome={
            "primary_modeling_task": "sales_amount_regression",
            "metrics_summary": {
                "best_model_r2": 0.0662,
                "best_model_mae": 241668.61,
                "baseline_model": "naive_last_value",
                "baseline_mae": 253982.68,
                "improvement_vs_baseline": 4.85,
            },
        },
        final_synthesis=synthesis,
        output_language="zh-CN",
    )

    assert "MAE=241668.61" in markdown
    assert "naive_last_value baseline" in markdown
    assert "改善 4.85%" in markdown


def test_clean_business_text_keeps_preformatted_small_percent() -> None:
    assert clean_business_text("负利润记录占比为 0.06%") == "负利润记录占比为 0.06%"
