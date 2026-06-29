from __future__ import annotations

import json

from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_evidence_pack import build_llm_evidence_pack, build_section_evidence_slice


def _report_with_large_payloads() -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=4,
        summary=["总销售额为 1250000，负利润记录占比为 18.72%。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1250000.0, "order_count": 500, "empty": None},
                tables={"monthly_totals": [{"month": index, "sales": index * 1000} for index in range(20)]},
                chart_payload={"huge_plotly_payload": "x" * 5000},
                findings=[f"趋势发现 {index}" for index in range(9)],
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="dual_axis_bar_line",
                summary_metrics={
                    "total_profit_amount": 180000.0,
                    "profit_margin": 0.144,
                    "negative_profit_rate": 0.1872,
                    "unknown_metric": "unknown",
                },
                tables={
                    "discount_threshold_candidates": [
                        {"discount_bucket": "20-30%", "negative_profit_rate": 0.24, "avg_profit": 12.0},
                        {"discount_bucket": "30%+", "negative_profit_rate": 0.72, "avg_profit": -44.5},
                    ],
                    "discount_cap_what_if": [
                        {
                            "scenario_name": "高风险折扣收紧情景估算",
                            "high_risk_bucket": "30%+",
                            "estimated_profit_delta": 21200.0,
                        }
                    ],
                    "discount_profit_risk_buckets": [
                        {"discount_bucket": "0-10%", "avg_profit": 30.0, "loss_rate": 0.04},
                        {"discount_bucket": "10-20%", "avg_profit": 15.0, "loss_rate": 0.12},
                        {"discount_bucket": "20-30%", "avg_profit": 12.0, "loss_rate": 0.24},
                        {"discount_bucket": "30%+", "avg_profit": -44.5, "loss_rate": 0.72},
                    ],
                    "wide_debug_table": [{"row": index, "value": index} for index in range(30)],
                },
                chart_payload={"raw_chart": "y" * 5000},
                findings=[f"折扣发现 {index}" for index in range(8)],
                warnings=["what-if 是静态估算。"],
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
                },
                tables={
                    "threshold_analysis": [
                        {"threshold": 0.4, "recall": 0.95, "precision": 0.68},
                        {"threshold": 0.5, "recall": 0.91, "precision": 0.72},
                        {"threshold": 0.6, "recall": 0.84, "precision": 0.77},
                    ],
                    "feature_importance_grouped": [
                        {"feature_group": "discount", "total_importance": 0.42},
                        {"feature_group": "category", "total_importance": 0.16},
                    ],
                    "confusion_matrix": [{"actual": "loss", "predicted": "loss", "count": 42}],
                    "high_risk_examples": [{"row_index": index, "loss_probability": 0.9 - index * 0.01} for index in range(10)],
                },
                findings=["模型仅用于人工复核排序，不代表因果关系。"],
            ),
        ],
    )


def _schema_mapping() -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            "Sales": "sales_amount",
            "Profit": "profit",
            "Discount": "discount",
            "Order ID": "order_id",
            "Product": "product_name",
            "Category": "category",
            "Segment": "segment",
            "Region": "region",
        },
        confidence=0.92,
        missing_required_fields=["customer_id"],
        uncertain_fields=["Ship Mode"],
    )


def test_llm_evidence_pack_trims_large_payloads_and_keeps_core_evidence() -> None:
    pack = build_llm_evidence_pack(
        report=_report_with_large_payloads(),
        schema_mapping=_schema_mapping(),
        dataset_profile={"row_count": 999, "column_count": 18},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        chart_selection_plan={
            "selected_charts": [
                {
                    "section_id": "discount_and_profit",
                    "chart_id": "discount_quality",
                    "title": "各折扣区间利润质量",
                    "chart_type": "dual_axis_bar_line",
                    "business_question": "哪个折扣区间风险最高？",
                    "reason": "同时比较平均利润和亏损率。",
                    "chart_payload": {"raw": "must-not-leak"},
                }
            ]
        },
        max_table_rows=3,
        max_findings_per_module=2,
    )
    serialized = json.dumps(pack, ensure_ascii=False)

    assert "chart_payload" not in serialized
    assert "huge_plotly_payload" not in serialized
    assert pack["dataset"]["row_count"] == 999
    assert pack["dataset"]["column_count"] == 18
    assert pack["dataset"]["mapped_fields"][0] == {"original": "Sales", "canonical": "sales_amount"}
    assert pack["dataset"]["missing_required_fields"] == ["customer_id"]
    assert pack["business_kpis"]["total_sales"] == 1250000.0
    assert pack["business_kpis"]["total_profit"] == 180000.0
    assert pack["business_kpis"]["negative_profit_rate"] == 0.1872

    discount_module = next(item for item in pack["module_evidence"] if item["module_id"] == "discount_profit_analysis")
    assert len(discount_module["top_findings"]) == 2
    assert len(discount_module["key_tables"]["discount_profit_risk_buckets"]) == 3
    assert "wide_debug_table" not in discount_module["key_tables"]
    assert "unknown_metric" not in discount_module["summary_metrics"]

    assert pack["discount_evidence"]["threshold_candidates"][1]["discount_bucket"] == "30%+"
    assert pack["discount_evidence"]["what_if"][0]["estimated_profit_delta"] == 21200.0
    assert len(pack["discount_evidence"]["risk_buckets"]) == 3
    assert pack["modeling_evidence"]["best_model"] == "LogisticRegression"
    assert pack["modeling_evidence"]["metrics"]["best_recall"] == 0.91
    assert pack["modeling_evidence"]["threshold_analysis"]
    assert pack["modeling_evidence"]["feature_importance_grouped"][0]["feature_group"] == "discount"
    assert pack["action_evidence"]["p1_actions"]
    assert pack["action_evidence"]["p2_actions"]
    assert pack["chart_evidence"][0]["chart_kind"] == "dual_axis_bar_line"
    assert pack["limitations"]


def test_section_evidence_slice_keeps_only_relevant_modules_and_common_evidence() -> None:
    pack = build_llm_evidence_pack(
        report=_report_with_large_payloads(),
        schema_mapping=_schema_mapping(),
        max_table_rows=2,
    )

    section_slice = build_section_evidence_slice(pack, "discount_and_profit")

    assert section_slice["dataset"]["dataset_type"] == "sales_transaction"
    assert section_slice["discount_evidence"]["what_if"][0]["high_risk_bucket"] == "30%+"
    assert [item["module_id"] for item in section_slice["module_evidence"]] == ["discount_profit_analysis"]
