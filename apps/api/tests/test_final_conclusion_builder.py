from __future__ import annotations

from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook.summary_builder import (
    build_final_conclusion_markdown,
    build_kaggle_analysis_brief_markdown,
)


def _schema(*fields: str) -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={field: field for field in fields},
        confidence=0.9,
    )


def _report(modules: list[ModuleReport]) -> AnalysisReport:
    return AnalysisReport(
        task_id="task-final",
        dataset_type="sales_transaction",
        module_count=len(modules),
        summary=[
            "本节汇总最重要的发现，并将它们转化成可执行的经营动作。",
            "一份分析 notebook 的最终价值，不在于图表多少。",
            "Top 商品为 Canon imageCLASS 2200 Advanced Copier。",
            "已按 segment x region 完成双维拆解。",
            "发现 68 条负销售额记录。",
            "峰值日期为 2010-12-24，谷值日期为 2011-01-14。",
            "period=2010-02-05，Weekly_Sales=2,813,663.33",
            "20% 以上折扣订单平均利润为 -18.4。",
        ],
        modules=modules,
    )


def test_final_conclusion_turns_strong_loss_risk_into_business_judgment() -> None:
    report = _report(
        [
            ModuleReport(
                module_id="loss_risk_modeling",
                title="亏损风险建模",
                chart_type="model",
                findings=["模型显示高折扣、低利润类目和部分区域组合更容易进入亏损复核清单。"],
                summary_metrics={
                    "best_model": "LogisticRegression",
                    "model_quality_status": "usable",
                    "best_precision": 0.74,
                    "best_recall": 0.82,
                    "best_f1": 0.78,
                    "best_roc_auc": 0.86,
                    "threshold_default": 0.5,
                    "false_positive_count": 11,
                    "false_negative_count": 5,
                },
                tables={
                    "threshold_analysis": [
                        {"threshold": 0.4, "predicted_loss_count": 120, "precision": 0.62, "recall": 0.91},
                        {"threshold": 0.7, "predicted_loss_count": 47, "precision": 0.83, "recall": 0.58},
                    ],
                    "feature_importance_grouped": [
                        {"feature_group": "discount", "total_importance": 0.42},
                        {"feature_group": "sub_category", "total_importance": 0.27},
                    ],
                },
            )
        ]
    )

    markdown = build_final_conclusion_markdown(
        report,
        _schema("sales_amount", "profit", "discount", "category", "segment", "region", "order_id"),
        dataset_profile={"negative_profit_rate": 0.18},
        analysis_focus={"selected_focuses": ["discount_erosion_focus", "profit_quality_focus"]},
        evidence_pack={
            "distinctive_facts": [
                {"text": "负利润记录占比为 18.00%，高折扣订单需要优先复核。"},
                {"text": "高销售低利润商品会让规模增长掩盖利润压力。"},
            ]
        },
        modeling_outcome={
            "primary_modeling_task": "loss_risk_classification",
            "modeling_status": "usable",
            "summary_metrics": report.modules[0].summary_metrics,
        },
        output_language="zh-CN",
    )

    assert "本节汇总最重要的发现" not in markdown
    assert "一份分析 notebook 的最终价值" not in markdown
    assert "Top 商品为 Canon" not in markdown
    assert "已按 segment x region 完成双维拆解" not in markdown
    assert "发现 68 条负销售额记录" not in markdown
    assert "峰值日期为" not in markdown
    assert "period=2010-02-05" not in markdown
    assert "人工复核" in markdown
    assert "LogisticRegression" in markdown
    assert "阈值" in markdown
    assert "误报" in markdown or "漏判" in markdown
    assert "高折扣" in markdown


def test_final_conclusion_downgrades_weak_loss_risk() -> None:
    report = _report(
        [
            ModuleReport(
                module_id="loss_risk_modeling",
                title="亏损风险建模",
                chart_type="model",
                findings=["亏损正例很少，测试集正例不足导致 precision 和 F1 不稳定。"],
                summary_metrics={
                    "best_model": "RandomForestClassifier-small",
                    "model_quality_status": "weak",
                    "target_positive_rate": 0.008,
                    "test_positive_count": 2,
                    "best_precision": 0.18,
                    "best_recall": 0.5,
                    "best_f1": 0.26,
                },
            )
        ]
    )

    markdown = build_final_conclusion_markdown(
        report,
        _schema("sales_amount", "profit", "category", "country"),
        dataset_profile={"negative_profit_rate": 0.008},
        analysis_focus={"selected_focuses": ["profit_quality_focus", "country_market_focus"]},
        modeling_outcome={
            "primary_modeling_task": "loss_risk_classification",
            "modeling_status": "weak",
            "summary_metrics": report.modules[0].summary_metrics,
        },
        output_language="zh-CN",
    )

    assert "亏损样本" in markdown or "正例" in markdown
    assert "探索性参考" in markdown
    assert "不建议直接用于复核排序" in markdown
    assert "自动决策" in markdown
    assert "EDA" in markdown or "业务复盘" in markdown
    assert "亏损占比=0.8%" in markdown


def test_final_conclusion_summarizes_sales_regression_limits_and_next_data() -> None:
    report = _report(
        [
            ModuleReport(
                module_id="forecast_analysis",
                title="销售额预测建模",
                chart_type="line",
                findings=["RandomForestRegressor-small 相对 naive_last_value baseline 有小幅改善，但 R2 仍然偏低。"],
                summary_metrics={
                    "forecast_status": "sales_regression_evaluated",
                    "best_model": "RandomForestRegressor-small",
                    "baseline_model": "naive_last_value",
                    "best_model_mae": 248.6,
                    "baseline_mae": 260.1,
                    "improvement_vs_baseline": 4.42,
                    "best_model_r2": 0.06,
                    "best_model_mape": 10.98,
                },
            )
        ]
    )

    markdown = build_final_conclusion_markdown(
        report,
        _schema("order_datetime", "sales_amount", "category", "store"),
        dataset_profile={"monthly_volatility": 0.1468},
        analysis_focus={"selected_focuses": ["trend_volatility_focus", "product_concentration_focus"]},
        modeling_outcome={
            "primary_modeling_task": "sales_amount_regression",
            "modeling_status": "usable",
            "metrics_summary": report.modules[0].summary_metrics,
        },
        output_language="zh-CN",
    )

    assert "RandomForestRegressor-small" in markdown
    assert "baseline" in markdown
    assert "4.42" in markdown
    assert "R2" in markdown or "R²" in markdown
    assert "生产级预测" in markdown
    assert "节假日" in markdown
    assert "促销" in markdown
    assert "库存" in markdown or "价格" in markdown
    assert "已使用时间顺序切分" not in markdown
    assert "围绕 " not in markdown


def test_english_notebook_summary_and_conclusion_prioritize_structured_facts() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=3,
        summary=["本节汇总最重要的发现，并将它们转化成可执行的经营动作。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1250000.0},
            ),
            ModuleReport(
                module_id="discount_profit_analysis",
                title="折扣与利润分析",
                chart_type="scatter",
                summary_metrics={"total_profit_amount": 180000.0},
                tables={
                    "discount_threshold_candidates": [
                        {
                            "discount_bucket": "30%+",
                            "negative_profit_rate": 0.72,
                            "profit_margin": -0.31,
                            "risk_level": "high",
                        }
                    ]
                },
            ),
            ModuleReport(
                module_id="product_contribution_analysis",
                title="商品分析",
                chart_type="bar",
                tables={"product_sales": [{"product_name": "Copier", "sales_amount": 200000.0}]},
            ),
        ],
    )

    brief = build_kaggle_analysis_brief_markdown(
        report,
        dataset_profile={"row_count": 240},
        analysis_focus=None,
        evidence_pack=None,
        output_language="en",
    )
    conclusion = build_final_conclusion_markdown(
        report,
        _schema("sales_amount", "profit", "discount", "product_name"),
        dataset_profile={"row_count": 240},
        output_language="en",
    )
    combined = brief + "\n" + conclusion

    assert "Total sales reached 1,250,000" in combined
    assert "Total profit was 180,000" in combined
    assert "30%+" in combined
    assert "margin-risk area" in combined
    first_finding = next(line for line in brief.splitlines() if line.startswith("1. "))
    assert not first_finding.startswith("1. The analysis completed")


def test_english_notebook_summary_without_profit_discount_states_scope_boundary() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        summary=[],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 110875937.33},
            ),
            ModuleReport(
                module_id="data_quality_check",
                title="数据质量检查",
                chart_type="table",
                summary_metrics={"missing_order_datetime": 8812, "duplicate_rows": 8811},
            ),
        ],
    )

    markdown = build_final_conclusion_markdown(
        report,
        _schema("sales_amount", "quantity", "order_datetime", "sku"),
        dataset_profile={"row_count": 20000},
        output_language="en",
    )

    assert "Total sales reached 110,875,937.33" in markdown
    assert "Profit and Discount fields were not mapped" in markdown
    assert "margin-risk area" not in markdown
    assert "high-risk discount tier" not in markdown.lower()
    assert "profit quality" not in markdown.lower()


def test_english_final_conclusion_filters_generic_template_sentence() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[
            "This section reviews conclusions and recommended actions using the mapped sales dataset and available module evidence.",
            "West region contributed the largest sales amount in the current period.",
        ],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend Analysis",
                chart_type="line",
                summary_metrics={"total_sales_amount": 1250000.0},
            )
        ],
    )

    markdown = build_final_conclusion_markdown(
        report,
        _schema("sales_amount", "order_datetime", "region", "sku"),
        narrative_observations=["Sales concentration is highest in the top three SKUs."],
        output_language="en",
    )

    assert "This section reviews conclusions and recommended actions using the mapped sales dataset and available module evidence." not in markdown
    assert "Total sales reached 1,250,000" in markdown
    assert "Sales concentration is highest in the top three SKUs." in markdown
    assert "West region contributed the largest sales amount in the current period." in markdown
    assert "### Recommendations" in markdown
    assert "### Usage Limits" in markdown


def test_english_final_conclusion_filters_section_purpose_variants() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        summary=[
            "Section purpose: To synthesize key findings, risks, and next actions.",
            "This section reviews dataset and field mapping using the mapped sales dataset.",
            "Available module evidence should summarize key findings, risks, and next actions.",
            "West region sales reached 42,000 in the latest complete month.",
        ],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend Analysis",
                chart_type="line",
                summary_metrics={"total_sales_amount": 42000.0},
            )
        ],
    )

    markdown = build_final_conclusion_markdown(
        report,
        _schema("sales_amount", "order_datetime", "region"),
        output_language="en",
    )

    assert "Section purpose:" not in markdown
    assert "This section reviews" not in markdown
    assert "mapped sales dataset" not in markdown
    assert "available module evidence" not in markdown
    assert "summarize key findings, risks, and next actions" not in markdown
    assert "West region sales reached 42,000" in markdown


def test_chinese_final_conclusion_without_profit_discount_keeps_field_supplement_action() -> None:
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["销售额已经完成基础趋势复盘。"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="销售趋势分析",
                chart_type="line",
                summary_metrics={"total_sales_amount": 860000.0},
            )
        ],
    )

    markdown = build_final_conclusion_markdown(
        report,
        _schema("sales_amount", "order_datetime", "sku"),
        dataset_profile={"row_count": 1200},
        output_language="zh-CN",
    )

    assert "### 行动建议" in markdown
    assert "补充" in markdown
    assert "字段" in markdown or "口径" in markdown
