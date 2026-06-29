from __future__ import annotations

import pandas as pd
import nbformat

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport
from app.schemas.report import ModuleReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.notebook_builder import build_notebook
from app.services.notebook.modeling_renderer import render_modeling_section_cells


def _loss_module(*, quality: str = "strong") -> ModuleReport:
    return ModuleReport(
        module_id="loss_risk_modeling",
        title="亏损风险建模",
        chart_type="modeling",
        summary_metrics={
            "model_quality_status": quality,
            "best_model": "LogisticRegression",
            "best_precision": 0.7136 if quality != "weak" else 0.0526,
            "best_recall": 0.96 if quality != "weak" else 0.6667,
            "best_f1": 0.8187 if quality != "weak" else 0.0976,
            "best_roc_auc": 0.9867,
            "target_positive_count": 1901 if quality != "weak" else 11,
            "target_positive_rate": 0.186482 if quality != "weak" else 0.00055,
            "test_positive_count": 475 if quality != "weak" else 3,
            "false_positive_count": 183 if quality != "weak" else 36,
            "false_negative_count": 19 if quality != "weak" else 1,
            "true_positive_count": 456 if quality != "weak" else 2,
            "true_negative_count": 1891 if quality != "weak" else 3961,
            "weak_reasons": ["too_few_test_positives", "low_precision"] if quality == "weak" else [],
        },
        tables={
            "model_comparison": [
                {"model": "DummyClassifier", "precision": 0.2, "recall": 1.0, "f1": 0.33},
                {"model": "LogisticRegression", "precision": 0.7136, "recall": 0.96, "f1": 0.8187},
            ],
            "threshold_analysis": [
                {
                    "threshold": 0.5,
                    "precision": 0.7136,
                    "recall": 0.96,
                    "f1": 0.8187,
                    "predicted_loss_count": 639,
                    "review_load_rate": 0.2507,
                }
            ],
            "confusion_matrix": [
                {"actual": "non_loss", "predicted": "non_loss", "count": 1891},
                {"actual": "non_loss", "predicted": "loss", "count": 183},
                {"actual": "loss", "predicted": "non_loss", "count": 19},
                {"actual": "loss", "predicted": "loss", "count": 456},
            ],
            "feature_importance_grouped": [
                {
                    "feature_group": "sub_category",
                    "total_importance": 25.7562,
                    "top_features": "sub_category_Storage, sub_category_Supplies",
                    "business_meaning": "子类目结构与亏损风险识别有关，需要结合商品结构复盘。",
                },
                {
                    "feature_group": "discount",
                    "total_importance": 5.5583,
                    "top_features": "discount",
                    "business_meaning": "折扣强度是模型识别亏损风险的重要预测信号。",
                },
            ],
            "feature_importance": [
                {"feature": "discount", "importance": 5.5583},
                {"feature": "sub_category_Storage", "importance": 4.2},
            ],
            "safe_features": [
                {"feature": "discount", "reason": "numeric feature"},
                {"feature": "sub_category", "reason": "low cardinality category"},
            ],
            "blocked_features": [
                {"feature": "profit", "reason": "target leakage"},
                {"feature": "order_id", "reason": "identifier"},
            ],
            "high_risk_examples": [
                {
                    "row_index": 10,
                    "loss_probability": 0.91,
                    "actual_is_loss": True,
                    "predicted_is_loss": True,
                    "discount": 0.7,
                    "sales_amount": 1200.0,
                    "category": "Furniture",
                    "sub_category": "Tables",
                    "segment": "Consumer",
                    "region": "West",
                }
            ],
            "false_negative_examples": [
                {
                    "row_index": 11,
                    "loss_probability": 0.42,
                    "actual_is_loss": True,
                    "predicted_is_loss": False,
                    "discount": 0.2,
                    "sales_amount": 300.0,
                    "category": "Office Supplies",
                    "sub_category": "Storage",
                    "segment": "Corporate",
                    "region": "East",
                }
            ],
        },
    )


def _forecast_module(*, mape: float = 11.66) -> ModuleReport:
    return ModuleReport(
        module_id="forecast_analysis",
        title="销售额预测",
        chart_type="line",
        summary_metrics={
            "forecast_status": "baseline_evaluated",
            "best_baseline": "naive_last_value",
            "best_baseline_mae": 253982.68,
            "best_baseline_rmse": 301514.77,
            "best_baseline_mape": mape,
            "time_split": "chronological",
            "series_granularity": "weekly",
            "train_period_count": 115,
            "backtest_period_count": 28,
        },
        tables={
            "baseline_metrics": [
                {"baseline": "naive_last_value", "mae": 253982.68, "rmse": 301514.77, "mape": mape},
                {"baseline": "moving_average_7", "mae": 280000.0, "rmse": 320000.0, "mape": mape + 3},
            ],
            "baseline_backtest": [
                {"period": "2012-01-06", "actual": 1600000.0, "forecast": 1580000.0},
                {"period": "2012-01-13", "actual": 1500000.0, "forecast": 1600000.0},
            ],
        },
    )


def _sales_regression_module(*, status: str = "usable", best_model: str = "Ridge Regression") -> ModuleReport:
    weak = status == "weak"
    return ModuleReport(
        module_id="forecast_analysis",
        title="销售额预测",
        chart_type="line",
        summary_metrics={
            "forecast_status": "regression_evaluated",
            "regression_status": status,
            "time_split": "chronological",
            "regression_time_split": "chronological",
            "series_granularity": "weekly",
            "day_span": 994,
            "observed_period_count": 143,
            "train_period_count": 115,
            "backtest_period_count": 28,
            "best_model": best_model,
            "best_model_mae": 253982.68 if weak else 180000.0,
            "best_model_rmse": 301514.77 if weak else 220000.0,
            "best_model_mape": 120.0 if weak else 8.9,
            "best_model_r2": -0.15 if weak else 0.42,
            "baseline_model": "naive_last_value",
            "baseline_mae": 253982.68,
            "improvement_vs_baseline": 0.0 if weak else 29.13,
            "regression_cv_strategy": "TimeSeriesSplit_3",
            "cv_mae_mean": 255000.0 if weak else 185000.0,
            "cv_mae_std": 42000.0 if weak else 12000.0,
            "cv_mape_mean": 125.0 if weak else 9.7,
            "cv_mape_std": 38.0 if weak else 1.2,
            "cv_r2_mean": -0.1 if weak else 0.31,
            "cv_r2_std": 0.2 if weak else 0.05,
            "selection_basis": {
                "selected_model": best_model,
                "holdout_mae_rank": 1,
                "cv_mae_rank": 3 if weak else 1,
                "cv_best_model": "moving_average_7" if weak else best_model,
                "cv_stability_note": "holdout_best_cv_not_stable" if weak else "holdout_and_cv_aligned",
                "selected_reason": "Holdout MAE and CV stability are considered together.",
            },
        },
        tables={
            "regression_cv_folds": [
                {
                    "fold": 1,
                    "train_start": "2010-01-01",
                    "train_end": "2010-08-01",
                    "valid_start": "2010-08-08",
                    "valid_end": "2010-10-01",
                    "model": "naive_last_value",
                    "mae": 260000.0,
                    "rmse": 310000.0,
                    "mape": 12.0,
                    "r2": -0.1,
                },
                {
                    "fold": 1,
                    "train_start": "2010-01-01",
                    "train_end": "2010-08-01",
                    "valid_start": "2010-08-08",
                    "valid_end": "2010-10-01",
                    "model": "Ridge Regression",
                    "mae": 190000.0 if not weak else 270000.0,
                    "rmse": 240000.0 if not weak else 330000.0,
                    "mape": 9.5 if not weak else 140.0,
                    "r2": 0.35 if not weak else -0.2,
                },
            ],
            "regression_cv_summary": [
                {
                    "model": "naive_last_value",
                    "cv_mae_mean": 255000.0,
                    "cv_mae_std": 42000.0,
                    "cv_mape_mean": 12.4,
                    "cv_mape_std": 2.0,
                    "cv_r2_mean": -0.05,
                    "cv_r2_std": 0.1,
                },
                {
                    "model": "Ridge Regression",
                    "cv_mae_mean": 185000.0 if not weak else 270000.0,
                    "cv_mae_std": 12000.0 if not weak else 46000.0,
                    "cv_mape_mean": 9.7 if not weak else 138.0,
                    "cv_mape_std": 1.2 if not weak else 42.0,
                    "cv_r2_mean": 0.31 if not weak else -0.18,
                    "cv_r2_std": 0.05 if not weak else 0.25,
                },
            ],
            "model_comparison": [
                {
                    "model": "naive_last_value",
                    "model_type": "baseline",
                    "mae": 253982.68,
                    "rmse": 301514.77,
                    "mape": 120.0 if weak else 11.66,
                    "r2": None,
                    "improvement_vs_baseline": 0.0,
                },
                {
                    "model": "Ridge Regression",
                    "model_type": "regression",
                    "mae": 260000.0 if weak else 180000.0,
                    "rmse": 320000.0 if weak else 220000.0,
                    "mape": 130.0 if weak else 8.9,
                    "r2": -0.15 if weak else 0.42,
                    "improvement_vs_baseline": -2.37 if weak else 29.13,
                },
            ],
            "regression_backtest": [
                {"period": "2012-01-06", "actual": 1600000.0, "predicted": 1580000.0, "model": best_model},
                {"period": "2012-01-13", "actual": 1500000.0, "predicted": 1520000.0, "model": best_model},
            ],
            "feature_importance": [
                {"feature": "lag_1", "importance": 0.52},
                {"feature": "rolling_mean_4", "importance": 0.31},
            ],
            "worst_error_periods": [
                {
                    "period": "2012-01-13",
                    "actual": 1500000.0,
                    "predicted": 1520000.0,
                    "absolute_error": 20000.0,
                    "percentage_error": 1.33,
                }
            ],
        },
    )


def _outcome(*, task: str, status: str, value: str = "medium") -> dict[str, object]:
    return {
        "primary_modeling_task": task,
        "modeling_status": status,
        "modeling_value_level": value,
        "metrics_summary": {},
        "main_findings": [],
        "limitations": [],
        "recommended_action": "",
    }


def _combined(cells: list) -> tuple[str, str]:
    markdown = "\n\n".join(str(cell.source) for cell in cells if cell.cell_type == "markdown")
    code = "\n\n".join(str(cell.source) for cell in cells if cell.cell_type == "code")
    return markdown, code


def test_strong_loss_risk_keeps_charts_but_removes_confusion_matrix_table_and_business_meaning() -> None:
    cells = render_modeling_section_cells(
        _loss_module(quality="strong"),
        modeling_outcome=_outcome(
            task="loss_risk_classification",
            status="strong_model",
            value="high",
        ),
    )

    markdown, code = _combined(cells)

    assert "## Modeling Analysis: Loss Risk Review" in markdown
    assert "| model | accuracy | precision | recall | f1 | roc_auc |" not in markdown
    assert "| threshold | precision | recall | f1 | predicted_loss_count | review_load_rate |" not in markdown
    assert "| actual | predicted | count |" not in markdown
    assert "TP=" in markdown
    assert "FP=" in markdown
    assert "FN=" in markdown
    assert "business_meaning" not in markdown
    assert "| feature_group | total_importance | top_features |" not in markdown
    assert "MODEL_CONFIG" in code
    assert "train_test_split" in code
    assert "LogisticRegression" in code
    assert "RandomForestClassifier" in code
    assert "display(model_comparison)" in code
    assert "display(threshold_analysis)" in code
    assert "confusion_matrix" in code
    assert "feature_importance_grouped" in code


def test_zh_cn_modeling_display_rejects_english_interpretation_paragraph() -> None:
    cells = render_modeling_section_cells(
        _loss_module(quality="strong"),
        modeling_outcome=_outcome(
            task="loss_risk_classification",
            status="strong_model",
            value="high",
        ),
        modeling_interpretation={
            "threshold_takeaway": "At the default threshold, TP=456, FP=183, FN=19, and TN=1891. The model can support human review prioritization, but it must not be used for automatic decisions.",
            "error_takeaway": "At the default threshold, TP=456, FP=183, FN=19, and TN=1891.",
            "business_use_warning": "The model can support human review prioritization, but it must not be used for automatic decisions.",
        },
        output_language="zh-CN",
    )

    markdown, _code = _combined(cells)

    assert "At the default threshold" not in markdown
    assert "默认阈值" in markdown
    assert "TP=" in markdown
    assert "FP=" in markdown
    assert "FN=" in markdown
    assert "不用于自动决策" in markdown


def test_strong_loss_risk_interleaves_four_display_blocks_with_chart_explanations() -> None:
    cells = render_modeling_section_cells(
        _loss_module(quality="strong"),
        modeling_outcome=_outcome(
            task="loss_risk_classification",
            status="strong_model",
            value="high",
        ),
        modeling_interpretation={
            "modeling_summary": "LLM 建模小结：亏损样本占比 18.65%，LogisticRegression recall=0.96、F1=0.8187，可用来辅助人工复核，不代表因果关系，不用于自动决策。",
            "model_choice_takeaway": "当前选择的LogisticRegression模型召回率达到96%，F1分数0.8187，适合优先保障高召回率。",
            "threshold_takeaway": "阈值从0.5降至0.4时，召回率从96%进一步提升至96.42%，复核负载率从25.07%上升至26.05%。",
            "feature_takeaway": "LLM 特征解释：discount 与 sub_category 是当前主要预测信号，需要回到订单明细复盘。",
            "review_guidance": "LLM 复核建议：优先人工复核模型预测分数最高的一批订单（例如排名前25%）。",
        },
        modeling_outcome_interpretation={
            "outcome_summary": "LLM outcome：strong loss-risk 模型有明确复核价值。",
            "business_interpretation": "LLM outcome 业务解释：它把亏损风险转成可排序的人工复核线索。",
            "recommended_action": "LLM outcome 动作：结合阈值、FN/FP 和订单明细制定复核批次。",
            "risk_warning": "模型输出仅反映数据关联，不代表因果关系，不用于自动决策。",
        },
    )

    cell_sources = [str(cell.source) for cell in cells]
    markdown, code = _combined(cells)

    assert "## Modeling Analysis: Loss Risk Review" in cell_sources[0]
    assert "Feature engineering and leakage guard" in cell_sources[0]
    assert "Main fields used" in cell_sources[0]
    assert "Excluded fields" in cell_sources[0]
    assert "safe_features" not in cell_sources[0]
    assert "blocked_features" not in cell_sources[0]
    assert "| feature | feature_type | reason |" not in cell_sources[0]
    assert "retrain" not in cell_sources[0].lower()
    assert "rerun" not in cell_sources[0].lower()
    assert "### Modeling Summary" in markdown

    training_idx = next(i for i, cell in enumerate(cells) if cell.cell_type == "code" and "MODEL_CONFIG" in str(cell.source))
    threshold_idx = next(
        i for i, cell in enumerate(cells) if cell.cell_type == "code" and "display(model_comparison)" in str(cell.source)
    )
    confusion_idx = next(i for i, cell in enumerate(cells) if cell.cell_type == "code" and "confusion_matrix_df" in str(cell.source))
    feature_idx = next(
        i
        for i, cell in enumerate(cells)
        if cell.cell_type == "code" and "feature_importance_grouped" in str(cell.source) and "barh" in str(cell.source)
    )

    assert training_idx < threshold_idx < confusion_idx < feature_idx
    assert "display(loss_risk_model_summary)" not in cell_sources[training_idx]
    assert "display(pd.DataFrame({'used_features': feature_cols[:20]}))" in cell_sources[training_idx]
    assert "display(pd.DataFrame({'blocked_fields': blocked_cols[:20]}))" in cell_sources[training_idx]
    assert "LOSS_RISK_FEATURE_PLAN" in cell_sources[training_idx]
    assert "The code below trains DummyClassifier, LogisticRegression, and RandomForestClassifier" in cell_sources[threshold_idx - 1]
    assert "review volume across thresholds" in cell_sources[threshold_idx - 1]
    assert "display(loss_risk_model_summary)" in cell_sources[threshold_idx]
    assert "LogisticRegression" in cell_sources[threshold_idx + 1]
    assert "recall=0.96" in cell_sources[threshold_idx + 1]
    assert "F1=0.8187" in cell_sources[threshold_idx + 1]
    assert "The confusion matrix at the default threshold" in cell_sources[confusion_idx - 1]
    assert "correctly identified" not in cell_sources[confusion_idx - 1]
    assert "FN represents missed loss-making orders" not in cell_sources[confusion_idx]
    assert "FN" in cell_sources[confusion_idx + 1]
    assert "FP" in cell_sources[confusion_idx + 1]
    assert "high-risk examples" in cell_sources[confusion_idx + 1]
    assert "false-negative loss examples" in cell_sources[confusion_idx + 1]
    assert "top 25%" not in cell_sources[confusion_idx + 1]
    assert "default threshold" in cell_sources[confusion_idx + 1]
    assert "25.07%" in cell_sources[confusion_idx + 1]
    assert "### Confusion Matrix Explanation" not in markdown
    assert "### Risk Signal Interpretation" in cell_sources[feature_idx - 1]
    assert "These fields are predictive signals only" not in cell_sources[feature_idx]
    assert "business_meaning" not in cell_sources[feature_idx]
    assert "Top raw features" in cell_sources[feature_idx + 1]
    assert "### Risk Signal Interpretation" not in cell_sources[feature_idx + 1]
    assert "### Modeling Summary" in cell_sources[feature_idx + 1]
    assert "recall=0.96" in cell_sources[feature_idx + 1]
    assert "F1=0.8187" in cell_sources[feature_idx + 1]
    assert "LLM outcome" not in cell_sources[feature_idx + 1]
    assert "modeling status" not in cell_sources[feature_idx + 1].lower()
    assert "modeling value" not in cell_sources[feature_idx + 1].lower()
    assert "strong" not in cell_sources[feature_idx + 1]
    assert "default threshold" in cell_sources[feature_idx + 1]
    assert "FP=" in cell_sources[feature_idx + 1]
    assert "FN=" in cell_sources[feature_idx + 1]
    assert "threshold trade-off" in cell_sources[feature_idx + 1]
    conclusion_text = cell_sources[feature_idx + 1].split("### Modeling Summary", maxsplit=1)[1]
    assert "main signals are concentrated in subcategory, discount" in conclusion_text
    assert "more likely" not in conclusion_text
    assert "top 25%" not in markdown
    assert cell_sources[feature_idx + 1].count("does not imply causality") == 1
    assert "risk probability estimate" not in cell_sources[feature_idx + 1]
    assert "automatic decisions" in cell_sources[feature_idx + 1]
    assert "| actual | predicted | count |" not in markdown
    assert "business_meaning" not in markdown
    assert "confusion_matrix" in code
    assert "feature_importance_grouped" in code


def test_loss_risk_code_first_cells_execute_on_small_dataframe() -> None:
    cells = render_modeling_section_cells(
        _loss_module(quality="strong"),
        modeling_outcome=_outcome(
            task="loss_risk_classification",
            status="strong_model",
            value="high",
        ),
    )
    df = pd.DataFrame(
        {
            "Profit": [-20, -15, -10, -8, -5, -3, 4, 5, 8, 10, 14, 18] * 8,
            "Discount": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.2] * 8,
            "Sales": [120, 100, 80, 70, 60, 50, 40, 55, 65, 75, 85, 95] * 8,
            "Quantity": [2, 1, 3, 2, 4, 1, 2, 2, 1, 3, 2, 1] * 8,
            "Row ID": list(range(96)),
            "Category": ["Furniture", "Office Supplies", "Technology", "Furniture"] * 24,
            "Sub-Category": ["Tables", "Storage", "Phones", "Chairs"] * 24,
            "Segment": ["Consumer", "Corporate", "Home Office"] * 32,
            "Region": ["West", "East", "Central", "South"] * 24,
            "Order ID": [f"order-{idx}" for idx in range(96)],
            "Order Date": pd.date_range("2024-01-01", periods=96, freq="D"),
        }
    )
    namespace = {"df": df, "display": lambda *args, **kwargs: None}

    training_cell = next(cell for cell in cells if cell.cell_type == "code" and "MODEL_CONFIG" in str(cell.source))
    exec(str(training_cell.source), namespace)

    assert set(namespace["model_comparison"]["model"]) == {
        "DummyClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    }
    assert not namespace["threshold_analysis"].empty
    assert "Row ID" not in namespace["feature_cols"]
    assert "Row ID" in namespace["blocked_cols"]


def test_weak_loss_risk_uses_compressed_exploratory_display() -> None:
    cells = render_modeling_section_cells(
        _loss_module(quality="weak"),
        modeling_outcome=_outcome(
            task="loss_risk_classification",
            status="weak_model",
            value="low",
        ),
        modeling_interpretation={
            "feature_takeaway": "当前较突出的预测信号组是 country, sub_category, sales_amount，需结合业务明细复盘。该字段仅为预测信号，不代表因果关系。",
        },
    )

    markdown, code = _combined(cells)

    assert "## Modeling Analysis: Loss Risk Exploration" in markdown
    assert "exploratory risk evidence" in markdown
    assert "not as a review-ranking or automatic-decision model" in markdown
    assert "### Analyst Explanation" not in markdown
    assert markdown.count("review-ranking") <= 1
    assert "### Modeling Task and Validation Design" not in markdown
    assert "safe_features" not in markdown
    assert "blocked_features" not in markdown
    assert "MODEL_CONFIG" in code
    assert "train_test_split" in code
    assert "LogisticRegression" in code
    assert "RandomForestClassifier" in code
    assert "display(model_comparison)" in code
    assert "display(loss_risk_model_summary)" in code
    assert "display(threshold_analysis)" not in code
    assert "confusion_matrix" not in code
    assert "feature_importance_grouped" in code
    assert "| model_quality_status | best_model | best_precision |" not in markdown
    assert "| feature_group | total_importance | top_features |" not in markdown
    assert "few positive examples" in markdown
    assert "candidate signals" in markdown
    assert "profit definition" in markdown
    assert "predictive signals only" in markdown or "candidate signals" in markdown
    assert "They do not explain why losses happen" in markdown


def test_forecast_baseline_gets_formal_modeling_display() -> None:
    cells = render_modeling_section_cells(
        None,
        forecast_module=_forecast_module(mape=11.66),
        modeling_outcome=_outcome(
            task="sales_amount_forecast_baseline",
            status="baseline_evaluated",
            value="medium",
        ),
        modeling_opportunity_decision={
            "notebook_message": "后续可评估销售额预测或回归，但本轮仅记录机会，不训练模型。",
        },
        modeling_outcome_interpretation={
            "outcome_summary": "当前任务为销售额预测基线建模，状态为基线已评估，价值水平为中等。这是一个对照基线，不建议作为生产级预测使用。",
            "business_interpretation": "简单历史均值的基线模型 MAPE 11.66% 可用于判断后续模型是否真正优于简单 baseline。",
            "recommended_action": "建议将此基线作为后续预测模型的对照基准。",
        },
    )

    cell_sources = [str(cell.source) for cell in cells]
    markdown, code = _combined(cells)
    forecast_idx = next(i for i, source in enumerate(cell_sources) if "baseline_backtest = pd.DataFrame" in source)

    assert "## Modeling Analysis:" in markdown and "baseline" in markdown
    assert "MAE" in markdown and "RMSE" in markdown and "MAPE" in markdown
    assert "chronological" in markdown
    assert "automatic decisions" in markdown
    assert "仅记录机会" not in markdown
    assert "不训练模型" not in markdown
    assert "### 回测设计与指标" in cell_sources[forecast_idx - 1]
    assert "MAE" in cell_sources[forecast_idx - 1]
    assert "### 结果解释" not in cell_sources[forecast_idx - 1]
    assert "### 结果解释" in cell_sources[forecast_idx + 1]
    assert "### Usage Limits" in cell_sources[forecast_idx + 1]
    assert "状态为" not in cell_sources[forecast_idx + 1]
    assert "价值水平" not in cell_sources[forecast_idx + 1]
    assert "基线已评估" not in cell_sources[forecast_idx + 1]
    assert "MAPE 11.66%" in cell_sources[forecast_idx + 1]
    assert "简单历史均值" not in cell_sources[forecast_idx + 1]
    assert "naive_last_value" in cell_sources[forecast_idx + 1]
    assert "baseline_backtest" in code
    assert "Actual vs Baseline" in code


def test_high_mape_forecast_is_downgraded_in_display() -> None:
    cells = render_modeling_section_cells(
        None,
        forecast_module=_forecast_module(mape=1053.75),
        modeling_outcome=_outcome(
            task="sales_amount_forecast_baseline",
            status="weak_baseline",
            value="low",
        ),
        modeling_outcome_interpretation={
            "outcome_summary": "当前任务为销售额预测基线评估，状态为弱基线，建模价值级别低。MAPE高达1053.75%。",
            "business_interpretation": "预测误差很大，只能提供粗略参考范围。",
            "recommended_action": "暂不建议基于此结果进行任何预测包装。",
        },
    )

    cell_sources = [str(cell.source) for cell in cells]
    markdown, code = _combined(cells)
    forecast_idx = next(i for i, source in enumerate(cell_sources) if "baseline_backtest = pd.DataFrame" in source)

    assert "## Modeling Analysis:" in markdown and "baseline" in markdown
    assert "MAPE" in markdown
    assert "forecast error is too large" in markdown or "MAPE 偏高" in markdown
    assert "monitoring" in markdown or "监控参照" in markdown
    assert "稳定预测能力" not in markdown
    assert "MAPE" in cell_sources[forecast_idx + 1]
    assert "状态为弱基线" not in cell_sources[forecast_idx + 1]
    assert "建模价值级别" not in cell_sources[forecast_idx + 1]
    assert "forecast error is too large" in cell_sources[forecast_idx + 1] or "预测误差很大" in cell_sources[forecast_idx + 1]
    assert "预测包装" not in cell_sources[forecast_idx + 1]
    assert "业务决策" in cell_sources[forecast_idx + 1]
    assert "baseline_backtest" in code


def test_sales_regression_usable_gets_compact_full_display() -> None:
    cells = render_modeling_section_cells(
        None,
        forecast_module=_sales_regression_module(status="usable", best_model="Ridge Regression"),
        modeling_outcome=_outcome(
            task="sales_amount_regression",
            status="regression_usable",
            value="medium",
        ),
        modeling_outcome_interpretation={
            "outcome_summary": "Ridge Regression 明显优于 naive baseline，可作为监控对照模型。",
            "business_interpretation": "该模型可帮助判断周销售额偏离，但不是生产级预测。",
            "recommended_action": "建议作为销售监控和后续模型增强的对照。",
            "model_selection_takeaway": (
                "Ridge Regression 的 MAE 明显低于 naive baseline，因此作为当前 best model。"
                "但仍要结合 R2 和 CV 波动判断业务可用性。"
            ),
            "cv_stability_takeaway": (
                "TimeSeriesSplit 显示 Ridge Regression 在多个时间窗口 MAE 更低且波动较小。"
                "这说明优势不是只来自最后一段回测，但仍只能作为监控对照。"
            ),
            "prediction_fit_takeaway": (
                "Actual vs Best Model 用来观察趋势是否跟得上。"
                "峰值和低谷仍需人工复核，避免把监控模型当成生产预测或补货依据。"
            ),
            "feature_importance_takeaway": (
                "lag_1 和 rolling_mean_4 说明模型主要依赖历史销售惯性和短期移动均值。"
                "这些信号不能解释促销或门店变化本身。"
            ),
            "error_analysis_takeaway": (
                "最大误差周期提示模型可能漏掉促销、节假日或异常订单。"
                "这类误差应作为补充业务变量的优先线索，而不是直接作为销售计划依据或补货依据。"
            ),
            "final_regression_synthesis": (
                "综合来看，模型有一定监控参考价值。"
                "它相比 baseline 有改善，但预测误差较大，不是生产级预测。"
                "后续应优先引入促销日历、节假日、门店活动等外部变量，并尝试更稳定的模型架构以提升预测价值。"
                "当前结论更适合作为销售监控和后续模型对照，而不是直接驱动经营计划。"
            ),
        },
    )

    cell_sources = [str(cell.source) for cell in cells]
    markdown, code = _combined(cells)
    feature_engineering_idx = next(i for i, source in enumerate(cell_sources) if "feature_df = sales_series.copy()" in source)
    split_idx = next(i for i, source in enumerate(cell_sources) if "train_df = feature_df.iloc[:-backtest_periods]" in source)
    cv_idx = next(i for i, source in enumerate(cell_sources) if "TimeSeriesSplit(n_splits=3)" in source)
    comparison_idx = next(i for i, source in enumerate(cell_sources) if "model_comparison = pd.DataFrame" in source)
    backtest_idx = next(i for i, source in enumerate(cell_sources) if "backtest_df = pd.DataFrame" in source)
    feature_idx = next(i for i, source in enumerate(cell_sources) if "feature_importance_df = " in source)
    error_idx = next(i for i, source in enumerate(cell_sources) if "error_top5 = " in source)

    assert "## Modeling Analysis: Sales Regression" in markdown
    assert "### Modeling Goal and Data Conditions" in markdown
    assert "### Feature Engineering" in markdown
    assert "### Time Split" in markdown
    assert "calendar" in markdown and "lag" in markdown and "rolling" in markdown
    assert "shift" in markdown
    assert "### Model Comparison and Selection" in markdown
    assert "### Time Series Cross-validation" in markdown
    assert "TimeSeriesSplit shows Ridge Regression" in markdown
    assert "cv_mape_mean" in code
    assert "cv_r2_std" in code
    assert "### Model Selection Explanation" not in markdown
    assert "Ridge Regression" in markdown and "naive_last_value" in markdown
    assert "CV is also part of the selection" in markdown
    assert "CV MAE rank is 1" in markdown
    assert "### Sales Regression Backtest" in markdown
    assert "### Prediction Fit Explanation" not in markdown
    assert "peaks or troughs" in markdown
    assert "### Feature Importance" in markdown
    assert "| feature | importance |" not in markdown
    assert "lag_1, rolling_mean_4" in markdown
    assert "historical sales inertia" in markdown
    assert "### Error Analysis" in markdown
    assert "### Error Analysis Explanation" not in markdown
    assert "promotions, holidays" in markdown
    assert "### Overall Summary" in markdown
    assert "Overall" in markdown
    assert "production-grade forecast" in markdown
    assert "### Analyst Explanation" not in markdown
    assert "### 1." not in markdown
    assert "### 2." not in markdown
    assert "### 3." not in markdown
    assert "Ridge Regression" in markdown
    assert "improvement_vs_baseline" in code
    assert "lag_1" in markdown
    assert "largest-error period" in markdown
    assert "feature_df = sales_series.copy()" in code
    assert "TimeSeriesSplit(n_splits=3)" in code
    assert "model_comparison = pd.DataFrame" in code
    assert "backtest_df = pd.DataFrame" in code
    assert "Actual vs Best Model" in code
    assert "feature_importance_df" in code
    assert "error_top5" in code
    assert feature_engineering_idx < split_idx < cv_idx < comparison_idx < backtest_idx < feature_idx < error_idx
    assert markdown.index("### Time Series Cross-validation") < markdown.index("### Model Comparison and Selection")
    assert "### Sales Regression Backtest" in cell_sources[backtest_idx - 1]
    assert "### Feature Importance" in cell_sources[feature_idx - 1]
    assert "### Error Analysis" in cell_sources[error_idx - 1]


def test_sales_regression_weak_uses_compressed_display_without_feature_chart() -> None:
    cells = render_modeling_section_cells(
        None,
        forecast_module=_sales_regression_module(status="weak", best_model="naive_last_value"),
        modeling_outcome=_outcome(
            task="sales_amount_regression",
            status="regression_weak",
            value="low",
        ),
        modeling_outcome_interpretation={
            "outcome_summary": "当前销售额回归模型没有形成稳定预测能力。",
            "business_interpretation": "预测难度高，更适合作为监控参照。",
            "recommended_action": "建议补充促销、节假日或聚合到周/月级别后再建模。",
            "model_selection_takeaway": (
                "best model 仍接近 baseline，说明复杂模型没有带来稳定增益。"
                "这类结果更适合说明预测难度，而不是作为经营计划模型。"
            ),
            "cv_stability_takeaway": (
                "TimeSeriesSplit 显示各折误差波动较大，复杂模型优势不稳定。"
                "因此不能只看最后 holdout 的表现来判断模型可用。"
            ),
            "prediction_fit_takeaway": (
                "Actual vs Best Model 显示模型难以跟上突发峰值或低谷。"
                "这些偏离提示当前历史特征不足，需要补充业务事件变量。"
            ),
            "error_analysis_takeaway": (
                "Top 误差集中在销售尖峰，可能缺少促销、节假日、门店或类目变量。"
                "这些周期应作为后续补变量和异常核查的重点，而不是直接解释为稳定预测失败。"
            ),
            "final_regression_synthesis": (
                "当前结果更适合作为预测难度证据。"
                "复杂模型没有形成稳定优势，不建议用于销售计划、库存、补货或经营目标制定。"
                "后续应补充业务变量或调整时间粒度后再评估。"
            ),
        },
    )

    markdown, code = _combined(cells)

    assert "## Modeling Analysis: Sales Forecast Attempt" in markdown
    assert "### Modeling Goal and Data Conditions" in markdown
    assert "### Feature Engineering" in markdown
    assert "### Time Split" in markdown
    assert "calendar" in markdown and "lag" in markdown and "rolling" in markdown
    assert "shift" in markdown
    assert "has not produced stable forecasting capability" in markdown
    assert "Model Comparison" in markdown
    assert "### Time Series Cross-validation" in markdown
    assert "cross-window instability" in markdown
    assert "### Model Selection Explanation" not in markdown
    assert "do not consistently outperform the simple baseline" in markdown
    assert "CV is also part of the selection" in markdown
    assert "conclusion should be downgraded" in markdown
    assert "### Sales Regression Backtest" in markdown
    assert "### Prediction Fit Explanation" not in markdown
    assert "sudden peaks or dips" in markdown
    assert "Error Analysis" in markdown
    assert "### Error Analysis Explanation" not in markdown
    assert "promotions, holidays, replenishment" in markdown
    assert "### Overall Summary" in markdown
    assert "not a production-grade forecast" in markdown
    assert "### Why This Should Not Be Used Directly" not in markdown
    assert "### Next Steps" not in markdown
    assert "### Analyst Explanation" not in markdown
    assert "### 1." not in markdown
    assert "Feature Importance" not in markdown
    assert "feature_importance" not in code
    assert "feature_df = sales_series.copy()" in code
    assert "TimeSeriesSplit(n_splits=3)" in code
    assert "model_comparison = pd.DataFrame" in code
    assert "backtest_df = pd.DataFrame" in code
    assert "error_top5" in code
    assert "Actual vs Best Model" in code
    assert "should not be used for sales planning" in markdown


def test_build_notebook_uses_modeling_outcome_for_forecast_modeling_section(tmp_path) -> None:
    report = AnalysisReport(
        task_id="task-modeling-display",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[_forecast_module(mape=11.66)],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Date": "order_datetime", "Weekly_Sales": "sales_amount"},
        confidence=1.0,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    outline = NotebookOutline(
        title="Modeling Display Test",
        sections=[
            NotebookSection(
                section_id="modeling",
                title="建模分析",
                purpose="展示建模分析。",
            )
        ],
    )

    notebook_path = build_notebook(
        task_id="task-modeling-display",
        output_dir=tmp_path,
        report=report,
        schema_mapping=schema_mapping,
        plan=AnalysisPlan(analysis_plan=["forecast_analysis"], chart_preferences={"forecast_analysis": "line"}),
        outline=outline,
        content_plan=NotebookContentPlan(sections=[]),
        modeling_outcome=_outcome(
            task="sales_amount_forecast_baseline",
            status="baseline_evaluated",
            value="medium",
        ),
        modeling_opportunity_decision={
            "notebook_message": "后续可评估销售额预测或回归，但本轮仅记录机会，不训练模型。",
        },
    )

    nb = nbformat.read(notebook_path, as_version=4)
    combined_markdown = "\n\n".join(str(cell.source) for cell in nb.cells if cell.cell_type == "markdown")

    assert "## Modeling Analysis:" in combined_markdown
    assert "MAE" in combined_markdown
    assert "仅记录机会" not in combined_markdown
    assert "不训练模型" not in combined_markdown
