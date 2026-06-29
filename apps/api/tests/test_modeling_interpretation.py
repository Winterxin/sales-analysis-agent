from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.schemas.report import AnalysisReport, ModuleReport
from app.services.llm_client import LLMClient
from app.services.modeling_interpretation import (
    _build_modeling_payload,
    _ensure_modeling_summary_limits,
    _sanitize_modeling_interpretation,
    build_modeling_interpretation_with_trace,
)
from app.services.notebook.modeling_renderer import (
    _modeling_plot_setup_code,
    render_modeling_section_cells,
    render_modeling_section_markdown,
)


INTERPRETATION_KEYS = {
    "modeling_summary",
    "model_choice_takeaway",
    "threshold_takeaway",
    "feature_takeaway",
    "error_takeaway",
    "review_guidance",
    "business_use_warning",
}


def _modeling_module(*, skipped: bool = False) -> ModuleReport:
    if skipped:
        return ModuleReport(
            module_id="loss_risk_modeling",
            title="亏损风险建模",
            chart_type="modeling",
            warnings=["modeling_plan_skipped: single_target_class"],
        )
    return ModuleReport(
        module_id="loss_risk_modeling",
        title="亏损风险建模",
        chart_type="modeling",
        summary_metrics={
            "target_positive_rate": 0.25,
            "target_positive_count": 25,
            "target_negative_count": 75,
            "train_rows": 80,
            "test_rows": 20,
            "train_positive_count": 20,
            "train_negative_count": 60,
            "test_positive_count": 5,
            "test_negative_count": 15,
            "class_imbalance_level": "moderate",
            "model_quality_status": "usable",
            "weak_reasons": [],
            "best_model": "LogisticRegression",
            "best_recall": 0.9,
            "best_f1": 0.78,
            "best_roc_auc": 0.88,
            "business_priority_metric": "recall",
            "threshold_default": 0.5,
            "false_positive_count": 3,
            "false_negative_count": 1,
            "true_positive_count": 9,
            "true_negative_count": 7,
        },
        tables={
            "model_comparison": [
                {
                    "model": "DummyClassifier",
                    "accuracy": 0.5,
                    "precision": 0.25,
                    "recall": 1.0,
                    "f1": 0.4,
                    "roc_auc": 0.5,
                    "is_baseline": True,
                    "lift_over_baseline_f1": 0.0,
                    "lift_over_baseline_recall": 0.0,
                },
                {
                    "model": "LogisticRegression",
                    "accuracy": 0.8,
                    "precision": 0.75,
                    "recall": 0.9,
                    "f1": 0.78,
                    "roc_auc": 0.88,
                    "is_baseline": False,
                    "lift_over_baseline_f1": 0.38,
                    "lift_over_baseline_recall": -0.1,
                }
            ],
            "threshold_analysis": [
                {
                    "threshold": 0.5,
                    "precision": 0.75,
                    "recall": 0.9,
                    "f1": 0.78,
                    "predicted_loss_count": 12,
                    "review_load_rate": 0.6,
                }
            ],
            "confusion_matrix": [
                {"actual": "loss", "predicted": "loss", "count": 9},
                {"actual": "loss", "predicted": "non_loss", "count": 1},
                {"actual": "non_loss", "predicted": "loss", "count": 3},
                {"actual": "non_loss", "predicted": "non_loss", "count": 7},
            ],
            "feature_importance_grouped": [
                {
                    "feature_group": "discount",
                    "total_importance": 5.1,
                    "top_features": "discount",
                    "business_meaning": "折扣强度是模型识别亏损风险的重要预测信号。",
                }
            ],
            "feature_importance": [
                {"feature": "discount", "importance": 5.1},
                {"feature": "sub_category_Storage", "importance": 2.0},
            ],
            "high_risk_examples": [
                {
                    "row_index": 1,
                    "loss_probability": 0.92,
                    "actual_is_loss": True,
                    "predicted_is_loss": True,
                    "discount": 0.4,
                    "sales_amount": 100,
                    "category": "Furniture",
                    "region": "West",
                }
            ],
            "false_negative_examples": [
                {
                    "row_index": 2,
                    "loss_probability": 0.43,
                    "actual_is_loss": True,
                    "predicted_is_loss": False,
                    "discount": 0.2,
                    "sales_amount": 50,
                    "category": "Office Supplies",
                    "region": "East",
                }
            ],
            "safe_features": [
                {"feature": "discount", "feature_type": "numeric", "reason": "supported_numeric_feature"}
            ],
            "blocked_features": [
                {"feature": "profit", "feature_type": "unknown", "reason": "target_leakage"},
                {"feature": "order_id", "feature_type": "categorical", "reason": "high_cardinality_guard"},
            ],
        },
        findings=["最佳模型为 LogisticRegression，Recall=0.9，F1=0.78，ROC AUC=0.88。"],
    )


def _report_with_modeling(*, skipped: bool = False) -> AnalysisReport:
    return AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=1,
        modules=[_modeling_module(skipped=skipped)],
    )


class DisabledLLMClient:
    enabled = False
    source = "disabled"
    configured_model = None
    last_completion_metrics: dict[str, Any] = {}


class SuccessfulModelingLLMClient:
    enabled = True
    source = "fake"
    configured_model = "fake-model"
    last_completion_metrics = {
        "elapsed_ms": 12.0,
        "prompt_chars": 120,
        "response_chars": 80,
        "attempt_count": 1,
        "error_type": None,
    }

    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def suggest_modeling_interpretation(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, Any]:
        self.payload = payload
        self.language_instruction = language_instruction
        return {
            "modeling_summary": "LLM整体小结：以 is_loss = profit < 0 定义亏损风险，样本中亏损占比为 0.25，LogisticRegression 的 recall 为 0.9，F1 为 0.78，ROC AUC 为 0.88；默认阈值下 FN 为 1、FP 为 3，discount 是主要预测信号，模型仅用于人工复核排序。",
            "model_choice_takeaway": "LLM模型选择解释：LogisticRegression 的 recall 为 0.9，适合优先减少漏判。",
            "threshold_takeaway": "LLM阈值解释：0.5 阈值下 FN 为 1，调整阈值需权衡复核量。",
            "feature_takeaway": "LLM特征解释：discount 是当前主要预测信号，应回到折扣策略复盘。",
            "error_takeaway": "LLM错误解释：FN 为 1，更需要优先复盘；FP 为 3，代表额外复核成本。",
            "review_guidance": "LLM复核建议：优先查看高风险订单和 FN 样本，并结合折扣、类目和订单明细人工判断。",
            "business_use_warning": "LLM限制：只用于亏损风险预警和人工复核优先级排序，不代表因果关系，不用于自动决策。",
        }


class ErrorModelingLLMClient(SuccessfulModelingLLMClient):
    def suggest_modeling_interpretation(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, Any]:
        raise RuntimeError("boom")


class NoModelingMethodLLMClient:
    enabled = True
    source = "fake"
    configured_model = "fake-model"
    last_completion_metrics: dict[str, Any] = {}


def test_llm_client_has_suggest_modeling_interpretation_method() -> None:
    assert hasattr(LLMClient, "suggest_modeling_interpretation")


def test_build_modeling_interpretation_disabled_returns_fallback() -> None:
    interpretation, trace = build_modeling_interpretation_with_trace(
        _report_with_modeling(),
        llm_client=DisabledLLMClient(),
    )

    assert set(interpretation) == INTERPRETATION_KEYS
    assert "LogisticRegression" in interpretation["modeling_summary"]
    assert len(interpretation["modeling_summary"]) <= 230
    assert "不代表因果关系" in interpretation["feature_takeaway"]
    assert trace.status == "disabled"
    assert trace.stage == "modeling_interpretation"
    assert trace.applied is False


def test_modeling_payload_includes_loss_risk_analysis_context() -> None:
    payload = _build_modeling_payload(_modeling_module())
    context = payload["loss_risk_analysis_context"]

    assert context["best_model"] == "LogisticRegression"
    assert context["best_recall"] == 0.9
    assert context["tp"] == 9
    assert context["fp"] == 3
    assert context["fn"] == 1
    assert context["threshold_0_5"]["predicted_loss_count"] == 12
    assert context["top_raw_features"][0]["feature"] == "discount"


def test_build_modeling_interpretation_fallback_downgrades_weak_modeling() -> None:
    report = _report_with_modeling()
    module = report.modules[0]
    module.summary_metrics["model_quality_status"] = "weak"
    module.summary_metrics["weak_reasons"] = ["extreme_class_imbalance", "too_few_test_positives", "low_precision"]

    interpretation, trace = build_modeling_interpretation_with_trace(
        report,
        llm_client=DisabledLLMClient(),
    )

    assert trace.status == "disabled"
    assert "探索性参考" in interpretation["modeling_summary"]
    assert "人工复核优先级排序" not in interpretation["modeling_summary"]
    assert "不建议直接用于复核排序" in interpretation["review_guidance"]


def test_build_modeling_interpretation_no_modeling_module_is_not_applicable() -> None:
    report = AnalysisReport(
        task_id="task-no-modeling",
        dataset_type="sales_transaction",
        module_count=0,
        modules=[],
    )

    interpretation, trace = build_modeling_interpretation_with_trace(
        report,
        llm_client=SuccessfulModelingLLMClient(),
    )

    assert interpretation == {}
    assert trace.status == "not_applicable"


def test_build_modeling_interpretation_skipped_modeling_is_not_applicable() -> None:
    interpretation, trace = build_modeling_interpretation_with_trace(
        _report_with_modeling(skipped=True),
        llm_client=SuccessfulModelingLLMClient(),
    )

    assert interpretation == {}
    assert trace.status == "not_applicable"


def test_build_modeling_interpretation_llm_success_uses_compact_payload() -> None:
    llm_client = SuccessfulModelingLLMClient()

    interpretation, trace = build_modeling_interpretation_with_trace(
        _report_with_modeling(),
        llm_client=llm_client,
    )

    assert trace.status == "llm_applied"
    assert interpretation["modeling_summary"].startswith("LLM整体小结")
    assert interpretation["model_choice_takeaway"].startswith("LLM模型选择解释")
    assert llm_client.payload is not None
    assert sorted(llm_client.payload) == [
        "business_priority_metric",
        "confusion_matrix",
        "false_negative_count",
        "false_positive_count",
        "feature_importance_grouped",
        "feature_importance_top_raw",
        "loss_risk_analysis_context",
        "model_comparison",
        "summary_metrics",
        "target_definition",
        "task",
        "threshold_analysis",
        "threshold_default",
        "top_feature_groups",
        "top_raw_features",
        "warnings",
    ]
    assert len(llm_client.payload["feature_importance_top_raw"]) == 2
    assert llm_client.payload["target_definition"] == "is_loss = profit < 0"
    assert llm_client.payload["business_priority_metric"] == "recall"
    assert llm_client.payload["threshold_default"] == 0.5
    assert llm_client.payload["top_raw_features"][0]["feature"] == "discount"
    assert llm_client.payload["top_feature_groups"][0]["feature_group"] == "discount"


def test_build_modeling_interpretation_llm_error_falls_back() -> None:
    interpretation, trace = build_modeling_interpretation_with_trace(
        _report_with_modeling(),
        llm_client=ErrorModelingLLMClient(),
    )

    assert trace.status == "fallback_on_error"
    assert set(interpretation) == INTERPRETATION_KEYS
    assert "LogisticRegression" in interpretation["model_choice_takeaway"]


def test_build_modeling_interpretation_missing_method_is_skipped_with_fallback() -> None:
    interpretation, trace = build_modeling_interpretation_with_trace(
        _report_with_modeling(),
        llm_client=NoModelingMethodLLMClient(),
    )

    assert trace.status == "skipped"
    assert set(interpretation) == INTERPRETATION_KEYS


def test_sanitize_modeling_interpretation_truncates_and_handles_banned_terms() -> None:
    fallback = {
        "modeling_summary": "fallback summary 不代表因果关系，不用于自动决策",
        "model_choice_takeaway": "fallback model",
        "threshold_takeaway": "fallback threshold",
        "feature_takeaway": "fallback feature",
        "error_takeaway": "fallback error",
        "review_guidance": "fallback guidance",
        "business_use_warning": "fallback warning 不代表因果关系",
    }
    raw = {
        "modeling_summary": "整体小结证明 discount 导致亏损。",
        "model_choice_takeaway": "A" * 300,
        "threshold_takeaway": "阈值证明模型可以自动决策。",
        "feature_takeaway": "discount 导致亏损。",
        "error_takeaway": "",
        "review_guidance": "B" * 300,
        "business_use_warning": "仅供参考。",
        "extra": "ignored",
    }

    sanitized = _sanitize_modeling_interpretation(raw, fallback)

    assert set(sanitized) == INTERPRETATION_KEYS
    assert sanitized["modeling_summary"] == fallback["modeling_summary"]
    assert len(sanitized["model_choice_takeaway"]) <= 180
    assert len(sanitized["review_guidance"]) <= 190
    assert sanitized["threshold_takeaway"] == fallback["threshold_takeaway"]
    assert sanitized["feature_takeaway"] == fallback["feature_takeaway"]
    assert sanitized["error_takeaway"] == fallback["error_takeaway"]
    assert "不代表因果关系" in sanitized["business_use_warning"]
    assert "不用于自动决策" in sanitized["business_use_warning"]


def test_sanitize_modeling_summary_appends_required_limits_within_length() -> None:
    fallback = {
        "modeling_summary": "fallback summary 不代表因果关系，不用于自动决策",
        "model_choice_takeaway": "fallback model",
        "threshold_takeaway": "fallback threshold",
        "feature_takeaway": "fallback feature",
        "error_takeaway": "fallback error",
        "review_guidance": "fallback guidance",
        "business_use_warning": "fallback warning 不代表因果关系，不用于自动决策",
    }
    raw = {
        "modeling_summary": "这是一段较长但没有限制说明的模型小结。" * 20,
        "model_choice_takeaway": "model",
        "threshold_takeaway": "threshold",
        "feature_takeaway": "feature 不代表因果关系",
        "error_takeaway": "error",
        "review_guidance": "guidance",
        "business_use_warning": "warning 不代表因果关系，不用于自动决策",
    }

    sanitized = _sanitize_modeling_interpretation(raw, fallback)

    assert len(sanitized["modeling_summary"]) <= 230
    assert "不代表因果关系" in sanitized["modeling_summary"]
    assert "不用于自动决策" in sanitized["modeling_summary"]


def test_sanitize_long_modeling_summary_preserves_complete_limits() -> None:
    fallback = {
        "modeling_summary": "fallback summary 不代表因果关系，不用于自动决策",
        "model_choice_takeaway": "fallback model",
        "threshold_takeaway": "fallback threshold",
        "feature_takeaway": "fallback feature 不代表因果关系",
        "error_takeaway": "fallback error",
        "review_guidance": "fallback guidance",
        "business_use_warning": "fallback warning 不代表因果关系，不用于自动决策",
    }
    raw = {
        "modeling_summary": (
            "本段已经有很长的模型解释。" * 20
            + "该模型不代表因果关系，不用于自动决策。"
        ),
        "model_choice_takeaway": "model",
        "threshold_takeaway": "threshold",
        "feature_takeaway": "feature 不代表因果关系",
        "error_takeaway": "error",
        "review_guidance": "guidance",
        "business_use_warning": "warning 不代表因果关系，不用于自动决策",
    }

    sanitized = _sanitize_modeling_interpretation(raw, fallback)

    assert len(sanitized["modeling_summary"]) <= 230
    assert "不代表因果关系" in sanitized["modeling_summary"]
    assert "不用于自动决策" in sanitized["modeling_summary"]
    assert "不用…" not in sanitized["modeling_summary"]


def test_modeling_summary_limit_trims_complete_sentences_without_bad_ellipsis() -> None:
    raw_summary = (
        "该模型把亏损风险转化为人工复核优先级。"
        "当前解释文本非常长，需要在完整句子边界进行裁剪，避免留下半句话。"
        "人工复核建议需要结合折扣、类目和订单明细判断。"
        "模型输出只能作为风险排序参考。"
        * 8
    )

    summary = _ensure_modeling_summary_limits(raw_summary)

    assert len(summary) <= 230
    assert "不代表因果关系" in summary
    assert "不用于自动决策" in summary
    assert "该…" not in summary
    assert "人工…" not in summary
    assert "模型…" not in summary
    assert "…" not in summary


def test_modeling_summary_limit_does_not_split_decimal_threshold() -> None:
    raw_summary = (
        "本次建模以利润小于零作为亏损目标，样本中亏损订单占比约18.6%，属于中度不平衡。"
        "默认阈值0.5下召回率为0.96，"
        "FP=183、FN=19，模型可作为人工复核优先级参考。"
    )

    summary = _ensure_modeling_summary_limits(raw_summary, limit=120)

    assert "默认阈值0.该模型" not in summary
    assert "召回率为0.该模型" not in summary
    assert "召回率为0." not in summary
    assert "不代表因果关系" in summary
    assert "不用于自动决策" in summary


def test_sanitize_feature_takeaway_appends_no_causality_note() -> None:
    fallback = {
        "modeling_summary": "fallback summary 不代表因果关系，不用于自动决策",
        "model_choice_takeaway": "fallback model",
        "threshold_takeaway": "fallback threshold",
        "feature_takeaway": "fallback feature 不代表因果关系",
        "error_takeaway": "fallback error",
        "review_guidance": "fallback guidance",
        "business_use_warning": "fallback warning 不代表因果关系，不用于自动决策",
    }
    raw = {
        "modeling_summary": "summary 不代表因果关系，不用于自动决策",
        "model_choice_takeaway": "model",
        "threshold_takeaway": "threshold",
        "feature_takeaway": "discount 是主要预测信号。",
        "error_takeaway": "error",
        "review_guidance": "guidance",
        "business_use_warning": "warning 不代表因果关系，不用于自动决策",
    }

    sanitized = _sanitize_modeling_interpretation(raw, fallback)

    assert "该字段仅为预测信号，不代表因果关系。" in sanitized["feature_takeaway"]
    assert len(sanitized["feature_takeaway"]) <= 180


def test_sanitize_review_guidance_forbidden_automation_phrases_fall_back() -> None:
    fallback = {
        "modeling_summary": "fallback summary 不代表因果关系，不用于自动决策",
        "model_choice_takeaway": "fallback model",
        "threshold_takeaway": "fallback threshold",
        "feature_takeaway": "fallback feature 不代表因果关系",
        "error_takeaway": "fallback error",
        "review_guidance": "fallback guidance",
        "business_use_warning": "fallback warning 不代表因果关系，不用于自动决策",
    }
    for forbidden in ["自动拦截", "自动审批", "自动拒单", "自动处置", "自动处理"]:
        raw = {
            "modeling_summary": "summary 不代表因果关系，不用于自动决策",
            "model_choice_takeaway": "model",
            "threshold_takeaway": "threshold",
            "feature_takeaway": "feature 不代表因果关系",
            "error_takeaway": "error",
            "review_guidance": f"建议对高风险订单进行{forbidden}。",
            "business_use_warning": "warning 不代表因果关系，不用于自动决策",
        }

        sanitized = _sanitize_modeling_interpretation(raw, fallback)

        assert sanitized["review_guidance"] == fallback["review_guidance"]


def test_render_modeling_section_uses_modeling_interpretation() -> None:
    interpretation = {
        "modeling_summary": "LLM整体小结：这段建模分析把亏损风险转化为可复核的订单优先级，LogisticRegression 的 recall 为 0.9。",
        "model_choice_takeaway": "LLM模型选择解释：当前最佳模型按 recall 优先选择。",
        "threshold_takeaway": "LLM阈值解释：降低阈值会增加复核量。",
        "feature_takeaway": "LLM特征解释：discount 是主要预测信号。",
        "error_takeaway": "LLM错误解释：FN 比 FP 更需要优先复盘。",
        "review_guidance": "LLM复核建议：优先复核高风险订单和 FN 样本。",
        "business_use_warning": "LLM限制：只用于风险预警，不代表因果关系，不用于自动决策。",
    }

    markdown = render_modeling_section_markdown(
        _modeling_module(),
        modeling_interpretation=interpretation,
    )

    assert "### 建模分析小结" in markdown
    assert interpretation["modeling_summary"] in markdown
    assert interpretation["model_choice_takeaway"] in markdown
    assert interpretation["threshold_takeaway"] in markdown
    assert interpretation["feature_takeaway"] in markdown
    assert interpretation["error_takeaway"] in markdown
    assert interpretation["review_guidance"] in markdown
    assert "不代表因果关系" in markdown or "不用于自动决策" in markdown
    assert markdown.count("不代表因果关系") <= 3
    assert "Top 原始特征" in markdown
    assert "feature_importance" not in markdown


def test_render_modeling_section_fallback_includes_modeling_summary() -> None:
    markdown = render_modeling_section_markdown(_modeling_module())

    assert "### 建模分析小结" in markdown
    assert "LogisticRegression" in markdown
    assert "0.25" in markdown
    assert "DummyClassifier" in markdown
    assert "target_positive_count" in markdown
    assert "class_imbalance_level" in markdown
    assert "blocked_features" not in markdown
    assert "target_leakage" in markdown
    assert "不代表因果关系" in markdown
    assert "不用于自动决策" in markdown


def test_render_modeling_section_downgrades_weak_modeling_language() -> None:
    module = _modeling_module()
    module.summary_metrics["model_quality_status"] = "weak"
    module.summary_metrics["weak_reasons"] = ["extreme_class_imbalance", "too_few_test_positives", "low_precision"]

    markdown = render_modeling_section_markdown(module)

    assert "探索性参考" in markdown
    assert "不建议直接用于复核排序" in markdown
    assert "人工复核优先级排序" not in markdown
    assert "人工复核优先级排序，不代表因果关系" not in markdown
    assert "当前模型适合亏损风险预警和人工复核优先级排序" not in markdown
    assert "适合亏损风险预警和复核排序" not in markdown


def test_render_modeling_section_skipped_does_not_show_modeling_summary() -> None:
    markdown = render_modeling_section_markdown(_modeling_module(skipped=True))

    assert "The current data is not suitable for loss-risk modeling" in markdown
    assert "### Modeling Summary" not in markdown
    assert "LogisticRegression" not in markdown
    assert "review ranking" in markdown
    assert "causal claims" in markdown
    assert "automatic decisions" in markdown


def test_render_skipped_modeling_section_uses_opportunity_decision_when_available() -> None:
    decision = {
        "decision_summary": "亏损风险 target 为单类别，后续可评估销售额预测机会。",
        "notebook_message": "当前亏损风险 target 不适合训练；后续可评估销售额预测 baseline。",
        "risk_warning": "G6-2 不训练新模型，不用于自动决策。",
    }

    markdown = render_modeling_section_markdown(
        _modeling_module(skipped=True),
        modeling_opportunity_decision=decision,
    )

    assert "## Modeling Analysis: Feasibility Review" in markdown
    assert "single_target_class" in markdown
    assert "No new model is trained" in markdown
    assert "LogisticRegression" not in markdown


def test_render_modeling_opportunity_only_section_without_model_cells() -> None:
    decision = {
        "decision_summary": "当前推荐建模方向为 sales_amount_forecast_or_regression，但 G6-2 只记录机会。",
        "notebook_message": "当前数据缺少 profit 字段，因此不构造亏损风险分类模型；后续可评估销售额预测 baseline。",
        "risk_warning": "G6-2 不训练新模型，不用于自动决策。",
    }

    cells = render_modeling_section_cells(
        None,
        modeling_opportunity_decision=decision,
    )
    combined_source = "\n\n".join(cell.source for cell in cells)

    assert len(cells) == 1
    assert cells[0].cell_type == "markdown"
    assert "## Modeling Analysis: Feasibility Review" in combined_source
    assert "No new model is trained" in combined_source
    assert "threshold_analysis" not in combined_source
    assert "confusion_matrix" not in combined_source


def test_render_modeling_section_cells_include_visualization_code_cells() -> None:
    cells = render_modeling_section_cells(_modeling_module())
    cell_sources = [cell.source for cell in cells]
    code_sources = [source for cell, source in zip(cells, cell_sources) if cell.cell_type == "code"]
    combined_code = "\n\n".join(code_sources)

    assert any("### Modeling Analysis" in source or "### 建模分析" in source for source in cell_sources)
    assert "threshold_analysis" in combined_code
    assert "confusion_matrix" in combined_code
    assert len(code_sources) == 3
    assert "threshold_analysis" in combined_code
    assert "confusion_matrix" in combined_code
    assert "feature_importance_grouped" in combined_code
    assert "Threshold" in combined_code or "threshold" in combined_code
    assert "Confusion" in combined_code or "confusion" in combined_code
    assert "feature" in combined_code


def test_render_modeling_section_cells_skip_visualization_for_skipped_module() -> None:
    cells = render_modeling_section_cells(_modeling_module(skipped=True))
    combined_source = "\n\n".join(cell.source for cell in cells)

    assert len(cells) == 1
    assert cells[0].cell_type == "markdown"
    assert "The current data is not suitable for loss-risk modeling" in combined_source
    assert "### Modeling Summary" not in combined_source
    assert "threshold_analysis" not in combined_source
    assert "confusion_matrix" not in combined_source
    assert "feature_importance_grouped" not in combined_source


def test_modeling_plot_setup_checks_available_fonts_before_setting_family() -> None:
    setup_code = _modeling_plot_setup_code()

    assert "matplotlib.font_manager" in setup_code
    assert "fontManager.ttflist" in setup_code
    assert "preferred_fonts" in setup_code
    assert 'plt.rcParams["axes.unicode_minus"] = False' in setup_code
    assert 'plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK JP"]' not in setup_code


def test_run_trace_contains_modeling_interpretation_stage(
    client: TestClient,
    sample_csv_path: Path,
) -> None:
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )
    assert upload_response.status_code == 200

    run_response = client.post(f"/api/v1/analysis/tasks/{task_id}/run")
    assert run_response.status_code == 202
    body = run_response.json()
    assert "modeling_interpretation" in body["llm_trace"]

    trace_response = client.get(f"/api/v1/analysis/tasks/{task_id}/artifacts/llm-trace")
    assert trace_response.status_code == 200
    trace_payload = json.loads(trace_response.content)
    assert "modeling_interpretation" in trace_payload
