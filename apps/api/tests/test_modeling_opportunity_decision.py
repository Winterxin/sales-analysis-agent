from __future__ import annotations

from typing import Any

from app.services.modeling_opportunity_decision import (
    DECISION_KEYS,
    build_modeling_opportunity_decision_with_trace,
)


class DisabledLLMClient:
    enabled = False
    source = "disabled"
    configured_model = None
    last_completion_metrics: dict[str, Any] = {}


class SuccessfulLLMClient:
    enabled = True
    source = "test"
    configured_model = "test-model"
    last_completion_metrics = {"elapsed_ms": 1, "prompt_chars": 10, "response_chars": 20}

    def suggest_modeling_opportunity_decision(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, str]:
        self.payload = payload
        self.language_instruction = language_instruction
        return {
            "decision_summary": "LLM判断：当前数据缺少 profit，因此不训练亏损模型，但可把销售额预测作为后续候选。",
            "notebook_message": "当前数据缺少 profit 字段，本轮不构造亏损风险模型；字段结构支持后续评估销售额预测 baseline。",
            "risk_warning": "G6-2 只记录建模机会，不训练新模型，也不用于自动决策。",
        }


class UnsafeLLMClient(SuccessfulLLMClient):
    def suggest_modeling_opportunity_decision(
        self, payload: dict[str, Any], language_instruction: str | None = None
    ) -> dict[str, str]:
        return {
            "decision_summary": "我已经训练了一个新回归模型，可以自动决策。",
            "notebook_message": "建议直接训练新模型并自动排序。",
            "risk_warning": "自动决策。",
        }


def _opportunity_plan() -> dict[str, Any]:
    return {
        "recommended_modeling_task": "sales_amount_forecast_or_regression",
        "decision_status": "opportunity_only",
        "should_train_model": False,
        "risk_level": "medium",
        "hard_gate_reasons": ["missing_profit_field"],
        "notebook_message": "当前数据缺少 profit 字段，因此不构造亏损风险分类模型。",
        "available_tasks": [
            {"task_type": "sales_amount_forecast_or_regression", "status": "candidate"},
            {"task_type": "loss_risk_classification", "status": "blocked"},
        ],
    }


def test_disabled_llm_returns_deterministic_modeling_opportunity_decision() -> None:
    decision, trace = build_modeling_opportunity_decision_with_trace(
        _opportunity_plan(),
        llm_client=DisabledLLMClient(),
    )

    assert set(decision) == DECISION_KEYS
    assert trace.status == "disabled"
    assert trace.applied is False
    assert "缺少 profit" in decision["notebook_message"]
    assert "不训练新模型" in decision["risk_warning"]


def test_llm_success_uses_payload_and_applies_sanitized_decision() -> None:
    llm_client = SuccessfulLLMClient()

    decision, trace = build_modeling_opportunity_decision_with_trace(
        _opportunity_plan(),
        llm_client=llm_client,
    )

    assert trace.status == "llm_applied"
    assert trace.applied is True
    assert decision["decision_summary"].startswith("LLM判断")
    assert llm_client.payload["recommended_modeling_task"] == "sales_amount_forecast_or_regression"


def test_unsafe_llm_output_falls_back_to_deterministic_decision() -> None:
    decision, trace = build_modeling_opportunity_decision_with_trace(
        _opportunity_plan(),
        llm_client=UnsafeLLMClient(),
    )

    assert trace.status == "fallback_invalid_payload"
    assert "已经训练" not in decision["decision_summary"]
    assert "自动决策" not in decision["notebook_message"]
    assert "不训练新模型" in decision["risk_warning"]
