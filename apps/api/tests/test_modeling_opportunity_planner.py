from __future__ import annotations

import pandas as pd

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.schema_mapping import SchemaMapping
from app.services.modeling_opportunity_planner import build_modeling_opportunity_plan


def _mapping(field_mapping: dict[str, str]) -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping=field_mapping,
        confidence=1.0,
        missing_required_fields=[],
        uncertain_fields=[],
    )


def _plan(modules: list[str] | None = None) -> AnalysisPlan:
    return AnalysisPlan(
        analysis_plan=modules or ["data_quality_check"],
        chart_preferences={},
        reasoning_summary=[],
    )


def test_profit_dataset_recommends_existing_loss_modeling_when_target_is_valid() -> None:
    frame = pd.DataFrame(
        {
            "Sales": [100, 120, 140, 160, 180, 200] * 20,
            "Profit": [-10, -5, 4, 8, 12, 16] * 20,
            "Discount": [0.1, 0.2, 0.0, 0.05, 0.1, 0.0] * 20,
            "Category": ["A", "B", "A", "C", "B", "A"] * 20,
        }
    )

    result = build_modeling_opportunity_plan(
        frame,
        _mapping({"Sales": "sales_amount", "Profit": "profit", "Discount": "discount", "Category": "category"}),
        dataset_profile={"has_profit": True},
        analysis_plan=_plan(["loss_risk_modeling"]),
    )

    assert result["recommended_modeling_task"] == "loss_risk_classification"
    assert result["decision_status"] == "existing_modeling"
    assert result["should_train_model"] is True
    loss_task = next(task for task in result["available_tasks"] if task["task_type"] == "loss_risk_classification")
    assert loss_task["status"] == "runnable"
    assert "valid_profit_target" in loss_task["reasons"]


def test_missing_profit_dataset_recommends_sales_forecast_opportunity_without_training() -> None:
    frame = pd.DataFrame(
        {
            "InvoiceDate": pd.date_range("2024-01-01", periods=90, freq="D"),
            "Sales": [100 + i for i in range(90)],
            "CustomerID": [f"C{i % 12}" for i in range(90)],
        }
    )

    result = build_modeling_opportunity_plan(
        frame,
        _mapping({"InvoiceDate": "order_datetime", "Sales": "sales_amount", "CustomerID": "customer_id"}),
        dataset_profile={"has_profit": False, "repeat_customer_rate": 0.5},
        analysis_plan=_plan(),
    )

    assert result["recommended_modeling_task"] == "sales_amount_forecast_or_regression"
    assert result["decision_status"] == "opportunity_only"
    assert result["should_train_model"] is False
    assert "missing_profit_field" in result["hard_gate_reasons"]
    assert "缺少 profit" in result["notebook_message"]
    forecast_task = next(task for task in result["available_tasks"] if task["task_type"] == "sales_amount_forecast_or_regression")
    assert forecast_task["status"] == "candidate"
    repeat_task = next(task for task in result["available_tasks"] if task["task_type"] == "customer_repeat_or_retention")
    assert repeat_task["status"] == "candidate"


def test_single_class_profit_dataset_is_skipped_and_does_not_train() -> None:
    frame = pd.DataFrame(
        {
            "Sales": [100] * 80,
            "Profit": [12] * 80,
            "Date": pd.date_range("2024-01-01", periods=80, freq="D"),
        }
    )

    result = build_modeling_opportunity_plan(
        frame,
        _mapping({"Sales": "sales_amount", "Profit": "profit", "Date": "order_datetime"}),
        dataset_profile={"has_profit": True},
        analysis_plan=_plan(["loss_risk_modeling"]),
    )

    assert result["recommended_modeling_task"] == "sales_amount_forecast_or_regression"
    assert result["should_train_model"] is False
    assert "single_target_class" in result["hard_gate_reasons"]
    loss_task = next(task for task in result["available_tasks"] if task["task_type"] == "loss_risk_classification")
    assert loss_task["status"] == "blocked"


def test_status_risk_candidate_requires_negative_status_values() -> None:
    frame = pd.DataFrame(
        {
            "Status": ["Complete"] * 70 + ["Cancelled"] * 15 + ["Returned"] * 15,
            "Sales": [100] * 100,
            "Date": pd.date_range("2024-01-01", periods=100, freq="D"),
        }
    )

    result = build_modeling_opportunity_plan(
        frame,
        _mapping({"Status": "order_status", "Sales": "sales_amount", "Date": "order_datetime"}),
        dataset_profile={"has_order_status": True},
        analysis_plan=_plan(),
    )

    status_task = next(task for task in result["available_tasks"] if task["task_type"] == "status_risk_classification")
    assert status_task["status"] == "candidate"
    assert "negative_status_values_detected" in status_task["reasons"]
    assert result["recommended_modeling_task"] == "sales_amount_forecast_or_regression"


def test_dataset_without_stable_target_recommends_descriptive_only() -> None:
    frame = pd.DataFrame({"Product": ["A", "B", "C"], "Quantity": [1, 2, 3]})

    result = build_modeling_opportunity_plan(
        frame,
        _mapping({"Product": "product_name", "Quantity": "quantity"}),
        dataset_profile={},
        analysis_plan=_plan(),
    )

    assert result["recommended_modeling_task"] == "no_modeling_descriptive_only"
    assert result["decision_status"] == "not_recommended"
    assert result["should_train_model"] is False
    assert "当前数据未形成稳定建模目标" in result["notebook_message"]
