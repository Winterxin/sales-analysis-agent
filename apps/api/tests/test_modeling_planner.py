from __future__ import annotations

import pandas as pd

from app.schemas.schema_mapping import SchemaMapping
from app.services.modeling_planner import build_modeling_plan


def _mapping(fields: dict[str, str]) -> SchemaMapping:
    return SchemaMapping(dataset_type="sales_transaction", field_mapping=fields, confidence=1.0)


def _runnable_frame() -> pd.DataFrame:
    rows = 60
    return pd.DataFrame(
        {
            "profit": [-5] * 20 + [10] * 40,
            "discount": [0.1] * rows,
            "sales_amount": [100.0] * rows,
            "quantity": [2] * rows,
            "category": ["Furniture"] * rows,
        }
    )


def test_profit_with_two_target_classes_returns_runnable_plan() -> None:
    plan = build_modeling_plan(
        _runnable_frame(),
        _mapping(
            {
                "Profit": "profit",
                "Discount": "discount",
                "Sales": "sales_amount",
                "Quantity": "quantity",
                "Category": "category",
            }
        ),
    )

    assert plan.status == "runnable"
    assert plan.task_type == "loss_risk_classification"
    assert plan.target.name == "is_loss"
    assert plan.target.source_field == "profit"
    assert plan.target.expression == "profit < 0"
    assert plan.business_priority_metric == "recall"
    assert plan.selected_models == ["LogisticRegression", "RandomForestClassifier"]
    assert plan.metrics == ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]


def test_string_numeric_profit_values_still_return_runnable_plan() -> None:
    frame = _runnable_frame()
    frame["profit"] = [str(value) for value in frame["profit"]]

    plan = build_modeling_plan(
        frame,
        _mapping({"Profit": "profit", "Discount": "discount", "Sales": "sales_amount"}),
    )

    assert plan.status == "runnable"


def test_missing_profit_returns_skipped_plan() -> None:
    frame = _runnable_frame().drop(columns=["profit"])

    plan = build_modeling_plan(
        frame,
        _mapping({"Discount": "discount", "Sales": "sales_amount", "Quantity": "quantity"}),
    )

    assert plan.status == "skipped"
    assert "missing_profit_field" in plan.skip_reasons


def test_valid_profit_sample_size_too_small_returns_skipped_plan() -> None:
    frame = _runnable_frame()
    frame["profit"] = frame["profit"].astype(object)
    frame.loc[45:, "profit"] = "not available"

    plan = build_modeling_plan(
        frame,
        _mapping({"Profit": "profit", "Discount": "discount", "Sales": "sales_amount"}),
    )

    assert plan.status == "skipped"
    assert "insufficient_sample_size" in plan.skip_reasons


def test_all_invalid_profit_values_return_no_valid_profit_values() -> None:
    frame = _runnable_frame()
    frame["profit"] = "not available"

    plan = build_modeling_plan(
        frame,
        _mapping({"Profit": "profit", "Discount": "discount", "Sales": "sales_amount"}),
    )

    assert plan.status == "skipped"
    assert "no_valid_profit_values" in plan.skip_reasons


def test_single_target_class_returns_skipped_for_all_positive_or_all_negative_profit() -> None:
    positive = _runnable_frame()
    positive["profit"] = 1
    negative = _runnable_frame()
    negative["profit"] = -1
    mapping = _mapping({"Profit": "profit", "Discount": "discount", "Sales": "sales_amount"})

    positive_plan = build_modeling_plan(positive, mapping)
    negative_plan = build_modeling_plan(negative, mapping)

    assert positive_plan.status == "skipped"
    assert negative_plan.status == "skipped"
    assert "single_target_class" in positive_plan.skip_reasons
    assert "single_target_class" in negative_plan.skip_reasons


def test_minority_class_too_small_returns_skipped_plan() -> None:
    frame = _runnable_frame()
    frame["profit"] = [-5] * 9 + [10] * 51

    plan = build_modeling_plan(
        frame,
        _mapping({"Profit": "profit", "Discount": "discount", "Sales": "sales_amount"}),
    )

    assert plan.status == "skipped"
    assert "insufficient_class_count" in plan.skip_reasons


def test_allowed_features_exclude_profit_and_blocked_features_mark_target_leakage() -> None:
    plan = build_modeling_plan(
        _runnable_frame(),
        _mapping(
            {
                "Profit": "profit",
                "Profit Margin": "profit_margin",
                "Loss Flag": "loss_flag",
                "Discount": "discount",
                "Sales": "sales_amount",
            }
        ),
    )

    assert "profit" not in [feature.name for feature in plan.allowed_features]
    blocked_by_name = {feature.name: feature.reason for feature in plan.blocked_features}
    assert blocked_by_name["profit"] == "target_leakage"
    assert blocked_by_name["profit_margin"] == "target_leakage"
    assert blocked_by_name["loss_flag"] == "target_leakage"


def test_order_datetime_adds_date_part_extraction_step() -> None:
    frame = _runnable_frame()
    frame["order_datetime"] = pd.date_range("2026-01-01", periods=len(frame), freq="D")

    plan = build_modeling_plan(
        frame,
        _mapping(
            {
                "Profit": "profit",
                "Discount": "discount",
                "Sales": "sales_amount",
                "Order Date": "order_datetime",
            }
        ),
    )

    step_by_name = {step["name"]: step for step in plan.feature_engineering_steps}
    assert "date_part_extraction" in step_by_name
    assert step_by_name["date_part_extraction"]["parts"] == ["year", "month", "quarter", "weekday"]


def test_high_cardinality_identifiers_are_guarded_and_not_allowed() -> None:
    frame = _runnable_frame()
    frame["product_name"] = [f"Product {i}" for i in range(len(frame))]
    frame["customer_id"] = [f"Customer {i}" for i in range(len(frame))]
    frame["order_id"] = [f"Order {i}" for i in range(len(frame))]

    plan = build_modeling_plan(
        frame,
        _mapping(
            {
                "Profit": "profit",
                "Discount": "discount",
                "Sales": "sales_amount",
                "Product Name": "product_name",
                "Customer ID": "customer_id",
                "Order ID": "order_id",
            }
        ),
    )

    allowed = {feature.name for feature in plan.allowed_features}
    assert "product_name" not in allowed
    assert "customer_id" not in allowed
    assert "order_id" not in allowed
    guarded_step = next(step for step in plan.feature_engineering_steps if step["name"] == "high_cardinality_guard")
    assert guarded_step["fields"] == ["customer_id", "order_id", "product_name"]


def test_planner_exception_returns_skipped_plan() -> None:
    plan = build_modeling_plan(
        frame=None,  # type: ignore[arg-type]
        schema_mapping=_mapping({"Profit": "profit", "Discount": "discount", "Sales": "sales_amount"}),
    )

    assert plan.status == "skipped"
    assert any(reason.startswith("planner_error") for reason in plan.skip_reasons)
