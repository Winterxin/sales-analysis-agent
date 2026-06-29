from __future__ import annotations

import pandas as pd

from app.analysis.modules import loss_risk_modeling


def _canonical_columns() -> dict[str, str]:
    return {
        "profit": "Profit",
        "discount": "Discount",
        "sales_amount": "Sales",
        "quantity": "Quantity",
        "category": "Category",
        "sub_category": "Sub-Category",
        "region": "Region",
        "order_datetime": "Order Date",
        "product_name": "Product Name",
        "customer_id": "Customer ID",
        "order_id": "Order ID",
    }


def _modeling_frame(rows: int = 120) -> pd.DataFrame:
    data: list[dict[str, object]] = []
    for index in range(rows):
        discount = 0.35 if index % 4 == 0 else 0.05
        sales = 80 + index
        quantity = 1 + (index % 5)
        category = "Furniture" if index % 3 == 0 else "Technology"
        is_loss = index % 4 == 0 or (category == "Furniture" and discount > 0.3)
        data.append(
            {
                "Profit": -20 - (index % 7) if is_loss else 20 + (index % 11),
                "Discount": discount,
                "Sales": sales,
                "Quantity": quantity,
                "Category": category,
                "Sub-Category": "Chairs" if category == "Furniture" else "Phones",
                "Region": "West" if index % 2 == 0 else "East",
                "Order Date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index),
                "Product Name": f"Product {index}",
                "Customer ID": f"Customer {index}",
                "Order ID": f"Order {index}",
            }
        )
    return pd.DataFrame(data)


def test_loss_risk_modeling_returns_module_result_for_runnable_data() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    assert result.module_id == "loss_risk_modeling"
    assert result.title == "亏损风险建模"
    assert result.chart_type == "modeling"
    assert result.summary_metrics["train_rows"] > 0
    assert result.summary_metrics["test_rows"] > 0
    assert result.summary_metrics["business_priority_metric"] == "recall"
    assert not result.warnings


def test_model_comparison_contains_supported_models() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    model_names = {row["model"] for row in result.tables["model_comparison"]}
    assert {"LogisticRegression", "RandomForestClassifier"} <= model_names


def test_model_comparison_includes_dummy_baseline_and_lift_fields() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    rows = result.tables["model_comparison"]
    baseline = next(row for row in rows if row["model"] == "DummyClassifier")
    assert baseline["is_baseline"] is True
    assert all("lift_over_baseline_f1" in row for row in rows)
    assert all("lift_over_baseline_recall" in row for row in rows)
    assert any(row["model"] != "DummyClassifier" and row["lift_over_baseline_f1"] >= 0 for row in rows)


def test_summary_metrics_include_best_model_scores() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    assert result.summary_metrics["best_model"]
    assert "best_recall" in result.summary_metrics
    assert "best_f1" in result.summary_metrics
    assert "best_roc_auc" in result.summary_metrics
    assert result.summary_metrics["threshold_default"] == 0.5
    assert "false_positive_count" in result.summary_metrics
    assert "false_negative_count" in result.summary_metrics
    assert "true_positive_count" in result.summary_metrics
    assert "true_negative_count" in result.summary_metrics
    assert result.summary_metrics["target_positive_count"] > 0
    assert result.summary_metrics["target_negative_count"] > 0
    assert result.summary_metrics["train_positive_count"] > 0
    assert result.summary_metrics["test_positive_count"] > 0
    assert result.summary_metrics["class_imbalance_level"] in {"balanced", "moderate", "high", "extreme"}
    assert result.summary_metrics["model_quality_status"] in {"strong", "usable", "weak"}
    assert isinstance(result.summary_metrics["weak_reasons"], list)


def test_feature_plan_tables_expose_safe_and_blocked_features() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    assert result.tables["safe_features"]
    blocked = result.tables["blocked_features"]
    blocked_by_name = {row["feature"]: row["reason"] for row in blocked}
    assert blocked_by_name["profit"] == "target_leakage"
    assert blocked_by_name["product_name"] == "high_cardinality_guard"
    assert blocked_by_name["customer_id"] == "high_cardinality_guard"
    assert blocked_by_name["order_id"] == "high_cardinality_guard"


def test_extreme_imbalance_is_marked_weak_even_when_model_runs() -> None:
    rows = 2000
    frame = pd.DataFrame(
        {
            "Profit": [-5] * 10 + [10] * (rows - 10),
            "Discount": [0.4] * 10 + [0.05] * (rows - 10),
            "Sales": list(range(rows)),
            "Quantity": [1] * rows,
            "Category": ["Loss"] * 10 + ["Normal"] * (rows - 10),
        }
    )

    result = loss_risk_modeling.run(
        frame,
        {
            "profit": "Profit",
            "discount": "Discount",
            "sales_amount": "Sales",
            "quantity": "Quantity",
            "category": "Category",
        },
    )

    assert result.tables["model_comparison"]
    assert result.summary_metrics["model_quality_status"] == "weak"
    assert "extreme_class_imbalance" in result.summary_metrics["weak_reasons"]
    assert "too_few_test_positives" in result.summary_metrics["weak_reasons"]



def test_best_model_selection_prioritizes_recall_over_f1(monkeypatch) -> None:
    original_metrics = loss_risk_modeling._model_metrics

    def fake_metrics(model_name, y_test, y_pred, y_score):
        if model_name == "LogisticRegression":
            return {
                "model": model_name,
                "accuracy": 0.7,
                "precision": 0.6,
                "recall": 0.95,
                "f1": 0.7,
                "roc_auc": 0.72,
            }
        if model_name == "RandomForestClassifier":
            return {
                "model": model_name,
                "accuracy": 0.9,
                "precision": 0.9,
                "recall": 0.75,
                "f1": 0.88,
                "roc_auc": 0.95,
            }
        return original_metrics(model_name, y_test, y_pred, y_score)

    monkeypatch.setattr(loss_risk_modeling, "_model_metrics", fake_metrics)

    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    assert result.summary_metrics["best_model"] == "LogisticRegression"
    assert result.summary_metrics["best_recall"] == 0.95
    assert any("Recall=0.95" in finding for finding in result.findings)


def test_feature_importance_and_high_risk_examples_are_not_empty() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    assert result.tables["feature_importance"]
    assert result.tables["high_risk_examples"]


def test_threshold_analysis_contains_business_review_load_metrics() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    rows = result.tables["threshold_analysis"]
    assert {row["threshold"] for row in rows} == {0.3, 0.4, 0.5, 0.6, 0.7}
    for row in rows:
        assert {
            "threshold",
            "precision",
            "recall",
            "f1",
            "predicted_loss_count",
            "review_load_rate",
        } <= set(row)


def test_feature_importance_grouped_summarizes_business_feature_groups() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    grouped = result.tables["feature_importance_grouped"]
    assert grouped
    group_names = {row["feature_group"] for row in grouped}
    assert "discount" in group_names
    assert all("business_meaning" in row for row in grouped)


def test_high_risk_examples_include_business_locator_fields() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    row = result.tables["high_risk_examples"][0]
    assert "discount" in row
    assert "sales_amount" in row
    assert "category" in row
    assert "region" in row
    assert "order_id" in row
    assert "product_name" in row


def test_outputs_false_positive_and_false_negative_example_tables() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    assert "false_positive_examples" in result.tables
    assert "false_negative_examples" in result.tables


def test_profit_is_not_used_as_a_feature() -> None:
    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    feature_names = {row["feature"] for row in result.tables["feature_importance"]}
    assert "profit" not in feature_names
    assert "Profit" not in feature_names


def test_single_target_class_gracefully_skips() -> None:
    frame = _modeling_frame()
    frame["Profit"] = 10

    result = loss_risk_modeling.run(frame, _canonical_columns())

    assert result.module_id == "loss_risk_modeling"
    assert result.warnings
    assert any("single_target_class" in warning for warning in result.warnings)


def test_sample_size_too_small_gracefully_skips() -> None:
    result = loss_risk_modeling.run(_modeling_frame(rows=40), _canonical_columns())

    assert result.module_id == "loss_risk_modeling"
    assert result.warnings
    assert any("insufficient_sample_size" in warning for warning in result.warnings)


def test_sklearn_exception_gracefully_skips(monkeypatch) -> None:
    class BrokenModel:
        def fit(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(loss_risk_modeling, "_model_candidates", lambda: [("BrokenModel", BrokenModel())])

    result = loss_risk_modeling.run(_modeling_frame(), _canonical_columns())

    assert result.module_id == "loss_risk_modeling"
    assert result.warnings
    assert any("sklearn_error" in warning for warning in result.warnings)
