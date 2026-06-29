from __future__ import annotations

from typing import Any

import pandas as pd

from app.schemas.modeling import ModelingFeaturePlan, ModelingPlan, ModelingTarget
from app.schemas.schema_mapping import SchemaMapping


MIN_SAMPLE_SIZE = 50
MIN_CLASS_COUNT = 10
TASK_TYPE = "loss_risk_classification"
LEAKAGE_FIELDS = {"profit", "is_loss", "profit_margin", "loss_flag", "negative_profit"}
NUMERIC_FEATURES = {"discount", "sales_amount", "quantity", "unit_price"}
CATEGORICAL_FEATURES = {
    "category",
    "sub_category",
    "segment",
    "region",
    "country",
    "productline",
    "deal_size",
    "order_status",
    "ship_mode",
}
DATE_FEATURES = {"order_datetime"}
HIGH_CARDINALITY_FIELDS = {"product_name", "customer_id", "order_id"}
SELECTED_MODELS = ["LogisticRegression", "RandomForestClassifier"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]


def _base_plan(status: str, skip_reasons: list[str] | None = None) -> ModelingPlan:
    return ModelingPlan(
        status=status,  # type: ignore[arg-type]
        task_type=TASK_TYPE,
        target=ModelingTarget(
            name="is_loss",
            source_field="profit",
            expression="profit < 0",
            positive_class="loss",
        ),
        selected_models=list(SELECTED_MODELS),
        metrics=list(METRICS),
        skip_reasons=skip_reasons or [],
        limitations=[
            "G2 phase 1 only plans loss risk classification and does not train or score models.",
            "Feature transformations are declarative only; no data is transformed in the planner.",
        ],
    )


def _canonical_fields(schema_mapping: SchemaMapping) -> set[str]:
    return {
        str(canonical)
        for canonical in schema_mapping.field_mapping.values()
        if str(canonical).strip()
    }


def _profit_series(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> pd.Series | None:
    if "profit" in frame.columns:
        return frame["profit"]
    for source_field, canonical_field in schema_mapping.field_mapping.items():
        if canonical_field == "profit" and source_field in frame.columns:
            return frame[source_field]
    return None


def _blocked_features(fields: set[str]) -> list[ModelingFeaturePlan]:
    blocked: list[ModelingFeaturePlan] = []
    for field in sorted(fields & LEAKAGE_FIELDS):
        blocked.append(
            ModelingFeaturePlan(name=field, feature_type="unknown", reason="target_leakage")
        )
    for field in sorted(fields & HIGH_CARDINALITY_FIELDS):
        blocked.append(
            ModelingFeaturePlan(
                name=field,
                feature_type="categorical",
                reason="high_cardinality_guard",
            )
        )
    return blocked


def _allowed_features(fields: set[str]) -> list[ModelingFeaturePlan]:
    allowed: list[ModelingFeaturePlan] = []
    for field in sorted(fields & NUMERIC_FEATURES):
        allowed.append(
            ModelingFeaturePlan(name=field, feature_type="numeric", reason="supported_numeric_feature")
        )
    for field in sorted(fields & CATEGORICAL_FEATURES):
        allowed.append(
            ModelingFeaturePlan(
                name=field,
                feature_type="categorical",
                reason="supported_categorical_feature",
            )
        )
    for field in sorted(fields & DATE_FEATURES):
        allowed.append(ModelingFeaturePlan(name=field, feature_type="date", reason="supported_date_feature"))
    return allowed


def _feature_engineering_steps(fields: set[str]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {
            "name": "numeric_imputation",
            "applies_to": sorted(fields & NUMERIC_FEATURES),
            "strategy": "median",
        },
        {
            "name": "categorical_imputation",
            "applies_to": sorted(fields & CATEGORICAL_FEATURES),
            "strategy": "most_frequent_or_unknown",
        },
        {
            "name": "one_hot_encoding",
            "applies_to": sorted(fields & CATEGORICAL_FEATURES),
            "strategy": "encode supported low-cardinality categorical fields",
        },
        {
            "name": "optional_scaling",
            "applies_to": sorted(fields & NUMERIC_FEATURES),
            "strategy": "consider standardization for linear models",
        },
    ]
    if "order_datetime" in fields:
        steps.append(
            {
                "name": "date_part_extraction",
                "applies_to": ["order_datetime"],
                "parts": ["year", "month", "quarter", "weekday"],
            }
        )
    guarded_fields = sorted(fields & HIGH_CARDINALITY_FIELDS)
    if guarded_fields:
        steps.append(
            {
                "name": "high_cardinality_guard",
                "fields": guarded_fields,
                "strategy": "exclude identifier-like high-cardinality fields from the first model version",
            }
        )
    return steps


def _with_common_plan_details(plan: ModelingPlan, fields: set[str]) -> ModelingPlan:
    plan.allowed_features = _allowed_features(fields)
    plan.blocked_features = _blocked_features(fields)
    plan.feature_engineering_steps = _feature_engineering_steps(fields)
    return plan


def build_modeling_plan(
    frame: pd.DataFrame,
    schema_mapping: SchemaMapping,
    dataset_profile: dict[str, Any] | None = None,
    analysis_focus: dict[str, Any] | None = None,
    evidence_pack: dict[str, Any] | None = None,
) -> ModelingPlan:
    del dataset_profile, analysis_focus, evidence_pack

    try:
        fields = _canonical_fields(schema_mapping)
        plan = _with_common_plan_details(_base_plan(status="skipped"), fields)

        profit = _profit_series(frame, schema_mapping)
        if "profit" not in fields or profit is None:
            plan.skip_reasons = ["missing_profit_field"]
            return plan

        valid_profit = pd.to_numeric(profit, errors="coerce").dropna()
        if valid_profit.empty:
            plan.skip_reasons = ["no_valid_profit_values"]
            return plan

        if len(valid_profit) < MIN_SAMPLE_SIZE:
            plan.skip_reasons = ["insufficient_sample_size"]
            return plan

        target = valid_profit < 0
        class_counts = target.value_counts()
        if len(class_counts) < 2:
            plan.skip_reasons = ["single_target_class"]
            return plan
        if int(class_counts.min()) < MIN_CLASS_COUNT:
            plan.skip_reasons = ["insufficient_class_count"]
            return plan

        if len(plan.allowed_features) < 2:
            plan.skip_reasons = ["insufficient_allowed_features"]
            return plan

        plan.status = "runnable"
        plan.skip_reasons = []
        return plan
    except Exception as exc:  # pragma: no cover - exact exception type is intentionally unconstrained.
        return _base_plan(status="skipped", skip_reasons=[f"planner_error: {type(exc).__name__}: {exc}"])
