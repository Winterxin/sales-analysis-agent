from __future__ import annotations

from typing import Any

import pandas as pd

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.schema_mapping import SchemaMapping


MIN_SAMPLE_SIZE = 50
MIN_CLASS_COUNT = 10
MIN_FORECAST_ROWS = 30
MIN_FORECAST_DAY_SPAN = 30
NEGATIVE_STATUS_KEYWORDS = (
    "cancel",
    "cancelled",
    "canceled",
    "return",
    "returned",
    "refund",
    "refunded",
    "fail",
    "failed",
    "closed",
)
TASK_PRIORITY = (
    "loss_risk_classification",
    "sales_amount_forecast_or_regression",
    "status_risk_classification",
    "customer_repeat_or_retention",
    "no_modeling_descriptive_only",
)


def _canonical_fields(schema_mapping: SchemaMapping) -> set[str]:
    return {
        str(canonical)
        for canonical in schema_mapping.field_mapping.values()
        if str(canonical).strip()
    }


def _source_column(schema_mapping: SchemaMapping, canonical_field: str, frame: pd.DataFrame) -> str | None:
    if canonical_field in frame.columns:
        return canonical_field
    for source_field, mapped_field in schema_mapping.field_mapping.items():
        if mapped_field == canonical_field and source_field in frame.columns:
            return source_field
    return None


def _missing(required: list[str], fields: set[str]) -> list[str]:
    return [field for field in required if field not in fields]


def _task(
    *,
    task_type: str,
    status: str,
    required_fields: list[str],
    fields: set[str],
    risk_level: str,
    reasons: list[str] | None = None,
    risk_reasons: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "task_type": task_type,
        "status": status,
        "required_fields": required_fields,
        "available_fields": [field for field in required_fields if field in fields],
        "missing_fields": _missing(required_fields, fields),
        "risk_level": risk_level,
        "reasons": reasons or [],
        "risk_reasons": risk_reasons or [],
    }


def _profit_series(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> pd.Series | None:
    column = _source_column(schema_mapping, "profit", frame)
    if column is None:
        return None
    return frame[column]


def _status_series(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> pd.Series | None:
    column = _source_column(schema_mapping, "order_status", frame)
    if column is None:
        return None
    return frame[column]


def _date_series(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> pd.Series | None:
    column = _source_column(schema_mapping, "order_datetime", frame)
    if column is None:
        return None
    return frame[column]


def _loss_task(
    frame: pd.DataFrame,
    schema_mapping: SchemaMapping,
    fields: set[str],
    hard_gate_reasons: list[str],
) -> dict[str, Any]:
    required = ["profit"]
    profit = _profit_series(frame, schema_mapping)
    if "profit" not in fields or profit is None:
        hard_gate_reasons.append("missing_profit_field")
        return _task(
            task_type="loss_risk_classification",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["missing_profit_field"],
        )

    valid_profit = pd.to_numeric(profit, errors="coerce").dropna()
    if len(valid_profit) < MIN_SAMPLE_SIZE:
        hard_gate_reasons.append("insufficient_sample_size")
        return _task(
            task_type="loss_risk_classification",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["insufficient_sample_size"],
        )

    target = valid_profit < 0
    class_counts = target.value_counts()
    if len(class_counts) < 2:
        hard_gate_reasons.append("single_target_class")
        return _task(
            task_type="loss_risk_classification",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["single_target_class"],
        )
    if int(class_counts.min()) < MIN_CLASS_COUNT:
        hard_gate_reasons.append("insufficient_class_count")
        return _task(
            task_type="loss_risk_classification",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["insufficient_class_count"],
        )

    return _task(
        task_type="loss_risk_classification",
        status="runnable",
        required_fields=required,
        fields=fields,
        risk_level="low",
        reasons=["valid_profit_target"],
    )


def _status_task(frame: pd.DataFrame, schema_mapping: SchemaMapping, fields: set[str]) -> dict[str, Any]:
    required = ["order_status"]
    status = _status_series(frame, schema_mapping)
    if "order_status" not in fields or status is None:
        return _task(
            task_type="status_risk_classification",
            status="unavailable",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["missing_order_status_field"],
        )

    normalized = status.dropna().astype(str).str.lower()
    negative_mask = normalized.apply(lambda value: any(keyword in value for keyword in NEGATIVE_STATUS_KEYWORDS))
    negative_count = int(negative_mask.sum())
    positive_count = int(len(negative_mask) - negative_count)
    if negative_count < MIN_CLASS_COUNT or positive_count < MIN_CLASS_COUNT:
        return _task(
            task_type="status_risk_classification",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["insufficient_negative_status_examples"],
        )

    return _task(
        task_type="status_risk_classification",
        status="candidate",
        required_fields=required,
        fields=fields,
        risk_level="medium",
        reasons=["negative_status_values_detected"],
        risk_reasons=["requires_leakage_review_before_training"],
    )


def _forecast_task(frame: pd.DataFrame, schema_mapping: SchemaMapping, fields: set[str]) -> dict[str, Any]:
    required = ["order_datetime", "sales_amount"]
    if _missing(required, fields):
        return _task(
            task_type="sales_amount_forecast_or_regression",
            status="unavailable",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["missing_date_or_sales_amount"],
        )

    date_values = pd.to_datetime(_date_series(frame, schema_mapping), errors="coerce").dropna()
    if len(date_values) < MIN_FORECAST_ROWS:
        return _task(
            task_type="sales_amount_forecast_or_regression",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["insufficient_time_rows"],
        )
    day_span = int((date_values.max() - date_values.min()).days)
    if day_span < MIN_FORECAST_DAY_SPAN:
        return _task(
            task_type="sales_amount_forecast_or_regression",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="medium",
            risk_reasons=["insufficient_time_span"],
        )

    return _task(
        task_type="sales_amount_forecast_or_regression",
        status="candidate",
        required_fields=required,
        fields=fields,
        risk_level="medium",
        reasons=["has_date_and_sales_amount"],
        risk_reasons=["forecast_requires_time_split"],
    )


def _repeat_task(
    frame: pd.DataFrame,
    schema_mapping: SchemaMapping,
    fields: set[str],
    dataset_profile: dict[str, Any],
) -> dict[str, Any]:
    required = ["customer_id", "order_datetime"]
    if _missing(required, fields):
        return _task(
            task_type="customer_repeat_or_retention",
            status="unavailable",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["missing_customer_or_date_field"],
        )

    customer_column = _source_column(schema_mapping, "customer_id", frame)
    repeat_rate = dataset_profile.get("repeat_customer_rate")
    customer_count = int(frame[customer_column].nunique(dropna=True)) if customer_column else 0
    if customer_count < MIN_CLASS_COUNT:
        return _task(
            task_type="customer_repeat_or_retention",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="high",
            risk_reasons=["insufficient_customer_count"],
        )
    if repeat_rate is not None and float(repeat_rate or 0) <= 0:
        return _task(
            task_type="customer_repeat_or_retention",
            status="blocked",
            required_fields=required,
            fields=fields,
            risk_level="medium",
            risk_reasons=["no_repeat_customer_signal"],
        )

    return _task(
        task_type="customer_repeat_or_retention",
        status="candidate",
        required_fields=required,
        fields=fields,
        risk_level="high",
        reasons=["has_customer_and_date_fields"],
        risk_reasons=["needs_observation_window_before_training"],
    )


def _descriptive_task(fields: set[str]) -> dict[str, Any]:
    return _task(
        task_type="no_modeling_descriptive_only",
        status="fallback",
        required_fields=[],
        fields=fields,
        risk_level="low",
        reasons=["descriptive_analysis_remains_primary"],
    )


def _recommended_task(tasks: list[dict[str, Any]], analysis_plan: AnalysisPlan | None) -> str:
    plan_modules = set(getattr(analysis_plan, "analysis_plan", []) or [])
    if "loss_risk_modeling" in plan_modules:
        loss = next((task for task in tasks if task["task_type"] == "loss_risk_classification"), None)
        if loss and loss["status"] == "runnable":
            return "loss_risk_classification"
    for task_type in TASK_PRIORITY[1:]:
        task = next((item for item in tasks if item["task_type"] == task_type), None)
        if task and task["status"] == "candidate":
            return task_type
    return "no_modeling_descriptive_only"


def _decision_status(recommended: str, tasks: list[dict[str, Any]], analysis_plan: AnalysisPlan | None) -> str:
    plan_modules = set(getattr(analysis_plan, "analysis_plan", []) or [])
    if recommended == "loss_risk_classification" and "loss_risk_modeling" in plan_modules:
        return "existing_modeling"
    if recommended == "no_modeling_descriptive_only":
        return "not_recommended"
    task = next((item for item in tasks if item["task_type"] == recommended), None)
    if task and task["status"] == "candidate":
        return "opportunity_only"
    return "skipped_by_hard_gate"


def _risk_level(recommended: str, tasks: list[dict[str, Any]]) -> str:
    task = next((item for item in tasks if item["task_type"] == recommended), None)
    return str(task.get("risk_level") if task else "low")


def _notebook_message(recommended: str, hard_gate_reasons: list[str], tasks: list[dict[str, Any]]) -> str:
    if recommended == "loss_risk_classification":
        return "当前数据具备 profit 字段且亏损目标分布可用，因此保留亏损风险分类模型；模型结果仍需结合 G6-1 质量状态判断。"
    if recommended == "sales_amount_forecast_or_regression":
        prefix = "当前数据缺少 profit 字段，因此不构造亏损风险分类模型。" if "missing_profit_field" in hard_gate_reasons else "当前亏损风险 target 不适合训练。"
        return prefix + " 从字段结构看，后续更适合评估销售额预测或回归 baseline，但 G6-2 只记录建模机会，不训练新模型。"
    if recommended == "status_risk_classification":
        return "当前数据存在订单状态字段和负向状态样本，后续可以评估订单状态风险分类；G6-2 只记录机会，不训练新模型。"
    if recommended == "customer_repeat_or_retention":
        return "当前数据存在客户和时间字段，后续可以评估复购或留存相关建模；G6-2 只记录机会，不训练新模型。"
    risk_reasons = sorted({reason for task in tasks for reason in task.get("risk_reasons", [])})
    if risk_reasons:
        return "当前数据未形成稳定建模目标，本轮保留描述性分析；主要限制包括：" + "、".join(risk_reasons[:4]) + "。"
    return "当前数据未形成稳定建模目标，本轮保留描述性分析。"


def build_modeling_opportunity_plan(
    frame: pd.DataFrame,
    schema_mapping: SchemaMapping,
    *,
    dataset_profile: dict[str, Any] | None = None,
    analysis_plan: AnalysisPlan | None = None,
) -> dict[str, Any]:
    profile = dataset_profile or {}
    fields = _canonical_fields(schema_mapping)
    hard_gate_reasons: list[str] = []
    tasks = [
        _loss_task(frame, schema_mapping, fields, hard_gate_reasons),
        _status_task(frame, schema_mapping, fields),
        _forecast_task(frame, schema_mapping, fields),
        _repeat_task(frame, schema_mapping, fields, profile),
        _descriptive_task(fields),
    ]
    recommended = _recommended_task(tasks, analysis_plan)
    decision_status = _decision_status(recommended, tasks, analysis_plan)
    should_train_model = decision_status == "existing_modeling"
    return {
        "available_tasks": tasks,
        "recommended_modeling_task": recommended,
        "should_train_model": should_train_model,
        "decision_status": decision_status,
        "risk_level": _risk_level(recommended, tasks),
        "hard_gate_reasons": sorted(set(hard_gate_reasons)),
        "notebook_message": _notebook_message(recommended, hard_gate_reasons, tasks),
        "llm_decision_explanation": None,
    }
