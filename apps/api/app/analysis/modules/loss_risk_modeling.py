from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.analysis.contracts import ModuleResult
from app.schemas.modeling import ModelingFeaturePlan
from app.schemas.schema_mapping import SchemaMapping
from app.services.modeling_planner import build_modeling_plan


MODULE_ID = "loss_risk_modeling"
TITLE = "亏损风险建模"
CHART_TYPE = "modeling"
LEAKAGE_FIELDS = {"profit", "is_loss", "profit_margin", "loss_flag", "negative_profit"}
HIGH_CARDINALITY_FIELDS = {"product_name", "customer_id", "order_id"}
DATE_PARTS = ["year", "month", "quarter", "weekday"]
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
DISPLAY_FIELDS = [
    "discount",
    "sales_amount",
    "quantity",
    "category",
    "sub_category",
    "segment",
    "region",
    "order_status",
    "deal_size",
    "productline",
    "order_id",
    "product_name",
]
FEATURE_GROUP_MEANINGS = {
    "discount": "折扣强度是模型识别亏损风险的重要预测信号。",
    "sales_amount": "销售额规模可作为识别亏损风险的辅助信号，需要结合利润和订单结构解释。",
    "quantity": "购买数量可作为订单规模信号，需要结合商品和折扣策略复盘。",
    "unit_price": "单价水平可作为价格带信号，需要结合商品结构解释。",
    "category": "类目结构与亏损风险识别有关，需要结合商品组合复盘。",
    "sub_category": "子类目结构与亏损风险识别有关，需要结合商品结构复盘。",
    "segment": "客群字段可帮助识别风险分布差异，但不能单独解释为风险来源。",
    "region": "区域字段可帮助识别风险分布差异，需要结合当地业务背景解释。",
    "country": "国家字段可帮助识别市场差异，需要结合市场结构解释。",
    "productline": "产品线结构与亏损风险识别有关，需要结合商品结构复盘。",
    "deal_size": "交易规模字段可辅助识别不同订单层级的风险差异。",
    "order_status": "订单状态可辅助识别履约或取消相关风险，需要结合订单流程复盘。",
    "date_parts": "时间字段仅作为辅助信号，不能单独解释为风险来源。",
    "other": "该组特征可作为辅助预测信号，需要结合业务明细解释。",
}


def _skip_result(warnings: list[str], summary_metrics: dict[str, object] | None = None) -> ModuleResult:
    return ModuleResult(
        module_id=MODULE_ID,
        title=TITLE,
        chart_type=CHART_TYPE,
        summary_metrics=summary_metrics or {},
        warnings=warnings,
    )


def _round(value: object, digits: int = 4) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(numeric):
        return 0.0
    return round(numeric, digits)


def _display_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            return value
    return value


def _schema_mapping(canonical_columns: dict[str, str]) -> SchemaMapping:
    return SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={
            source_field: canonical_field
            for canonical_field, source_field in canonical_columns.items()
            if source_field
        },
        confidence=1.0,
    )


def _canonical_frame(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> pd.DataFrame:
    canonical = pd.DataFrame(index=frame.index)
    for canonical_field, source_field in canonical_columns.items():
        if source_field in frame.columns:
            canonical[canonical_field] = frame[source_field]
    return canonical


def _model_candidates() -> list[tuple[str, object]]:
    return [
        (
            "DummyClassifier",
            DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        ),
        (
            "LogisticRegression",
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def _feature_columns(
    features: list[ModelingFeaturePlan],
    working: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    date_features: list[str] = []

    for feature in features:
        if feature.name in LEAKAGE_FIELDS or feature.name in HIGH_CARDINALITY_FIELDS:
            continue
        if feature.name not in working.columns:
            continue
        if feature.feature_type == "numeric":
            numeric_features.append(feature.name)
        elif feature.feature_type == "categorical":
            categorical_features.append(feature.name)
        elif feature.feature_type == "date":
            date_features.append(feature.name)
    return numeric_features, categorical_features, date_features


def _add_date_parts(
    working: pd.DataFrame,
    date_features: list[str],
) -> list[str]:
    date_part_features: list[str] = []
    for feature in date_features:
        parsed = pd.to_datetime(working[feature], errors="coerce")
        for part in DATE_PARTS:
            column = f"{feature}_{part}"
            if part == "year":
                working[column] = parsed.dt.year
            elif part == "month":
                working[column] = parsed.dt.month
            elif part == "quarter":
                working[column] = parsed.dt.quarter
            elif part == "weekday":
                working[column] = parsed.dt.weekday
            date_part_features.append(column)
    return date_part_features


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility for older sklearn releases.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:
        return []


def _feature_importance_rows(
    model_name: str,
    pipeline: Pipeline,
) -> list[dict[str, object]]:
    model = pipeline.named_steps["model"]
    names = _feature_names(pipeline)
    values: list[float] = []
    if hasattr(model, "feature_importances_"):
        values = [float(value) for value in model.feature_importances_]
    elif hasattr(model, "coef_"):
        values = [abs(float(value)) for value in model.coef_[0]]
    if not names or not values:
        return []
    rows = [
        {
            "model": model_name,
            "feature": name.split("__", 1)[-1],
            "importance": _round(value),
        }
        for name, value in zip(names, values, strict=False)
    ]
    return sorted(rows, key=lambda row: float(row["importance"]), reverse=True)[:15]


def _model_metrics(model_name: str, y_test: pd.Series, y_pred: Any, y_score: Any) -> dict[str, object]:
    row: dict[str, object] = {
        "model": model_name,
        "accuracy": _round(accuracy_score(y_test, y_pred)),
        "precision": _round(precision_score(y_test, y_pred, zero_division=0)),
        "recall": _round(recall_score(y_test, y_pred, zero_division=0)),
        "f1": _round(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": 0.0,
    }
    try:
        row["roc_auc"] = _round(roc_auc_score(y_test, y_score))
    except ValueError:
        row["roc_auc"] = 0.0
    return row


def _class_imbalance_level(positive_rate: float, positive_count: int, negative_count: int) -> str:
    if positive_count == 0 or negative_count == 0:
        return "single_class"
    minority_rate = min(positive_rate, 1 - positive_rate)
    if minority_rate < 0.01:
        return "extreme"
    if minority_rate < 0.05:
        return "high"
    if minority_rate < 0.2:
        return "moderate"
    return "balanced"


def _comparison_with_baseline_lift(comparison: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = next((row for row in comparison if row.get("model") == "DummyClassifier"), None)
    baseline_f1 = float(baseline.get("f1") or 0) if baseline else 0.0
    baseline_recall = float(baseline.get("recall") or 0) if baseline else 0.0
    rows: list[dict[str, object]] = []
    for row in comparison:
        enriched = dict(row)
        enriched["is_baseline"] = row.get("model") == "DummyClassifier"
        enriched["lift_over_baseline_f1"] = _round(float(row.get("f1") or 0) - baseline_f1)
        enriched["lift_over_baseline_recall"] = _round(float(row.get("recall") or 0) - baseline_recall)
        rows.append(enriched)
    return rows


def _weak_reasons(
    *,
    target_positive_rate: float,
    test_positive_count: int,
    precision: float,
    recall: float,
    f1: float,
    lift_over_baseline_f1: float,
) -> list[str]:
    reasons: list[str] = []
    if target_positive_rate < 0.01:
        reasons.append("extreme_class_imbalance")
    if test_positive_count < 20:
        reasons.append("too_few_test_positives")
    if precision < 0.1:
        reasons.append("low_precision")
    if f1 < 0.25:
        reasons.append("low_f1")
    if recall < 0.5:
        reasons.append("low_recall")
    if lift_over_baseline_f1 < 0.05:
        reasons.append("low_baseline_lift")
    return reasons


def _model_quality_status(
    *,
    target_positive_rate: float,
    test_positive_count: int,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float,
    lift_over_baseline_f1: float,
) -> tuple[str, list[str]]:
    reasons = _weak_reasons(
        target_positive_rate=target_positive_rate,
        test_positive_count=test_positive_count,
        precision=precision,
        recall=recall,
        f1=f1,
        lift_over_baseline_f1=lift_over_baseline_f1,
    )
    if reasons:
        return "weak", reasons
    if test_positive_count >= 50 and precision >= 0.3 and recall >= 0.7 and f1 >= 0.5 and roc_auc >= 0.75:
        return "strong", []
    return "usable", []


def _feature_plan_rows(features: list[ModelingFeaturePlan]) -> list[dict[str, object]]:
    return [
        {
            "feature": feature.name,
            "feature_type": "blocked" if feature.feature_type == "unknown" else feature.feature_type,
            "reason": feature.reason,
        }
        for feature in features
    ]


def _confusion_rows(y_test: pd.Series, y_pred: Any) -> list[dict[str, object]]:
    labels = [False, True]
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    rows: list[dict[str, object]] = []
    for actual_index, actual in enumerate(labels):
        for predicted_index, predicted in enumerate(labels):
            rows.append(
                {
                    "actual": "loss" if actual else "non_loss",
                    "predicted": "loss" if predicted else "non_loss",
                    "count": int(matrix[actual_index][predicted_index]),
                }
            )
    return rows


def _prediction_rows(
    display_frame: pd.DataFrame,
    row_index,
    y_test: pd.Series,
    y_score: Any,
    y_pred: Any,
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    scored = display_frame.loc[row_index].copy()
    scored["actual_is_loss"] = y_test.astype(bool)
    scored["predicted_is_loss"] = [bool(value) for value in y_pred]
    scored["loss_probability"] = [float(score) for score in y_score]
    rows: list[dict[str, object]] = []
    for index, row in scored.sort_values("loss_probability", ascending=False).head(10).iterrows():
        output: dict[str, object] = {
            "row_index": int(index) if isinstance(index, int) else str(index),
            "loss_probability": _round(row["loss_probability"]),
            "actual_is_loss": bool(row["actual_is_loss"]),
            "predicted_is_loss": bool(row["predicted_is_loss"]),
        }
        for column in DISPLAY_FIELDS:
            if column in scored.columns:
                output[column] = _display_value(row[column])
        rows.append(output)
    return rows[:limit]


def _filtered_prediction_rows(
    display_frame: pd.DataFrame,
    row_index,
    y_test: pd.Series,
    y_score: Any,
    y_pred: Any,
    *,
    predicted: bool,
    actual: bool,
) -> list[dict[str, object]]:
    scored = pd.DataFrame(
        {
            "actual_is_loss": y_test.astype(bool),
            "predicted_is_loss": [bool(value) for value in y_pred],
            "loss_probability": [float(score) for score in y_score],
        },
        index=row_index,
    )
    selected_index = scored[
        (scored["predicted_is_loss"] == predicted) & (scored["actual_is_loss"] == actual)
    ].index
    if selected_index.empty:
        return []
    return _prediction_rows(
        display_frame,
        selected_index,
        scored.loc[selected_index, "actual_is_loss"],
        scored.loc[selected_index, "loss_probability"],
        scored.loc[selected_index, "predicted_is_loss"],
    )


def _threshold_analysis_rows(y_test: pd.Series, y_score: Any) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    test_rows = max(1, int(len(y_test)))
    scores = pd.Series([float(score) for score in y_score], index=y_test.index)
    for threshold in THRESHOLDS:
        y_pred = scores >= threshold
        predicted_loss_count = int(y_pred.sum())
        rows.append(
            {
                "threshold": threshold,
                "precision": _round(precision_score(y_test, y_pred, zero_division=0)),
                "recall": _round(recall_score(y_test, y_pred, zero_division=0)),
                "f1": _round(f1_score(y_test, y_pred, zero_division=0)),
                "predicted_loss_count": predicted_loss_count,
                "review_load_rate": _round(predicted_loss_count / test_rows),
            }
        )
    return rows


def _feature_group(feature: str) -> str:
    if feature == "discount" or feature.startswith("discount"):
        return "discount"
    if feature == "sales_amount":
        return "sales_amount"
    if feature == "quantity":
        return "quantity"
    if feature == "unit_price":
        return "unit_price"
    for prefix, group in [
        ("category_", "category"),
        ("sub_category_", "sub_category"),
        ("segment_", "segment"),
        ("region_", "region"),
        ("country_", "country"),
        ("productline_", "productline"),
        ("deal_size_", "deal_size"),
        ("order_status_", "order_status"),
        ("order_datetime_", "date_parts"),
    ]:
        if feature.startswith(prefix):
            return group
    return "other"


def _feature_importance_grouped_rows(feature_importance: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in feature_importance:
        feature = str(row.get("feature", ""))
        group = _feature_group(feature)
        bucket = grouped.setdefault(group, {"feature_group": group, "total_importance": 0.0, "features": []})
        bucket["total_importance"] = float(bucket["total_importance"]) + float(row.get("importance") or 0)
        bucket["features"].append((feature, float(row.get("importance") or 0)))

    result: list[dict[str, object]] = []
    for group, bucket in grouped.items():
        top_features = [
            feature
            for feature, _importance in sorted(
                bucket["features"],
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]
        result.append(
            {
                "feature_group": group,
                "total_importance": _round(bucket["total_importance"]),
                "top_features": ", ".join(top_features),
                "business_meaning": FEATURE_GROUP_MEANINGS.get(group, FEATURE_GROUP_MEANINGS["other"]),
            }
        )
    return sorted(result, key=lambda row: float(row["total_importance"]), reverse=True)


def _findings(
    best_row: dict[str, object],
    feature_importance: list[dict[str, object]],
) -> list[str]:
    top_features = ", ".join(str(row["feature"]) for row in feature_importance[:3]) or "暂无稳定特征"
    return [
        f"最佳模型为 {best_row['model']}，Recall={best_row['recall']}，F1={best_row['f1']}，ROC AUC={best_row['roc_auc']}。",
        f"recall 用于衡量亏损样本被识别出来的比例；F1 平衡召回率和精确率；ROC AUC 衡量模型区分亏损与非亏损记录的整体能力。",
        f"当前主要风险特征包括：{top_features}。",
        "阈值取舍上，低阈值通常提升召回但增加人工复核量，高阈值减少复核量但可能漏掉更多亏损订单。",
        "错误分析上，false negatives 是最需要关注的漏判亏损样本，false positives 是复核成本来源。",
        "该模型只能用于亏损风险预警和人工复核优先级排序，不代表特征与亏损之间存在因果关系。",
    ]


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    try:
        schema_mapping = _schema_mapping(canonical_columns)
        plan = build_modeling_plan(frame, schema_mapping)
        if plan.status != "runnable":
            return _skip_result([f"modeling_plan_skipped: {reason}" for reason in plan.skip_reasons])

        working = _canonical_frame(frame, canonical_columns)
        if "profit" not in working.columns:
            return _skip_result(["missing_profit_field"])
        working["profit"] = pd.to_numeric(working["profit"], errors="coerce")
        working = working.dropna(subset=["profit"]).copy()
        if working.empty:
            return _skip_result(["no_valid_profit_values"])

        y = working["profit"] < 0
        numeric_features, categorical_features, date_features = _feature_columns(
            plan.allowed_features,
            working,
        )
        numeric_features = numeric_features + _add_date_parts(working, date_features)
        feature_columns = numeric_features + categorical_features
        if len(feature_columns) < 2:
            return _skip_result(["insufficient_allowed_features"])

        X = working[feature_columns].copy()
        display_frame = working.copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        comparison: list[dict[str, object]] = []
        warnings: list[str] = []
        fitted_models: dict[str, Pipeline] = {}
        scores_by_model: dict[str, Any] = {}
        predictions_by_model: dict[str, Any] = {}

        for model_name, model in _model_candidates():
            try:
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", _preprocessor(numeric_features, categorical_features)),
                        ("model", model),
                    ]
                )
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                if hasattr(pipeline, "predict_proba"):
                    y_score = pipeline.predict_proba(X_test)[:, 1]
                else:
                    y_score = y_pred
                comparison.append(_model_metrics(model_name, y_test, y_pred, y_score))
                fitted_models[model_name] = pipeline
                scores_by_model[model_name] = y_score
                predictions_by_model[model_name] = y_pred
            except Exception as exc:
                warnings.append(f"sklearn_error: {model_name}: {type(exc).__name__}: {exc}")

        if not comparison:
            return _skip_result(warnings or ["sklearn_error: no model could be fitted"])

        comparison = _comparison_with_baseline_lift(comparison)
        non_baseline_comparison = [row for row in comparison if not row.get("is_baseline")]
        best_row = max(
            non_baseline_comparison or comparison,
            key=lambda row: (
                float(row.get("recall") or 0),
                float(row.get("f1") or 0),
                float(row.get("roc_auc") or 0),
            ),
        )
        best_model_name = str(best_row["model"])
        best_pipeline = fitted_models[best_model_name]
        best_scores = scores_by_model[best_model_name]
        best_predictions = pd.Series(best_scores, index=X_test.index) >= DEFAULT_THRESHOLD
        feature_importance = _feature_importance_rows(best_model_name, best_pipeline)
        try:
            threshold_analysis = _threshold_analysis_rows(y_test, best_scores)
        except Exception as exc:
            threshold_analysis = []
            warnings.append(f"threshold_analysis_error: {type(exc).__name__}: {exc}")
        false_positive_examples = _filtered_prediction_rows(
            display_frame,
            X_test.index,
            y_test,
            best_scores,
            best_predictions,
            predicted=True,
            actual=False,
        )
        false_negative_examples = _filtered_prediction_rows(
            display_frame,
            X_test.index,
            y_test,
            best_scores,
            best_predictions,
            predicted=False,
            actual=True,
        )
        matrix = confusion_matrix(y_test, best_predictions, labels=[False, True])
        target_positive_count = int(y.sum())
        target_negative_count = int((~y).sum())
        train_positive_count = int(y_train.sum())
        train_negative_count = int((~y_train).sum())
        test_positive_count = int(y_test.sum())
        test_negative_count = int((~y_test).sum())
        target_positive_rate = _round(y.mean(), digits=6)
        model_quality_status, weak_reasons = _model_quality_status(
            target_positive_rate=target_positive_rate,
            test_positive_count=test_positive_count,
            precision=float(best_row.get("precision") or 0),
            recall=float(best_row.get("recall") or 0),
            f1=float(best_row.get("f1") or 0),
            roc_auc=float(best_row.get("roc_auc") or 0),
            lift_over_baseline_f1=float(best_row.get("lift_over_baseline_f1") or 0),
        )
        tables = {
            "model_comparison": comparison,
            "confusion_matrix": _confusion_rows(y_test, best_predictions),
            "feature_importance": feature_importance,
            "feature_importance_grouped": _feature_importance_grouped_rows(feature_importance),
            "safe_features": _feature_plan_rows(plan.allowed_features),
            "blocked_features": _feature_plan_rows(plan.blocked_features),
            "feature_engineering_steps": plan.feature_engineering_steps,
            "threshold_analysis": threshold_analysis,
            "high_risk_examples": _prediction_rows(
                display_frame,
                X_test.index,
                y_test,
                best_scores,
                best_predictions,
            ),
            "false_positive_examples": false_positive_examples,
            "false_negative_examples": false_negative_examples,
        }
        summary_metrics: dict[str, object] = {
            "target_positive_count": target_positive_count,
            "target_negative_count": target_negative_count,
            "target_positive_rate": target_positive_rate,
            "train_positive_count": train_positive_count,
            "train_negative_count": train_negative_count,
            "test_positive_count": test_positive_count,
            "test_negative_count": test_negative_count,
            "class_imbalance_level": _class_imbalance_level(
                target_positive_rate,
                target_positive_count,
                target_negative_count,
            ),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "best_model": best_model_name,
            "best_precision": best_row.get("precision"),
            "best_recall": best_row["recall"],
            "best_f1": best_row["f1"],
            "best_roc_auc": best_row["roc_auc"],
            "best_lift_over_baseline_f1": best_row.get("lift_over_baseline_f1"),
            "best_lift_over_baseline_recall": best_row.get("lift_over_baseline_recall"),
            "model_quality_status": model_quality_status,
            "weak_reasons": weak_reasons,
            "business_priority_metric": "recall",
            "threshold_default": DEFAULT_THRESHOLD,
            "true_negative_count": int(matrix[0][0]),
            "false_positive_count": int(matrix[0][1]),
            "false_negative_count": int(matrix[1][0]),
            "true_positive_count": int(matrix[1][1]),
        }
        return ModuleResult(
            module_id=MODULE_ID,
            title=TITLE,
            chart_type=CHART_TYPE,
            summary_metrics=summary_metrics,
            tables=tables,
            findings=_findings(best_row, feature_importance),
            warnings=warnings,
        )
    except Exception as exc:
        return _skip_result([f"modeling_module_error: {type(exc).__name__}: {exc}"])
