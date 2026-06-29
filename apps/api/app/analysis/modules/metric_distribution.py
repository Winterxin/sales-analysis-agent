from __future__ import annotations

import math

import pandas as pd

from app.analysis.contracts import ModuleResult


CORE_METRICS = {
    "sales_amount": "销售额",
    "quantity": "销量",
    "discount": "折扣",
    "profit": "利润",
}

RELATIONSHIP_PRIORITIES = {
    frozenset({"discount", "profit"}): 1.0,
    frozenset({"sales_amount", "profit"}): 0.9,
    frozenset({"sales_amount", "quantity"}): 0.8,
    frozenset({"discount", "sales_amount"}): 0.7,
    frozenset({"discount", "quantity"}): 0.55,
    frozenset({"quantity", "profit"}): 0.5,
}

RELATIONSHIP_COLOR_PREFERENCES = {
    frozenset({"discount", "profit"}): "sales_amount",
    frozenset({"sales_amount", "profit"}): "discount",
    frozenset({"sales_amount", "quantity"}): "discount",
    frozenset({"discount", "sales_amount"}): "profit",
    frozenset({"discount", "quantity"}): "profit",
    frozenset({"quantity", "profit"}): "discount",
}


def _safe_float(value: object, digits: int = 4) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return round(numeric, digits)


def _safe_abs_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return _safe_float(abs(numerator) / abs(denominator))


def _profile_metric(
    frame: pd.DataFrame,
    canonical_name: str,
    column_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    series = pd.to_numeric(frame[column_name], errors="coerce").dropna()
    label = CORE_METRICS.get(canonical_name, canonical_name)

    if series.empty:
        empty_quantiles = {
            "metric": canonical_name,
            "column": column_name,
            "label": label,
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "skew": None,
            "mean_median_ratio": None,
            "p99_median_ratio": None,
            "non_negative": None,
        }
        empty_outliers = {
            "metric": canonical_name,
            "column": column_name,
            "label": label,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "iqr_low": None,
            "iqr_high": None,
        }
        return empty_quantiles, empty_outliers

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outlier_mask = (series < low) | (series > high)
    outlier_count = int(outlier_mask.sum())
    mean = _safe_float(series.mean())
    median = _safe_float(series.median())
    p99 = _safe_float(series.quantile(0.99))

    quantiles = {
        "metric": canonical_name,
        "column": column_name,
        "label": label,
        "count": int(series.count()),
        "mean": mean,
        "median": median,
        "p90": _safe_float(series.quantile(0.9)),
        "p95": _safe_float(series.quantile(0.95)),
        "p99": p99,
        "min": _safe_float(series.min()),
        "max": _safe_float(series.max()),
        "skew": _safe_float(series.skew()),
        "mean_median_ratio": _safe_abs_ratio(mean, median),
        "p99_median_ratio": _safe_abs_ratio(p99, median),
        "non_negative": bool(series.min() >= 0),
    }
    outliers = {
        "metric": canonical_name,
        "column": column_name,
        "label": label,
        "outlier_count": outlier_count,
        "outlier_ratio": _safe_float(outlier_count / int(series.count()), digits=6) or 0.0,
        "iqr_low": _safe_float(low),
        "iqr_high": _safe_float(high),
    }
    return quantiles, outliers


def _preferred_color_metric(
    x_metric: str,
    y_metric: str,
    available_metric_names: list[str],
) -> str | None:
    preferred = RELATIONSHIP_COLOR_PREFERENCES.get(frozenset({x_metric, y_metric}))
    if preferred and preferred in available_metric_names and preferred not in {x_metric, y_metric}:
        return preferred
    for metric_name in available_metric_names:
        if metric_name not in {x_metric, y_metric}:
            return metric_name
    return None


def _relationship_rows(
    frame: pd.DataFrame,
    available_metrics: list[tuple[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    numeric_frame = pd.DataFrame(
        {
            canonical_name: pd.to_numeric(frame[column_name], errors="coerce")
            for canonical_name, column_name in available_metrics
        }
    )
    available_metric_names = [canonical_name for canonical_name, _ in available_metrics]
    metric_columns = {canonical_name: column_name for canonical_name, column_name in available_metrics}
    metric_labels = {
        canonical_name: CORE_METRICS.get(canonical_name, canonical_name)
        for canonical_name, _ in available_metrics
    }

    correlation_rows: list[dict[str, object]] = []
    for index, (x_metric, x_column) in enumerate(available_metrics):
        for y_metric, y_column in available_metrics[index + 1 :]:
            pair_frame = numeric_frame[[x_metric, y_metric]].dropna()
            if len(pair_frame) < 3:
                continue
            if pair_frame[x_metric].nunique(dropna=True) <= 1:
                continue
            if pair_frame[y_metric].nunique(dropna=True) <= 1:
                continue

            correlation = _safe_float(pair_frame[x_metric].corr(pair_frame[y_metric]))
            if correlation is None:
                continue

            color_metric = _preferred_color_metric(
                x_metric,
                y_metric,
                available_metric_names,
            )
            pair_key = frozenset({x_metric, y_metric})
            correlation_rows.append(
                {
                    "x_metric": x_metric,
                    "x_column": x_column,
                    "x_label": metric_labels.get(x_metric, x_metric),
                    "y_metric": y_metric,
                    "y_column": y_column,
                    "y_label": metric_labels.get(y_metric, y_metric),
                    "correlation": correlation,
                    "abs_correlation": _safe_float(abs(correlation)),
                    "business_priority": RELATIONSHIP_PRIORITIES.get(pair_key, 0.35),
                    "observation_count": int(len(pair_frame)),
                    "color_metric": color_metric,
                    "color_column": (
                        metric_columns.get(color_metric) if color_metric is not None else None
                    ),
                    "color_label": (
                        metric_labels.get(color_metric, color_metric)
                        if color_metric is not None
                        else None
                    ),
                }
            )

    relationship_candidates = sorted(
        correlation_rows,
        key=lambda row: (
            -float(row.get("business_priority") or 0),
            -float(row.get("abs_correlation") or 0),
            -int(row.get("observation_count") or 0),
            str(row.get("x_metric") or ""),
            str(row.get("y_metric") or ""),
        ),
    )[:3]
    return correlation_rows, relationship_candidates


def run(frame: pd.DataFrame, canonical_columns: dict[str, str]) -> ModuleResult:
    available_metrics = [
        (canonical_name, column_name)
        for canonical_name, column_name in canonical_columns.items()
        if canonical_name in CORE_METRICS and column_name in frame.columns
    ]

    quantile_rows: list[dict[str, object]] = []
    outlier_rows: list[dict[str, object]] = []
    for canonical_name, column_name in available_metrics:
        quantiles, outliers = _profile_metric(frame, canonical_name, column_name)
        quantile_rows.append(quantiles)
        outlier_rows.append(outliers)
    correlation_rows, relationship_candidates = _relationship_rows(frame, available_metrics)

    metric_count = len(quantile_rows)
    highest_outlier = max(
        outlier_rows,
        key=lambda row: float(row.get("outlier_ratio") or 0),
        default={},
    )
    highest_tail = max(
        quantile_rows,
        key=lambda row: float(row.get("p99") or 0) / max(float(row.get("median") or 0), 1e-9),
        default={},
    )
    strongest_relationship = max(
        correlation_rows,
        key=lambda row: float(row.get("abs_correlation") or 0),
        default={},
    )

    findings = [
        f"已完成 {metric_count} 个核心数值指标的分布画像。",
    ]
    if highest_outlier:
        findings.append(
            "异常值占比最高的指标是 "
            f"{highest_outlier.get('label')}，占比约 {highest_outlier.get('outlier_ratio')}。"
        )
    if highest_tail:
        findings.append(
            "长尾特征最需要优先复盘的指标是 "
            f"{highest_tail.get('label')}，其 P99 与中位数差距较大。"
        )
    if strongest_relationship:
        findings.append(
            "数值关系中最值得优先拆解的是 "
            f"{strongest_relationship.get('x_label')} 与 {strongest_relationship.get('y_label')}，"
            f"相关程度约为 {strongest_relationship.get('correlation')}。"
        )

    return ModuleResult(
        module_id="metric_distribution_analysis",
        title="指标分布分析",
        chart_type="histogram",
        summary_metrics={
            "metric_count": metric_count,
            "highest_outlier_metric": highest_outlier.get("metric"),
            "highest_tail_metric": highest_tail.get("metric"),
            "strongest_relationship_pair": (
                f"{strongest_relationship.get('x_metric')}__{strongest_relationship.get('y_metric')}"
                if strongest_relationship
                else None
            ),
        },
        tables={
            "metric_quantiles": quantile_rows,
            "metric_outliers": outlier_rows,
            "metric_correlations": correlation_rows,
            "relationship_candidates": relationship_candidates,
        },
        chart_payload={
            "metrics": [row["metric"] for row in quantile_rows],
            "labels": [row["label"] for row in quantile_rows],
        },
        findings=findings,
    )
