from __future__ import annotations

from typing import Any

from app.schemas.report import ModuleReport

ALLOWED_CHARTS = [
    "raw_histogram",
    "log_histogram",
    "p99_clipped_histogram",
    "boxplot",
    "quantile_markers",
    "correlation_heatmap",
    "scatter_relationships",
]

ALLOWED_DECISIONS = {
    "use_raw",
    "use_log",
    "use_p99_clipped",
    "use_log_and_p99_clipped",
    "use_boxplot_only",
}

ALLOWED_RELATIONSHIP_CHARTS = {
    "correlation_heatmap",
    "scatter_relationships",
}


def _rows_by_metric(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(row.get("metric")): row
        for row in rows
        if isinstance(row, dict) and row.get("metric")
    }


def _float_value(row: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _fallback_metric_decision(
    quantile_row: dict[str, object],
    outlier_row: dict[str, object] | None,
) -> dict[str, object]:
    metric = str(quantile_row.get("metric", "unknown"))
    label = str(quantile_row.get("label") or metric)
    column = str(quantile_row.get("column") or label)
    mean_median_ratio = _float_value(quantile_row, "mean_median_ratio")
    p99_median_ratio = _float_value(quantile_row, "p99_median_ratio")
    non_negative = bool(quantile_row.get("non_negative"))
    outlier_ratio = _float_value(outlier_row or {}, "outlier_ratio")

    is_severely_long_tailed = (
        non_negative
        and (
            mean_median_ratio >= 2.0
            or p99_median_ratio >= 10.0
            or outlier_ratio >= 0.08
        )
    )
    if is_severely_long_tailed:
        return {
            "metric": metric,
            "column": column,
            "label": label,
            "decision": "use_log_and_p99_clipped",
            "charts": ["log_histogram", "p99_clipped_histogram", "quantile_markers"],
            "reason": (
                f"{label} is non-negative and strongly long-tailed: mean/median is "
                f"{mean_median_ratio:.2f}, P99/median is {p99_median_ratio:.2f}. "
                "Use transformed and clipped views instead of a raw histogram."
            ),
        }

    if not non_negative and (outlier_ratio >= 0.08 or p99_median_ratio >= 10.0):
        return {
            "metric": metric,
            "column": column,
            "label": label,
            "decision": "use_boxplot_only",
            "charts": ["boxplot", "quantile_markers"],
            "reason": (
                f"{label} can be negative and has meaningful outliers, so a boxplot "
                "is safer than log transformation."
            ),
        }

    return {
        "metric": metric,
        "column": column,
        "label": label,
        "decision": "use_raw",
        "charts": ["raw_histogram", "boxplot", "quantile_markers"],
        "reason": f"{label} does not show enough tail risk to require transformation.",
    }


def _relationship_pair_key(row: dict[str, object]) -> str | None:
    x_metric = str(row.get("x_metric", "")).strip()
    y_metric = str(row.get("y_metric", "")).strip()
    if not x_metric or not y_metric:
        return None
    return f"{x_metric}::{y_metric}"


def _fallback_relationship_views(module: ModuleReport) -> list[dict[str, object]]:
    quantile_rows = [
        row for row in module.tables.get("metric_quantiles", []) if isinstance(row, dict)
    ]
    correlation_rows = [
        row for row in module.tables.get("metric_correlations", []) if isinstance(row, dict)
    ]
    relationship_candidates = [
        row for row in module.tables.get("relationship_candidates", []) if isinstance(row, dict)
    ]
    metric_names = [
        str(row.get("metric"))
        for row in quantile_rows
        if row.get("metric")
    ]
    metric_labels = {
        str(row.get("metric")): str(row.get("label") or row.get("metric"))
        for row in quantile_rows
        if row.get("metric")
    }
    metric_count = int(module.summary_metrics.get("metric_count") or 0)

    views: list[dict[str, object]] = []
    if len(metric_names) >= 2 or metric_count >= 2:
        views.append(
            {
                "chart": "correlation_heatmap",
                "metrics": metric_names,
                "labels": [metric_labels.get(metric, metric) for metric in metric_names],
                "reason": (
                    "Use a correlation heatmap to scan which numeric metrics move together "
                    "before drilling into specific pairs."
                ),
            }
        )

    scatter_pairs = relationship_candidates or sorted(
        correlation_rows,
        key=lambda row: (
            -float(row.get("business_priority") or 0),
            -float(row.get("abs_correlation") or 0),
            -int(row.get("observation_count") or 0),
        ),
    )[:3]
    if scatter_pairs:
        views.append(
            {
                "chart": "scatter_relationships",
                "pairs": scatter_pairs,
                "reason": (
                    "Drill into the highest-priority metric pairs to see whether discount, "
                    "sales amount, quantity, and profit form visible relationship clusters."
                ),
            }
        )
    return views


def _fallback_strategy(module: ModuleReport) -> dict[str, object]:
    quantile_rows = [
        row for row in module.tables.get("metric_quantiles", []) if isinstance(row, dict)
    ]
    outlier_rows = _rows_by_metric(module.tables.get("metric_outliers", []))
    return {
        "source": "deterministic_fallback",
        "allowed_charts": ALLOWED_CHARTS,
        "relationship_views": _fallback_relationship_views(module),
        "decisions": [
            _fallback_metric_decision(row, outlier_rows.get(str(row.get("metric"))))
            for row in quantile_rows
        ],
    }


def _sanitize_relationship_views(
    payload_views: Any,
    fallback: dict[str, object],
) -> list[dict[str, object]]:
    fallback_views = [
        view for view in fallback.get("relationship_views", []) if isinstance(view, dict)
    ]
    if not isinstance(payload_views, list):
        return fallback_views

    fallback_by_chart = {
        str(view.get("chart")): view
        for view in fallback_views
        if view.get("chart")
    }
    scatter_fallback = fallback_by_chart.get("scatter_relationships", {})
    fallback_pairs_by_key = {
        key: pair
        for pair in scatter_fallback.get("pairs", [])
        if isinstance(pair, dict)
        for key in [_relationship_pair_key(pair)]
        if key is not None
    }

    sanitized_views: list[dict[str, object]] = []
    seen_charts: set[str] = set()
    for raw_view in payload_views:
        if not isinstance(raw_view, dict):
            continue
        chart = str(raw_view.get("chart", "")).strip()
        if chart not in ALLOWED_RELATIONSHIP_CHARTS or chart in seen_charts:
            continue
        fallback_view = fallback_by_chart.get(chart)
        if fallback_view is None:
            continue

        if chart == "correlation_heatmap":
            sanitized_views.append(
                {
                    **fallback_view,
                    "reason": str(raw_view.get("reason") or fallback_view.get("reason", "")),
                }
            )
            seen_charts.add(chart)
            continue

        raw_pairs = raw_view.get("pairs", [])
        sanitized_pairs: list[dict[str, object]] = []
        if isinstance(raw_pairs, list):
            for pair in raw_pairs:
                if not isinstance(pair, dict):
                    continue
                pair_key = _relationship_pair_key(pair)
                if pair_key is None or pair_key not in fallback_pairs_by_key:
                    continue
                if fallback_pairs_by_key[pair_key] in sanitized_pairs:
                    continue
                sanitized_pairs.append(fallback_pairs_by_key[pair_key])

        if sanitized_pairs:
            sanitized_views.append(
                {
                    **fallback_view,
                    "pairs": sanitized_pairs,
                    "reason": str(raw_view.get("reason") or fallback_view.get("reason", "")),
                }
            )
            seen_charts.add(chart)

    for fallback_view in fallback_views:
        chart = str(fallback_view.get("chart", "")).strip()
        if chart and chart not in seen_charts:
            sanitized_views.append(fallback_view)
    return sanitized_views


def _sanitize_strategy(payload: Any, fallback: dict[str, object]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return fallback
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        return fallback

    fallback_by_metric = {
        str(item.get("metric")): item
        for item in fallback.get("decisions", [])
        if isinstance(item, dict)
    }
    sanitized_decisions: list[dict[str, object]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        metric = str(decision.get("metric", "")).strip()
        fallback_decision = fallback_by_metric.get(metric)
        if fallback_decision is None:
            continue

        charts = [
            str(chart)
            for chart in decision.get("charts", [])
            if str(chart) in ALLOWED_CHARTS
        ]
        if not charts:
            charts = list(fallback_decision.get("charts", []))

        decision_name = str(decision.get("decision", "")).strip()
        if decision_name not in ALLOWED_DECISIONS:
            decision_name = str(fallback_decision.get("decision", "use_raw"))

        sanitized_decisions.append(
            {
                **fallback_decision,
                "decision": decision_name,
                "charts": charts,
                "reason": str(decision.get("reason") or fallback_decision.get("reason", "")),
            }
        )

    missing_metrics = set(fallback_by_metric) - {
        str(item.get("metric")) for item in sanitized_decisions
    }
    for metric in missing_metrics:
        sanitized_decisions.append(fallback_by_metric[metric])

    return {
        "source": "llm_sanitized" if sanitized_decisions else "deterministic_fallback",
        "allowed_charts": ALLOWED_CHARTS,
        "relationship_views": _sanitize_relationship_views(
            payload.get("relationship_views"),
            fallback,
        ),
        "decisions": sanitized_decisions or fallback.get("decisions", []),
    }


def build_metric_distribution_strategy(
    module: ModuleReport | None,
    llm_client=None,
) -> dict[str, object]:
    if module is None:
        return {
            "source": "empty",
            "allowed_charts": ALLOWED_CHARTS,
            "decisions": [],
            "relationship_views": [],
        }

    fallback = _fallback_strategy(module)
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return fallback

    suggest_strategy = getattr(llm_client, "suggest_metric_distribution_strategy", None)
    if suggest_strategy is None:
        return fallback

    payload = {
        "metric_diagnostics": module.tables.get("metric_quantiles", []),
        "metric_outliers": module.tables.get("metric_outliers", []),
        "metric_correlations": module.tables.get("metric_correlations", []),
        "relationship_candidates": module.tables.get("relationship_candidates", []),
        "allowed_charts": ALLOWED_CHARTS,
        "allowed_decisions": sorted(ALLOWED_DECISIONS),
        "business_rules": [
            "Do not treat right skew of non-negative sales amount as a standalone insight.",
            "Prefer log1p or P99-clipped views when the raw histogram would compress normal orders near zero.",
            "Use boxplot for metrics that can be negative, such as profit.",
            "Only choose relationship charts from the provided relationship_candidates and metric_correlations.",
            "Use a correlation heatmap for broad scanning, then use scatter relationships for the highest-priority pairs.",
        ],
    }
    try:
        return _sanitize_strategy(suggest_strategy(payload), fallback)
    except Exception:
        return fallback
