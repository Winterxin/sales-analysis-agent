from __future__ import annotations

from typing import Any

from app.schemas.report import ModuleReport


ALLOWED_CHARTS = [
    "discount_profit_scatter",
    "discount_bucket_boxplot",
    "bucket_profit_quality_bar",
    "high_sales_low_profit_bar",
    "category_discount_risk_heatmap",
]


def _table_rows(module: ModuleReport, table_name: str) -> list[dict[str, object]]:
    return [row for row in module.tables.get(table_name, []) if isinstance(row, dict)]


def _fallback_views(module: ModuleReport) -> list[dict[str, object]]:
    bucket_rows = (
        _table_rows(module, "discount_threshold_candidates")
        or _table_rows(module, "discount_profit_risk_buckets")
        or _table_rows(module, "discount_buckets")
    )
    high_sales_low_profit_rows = _table_rows(module, "high_sales_low_profit_items")
    category_risk_rows = _table_rows(module, "category_discount_risk")
    has_discount_payload = module.chart_payload.get("series_name") == "discount_vs_profit"

    views: list[dict[str, object]] = []
    if has_discount_payload or bucket_rows:
        views.append(
            {
                "chart": "discount_profit_scatter",
                "reason": (
                    "Use a discount-profit scatter plot to see whether higher discounts "
                    "cluster around negative profit orders."
                ),
            }
        )
    if bucket_rows:
        views.extend(
            [
                {
                    "chart": "discount_bucket_boxplot",
                    "reason": (
                        "Use a bucketed boxplot to compare profit spread and outliers across "
                        "discount levels."
                    ),
                },
                {
                    "chart": "bucket_profit_quality_bar",
                    "reason": (
                        "Compare average profit with loss rate by discount bucket to locate "
                        "the discount threshold where profit quality breaks."
                    ),
                },
            ]
        )
    if high_sales_low_profit_rows:
        views.append(
            {
                "chart": "high_sales_low_profit_bar",
                "reason": (
                    "Show high-sales low-profit items so the notebook can move from an "
                    "aggregate discount pattern to concrete SKU actions."
                ),
            }
        )
    if category_risk_rows:
        views.append(
            {
                "chart": "category_discount_risk_heatmap",
                "reason": (
                    "Use category x discount bucket risk to see whether discount erosion is "
                    "concentrated in specific categories."
                ),
            }
        )
    return views


def _sanitize_strategy(payload: Any, fallback: dict[str, object]) -> dict[str, object]:
    if not isinstance(payload, dict):
        return fallback
    raw_views = payload.get("views")
    if not isinstance(raw_views, list):
        return fallback

    fallback_by_chart = {
        str(view.get("chart")): view
        for view in fallback.get("views", [])
        if isinstance(view, dict) and view.get("chart")
    }
    sanitized_views: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_view in raw_views:
        if not isinstance(raw_view, dict):
            continue
        chart = str(raw_view.get("chart", "")).strip()
        fallback_view = fallback_by_chart.get(chart)
        if chart not in ALLOWED_CHARTS or fallback_view is None or chart in seen:
            continue
        sanitized_views.append(
            {
                **fallback_view,
                "reason": str(raw_view.get("reason") or fallback_view.get("reason", "")),
            }
        )
        seen.add(chart)

    for chart, fallback_view in fallback_by_chart.items():
        if chart not in seen:
            sanitized_views.append(fallback_view)

    return {
        "source": "llm_sanitized" if sanitized_views else "deterministic_fallback",
        "allowed_charts": ALLOWED_CHARTS,
        "views": sanitized_views or fallback.get("views", []),
    }


def build_discount_profit_strategy(
    module: ModuleReport | None,
    llm_client=None,
) -> dict[str, object]:
    if module is None:
        return {"source": "empty", "allowed_charts": ALLOWED_CHARTS, "views": []}

    fallback = {
        "source": "deterministic_fallback",
        "allowed_charts": ALLOWED_CHARTS,
        "views": _fallback_views(module),
    }
    if llm_client is None or not getattr(llm_client, "enabled", False):
        return fallback

    suggest_strategy = getattr(llm_client, "suggest_discount_profit_strategy", None)
    if suggest_strategy is None:
        return fallback

    payload = {
        "summary_metrics": module.summary_metrics,
        "tables": {
            "discount_threshold_candidates": module.tables.get(
                "discount_threshold_candidates",
                [],
            ),
            "discount_profit_risk_buckets": module.tables.get(
                "discount_profit_risk_buckets",
                module.tables.get("discount_buckets", []),
            ),
            "high_sales_low_profit_items": module.tables.get(
                "high_sales_low_profit_items", []
            ),
            "category_discount_risk": module.tables.get("category_discount_risk", []),
        },
        "allowed_charts": ALLOWED_CHARTS,
        "business_rules": [
            "Do not generate Python code.",
            "Choose only charts from allowed_charts.",
            "Prefer bucket_profit_quality_bar when discount buckets show negative profit or high loss rate.",
            "Use high_sales_low_profit_bar only when concrete item rows are available.",
            "Use category_discount_risk_heatmap only when category risk rows are available.",
        ],
    }
    try:
        return _sanitize_strategy(suggest_strategy(payload), fallback)
    except Exception:
        return fallback
