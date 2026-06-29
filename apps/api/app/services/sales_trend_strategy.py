from __future__ import annotations

from typing import Any

from app.schemas.report import ModuleReport

ALLOWED_CHARTS = [
    "daily_rolling_line",
    "monthly_line",
    "weekday_bar",
    "year_month_heatmap",
]


def _fallback_views(module: ModuleReport) -> list[dict[str, object]]:
    summary = module.summary_metrics
    daily_count = int(summary.get("daily_grain_count") or len(module.tables.get("daily_totals", [])))
    monthly_count = int(summary.get("monthly_grain_count") or len(module.tables.get("monthly_totals", [])))
    year_count = int(summary.get("year_count") or 0)
    has_multi_year = bool(summary.get("has_multi_year"))

    views: list[dict[str, object]] = []
    if daily_count >= 14:
        views.append(
            {
                "chart": "daily_rolling_line",
                "rolling_window_days": 30 if daily_count >= 30 else 7,
                "reason": (
                    "Use daily sales with a rolling average to separate baseline direction "
                    "from short-term volatility."
                ),
            }
        )
    if monthly_count >= 2:
        views.append(
            {
                "chart": "monthly_line",
                "reason": (
                    "Use a monthly line to compare medium-term trend and avoid reading too much "
                    "into daily noise."
                ),
            }
        )
    if daily_count >= 14 and module.tables.get("weekday_profile"):
        views.append(
            {
                "chart": "weekday_bar",
                "reason": "Compare weekday averages to see whether weekly rhythm is stable.",
            }
        )
    if (has_multi_year or year_count >= 2) and monthly_count >= 12 and module.tables.get("year_month_totals"):
        views.append(
            {
                "chart": "year_month_heatmap",
                "reason": (
                    "Use a year-month heatmap when coverage spans multiple years so seasonality "
                    "or repeated peaks become easier to spot."
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


def build_sales_trend_strategy(
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

    suggest_strategy = getattr(llm_client, "suggest_sales_trend_strategy", None)
    if suggest_strategy is None:
        return fallback

    payload = {
        "summary_metrics": module.summary_metrics,
        "tables": {
            "monthly_totals": module.tables.get("monthly_totals", []),
            "weekday_profile": module.tables.get("weekday_profile", []),
            "year_month_totals": module.tables.get("year_month_totals", []),
        },
        "allowed_charts": ALLOWED_CHARTS,
        "business_rules": [
            "Do not use a year-month heatmap unless the data spans at least two calendar years.",
            "Prefer a daily rolling line when there are enough daily points to separate baseline and noise.",
            "Use weekday bars only when there is enough daily coverage to make weekday comparison meaningful.",
        ],
    }
    try:
        return _sanitize_strategy(suggest_strategy(payload), fallback)
    except Exception:
        return fallback
