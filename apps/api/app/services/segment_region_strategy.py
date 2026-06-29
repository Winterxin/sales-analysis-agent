from __future__ import annotations

from typing import Any

from app.schemas.report import ModuleReport


ALLOWED_CHARTS = [
    "dimension_sales_bar",
    "segment_region_sales_heatmap",
    "segment_region_profit_margin_heatmap",
    "segment_profit_quality_bar",
    "segment_region_bubble",
    "weak_segment_region_bar",
]
MAX_LLM_VIEWS = 4
FALLBACK_VIEW_PRIORITY = [
    "segment_region_sales_heatmap",
    "segment_region_profit_margin_heatmap",
    "segment_region_bubble",
    "weak_segment_region_bar",
    "segment_profit_quality_bar",
    "dimension_sales_bar",
]


def _table_rows(module: ModuleReport, table_name: str) -> list[dict[str, object]]:
    return [row for row in module.tables.get(table_name, []) if isinstance(row, dict)]


def _row_has_any(row: dict[str, object], keys: set[str]) -> bool:
    return any(key in row for key in keys)


def _fallback_views(module: ModuleReport) -> list[dict[str, object]]:
    dimension_totals = _table_rows(module, "dimension_totals")
    matrix_rows = _table_rows(module, "segment_region_matrix")
    primary_profit_quality = _table_rows(module, "primary_profit_quality")
    weak_cuts = _table_rows(module, "weak_performance_cuts")

    views: list[dict[str, object]] = []
    if dimension_totals:
        views.append(
            {
                "chart": "dimension_sales_bar",
                "reason": "Compare primary segment or region sales scale before judging profitability.",
            }
        )
    if matrix_rows:
        views.append(
            {
                "chart": "segment_region_sales_heatmap",
                "reason": "Use a two-dimensional sales heatmap to locate strong and weak segment-region combinations.",
            }
        )
    if matrix_rows and _row_has_any(matrix_rows[0], {"Profit", "profit", "profit_margin"}):
        views.extend(
            [
                {
                    "chart": "segment_region_profit_margin_heatmap",
                    "reason": "Compare margin quality across segment-region combinations, not only sales size.",
                },
                {
                    "chart": "segment_region_bubble",
                    "reason": "Bridge sales and profit so high-scale but low-margin cuts become visible.",
                },
            ]
        )
    if primary_profit_quality and _row_has_any(primary_profit_quality[0], {"Profit", "profit", "profit_margin"}):
        views.append(
            {
                "chart": "segment_profit_quality_bar",
                "reason": "Rank primary dimensions by profit margin to avoid mistaking scale for quality.",
            }
        )
    if weak_cuts:
        views.append(
            {
                "chart": "weak_segment_region_bar",
                "reason": "Highlight concrete high-sales low-profit cuts for follow-up action.",
            }
        )
    priority = {chart: index for index, chart in enumerate(FALLBACK_VIEW_PRIORITY)}
    views.sort(key=lambda view: priority.get(str(view.get("chart")), len(priority)))
    return views[:MAX_LLM_VIEWS]


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

    if sanitized_views:
        return {
            "source": "llm_sanitized",
            "allowed_charts": ALLOWED_CHARTS,
            "views": sanitized_views[:MAX_LLM_VIEWS],
        }

    return fallback


def build_segment_region_strategy(
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

    suggest_strategy = getattr(llm_client, "suggest_segment_region_strategy", None)
    if suggest_strategy is None:
        return fallback

    payload = {
        "summary_metrics": module.summary_metrics,
        "tables": {
            "dimension_totals": module.tables.get("dimension_totals", []),
            "segment_region_matrix": module.tables.get("segment_region_matrix", []),
            "primary_profit_quality": module.tables.get("primary_profit_quality", []),
            "weak_performance_cuts": module.tables.get("weak_performance_cuts", []),
        },
        "allowed_charts": ALLOWED_CHARTS,
        "business_rules": [
            "Do not generate Python code.",
            "Choose only charts from allowed_charts.",
            "Choose at most 4 charts; prefer 3 when the story is already clear.",
            "Avoid choosing both dimension_sales_bar and segment_region_sales_heatmap unless scale ranking itself is the main story.",
            "Prefer profit-margin heatmap, bubble, and weak-cut views when profit is available.",
            "Prefer heatmaps when two-dimensional segment-region rows are available.",
            "Prefer profit-margin views when profit columns are available.",
            "Use weak_segment_region_bar only when concrete weak cut rows exist.",
        ],
    }
    try:
        return _sanitize_strategy(suggest_strategy(payload), fallback)
    except Exception:
        return fallback
