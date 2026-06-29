from __future__ import annotations

from typing import Any

from app.schemas.report import ModuleReport


ALLOWED_CHARTS = [
    "category_treemap",
    "top_product_bar",
    "product_pareto",
    "category_profit_margin_bar",
    "product_profit_bridge_scatter",
    "high_sales_low_profit_bar",
]
MAX_LLM_VIEWS = 4
FALLBACK_VIEW_PRIORITY = [
    "category_treemap",
    "product_pareto",
    "category_profit_margin_bar",
    "product_profit_bridge_scatter",
    "high_sales_low_profit_bar",
    "top_product_bar",
]


def _table_rows(module: ModuleReport, table_name: str) -> list[dict[str, object]]:
    return [row for row in module.tables.get(table_name, []) if isinstance(row, dict)]


def _row_has_any(row: dict[str, object], keys: set[str]) -> bool:
    return any(key in row for key in keys)


def _fallback_views(module: ModuleReport) -> list[dict[str, object]]:
    top_products = _table_rows(module, "top_products")
    category_sales = _table_rows(module, "category_sales")
    category_profit_quality = _table_rows(module, "category_profit_quality")
    high_sales_low_profit = _table_rows(module, "high_sales_low_profit_products")

    views: list[dict[str, object]] = []
    if category_sales:
        views.append(
            {
                "chart": "category_treemap",
                "reason": "Use category structure to see which categories and sub-categories carry sales volume.",
            }
        )
    if top_products:
        views.append(
            {
                "chart": "top_product_bar",
                "reason": "Show top products by sales to identify the main assortment contributors.",
            }
        )
        views.append(
            {
                "chart": "product_pareto",
                "reason": "Use Pareto contribution to judge whether sales depend on a small product set.",
            }
        )
    if category_profit_quality:
        views.append(
            {
                "chart": "category_profit_margin_bar",
                "reason": "Compare category profit margins so sales scale is not mistaken for healthy assortment quality.",
            }
        )
    if top_products and _row_has_any(top_products[0], {"Profit", "profit", "profit_margin"}):
        views.append(
            {
                "chart": "product_profit_bridge_scatter",
                "reason": "Bridge sales and profit to see whether top products also contribute margin.",
            }
        )
    if high_sales_low_profit:
        views.append(
            {
                "chart": "high_sales_low_profit_bar",
                "reason": "Highlight products that sell well but contribute weak profit for action planning.",
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


def build_product_category_strategy(
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

    suggest_strategy = getattr(llm_client, "suggest_product_category_strategy", None)
    if suggest_strategy is None:
        return fallback

    payload = {
        "summary_metrics": module.summary_metrics,
        "tables": {
            "top_products": module.tables.get("top_products", []),
            "category_sales": module.tables.get("category_sales", []),
            "category_profit_quality": module.tables.get("category_profit_quality", []),
            "high_sales_low_profit_products": module.tables.get(
                "high_sales_low_profit_products", []
            ),
        },
        "allowed_charts": ALLOWED_CHARTS,
        "business_rules": [
            "Do not generate Python code.",
            "Choose only charts from allowed_charts.",
            "Choose at most 4 charts; prefer 3 when the story is already clear.",
            "Do not choose both category_treemap and top_product_bar unless they answer clearly different questions.",
            "Avoid basic top bars when Pareto or profit bridge views already express the same ranking with more context.",
            "Prefer Pareto when top product concentration is material.",
            "Prefer profit bridge views when product or category profit columns are available.",
            "Use high_sales_low_profit_bar only when concrete product rows exist.",
        ],
    }
    try:
        return _sanitize_strategy(suggest_strategy(payload), fallback)
    except Exception:
        return fallback
