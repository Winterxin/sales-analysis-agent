from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from app.schemas.schema_mapping import SchemaMapping


NUMERIC_CANONICAL_FIELDS = {
    "sales_amount",
    "profit",
    "cost",
    "unit_price",
    "quantity",
    "discount",
}

_MONEY_CHARS_RE = re.compile(r"[\s,\$£€¥₹₨]")
_PAREN_NEGATIVE_RE = re.compile(r"^\((.*)\)$")

_NAME_PRIORITIES = {
    "sales_amount": (
        "grandtotal",
        "grand_total",
        "weeklysales",
        "weekly_sales",
        "totalsales",
        "total",
        "sales",
        "revenue",
        "amount",
        "price",
        "gmv",
    ),
    "profit": ("profit", "margin"),
    "cost": ("cost", "expense"),
    "unit_price": ("unitprice", "unit_price", "priceeach", "price"),
    "quantity": ("quantity", "qty", "units", "volume"),
    "discount": ("discount", "disc"),
}

_ID_NAME_TOKENS = ("id", "code", "number", "no", "sku", "status", "name", "title", "description")


@dataclass(frozen=True)
class CanonicalSourceDecision:
    canonical_field: str
    selected_original_column: str
    rejected_duplicate_columns: list[str]
    numeric_conversion_rate: float | None
    non_empty_rate: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_field": self.canonical_field,
            "selected_original_column": self.selected_original_column,
            "rejected_duplicate_columns": self.rejected_duplicate_columns,
            "numeric_conversion_rate": self.numeric_conversion_rate,
            "non_empty_rate": self.non_empty_rate,
            "reason": self.reason,
        }


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip()
    text = text.str.replace(_MONEY_CHARS_RE, "", regex=True)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace(_PAREN_NEGATIVE_RE, r"-\1", regex=True)
    text = text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA, "null": pd.NA})
    return pd.to_numeric(text, errors="coerce")


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _candidate_stats(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    series = frame[column]
    non_empty = series.notna() & series.astype("string").str.strip().ne("")
    numeric = clean_numeric_series(series)
    valid = numeric.notna()
    total = max(len(series), 1)
    return {
        "non_empty_rate": round(float(non_empty.mean()), 4) if len(series) else 0.0,
        "numeric_conversion_rate": round(float(valid.sum() / total), 4),
        "valid_count": int(valid.sum()),
    }


def _name_priority(canonical: str, column: str) -> float:
    normalized = _normalize_name(column)
    priorities = _NAME_PRIORITIES.get(canonical, ())
    for index, token in enumerate(priorities):
        if token in normalized:
            return round(1.0 - index * 0.04, 4)
    if canonical in NUMERIC_CANONICAL_FIELDS and any(token in normalized for token in _ID_NAME_TOKENS):
        return -0.5
    return 0.0


def _best_numeric_column(frame: pd.DataFrame, canonical: str, columns: list[str]) -> tuple[str, dict[str, Any], str]:
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for column in columns:
        stats = _candidate_stats(frame, column)
        score = (
            float(stats["numeric_conversion_rate"]) * 100
            + float(stats["non_empty_rate"]) * 10
            + _name_priority(canonical, column)
        )
        scored.append((score, column, stats))
    scored.sort(key=lambda item: item[0], reverse=True)
    _score, selected, stats = scored[0]
    reason = (
        "selected by numeric conversion rate, non-empty rate, and sales-domain column-name priority"
        if len(columns) > 1
        else "single mapped source"
    )
    return selected, stats, reason


def canonical_field_source_decisions(
    frame: pd.DataFrame,
    schema_mapping: SchemaMapping,
) -> dict[str, CanonicalSourceDecision]:
    grouped: dict[str, list[str]] = {}
    for original, canonical in schema_mapping.field_mapping.items():
        if original in frame.columns:
            grouped.setdefault(canonical, []).append(original)

    decisions: dict[str, CanonicalSourceDecision] = {}
    for canonical, originals in grouped.items():
        if canonical in NUMERIC_CANONICAL_FIELDS:
            selected, stats, reason = _best_numeric_column(frame, canonical, originals)
            decisions[canonical] = CanonicalSourceDecision(
                canonical_field=canonical,
                selected_original_column=selected,
                rejected_duplicate_columns=[column for column in originals if column != selected],
                numeric_conversion_rate=float(stats["numeric_conversion_rate"]),
                non_empty_rate=float(stats["non_empty_rate"]),
                reason=reason,
            )
        else:
            selected = originals[0]
            decisions[canonical] = CanonicalSourceDecision(
                canonical_field=canonical,
                selected_original_column=selected,
                rejected_duplicate_columns=[column for column in originals if column != selected],
                numeric_conversion_rate=None,
                non_empty_rate=None,
                reason="selected first mapped source for non-numeric canonical field",
            )
    return decisions


def canonical_columns_for_frame(frame: pd.DataFrame, schema_mapping: SchemaMapping) -> dict[str, str]:
    return {
        canonical: decision.selected_original_column
        for canonical, decision in canonical_field_source_decisions(frame, schema_mapping).items()
    }


def numeric_coercion_summary(frame: pd.DataFrame, columns: dict[str, str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for canonical in sorted(NUMERIC_CANONICAL_FIELDS & set(columns)):
        column = columns[canonical]
        if column not in frame.columns:
            continue
        stats = _candidate_stats(frame, column)
        summary[canonical] = {
            "selected_column": column,
            **stats,
        }
    return summary


def drop_empty_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [
        column
        for column in frame.columns
        if str(column).strip().lower().startswith("unnamed")
        and frame[column].isna().all()
    ]
    return frame.drop(columns=columns_to_drop) if columns_to_drop else frame
