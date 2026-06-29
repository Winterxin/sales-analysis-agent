from __future__ import annotations

from typing import Any

import pandas as pd


def distribution_readability(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "distribution_readability_mode": "raw",
            "p99": None,
            "max": None,
            "median": None,
            "clipped_count": 0,
            "clipped_ratio": 0.0,
            "reason": "no numeric values",
        }
    p99 = float(numeric.quantile(0.99))
    max_value = float(numeric.max())
    median = float(numeric.median())
    long_tail = False
    if p99 > 0 and max_value > p99 * 2:
        long_tail = True
    if median > 0 and p99 / median > 10:
        long_tail = True
    clipped_count = int((numeric > p99).sum()) if long_tail else 0
    return {
        "distribution_readability_mode": "p99_clipped" if long_tail else "raw",
        "p99": round(p99, 6),
        "max": round(max_value, 6),
        "median": round(median, 6),
        "clipped_count": clipped_count,
        "clipped_ratio": round(clipped_count / len(numeric), 6) if len(numeric) else 0.0,
        "reason": (
            "long tail detected; plotted values are clipped at P99 for readability"
            if long_tail
            else "distribution is readable at full range"
        ),
    }


def readable_distribution_frame(values: pd.Series, value_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    trace = distribution_readability(numeric)
    frame = numeric.to_frame(name=value_name)
    if trace["distribution_readability_mode"] == "p99_clipped" and trace["p99"] is not None:
        frame = frame[frame[value_name] <= float(trace["p99"])].copy()
    return frame, trace


def heatmap_readability(matrix: pd.DataFrame, *, max_rows: int = 8, max_cols: int = 8) -> dict[str, Any]:
    original_shape = [int(matrix.shape[0]), int(matrix.shape[1])]
    too_large = matrix.shape[0] * matrix.shape[1] > max_rows * max_cols or matrix.shape[0] > max_rows or matrix.shape[1] > max_cols
    rendered_shape = [
        min(int(matrix.shape[0]), max_rows),
        min(int(matrix.shape[1]), max_cols),
    ]
    return {
        "heatmap_readability_mode": "top_n_no_annotation" if too_large else "raw",
        "original_shape": original_shape,
        "rendered_shape": rendered_shape if too_large else original_shape,
        "top_n_applied": bool(too_large),
        "annotation_enabled": not too_large,
        "reason": (
            "large heatmap; render top rows/columns and disable cell annotations"
            if too_large
            else "heatmap size is readable"
        ),
    }


def readable_heatmap_matrix(matrix: pd.DataFrame, *, max_rows: int = 8, max_cols: int = 8) -> tuple[pd.DataFrame, dict[str, Any]]:
    trace = heatmap_readability(matrix, max_rows=max_rows, max_cols=max_cols)
    if trace["top_n_applied"]:
        rendered = matrix.iloc[:max_rows, :max_cols].copy()
    else:
        rendered = matrix.copy()
    return rendered, trace
