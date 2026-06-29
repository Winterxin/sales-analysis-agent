from __future__ import annotations

import warnings

import pandas as pd

COMMON_DATETIME_FORMATS = [
    "%d-%b-%y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d/%m/%Y",
]


def parse_datetime_series(series: pd.Series) -> pd.Series:
    sample = series.dropna().astype(str)
    if sample.empty:
        return pd.to_datetime(series, errors="coerce")

    for date_format in COMMON_DATETIME_FORMATS:
        parsed_sample = pd.to_datetime(sample, format=date_format, errors="coerce")
        if parsed_sample.notna().mean() >= 0.7:
            return pd.to_datetime(series, format=date_format, errors="coerce")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")
