from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from app.schemas.ingestion import IngestionSummary
from app.services.time_parser import parse_datetime_series

ENCODINGS = ["utf-8", "utf-8-sig", "gbk", "latin1"]


def _detect_csv_format(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    for encoding in ENCODINGS:
        try:
            sample = raw.decode(encoding)
            dialect = csv.Sniffer().sniff(sample[:2048], delimiters=",;\t|")
            return dialect.delimiter, encoding
        except (UnicodeDecodeError, csv.Error):
            continue
    return ",", "utf-8"


def _infer_datetime_candidates(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in frame.columns:
        sample = frame[column].dropna().astype(str).head(10)
        if sample.empty:
            continue
        parsed = parse_datetime_series(sample)
        if parsed.notna().mean() >= 0.7:
            candidates.append(str(column))
    return candidates


def _infer_numeric_candidates(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for column in frame.columns:
        sample = pd.to_numeric(frame[column], errors="coerce")
        if sample.notna().mean() >= 0.7:
            candidates.append(str(column))
    return candidates


def ingest_csv(path: Path) -> IngestionSummary:
    delimiter, encoding = _detect_csv_format(path)
    frame = pd.read_csv(path, sep=delimiter, encoding=encoding)
    preview_rows = frame.head(5).fillna("").to_dict(orient="records")
    return IngestionSummary(
        columns=[str(column) for column in frame.columns],
        preview_rows=preview_rows,
        datetime_candidates=_infer_datetime_candidates(frame),
        numeric_candidates=_infer_numeric_candidates(frame),
        row_count=len(frame),
        column_count=len(frame.columns),
        delimiter=delimiter,
        encoding=encoding,
    )
