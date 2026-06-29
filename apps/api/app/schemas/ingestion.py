from __future__ import annotations

from pydantic import BaseModel


class IngestionSummary(BaseModel):
    columns: list[str]
    preview_rows: list[dict[str, object]]
    datetime_candidates: list[str]
    numeric_candidates: list[str]
    row_count: int
    column_count: int
    delimiter: str
    encoding: str
