from __future__ import annotations

from pydantic import BaseModel, Field


class SchemaMapping(BaseModel):
    dataset_type: str = "unknown"
    field_mapping: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0
    missing_required_fields: list[str] = Field(default_factory=list)
    uncertain_fields: list[str] = Field(default_factory=list)
