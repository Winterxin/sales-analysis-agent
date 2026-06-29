from __future__ import annotations

from pydantic import BaseModel, Field


class NotebookRevisionDecision(BaseModel):
    revision_key: str
    reason: str


class NotebookRevisionPlan(BaseModel):
    decisions: list[NotebookRevisionDecision] = Field(default_factory=list)
