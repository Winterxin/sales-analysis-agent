from __future__ import annotations

from pydantic import BaseModel, Field


class NotebookSection(BaseModel):
    section_id: str
    title: str
    purpose: str


class NotebookOutline(BaseModel):
    title: str = "Sales Analysis Notebook"
    sections: list[NotebookSection] = Field(default_factory=list)
