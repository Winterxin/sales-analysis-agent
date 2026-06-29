from __future__ import annotations

from pydantic import BaseModel, Field


class NotebookSectionContent(BaseModel):
    section_id: str
    markdown_blocks: list[str] = Field(default_factory=list)
    code_cells: list[str] = Field(default_factory=list)


class NotebookContentPlan(BaseModel):
    sections: list[NotebookSectionContent] = Field(default_factory=list)
