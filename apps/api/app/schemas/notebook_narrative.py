from __future__ import annotations

from pydantic import BaseModel, Field


class NotebookSectionNarrative(BaseModel):
    section_id: str
    intro: str
    key_observations: list[str] = Field(default_factory=list)
    business_takeaway: str = ""
    followup_question: str = ""


class NotebookNarrative(BaseModel):
    sections: list[NotebookSectionNarrative] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
