from __future__ import annotations

from pydantic import BaseModel, Field


class NotebookAgentLoopStage(BaseModel):
    stage: str
    status: str
    summary: str
    metadata: dict[str, object] = Field(default_factory=dict)


class NotebookAgentLoopState(BaseModel):
    task_id: str
    dataset_type: str
    round: int = 1
    revision_count: int = 0
    stages: list[NotebookAgentLoopStage] = Field(default_factory=list)
