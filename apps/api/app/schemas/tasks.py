from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.ingestion import IngestionSummary
from app.schemas.llm_trace import LLMStageTrace
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping


class TaskCreateResponse(BaseModel):
    task_id: str
    status: str


class UploadResponse(BaseModel):
    task_id: str
    status: str
    ingestion: IngestionSummary
    schema_mapping: SchemaMapping
    analysis_plan: AnalysisPlan
    llm_trace: dict[str, LLMStageTrace] = Field(default_factory=dict)


class NotebookArtifact(BaseModel):
    filename: str
    path: str


class BusinessReviewArtifact(BaseModel):
    filename: str
    path: str


class RunResponse(BaseModel):
    task_id: str
    status: str
    output_language: str = "en"
    current_stage: str | None = None
    current_stage_label: str | None = None
    status_message: str | None = None
    cancel_requested: bool = False
    llm_status: str = "unknown"
    llm_message: str | None = None
    started_at: str | None = None
    heartbeat_at: str | None = None
    finished_at: str | None = None
    error_message: str | None = None


class TaskDetailResponse(BaseModel):
    task_id: str
    status: str
    dataset_type: str | None = None
    current_stage: str | None = None
    current_stage_label: str | None = None
    status_message: str | None = None
    cancel_requested: bool = False
    llm_status: str = "unknown"
    llm_message: str | None = None
    started_at: str | None = None
    heartbeat_at: str | None = None
    finished_at: str | None = None
    artifact_manifest: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None
