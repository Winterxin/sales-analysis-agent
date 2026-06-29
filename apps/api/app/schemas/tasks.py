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
    report: AnalysisReport
    business_review: BusinessReviewArtifact
    notebook: NotebookArtifact
    llm_trace: dict[str, LLMStageTrace] = Field(default_factory=dict)


class TaskDetailResponse(BaseModel):
    task_id: str
    status: str
    dataset_type: str | None = None
    artifact_manifest: dict[str, object] = Field(default_factory=dict)
    error_message: str | None = None
