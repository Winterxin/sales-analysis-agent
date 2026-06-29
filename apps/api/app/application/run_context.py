from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.db.models import AnalysisTask
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline
from app.schemas.notebook_revision import NotebookRevisionPlan
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.artifact_store import ArtifactStore
from app.services.run_budget import RunBudget


@dataclass
class AnalysisRunContext:
    task_id: str
    task: AnalysisTask
    store: ArtifactStore
    settings: Any
    llm_client: Any
    resolved_llm_profile: str
    output_language: str
    llm_profile_policy: dict[str, object]
    run_budget: RunBudget
    manifest: dict[str, object]
    llm_trace: dict[str, object] = field(default_factory=dict)
    schema_mapping: SchemaMapping | None = None
    analysis_plan: AnalysisPlan | None = None
    dataset_profile: dict[str, object] = field(default_factory=dict)
    analysis_focus: dict[str, object] = field(default_factory=dict)
    report: AnalysisReport | None = None
    section_priority: dict[str, object] | None = None
    evidence_pack: dict[str, object] | None = None
    llm_evidence_pack: dict[str, object] | None = None
    chart_selection_plan: dict[str, object] | None = None
    chart_intent_plan: dict[str, object] | None = None
    modeling_opportunity_plan: dict[str, object] | None = None
    modeling_opportunity_decision: dict[str, str] | None = None
    modeling_outcome: dict[str, object] | None = None
    modeling_outcome_interpretation: dict[str, str] | None = None
    final_synthesis: dict[str, object] | None = None
    raw_csv_path: Path | None = None
    report_json_path: Path | None = None
    notebook_outline: NotebookOutline | None = None
    notebook_narrative: NotebookNarrative | None = None
    notebook_content: NotebookContentPlan | None = None
    notebook_outline_path: Path | None = None
    notebook_narrative_path: Path | None = None
    notebook_content_path: Path | None = None
    modeling_interpretation: Any | None = None
    report_html_path: Path | None = None
    business_review_path: Path | None = None
    notebook_path: Path | None = None
    executed_notebook_path: Path | None = None
    named_notebook_path: Path | None = None
    chart_contexts: list[dict[str, object]] = field(default_factory=list)
    initial_chart_contexts: list[dict[str, object]] = field(default_factory=list)
    chart_contexts_path: Path | None = None
    modeling_opportunity_plan_path: Path | None = None
    modeling_opportunity_decision_path: Path | None = None
    modeling_outcome_path: Path | None = None
    final_synthesis_path: Path | None = None
    revision_plan: NotebookRevisionPlan | None = None
    revision_decisions_path: Path | None = None
    postrun_reflections: dict[str, object] = field(default_factory=dict)
    postrun_reflections_path: Path | None = None
    client_report_payload: dict[str, object] | None = None
    client_report_json_path: Path | None = None
    client_report_html_path: Path | None = None
    notebook_agent_state_path: Path | None = None
    llm_trace_path: Path | None = None
    profile_limited_stage_count: int = 0
    max_postrun_reflections_resolved: int | None = None
    allow_postrun_fallback_reflections: bool = True
