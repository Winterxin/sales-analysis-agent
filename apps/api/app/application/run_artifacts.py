from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session

from app.application.run_context import AnalysisRunContext
from app.services.artifact_store import ArtifactStore
from app.services.llm_trace_utils import with_llm_trace_summary


def _llm_trace_artifact_payload(
    llm_trace: dict[str, object],
    *,
    llm_profile: str | None = None,
    output_language: str | None = None,
    llm_profile_policy: dict[str, object] | None = None,
    profile_limited_stage_count: int | None = None,
    max_llm_postrun_reflections_resolved: int | None = None,
    allow_postrun_fallback_reflections: bool | None = None,
) -> dict[str, object]:
    return with_llm_trace_summary(
        llm_trace,
        llm_profile=llm_profile,
        output_language=output_language,
        llm_profile_policy=llm_profile_policy,
        profile_limited_stage_count=profile_limited_stage_count,
        max_llm_postrun_reflections_resolved=max_llm_postrun_reflections_resolved,
        allow_postrun_fallback_reflections=allow_postrun_fallback_reflections,
    )


class RunArtifacts:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record_output_language(
        self,
        *,
        store: ArtifactStore,
        task_id: str,
        manifest: dict[str, object],
        output_language: str,
    ) -> None:
        manifest["output_language"] = output_language
        store.save_manifest(task_id, manifest)

    def llm_trace_payload(self, ctx: AnalysisRunContext) -> dict[str, object]:
        return _llm_trace_artifact_payload(
            ctx.llm_trace,
            llm_profile=ctx.resolved_llm_profile,
            output_language=ctx.output_language,
            llm_profile_policy=ctx.llm_profile_policy,
            profile_limited_stage_count=ctx.profile_limited_stage_count,
            max_llm_postrun_reflections_resolved=ctx.max_postrun_reflections_resolved,
            allow_postrun_fallback_reflections=ctx.allow_postrun_fallback_reflections,
        )

    def persist_llm_trace(self, ctx: AnalysisRunContext) -> None:
        ctx.llm_trace_path = ctx.store.save_json(
            ctx.task_id, "llm_trace.json", self.llm_trace_payload(ctx)
        )

    def record_evidence_and_chart_plan(
        self,
        ctx: AnalysisRunContext,
        *,
        section_priority_path: Path,
        evidence_pack_path: Path,
        llm_evidence_pack_path: Path,
        chart_selection_plan_path: Path,
        chart_selection_plan_pre_intent_path: Path,
        chart_intent_plan_path: Path,
    ) -> None:
        ctx.manifest.update(
            {
                "section_priority": ctx.section_priority,
                "evidence_pack": ctx.evidence_pack,
                "llm_evidence_pack": ctx.llm_evidence_pack,
                "chart_selection_plan": ctx.chart_selection_plan,
                "chart_intent_plan": ctx.chart_intent_plan,
                "chart_intent_planning_status": (
                    ctx.chart_intent_plan.get("trace", {}) if ctx.chart_intent_plan else {}
                ).get("status"),
                "chart_intent_planning_mode": "shadow",
                "llm_profile": ctx.resolved_llm_profile,
                "llm_profile_policy": ctx.llm_profile_policy,
                "files": {
                    **ctx.manifest.get("files", {}),
                    "section_priority_json": str(section_priority_path),
                    "evidence_pack_json": str(evidence_pack_path),
                    "llm_evidence_pack_json": str(llm_evidence_pack_path),
                    "chart_selection_plan_json": str(chart_selection_plan_path),
                    "chart_selection_plan_pre_intent_json": str(
                        chart_selection_plan_pre_intent_path
                    ),
                    "chart_intent_plan_json": str(chart_intent_plan_path),
                },
            }
        )
        ctx.store.save_manifest(ctx.task_id, ctx.manifest)

    def record_modeling_opportunity_plan(self, ctx: AnalysisRunContext) -> None:
        ctx.manifest.update(
            {
                "modeling_opportunity_plan": ctx.modeling_opportunity_plan,
                "files": {
                    **ctx.manifest.get("files", {}),
                    "modeling_opportunity_plan_json": str(ctx.modeling_opportunity_plan_path),
                },
            }
        )
        ctx.store.save_manifest(ctx.task_id, ctx.manifest)

    def record_modeling_outcome(self, ctx: AnalysisRunContext) -> None:
        ctx.manifest.update(
            {
                "modeling_outcome": ctx.modeling_outcome,
                "files": {
                    **ctx.manifest.get("files", {}),
                    "modeling_outcome_json": str(ctx.modeling_outcome_path),
                },
            }
        )
        ctx.store.save_manifest(ctx.task_id, ctx.manifest)

    def record_final_synthesis(self, ctx: AnalysisRunContext) -> None:
        ctx.manifest.update(
            {
                "files": {
                    **ctx.manifest.get("files", {}),
                    "final_synthesis_json": str(ctx.final_synthesis_path),
                },
            }
        )
        ctx.store.save_manifest(ctx.task_id, ctx.manifest)

    def persist_completed_artifacts(self, ctx: AnalysisRunContext) -> None:
        ctx.manifest.update(
            {
                "status": "completed",
                "output_language": ctx.output_language,
                "report": _required_report(ctx).model_dump(),
                "llm_trace": self.llm_trace_payload(ctx),
                "files": {
                    **ctx.manifest.get("files", {}),
                    "llm_trace_json": str(_required_path(ctx.llm_trace_path, "llm_trace_path")),
                    "report_json": str(_required_path(ctx.report_json_path, "report_json_path")),
                    "notebook_outline_json": str(
                        _required_path(ctx.notebook_outline_path, "notebook_outline_path")
                    ),
                    "notebook_narrative_json": str(
                        _required_path(ctx.notebook_narrative_path, "notebook_narrative_path")
                    ),
                    "notebook_content_json": str(
                        _required_path(ctx.notebook_content_path, "notebook_content_path")
                    ),
                    "modeling_opportunity_plan_json": str(ctx.modeling_opportunity_plan_path)
                    if ctx.modeling_opportunity_plan_path
                    else "",
                    "modeling_opportunity_decision_json": str(
                        ctx.modeling_opportunity_decision_path
                    )
                    if ctx.modeling_opportunity_decision_path
                    else "",
                    "revision_decisions_json": str(
                        _required_path(ctx.revision_decisions_path, "revision_decisions_path")
                    ),
                    "notebook_agent_state_json": str(
                        _required_path(ctx.notebook_agent_state_path, "notebook_agent_state_path")
                    ),
                    "postrun_chart_contexts_json": str(
                        _required_path(ctx.chart_contexts_path, "chart_contexts_path")
                    ),
                    "postrun_chart_reflections_json": str(
                        _required_path(ctx.postrun_reflections_path, "postrun_reflections_path")
                    ),
                    "report_html": str(_required_path(ctx.report_html_path, "report_html_path")),
                    "business_review_md": str(
                        _required_path(ctx.business_review_path, "business_review_path")
                    ),
                    "client_report_html": str(
                        _required_path(ctx.client_report_html_path, "client_report_html_path").resolve()
                    ),
                    "client_report_json": str(
                        _required_path(ctx.client_report_json_path, "client_report_json_path").resolve()
                    ),
                    "analysis_source_notebook": str(
                        _required_path(ctx.notebook_path, "notebook_path")
                    ),
                    "analysis_notebook_canonical": str(
                        _required_path(ctx.executed_notebook_path, "executed_notebook_path")
                    ),
                    "analysis_notebook": str(
                        _required_path(ctx.named_notebook_path, "named_notebook_path")
                    ),
                },
            }
        )
        ctx.store.save_manifest(ctx.task_id, ctx.manifest)

    def mark_task_completed(self, ctx: AnalysisRunContext) -> None:
        ctx.task.status = "completed"
        ctx.task.artifact_manifest_path = str(ctx.store.manifest_path(ctx.task_id))
        ctx.task.dataset_type = _required_report(ctx).dataset_type
        self.session.add(ctx.task)
        self.session.commit()


def _required_report(ctx: AnalysisRunContext):
    if ctx.report is None:
        raise RuntimeError("report has not been built")
    return ctx.report


def _required_path(path: Path | None, name: str) -> Path:
    if path is None:
        raise RuntimeError(f"{name} has not been built")
    return path
