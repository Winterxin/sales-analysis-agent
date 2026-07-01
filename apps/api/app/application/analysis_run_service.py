from __future__ import annotations

import json
from pathlib import Path
import shutil

from fastapi import HTTPException
import pandas as pd
from sqlalchemy.orm import Session

from app.application.run_artifacts import RunArtifacts
from app.application.run_stages import RunStage, RunStageExecutor
from app.application.runtime_state import RuntimeStateStore
from app.analysis.runner import run_analysis
from app.application.run_context import AnalysisRunContext
from app.core.config import get_settings
from app.db.models import AnalysisTask
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_revision import NotebookRevisionPlan
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.schemas.tasks import BusinessReviewArtifact, NotebookArtifact, RunResponse
from app.services.artifact_store import ArtifactStore
from app.services.business_review_builder import build_business_review
from app.services.chart_selection_planner import build_chart_selection_plan
from app.services.client_report_builder import (
    build_client_report_payload_with_trace,
    render_client_report_html,
)
from app.services.evidence_pack_builder import build_evidence_pack
from app.services.insight_generator import generate_report_summary_with_trace
from app.services.intent_guided_chart_selection import apply_intent_guided_chart_selection
from app.services.llm_chart_intent_planner import build_llm_chart_intent_plan
from app.services.llm_client import get_default_llm_client
from app.services.llm_evidence_pack import build_llm_evidence_pack
from app.services.llm_profile_policy import (
    build_llm_profile_policy,
    normalize_llm_profile,
    resolve_max_postrun_reflections,
)
from app.services.modeling_interpretation import build_modeling_interpretation_with_trace
from app.services.modeling_opportunity_decision import build_modeling_opportunity_decision_with_trace
from app.services.modeling_opportunity_planner import build_modeling_opportunity_plan
from app.services.modeling_outcome_builder import build_modeling_outcome
from app.services.modeling_outcome_interpreter import build_modeling_outcome_interpretation_with_trace
from app.services.notebook_agent_loop import (
    apply_notebook_revision_plan,
    augment_outline_with_revision_sections,
    build_notebook_agent_loop_state,
    build_notebook_revision_plan_with_trace,
    revision_plan_for_output,
)
from app.services.notebook_builder import build_notebook
from app.services.notebook.action_plan_builder import action_plan_rows
from app.services.notebook.final_synthesis import build_final_synthesis_with_trace
from app.services.notebook_content_planner import build_notebook_content_with_trace
from app.services.notebook_executor import execute_notebook
from app.services.notebook_narrative import build_notebook_narrative_with_trace
from app.services.notebook_planner import build_notebook_outline_with_trace, build_section_priority
from app.services.output_language import is_english_output, normalize_output_language
from app.services.notebook_postrun_reflection import (
    apply_postrun_chart_reflections,
    build_postrun_chart_reflections,
    extract_postrun_chart_contexts,
)
from app.services.report_builder import build_html_report
from app.services.run_budget import (
    DemoSafeChartDecisionLLMClient,
    RunBudget,
    build_skipped_stage_trace,
)


def _llm_stage_trace_payload(llm_trace: dict[str, object]) -> dict[str, object]:
    return {
        stage: trace_payload
        for stage, trace_payload in llm_trace.items()
        if stage != "summary"
    }


def _content_has_llm_chart_decision(notebook_content) -> bool:
    for section in getattr(notebook_content, "sections", []):
        for code_cell in getattr(section, "code_cells", []):
            if "llm_sanitized" in str(code_cell):
                return True
    return False


def _stage_enabled_by_policy(ctx, policy_key: str, default: bool = True) -> bool:
    return bool(ctx.llm_profile_policy.get(policy_key, default))


def _chart_selection_enabled(ctx) -> bool:
    return _stage_enabled_by_policy(ctx, "chart_selection_enabled", default=False)


def _chart_intent_planning_enabled(ctx) -> bool:
    return _stage_enabled_by_policy(ctx, "chart_intent_planning_enabled", default=False)


def _intent_guided_chart_selection_enabled(ctx) -> bool:
    return _stage_enabled_by_policy(
        ctx,
        "intent_guided_chart_selection_enabled",
        default=False,
    )


def _profile_disabled_trace(ctx, *, stage: str, policy_key: str):
    return build_skipped_stage_trace(
        stage=stage,
        llm_client=ctx.llm_client,
        budget=ctx.run_budget,
        status="skipped_by_profile",
        reason=(
            f"Skipped {stage} LLM stage because llm_profile={ctx.resolved_llm_profile} "
            f"sets {policy_key}=False."
        ),
    )


_UNSUPPORTED_ENGLISH_DISCOUNT_SYNTHESIS_TERMS = (
    "high-discount",
    "discount risk",
    "discount approval",
    "discount threshold",
    "discount tier",
    "scoped discount tier",
    "discount-profit relationship",
    "what-if profit lift",
)

_ALLOWED_ENGLISH_MISSING_DISCOUNT_TERMS = (
    "discount field is not mapped",
    "discount is missing",
    "discount conclusions are out of scope",
    "add discount fields before",
)


def _has_discount_risk_evidence(report: AnalysisReport, schema_mapping: SchemaMapping) -> bool:
    mapped = set(schema_mapping.field_mapping.values())
    if not {"profit", "discount"}.issubset(mapped):
        return False
    discount_module = next(
        (module for module in report.modules if module.module_id == "discount_profit_analysis"),
        None,
    )
    if discount_module is None:
        return False
    tables = discount_module.tables or {}
    return any(
        bool(tables.get(table_name))
        for table_name in ("discount_buckets", "discount_threshold_candidates", "discount_cap_what_if")
    )


def _english_discount_synthesis_supported(value: object) -> bool:
    text = str(value or "").lower()
    if not text:
        return False
    if any(term in text for term in _ALLOWED_ENGLISH_MISSING_DISCOUNT_TERMS):
        return True
    return not any(term in text for term in _UNSUPPORTED_ENGLISH_DISCOUNT_SYNTHESIS_TERMS)


def _filter_final_synthesis_without_discount_risk(value: object) -> object:
    if isinstance(value, list):
        return [
            item
            for item in (_filter_final_synthesis_without_discount_risk(item) for item in value)
            if item not in ("", [], {})
            and _english_discount_synthesis_supported(json.dumps(item, ensure_ascii=False))
        ]
    if isinstance(value, dict):
        filtered = {
            key: _filter_final_synthesis_without_discount_risk(item)
            for key, item in value.items()
        }
        return {
            key: item
            for key, item in filtered.items()
            if item not in ("", [], {})
            and _english_discount_synthesis_supported(json.dumps(item, ensure_ascii=False))
        }
    return value


def _guard_final_synthesis_for_capabilities(
    final_synthesis: dict[str, object] | None,
    *,
    report: AnalysisReport,
    schema_mapping: SchemaMapping,
    output_language: str | None,
) -> dict[str, object] | None:
    if (
        not isinstance(final_synthesis, dict)
        or not is_english_output(output_language)
        or _has_discount_risk_evidence(report, schema_mapping)
    ):
        return final_synthesis
    guarded = _filter_final_synthesis_without_discount_risk(final_synthesis)
    return guarded if isinstance(guarded, dict) else final_synthesis


class AnalysisRunService:
    def __init__(
        self,
        session: Session,
        runtime_state: RuntimeStateStore | None = None,
    ):
        self.session = session
        self.artifacts = RunArtifacts(session)
        self.stage_executor = RunStageExecutor()
        self.runtime_state = runtime_state

    def run(
        self,
        task_id: str,
        llm_profile: str | None = None,
        output_language: str | None = None,
    ) -> RunResponse:
        ctx = self._prepare_context(
            task_id,
            llm_profile=llm_profile,
            output_language=output_language,
        )
        for stage in self._build_stages(ctx):
            if self.runtime_state is not None:
                self.runtime_state.mark_stage_started(
                    task_id=ctx.task_id,
                    stage_id=stage.stage_id,
                    stage_label=stage.label,
                    llm_enabled=bool(getattr(ctx.llm_client, "enabled", False)),
                )
            self.stage_executor.run_stage(stage)
            if self.runtime_state is not None:
                self.runtime_state.mark_stage_finished(
                    task_id=ctx.task_id,
                    llm_trace=ctx.llm_trace,
                )
        return self._build_run_response(ctx)

    def _build_stages(self, ctx: AnalysisRunContext) -> list[RunStage]:
        return [
            RunStage(
                "load_uploaded_inputs",
                "Load uploaded inputs",
                lambda: self._load_uploaded_inputs(ctx),
            ),
            RunStage(
                "deterministic_analysis",
                "Run deterministic analysis",
                lambda: self._run_analysis_modules(ctx),
            ),
            RunStage(
                "evidence_and_chart_planning",
                "Build evidence and chart plan",
                lambda: self._build_evidence_and_chart_plan(ctx),
            ),
            RunStage(
                "modeling_opportunity_planning",
                "Build modeling opportunity plan",
                lambda: self._build_modeling_opportunity_plan(ctx),
            ),
            RunStage(
                "notebook_planning",
                "Build notebook plans",
                lambda: self._build_notebook_plans(ctx),
            ),
            RunStage(
                "summary_and_modeling_interpretation",
                "Build summary and modeling interpretation",
                lambda: self._build_report_summary_and_modeling_interpretation(ctx),
            ),
            RunStage(
                "final_synthesis",
                "Build final synthesis",
                lambda: self._build_final_synthesis(ctx),
            ),
            RunStage(
                "notebook_build_and_execution",
                "Build and execute notebook",
                lambda: self._build_and_execute_notebook(ctx),
            ),
            RunStage(
                "notebook_revision",
                "Apply notebook revision",
                lambda: self._apply_revision_if_needed(ctx),
            ),
            RunStage(
                "postrun_reflection",
                "Apply post-run reflection",
                lambda: self._apply_postrun_reflections(ctx),
            ),
            RunStage(
                "client_report",
                "Build client report",
                lambda: self._build_client_report(ctx),
            ),
            RunStage(
                "agent_state",
                "Build agent state",
                lambda: self._build_agent_state(ctx),
            ),
            RunStage(
                "completed_artifact_persistence",
                "Persist completed artifacts",
                lambda: self._persist_completed_artifacts(ctx),
            ),
            RunStage(
                "task_completion",
                "Mark task completed",
                lambda: self._mark_task_completed(ctx),
            ),
        ]

    def _prepare_context(
        self,
        task_id: str,
        *,
        llm_profile: str | None = None,
        output_language: str | None = None,
    ) -> AnalysisRunContext:
        task = self._get_task_or_404(task_id)
        store = ArtifactStore()
        settings = get_settings()
        llm_client = get_default_llm_client()
        resolved_llm_profile = normalize_llm_profile(llm_profile or settings.llm_profile)
        manifest = store.load_manifest(task_id)
        resolved_output_language = normalize_output_language(
            output_language or manifest.get("output_language")
        )
        llm_profile_policy = build_llm_profile_policy(resolved_llm_profile)
        self.artifacts.record_output_language(
            store=store,
            task_id=task_id,
            manifest=manifest,
            output_language=resolved_output_language,
        )
        run_budget = RunBudget(
            demo_safe=settings.demo_safe,
            total_seconds=settings.demo_safe_run_budget_seconds if settings.demo_safe else None,
            min_stage_seconds=settings.demo_safe_min_stage_budget_seconds,
        )
        return AnalysisRunContext(
            task_id=task_id,
            task=task,
            store=store,
            settings=settings,
            llm_client=llm_client,
            resolved_llm_profile=resolved_llm_profile,
            output_language=resolved_output_language,
            llm_profile_policy=llm_profile_policy,
            run_budget=run_budget,
            manifest=manifest,
            allow_postrun_fallback_reflections=bool(
                llm_profile_policy.get("allow_postrun_fallback_reflections", True)
            ),
        )

    def _get_task_or_404(self, task_id: str) -> AnalysisTask:
        task = self.session.get(AnalysisTask, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    def _load_uploaded_inputs(self, ctx: AnalysisRunContext) -> None:
        try:
            ctx.raw_csv_path = Path(str(ctx.manifest["files"]["raw_csv"]))  # type: ignore[index]
            ctx.schema_mapping = SchemaMapping.model_validate(ctx.manifest["schema_mapping"])
            ctx.dataset_profile = dict(ctx.manifest.get("dataset_profile", {}))
            ctx.analysis_focus = dict(ctx.manifest.get("analysis_focus", {}))
            ctx.analysis_plan = AnalysisPlan.model_validate(ctx.manifest["analysis_plan"])
        except KeyError as exc:
            raise HTTPException(status_code=400, detail="Task has not been uploaded") from exc

    def _run_analysis_modules(self, ctx: AnalysisRunContext) -> None:
        ctx.report = run_analysis(
            task_id=ctx.task_id,
            csv_path=self._raw_csv_path(ctx),
            schema_mapping=self._schema_mapping(ctx),
            plan=self._analysis_plan(ctx),
        )
        ctx.llm_trace = _llm_stage_trace_payload(dict(ctx.manifest.get("llm_trace", {})))

    def _build_evidence_and_chart_plan(self, ctx: AnalysisRunContext) -> None:
        ctx.section_priority = build_section_priority(
            schema_mapping=self._schema_mapping(ctx),
            analysis_plan=self._analysis_plan(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_focus=ctx.analysis_focus,
        )
        section_priority_path = ctx.store.save_json(
            ctx.task_id, "section_priority.json", ctx.section_priority
        )
        ctx.evidence_pack = build_evidence_pack(
            report=self._report(ctx),
            schema_mapping=self._schema_mapping(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_focus=ctx.analysis_focus,
            section_priority=self._section_priority(ctx),
        )
        evidence_pack_path = ctx.store.save_json(ctx.task_id, "evidence_pack.json", ctx.evidence_pack)
        ctx.chart_selection_plan = build_chart_selection_plan(
            schema_mapping=self._schema_mapping(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_focus=ctx.analysis_focus,
            section_priority=self._section_priority(ctx),
            evidence_pack=self._evidence_pack(ctx),
            llm_client=ctx.llm_client if getattr(ctx.llm_client, "enabled", False) else None,
            llm_profile=ctx.resolved_llm_profile,
            llm_chart_selection_enabled=_chart_selection_enabled(ctx),
        )
        chart_selection_plan_pre_intent_path = ctx.store.save_json(
            ctx.task_id, "chart_selection_plan_pre_intent.json", ctx.chart_selection_plan
        )
        ctx.chart_intent_plan = build_llm_chart_intent_plan(
            schema_mapping=self._schema_mapping(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_focus=ctx.analysis_focus,
            section_priority=self._section_priority(ctx),
            evidence_pack=self._evidence_pack(ctx),
            chart_selection_plan=self._chart_selection_plan(ctx),
            llm_client=ctx.llm_client if getattr(ctx.llm_client, "enabled", False) else None,
            llm_profile=ctx.resolved_llm_profile,
            llm_chart_intent_planning_enabled=_chart_intent_planning_enabled(ctx),
        )
        chart_intent_plan_path = ctx.store.save_json(
            ctx.task_id, "chart_intent_plan.json", ctx.chart_intent_plan
        )
        ctx.chart_selection_plan = apply_intent_guided_chart_selection(
            base_chart_selection_plan=self._chart_selection_plan(ctx),
            chart_intent_plan=ctx.chart_intent_plan,
            schema_mapping=self._schema_mapping(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_focus=ctx.analysis_focus,
            section_priority=self._section_priority(ctx),
            evidence_pack=self._evidence_pack(ctx),
            llm_client=ctx.llm_client if getattr(ctx.llm_client, "enabled", False) else None,
            llm_profile=ctx.resolved_llm_profile,
            enabled=_intent_guided_chart_selection_enabled(ctx),
        )
        chart_selection_plan_path = ctx.store.save_json(
            ctx.task_id, "chart_selection_plan.json", ctx.chart_selection_plan
        )
        ctx.llm_evidence_pack = build_llm_evidence_pack(
            report=self._report(ctx),
            schema_mapping=self._schema_mapping(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_focus=ctx.analysis_focus,
            chart_selection_plan=self._chart_selection_plan(ctx),
            action_source_evidence_pack=self._evidence_pack(ctx),
        )
        llm_evidence_pack_path = ctx.store.save_json(
            ctx.task_id, "llm_evidence_pack.json", ctx.llm_evidence_pack
        )
        self.artifacts.record_evidence_and_chart_plan(
            ctx,
            section_priority_path=section_priority_path,
            evidence_pack_path=evidence_pack_path,
            llm_evidence_pack_path=llm_evidence_pack_path,
            chart_selection_plan_path=chart_selection_plan_path,
            chart_selection_plan_pre_intent_path=chart_selection_plan_pre_intent_path,
            chart_intent_plan_path=chart_intent_plan_path,
        )

    def _build_modeling_opportunity_plan(self, ctx: AnalysisRunContext) -> None:
        frame = pd.read_csv(self._raw_csv_path(ctx))
        ctx.modeling_opportunity_plan = build_modeling_opportunity_plan(
            frame,
            self._schema_mapping(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_plan=self._analysis_plan(ctx),
        )
        ctx.modeling_opportunity_plan_path = ctx.store.save_json(
            ctx.task_id,
            "modeling_opportunity_plan.json",
            ctx.modeling_opportunity_plan,
        )
        self.artifacts.record_modeling_opportunity_plan(ctx)

    def _build_notebook_plans(self, ctx: AnalysisRunContext) -> None:
        if ctx.run_budget.has_budget_for_stage("notebook_outline"):
            ctx.notebook_outline, outline_trace = build_notebook_outline_with_trace(
                schema_mapping=self._schema_mapping(ctx),
                analysis_plan=self._analysis_plan(ctx),
                llm_client=ctx.llm_client,
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                modeling_opportunity_plan=ctx.modeling_opportunity_plan,
                output_language=ctx.output_language,
            )
        else:
            ctx.notebook_outline, _ = build_notebook_outline_with_trace(
                schema_mapping=self._schema_mapping(ctx),
                analysis_plan=self._analysis_plan(ctx),
                llm_client=None,
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                modeling_opportunity_plan=ctx.modeling_opportunity_plan,
                output_language=ctx.output_language,
            )
            outline_trace = build_skipped_stage_trace(
                stage="notebook_outline",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped notebook outline LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["notebook_outline"] = outline_trace.model_dump()
        self.artifacts.persist_llm_trace(ctx)

        if not _stage_enabled_by_policy(ctx, "notebook_narrative_enabled"):
            ctx.notebook_narrative, _ = build_notebook_narrative_with_trace(
                outline=self._notebook_outline(ctx),
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                llm_client=None,
                evidence_pack=self._llm_evidence_pack(ctx),
                output_language=ctx.output_language,
            )
            narrative_trace = _profile_disabled_trace(
                ctx,
                stage="notebook_narrative",
                policy_key="notebook_narrative_enabled",
            )
        elif not ctx.run_budget.stage_allowed("notebook_narrative"):
            ctx.notebook_narrative, _ = build_notebook_narrative_with_trace(
                outline=self._notebook_outline(ctx),
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                llm_client=None,
                evidence_pack=self._llm_evidence_pack(ctx),
                output_language=ctx.output_language,
            )
            narrative_trace = build_skipped_stage_trace(
                stage="notebook_narrative",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_demo_safe",
                reason="Skipped notebook narrative LLM stage in demo-safe profile.",
            )
        elif ctx.run_budget.has_budget_for_stage("notebook_narrative"):
            ctx.notebook_narrative, narrative_trace = build_notebook_narrative_with_trace(
                outline=self._notebook_outline(ctx),
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                llm_client=ctx.llm_client,
                evidence_pack=self._llm_evidence_pack(ctx),
                output_language=ctx.output_language,
            )
        else:
            ctx.notebook_narrative, _ = build_notebook_narrative_with_trace(
                outline=self._notebook_outline(ctx),
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                llm_client=None,
                evidence_pack=self._llm_evidence_pack(ctx),
                output_language=ctx.output_language,
            )
            narrative_trace = build_skipped_stage_trace(
                stage="notebook_narrative",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped notebook narrative LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["notebook_narrative"] = narrative_trace.model_dump()
        self.artifacts.persist_llm_trace(ctx)

        if ctx.run_budget.has_budget_for_stage("notebook_content"):
            content_llm_client = (
                DemoSafeChartDecisionLLMClient(ctx.llm_client)
                if ctx.settings.demo_safe and getattr(ctx.llm_client, "enabled", False)
                else ctx.llm_client
            )
            ctx.notebook_content, content_trace = build_notebook_content_with_trace(
                outline=self._notebook_outline(ctx),
                narrative=self._notebook_narrative(ctx),
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                analysis_plan=self._analysis_plan(ctx),
                llm_client=content_llm_client,
                chart_selection_plan=self._chart_selection_plan(ctx),
                evidence_pack=self._llm_evidence_pack(ctx),
                output_language=ctx.output_language,
            )
            if (
                ctx.settings.demo_safe
                and getattr(ctx.llm_client, "enabled", False)
                and content_trace.status == "skipped"
                and _content_has_llm_chart_decision(ctx.notebook_content)
            ):
                content_trace = content_trace.model_copy(
                    update={
                        "status": "llm_partial",
                        "reason": (
                            "Demo-safe ran allowlisted LLM chart-decision calls and "
                            "skipped notebook section prose generation."
                        ),
                        "attempted": True,
                        "applied": True,
                        "fallback_type": None,
                    }
                )
        else:
            ctx.notebook_content, _ = build_notebook_content_with_trace(
                outline=self._notebook_outline(ctx),
                narrative=self._notebook_narrative(ctx),
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                analysis_plan=self._analysis_plan(ctx),
                llm_client=None,
                chart_selection_plan=self._chart_selection_plan(ctx),
                evidence_pack=self._llm_evidence_pack(ctx),
                output_language=ctx.output_language,
            )
            content_trace = build_skipped_stage_trace(
                stage="notebook_content",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped notebook content LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["notebook_content"] = content_trace.model_dump()
        self.artifacts.persist_llm_trace(ctx)

    def _build_report_summary_and_modeling_interpretation(
        self, ctx: AnalysisRunContext
    ) -> None:
        if not _stage_enabled_by_policy(ctx, "report_summary_enabled"):
            summary_trace = _profile_disabled_trace(
                ctx,
                stage="report_summary",
                policy_key="report_summary_enabled",
            )
        elif not ctx.run_budget.stage_allowed("report_summary"):
            summary_trace = build_skipped_stage_trace(
                stage="report_summary",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_demo_safe",
                reason="Skipped report summary LLM stage in demo-safe profile.",
            )
        elif ctx.run_budget.has_budget_for_stage("report_summary"):
            ctx.report, summary_trace = generate_report_summary_with_trace(
                self._report(ctx),
                llm_client=ctx.llm_client,
                evidence_pack=self._llm_evidence_pack(ctx),
                output_language=ctx.output_language,
            )
        else:
            summary_trace = build_skipped_stage_trace(
                stage="report_summary",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped report summary LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["report_summary"] = summary_trace.model_dump()
        self.artifacts.persist_llm_trace(ctx)

        if not _stage_enabled_by_policy(ctx, "modeling_interpretation_enabled"):
            modeling_interpretation, _ = build_modeling_interpretation_with_trace(
                self._report(ctx),
                llm_client=None,
                output_language=ctx.output_language,
            )
            modeling_interpretation_trace = _profile_disabled_trace(
                ctx,
                stage="modeling_interpretation",
                policy_key="modeling_interpretation_enabled",
            )
        elif ctx.run_budget.has_budget_for_stage("modeling_interpretation"):
            modeling_interpretation, modeling_interpretation_trace = (
                build_modeling_interpretation_with_trace(
                    self._report(ctx),
                    llm_client=ctx.llm_client,
                    output_language=ctx.output_language,
                )
            )
        else:
            modeling_interpretation, _ = build_modeling_interpretation_with_trace(
                self._report(ctx),
                llm_client=None,
                output_language=ctx.output_language,
            )
            modeling_interpretation_trace = build_skipped_stage_trace(
                stage="modeling_interpretation",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped modeling interpretation LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["modeling_interpretation"] = modeling_interpretation_trace.model_dump()
        self.artifacts.persist_llm_trace(ctx)
        ctx.modeling_interpretation = modeling_interpretation if modeling_interpretation_trace.applied else None

        if not _stage_enabled_by_policy(ctx, "modeling_opportunity_decision_enabled"):
            ctx.modeling_opportunity_decision, _ = build_modeling_opportunity_decision_with_trace(
                ctx.modeling_opportunity_plan or {},
                llm_client=None,
                output_language=ctx.output_language,
            )
            opportunity_trace = _profile_disabled_trace(
                ctx,
                stage="modeling_opportunity_decision",
                policy_key="modeling_opportunity_decision_enabled",
            )
        elif ctx.run_budget.has_budget_for_stage("modeling_opportunity_decision"):
            ctx.modeling_opportunity_decision, opportunity_trace = build_modeling_opportunity_decision_with_trace(
                ctx.modeling_opportunity_plan or {},
                llm_client=ctx.llm_client,
                output_language=ctx.output_language,
            )
        else:
            ctx.modeling_opportunity_decision, _ = build_modeling_opportunity_decision_with_trace(
                ctx.modeling_opportunity_plan or {},
                llm_client=None,
                output_language=ctx.output_language,
            )
            opportunity_trace = build_skipped_stage_trace(
                stage="modeling_opportunity_decision",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped modeling opportunity decision LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["modeling_opportunity_decision"] = opportunity_trace.model_dump()
        ctx.modeling_opportunity_decision_path = ctx.store.save_json(
            ctx.task_id,
            "modeling_opportunity_decision.json",
            ctx.modeling_opportunity_decision,
        )
        self.artifacts.persist_llm_trace(ctx)

        ctx.modeling_outcome = build_modeling_outcome(
            self._report(ctx),
            modeling_opportunity_plan=ctx.modeling_opportunity_plan,
            modeling_opportunity_decision=ctx.modeling_opportunity_decision,
            dataset=str(ctx.manifest.get("uploaded_filename") or ctx.task_id),
        )
        if not _stage_enabled_by_policy(ctx, "modeling_outcome_interpretation_enabled"):
            outcome_interpretation, _ = build_modeling_outcome_interpretation_with_trace(
                ctx.modeling_outcome,
                llm_client=None,
                output_language=ctx.output_language,
            )
            outcome_trace = _profile_disabled_trace(
                ctx,
                stage="modeling_outcome_interpretation",
                policy_key="modeling_outcome_interpretation_enabled",
            )
        elif ctx.run_budget.has_budget_for_stage("modeling_outcome_interpretation"):
            outcome_interpretation, outcome_trace = build_modeling_outcome_interpretation_with_trace(
                ctx.modeling_outcome,
                llm_client=ctx.llm_client,
                output_language=ctx.output_language,
            )
        else:
            outcome_interpretation, _ = build_modeling_outcome_interpretation_with_trace(
                ctx.modeling_outcome,
                llm_client=None,
                output_language=ctx.output_language,
            )
            outcome_trace = build_skipped_stage_trace(
                stage="modeling_outcome_interpretation",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped modeling outcome interpretation LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["modeling_outcome_interpretation"] = outcome_trace.model_dump()
        ctx.modeling_outcome_interpretation = (
            None if outcome_trace.status in {"disabled", "skipped_by_profile"} else outcome_interpretation
        )
        ctx.modeling_outcome_path = ctx.store.save_json(
            ctx.task_id,
            "modeling_outcome.json",
            {
                **ctx.modeling_outcome,
                "llm_outcome_interpretation": ctx.modeling_outcome_interpretation or {},
                "llm_outcome_trace": outcome_trace.model_dump(),
            },
        )
        self.artifacts.record_modeling_outcome(ctx)
        self.artifacts.persist_llm_trace(ctx)

    def _build_final_synthesis(self, ctx: AnalysisRunContext) -> None:
        rows = action_plan_rows(
            self._report(ctx),
            self._schema_mapping(ctx),
            ctx.dataset_profile,
            ctx.analysis_focus,
            self._evidence_pack(ctx),
        )
        if not _stage_enabled_by_policy(ctx, "final_synthesis_enabled"):
            ctx.final_synthesis, _ = build_final_synthesis_with_trace(
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                evidence_pack=self._evidence_pack(ctx),
                modeling_outcome=ctx.modeling_outcome,
                modeling_outcome_interpretation=ctx.modeling_outcome_interpretation,
                action_plan_rows=rows,
                chart_selection_plan=ctx.chart_selection_plan,
                llm_client=None,
                output_language=ctx.output_language,
            )
            synthesis_trace = _profile_disabled_trace(
                ctx,
                stage="final_synthesis",
                policy_key="final_synthesis_enabled",
            )
        elif ctx.run_budget.has_budget_for_stage("final_synthesis"):
            ctx.final_synthesis, synthesis_trace = build_final_synthesis_with_trace(
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                evidence_pack=self._evidence_pack(ctx),
                modeling_outcome=ctx.modeling_outcome,
                modeling_outcome_interpretation=ctx.modeling_outcome_interpretation,
                action_plan_rows=rows,
                chart_selection_plan=ctx.chart_selection_plan,
                llm_client=ctx.llm_client,
                output_language=ctx.output_language,
            )
        else:
            ctx.final_synthesis, synthesis_trace = build_final_synthesis_with_trace(
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                evidence_pack=self._evidence_pack(ctx),
                modeling_outcome=ctx.modeling_outcome,
                modeling_outcome_interpretation=ctx.modeling_outcome_interpretation,
                action_plan_rows=rows,
                chart_selection_plan=ctx.chart_selection_plan,
                llm_client=None,
                output_language=ctx.output_language,
            )
            synthesis_trace = build_skipped_stage_trace(
                stage="final_synthesis",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped final synthesis LLM stage because run budget was exhausted.",
            )
        ctx.final_synthesis = _guard_final_synthesis_for_capabilities(
            ctx.final_synthesis,
            report=self._report(ctx),
            schema_mapping=self._schema_mapping(ctx),
            output_language=ctx.output_language,
        )
        trace_payload = synthesis_trace.model_dump()
        metadata = (ctx.final_synthesis or {}).get("metadata") if isinstance(ctx.final_synthesis, dict) else {}
        if isinstance(metadata, dict):
            trace_payload.update(
                {
                    "final_synthesis_llm_status": metadata.get("final_synthesis_llm_status"),
                    "final_synthesis_used_llm": metadata.get("final_synthesis_used_llm"),
                    "final_synthesis_fallback_reason": metadata.get("final_synthesis_fallback_reason"),
                    "dropped_generic_bullets_count": metadata.get("dropped_generic_bullets_count", 0),
                    "dropped_process_log_bullets_count": metadata.get("dropped_process_log_bullets_count", 0),
                    "dropped_unsupported_bullets_count": metadata.get("dropped_unsupported_bullets_count", 0),
                    "final_action_display_text_used_count": metadata.get("final_action_display_text_used_count", 0),
                    "final_action_display_text_missing_count": metadata.get("final_action_display_text_missing_count", 0),
                    "final_action_display_text_fallback_count": metadata.get("final_action_display_text_fallback_count", 0),
                    "final_action_display_text_rejected_count": metadata.get("final_action_display_text_rejected_count", 0),
                    "final_action_deduped_count": metadata.get("final_action_deduped_count", 0),
                    "final_action_render_mode": metadata.get("final_action_render_mode"),
                }
            )
        ctx.llm_trace["final_synthesis"] = trace_payload
        ctx.final_synthesis_path = ctx.store.save_json(
            ctx.task_id,
            "final_synthesis.json",
            ctx.final_synthesis or {},
        )
        self.artifacts.record_final_synthesis(ctx)
        self.artifacts.persist_llm_trace(ctx)

    def _build_and_execute_notebook(self, ctx: AnalysisRunContext) -> None:
        report_payload = self._report(ctx).model_dump()
        report_payload["output_language"] = ctx.output_language
        ctx.report_json_path = ctx.store.save_json(
            ctx.task_id, "report.json", report_payload
        )
        ctx.notebook_outline_path = ctx.store.save_json(
            ctx.task_id, "notebook_outline.json", self._notebook_outline(ctx).model_dump()
        )
        ctx.notebook_narrative_path = ctx.store.save_json(
            ctx.task_id, "notebook_narrative.json", self._notebook_narrative(ctx).model_dump()
        )
        ctx.notebook_content_path = ctx.store.save_json(
            ctx.task_id, "notebook_content.json", self._notebook_content(ctx).model_dump()
        )
        ctx.report_html_path = ctx.store.save_text(
            ctx.task_id, "report.html", build_html_report(self._report(ctx), output_language=ctx.output_language)
        )
        ctx.business_review_path = ctx.store.save_text(
            ctx.task_id,
            "business_review.md",
            build_business_review(self._report(ctx), output_language=ctx.output_language),
        )
        ctx.notebook_path = build_notebook(
            task_id=ctx.task_id,
            output_dir=ctx.store.task_dir(ctx.task_id),
            report=self._report(ctx),
            schema_mapping=self._schema_mapping(ctx),
            plan=self._analysis_plan(ctx),
            dataset_profile=ctx.dataset_profile,
            analysis_focus=ctx.analysis_focus,
            outline=self._notebook_outline(ctx),
            narrative=self._notebook_narrative(ctx),
            content_plan=self._notebook_content(ctx),
            evidence_pack=self._evidence_pack(ctx),
            modeling_interpretation=ctx.modeling_interpretation,
            modeling_opportunity_decision=ctx.modeling_opportunity_decision,
            modeling_outcome=ctx.modeling_outcome,
            modeling_outcome_interpretation=ctx.modeling_outcome_interpretation,
            final_synthesis=ctx.final_synthesis,
            notebook_output_mode=ctx.settings.notebook_output_mode,
            output_language=ctx.output_language,
        )
        ctx.executed_notebook_path = ctx.store.task_dir(ctx.task_id) / "analysis.executed.ipynb"
        ctx.executed_notebook_path = execute_notebook(
            notebook_path=self._notebook_path(ctx),
            output_path=self._executed_notebook_path(ctx),
            working_dir=ctx.store.task_dir(ctx.task_id),
        )
        ctx.chart_contexts = extract_postrun_chart_contexts(
            self._executed_notebook_path(ctx),
            self._notebook_outline(ctx),
            chart_selection_plan=self._chart_selection_plan(ctx),
        )
        ctx.initial_chart_contexts = list(ctx.chart_contexts)

    def _apply_revision_if_needed(self, ctx: AnalysisRunContext) -> None:
        profile_max_revision_decisions = int(ctx.llm_profile_policy.get("max_revision_decisions") or 0)
        if not ctx.llm_profile_policy.get("notebook_revision_enabled", True):
            ctx.profile_limited_stage_count += 1
        if not ctx.run_budget.stage_allowed("notebook_revision_decision"):
            ctx.revision_plan = NotebookRevisionPlan()
            revision_trace = build_skipped_stage_trace(
                stage="notebook_revision_decision",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_demo_safe",
                reason="Skipped notebook revision decision LLM stage in demo-safe profile.",
            )
        elif ctx.run_budget.has_budget_for_stage("notebook_revision_decision"):
            ctx.revision_plan, revision_trace = build_notebook_revision_plan_with_trace(
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                chart_contexts=ctx.chart_contexts,
                llm_client=ctx.llm_client,
                max_revision_decisions=(
                    min(1, profile_max_revision_decisions)
                    if ctx.settings.demo_safe
                    else profile_max_revision_decisions
                ),
                evidence_pack=self._llm_evidence_pack(ctx),
                llm_profile=ctx.resolved_llm_profile,
            )
        else:
            ctx.revision_plan = NotebookRevisionPlan()
            revision_trace = build_skipped_stage_trace(
                stage="notebook_revision_decision",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped notebook revision decision LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["notebook_revision_decision"] = revision_trace.model_dump()
        self.artifacts.persist_llm_trace(ctx)
        ctx.revision_decisions_path = ctx.store.save_json(
            ctx.task_id, "revision_decisions.json", self._revision_plan(ctx).model_dump()
        )
        if self._revision_plan(ctx).decisions:
            output_revision_plan = revision_plan_for_output(
                self._revision_plan(ctx),
                notebook_output_mode=ctx.settings.notebook_output_mode,
            )
            apply_notebook_revision_plan(
                notebook_path=self._notebook_path(ctx),
                output_path=self._notebook_path(ctx),
                revision_plan=self._revision_plan(ctx),
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                notebook_output_mode=ctx.settings.notebook_output_mode,
            )
            ctx.executed_notebook_path = execute_notebook(
                notebook_path=self._notebook_path(ctx),
                output_path=self._executed_notebook_path(ctx),
                working_dir=ctx.store.task_dir(ctx.task_id),
            )
            ctx.chart_contexts = extract_postrun_chart_contexts(
                self._executed_notebook_path(ctx),
                augment_outline_with_revision_sections(
                    self._notebook_outline(ctx), output_revision_plan
                ),
                chart_selection_plan=self._chart_selection_plan(ctx),
            )

    def _apply_postrun_reflections(self, ctx: AnalysisRunContext) -> None:
        ctx.chart_contexts_path = ctx.store.save_json(
            ctx.task_id, "postrun_chart_contexts.json", ctx.chart_contexts
        )
        ctx.max_postrun_reflections_resolved = min(
            ctx.settings.max_postrun_reflections,
            resolve_max_postrun_reflections(ctx.llm_profile_policy, len(ctx.chart_contexts)),
        )
        if (
            ctx.max_postrun_reflections_resolved < len(ctx.chart_contexts)
            or not ctx.allow_postrun_fallback_reflections
        ):
            ctx.profile_limited_stage_count += 1
        if not _stage_enabled_by_policy(ctx, "postrun_reflection_enabled"):
            ctx.postrun_reflections = {}
            reflection_trace = _profile_disabled_trace(
                ctx,
                stage="postrun_chart_reflection",
                policy_key="postrun_reflection_enabled",
            )
        elif ctx.run_budget.has_budget_for_stage("postrun_chart_reflection"):
            ctx.postrun_reflections, reflection_trace = build_postrun_chart_reflections(
                chart_contexts=ctx.chart_contexts,
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                llm_client=ctx.llm_client,
                max_llm_reflections=ctx.max_postrun_reflections_resolved,
                allow_fallback_reflections=ctx.allow_postrun_fallback_reflections,
                llm_profile=ctx.resolved_llm_profile,
                output_language=ctx.output_language,
            )
        else:
            ctx.postrun_reflections = {}
            reflection_trace = build_skipped_stage_trace(
                stage="postrun_chart_reflection",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status="skipped_by_budget",
                reason="Skipped post-run chart reflection LLM stage because run budget was exhausted.",
            )
        ctx.llm_trace["postrun_chart_reflection"] = reflection_trace.model_dump()
        self.artifacts.persist_llm_trace(ctx)
        if ctx.postrun_reflections:
            apply_postrun_chart_reflections(
                notebook_path=self._executed_notebook_path(ctx),
                output_path=self._executed_notebook_path(ctx),
                reflections_by_chart_cell=ctx.postrun_reflections,
                notebook_output_mode=ctx.settings.notebook_output_mode,
                output_language=ctx.output_language,
            )
        ctx.postrun_reflections_path = ctx.store.save_json(
            ctx.task_id,
            "postrun_chart_reflections.json",
            ctx.postrun_reflections,
        )

    def _build_client_report(self, ctx: AnalysisRunContext) -> None:
        if not _stage_enabled_by_policy(ctx, "client_report_enabled"):
            ctx.client_report_payload, _ = build_client_report_payload_with_trace(
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                evidence_pack=self._evidence_pack(ctx),
                section_priority=self._section_priority(ctx),
                llm_client=None,
                llm_evidence_pack=self._llm_evidence_pack(ctx),
                final_synthesis=ctx.final_synthesis,
                modeling_outcome=ctx.modeling_outcome,
                modeling_outcome_interpretation=ctx.modeling_outcome_interpretation,
                output_language=ctx.output_language,
            )
            client_report_trace = _profile_disabled_trace(
                ctx,
                stage="client_report",
                policy_key="client_report_enabled",
            )
        elif ctx.run_budget.has_budget_for_stage("client_report"):
            ctx.client_report_payload, client_report_trace = build_client_report_payload_with_trace(
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                evidence_pack=self._evidence_pack(ctx),
                section_priority=self._section_priority(ctx),
                llm_client=ctx.llm_client,
                llm_evidence_pack=self._llm_evidence_pack(ctx),
                final_synthesis=ctx.final_synthesis,
                modeling_outcome=ctx.modeling_outcome,
                modeling_outcome_interpretation=ctx.modeling_outcome_interpretation,
                output_language=ctx.output_language,
            )
        else:
            ctx.client_report_payload, client_report_trace = build_client_report_payload_with_trace(
                report=self._report(ctx),
                schema_mapping=self._schema_mapping(ctx),
                dataset_profile=ctx.dataset_profile,
                analysis_focus=ctx.analysis_focus,
                evidence_pack=self._evidence_pack(ctx),
                section_priority=self._section_priority(ctx),
                llm_client=None,
                llm_evidence_pack=self._llm_evidence_pack(ctx),
                final_synthesis=ctx.final_synthesis,
                modeling_outcome=ctx.modeling_outcome,
                modeling_outcome_interpretation=ctx.modeling_outcome_interpretation,
                output_language=ctx.output_language,
            )
            client_report_trace = build_skipped_stage_trace(
                stage="client_report",
                llm_client=ctx.llm_client,
                budget=ctx.run_budget,
                status=(
                    "skipped_by_budget"
                    if ctx.run_budget.stage_allowed("client_report")
                    else "skipped_by_demo_safe"
                ),
                reason="Skipped client report LLM stage because run budget or demo-safe settings did not allow it.",
            )
        ctx.llm_trace["client_report"] = client_report_trace.model_dump()
        ctx.client_report_json_path = ctx.store.save_json(
            ctx.task_id, "client_report.json", self._client_report_payload(ctx)
        )
        ctx.client_report_html_path = ctx.store.save_text(
            ctx.task_id,
            "client_report.html",
            render_client_report_html(
                self._client_report_payload(ctx),
                output_language=ctx.output_language,
            ),
        )
        self.artifacts.persist_llm_trace(ctx)

    def _build_agent_state(self, ctx: AnalysisRunContext) -> None:
        notebook_agent_state = build_notebook_agent_loop_state(
            task_id=ctx.task_id,
            dataset_type=self._report(ctx).dataset_type,
            outline_section_ids=[section.section_id for section in self._notebook_outline(ctx).sections],
            initial_chart_contexts=ctx.initial_chart_contexts,
            final_chart_contexts=ctx.chart_contexts,
            revision_plan=self._revision_plan(ctx),
            llm_trace_payload=self.artifacts.llm_trace_payload(ctx),
        )
        ctx.notebook_agent_state_path = ctx.store.save_json(
            ctx.task_id, "notebook_agent_state.json", notebook_agent_state.model_dump()
        )
        artifact_name_stem = str(ctx.manifest.get("artifact_name_stem") or "analysis").strip() or "analysis"
        ctx.named_notebook_path = ctx.store.task_dir(ctx.task_id) / f"{artifact_name_stem}.ipynb"
        if ctx.named_notebook_path != self._executed_notebook_path(ctx):
            shutil.copyfile(self._executed_notebook_path(ctx), ctx.named_notebook_path)

    def _persist_completed_artifacts(self, ctx: AnalysisRunContext) -> None:
        self.artifacts.persist_completed_artifacts(ctx)

    def _mark_task_completed(self, ctx: AnalysisRunContext) -> None:
        self.artifacts.mark_task_completed(ctx)

    def _build_run_response(self, ctx: AnalysisRunContext) -> RunResponse:
        return RunResponse(
            task_id=ctx.task_id,
            status="completed",
            output_language=ctx.output_language,
            report=self._report(ctx),
            business_review=BusinessReviewArtifact(
                filename=self._business_review_path(ctx).name,
                path=str(self._business_review_path(ctx)),
            ),
            notebook=NotebookArtifact(
                filename=self._named_notebook_path(ctx).name,
                path=str(self._named_notebook_path(ctx)),
            ),
            llm_trace={
                stage: trace_payload
                for stage, trace_payload in ctx.llm_trace.items()
            },
        )

    def _analysis_plan(self, ctx: AnalysisRunContext) -> AnalysisPlan:
        if ctx.analysis_plan is None:
            raise RuntimeError("analysis_plan has not been loaded")
        return ctx.analysis_plan

    def _business_review_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.business_review_path is None:
            raise RuntimeError("business_review_path has not been built")
        return ctx.business_review_path

    def _chart_contexts_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.chart_contexts_path is None:
            raise RuntimeError("chart_contexts_path has not been built")
        return ctx.chart_contexts_path

    def _chart_selection_plan(self, ctx: AnalysisRunContext) -> dict[str, object]:
        if ctx.chart_selection_plan is None:
            raise RuntimeError("chart_selection_plan has not been built")
        return ctx.chart_selection_plan

    def _client_report_html_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.client_report_html_path is None:
            raise RuntimeError("client_report_html_path has not been built")
        return ctx.client_report_html_path

    def _client_report_json_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.client_report_json_path is None:
            raise RuntimeError("client_report_json_path has not been built")
        return ctx.client_report_json_path

    def _client_report_payload(self, ctx: AnalysisRunContext) -> dict[str, object]:
        if ctx.client_report_payload is None:
            raise RuntimeError("client_report_payload has not been built")
        return ctx.client_report_payload

    def _evidence_pack(self, ctx: AnalysisRunContext) -> dict[str, object]:
        if ctx.evidence_pack is None:
            raise RuntimeError("evidence_pack has not been built")
        return ctx.evidence_pack

    def _executed_notebook_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.executed_notebook_path is None:
            raise RuntimeError("executed_notebook_path has not been built")
        return ctx.executed_notebook_path

    def _llm_evidence_pack(self, ctx: AnalysisRunContext) -> dict[str, object]:
        if ctx.llm_evidence_pack is None:
            raise RuntimeError("llm_evidence_pack has not been built")
        return ctx.llm_evidence_pack

    def _llm_trace_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.llm_trace_path is None:
            raise RuntimeError("llm_trace_path has not been built")
        return ctx.llm_trace_path

    def _named_notebook_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.named_notebook_path is None:
            raise RuntimeError("named_notebook_path has not been built")
        return ctx.named_notebook_path

    def _notebook_agent_state_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.notebook_agent_state_path is None:
            raise RuntimeError("notebook_agent_state_path has not been built")
        return ctx.notebook_agent_state_path

    def _notebook_content(self, ctx: AnalysisRunContext):
        if ctx.notebook_content is None:
            raise RuntimeError("notebook_content has not been built")
        return ctx.notebook_content

    def _notebook_content_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.notebook_content_path is None:
            raise RuntimeError("notebook_content_path has not been built")
        return ctx.notebook_content_path

    def _notebook_narrative(self, ctx: AnalysisRunContext):
        if ctx.notebook_narrative is None:
            raise RuntimeError("notebook_narrative has not been built")
        return ctx.notebook_narrative

    def _notebook_narrative_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.notebook_narrative_path is None:
            raise RuntimeError("notebook_narrative_path has not been built")
        return ctx.notebook_narrative_path

    def _notebook_outline(self, ctx: AnalysisRunContext):
        if ctx.notebook_outline is None:
            raise RuntimeError("notebook_outline has not been built")
        return ctx.notebook_outline

    def _notebook_outline_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.notebook_outline_path is None:
            raise RuntimeError("notebook_outline_path has not been built")
        return ctx.notebook_outline_path

    def _notebook_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.notebook_path is None:
            raise RuntimeError("notebook_path has not been built")
        return ctx.notebook_path

    def _postrun_reflections_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.postrun_reflections_path is None:
            raise RuntimeError("postrun_reflections_path has not been built")
        return ctx.postrun_reflections_path

    def _raw_csv_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.raw_csv_path is None:
            raise RuntimeError("raw_csv_path has not been loaded")
        return ctx.raw_csv_path

    def _report(self, ctx: AnalysisRunContext) -> AnalysisReport:
        if ctx.report is None:
            raise RuntimeError("report has not been built")
        return ctx.report

    def _report_html_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.report_html_path is None:
            raise RuntimeError("report_html_path has not been built")
        return ctx.report_html_path

    def _report_json_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.report_json_path is None:
            raise RuntimeError("report_json_path has not been built")
        return ctx.report_json_path

    def _revision_decisions_path(self, ctx: AnalysisRunContext) -> Path:
        if ctx.revision_decisions_path is None:
            raise RuntimeError("revision_decisions_path has not been built")
        return ctx.revision_decisions_path

    def _revision_plan(self, ctx: AnalysisRunContext) -> NotebookRevisionPlan:
        if ctx.revision_plan is None:
            raise RuntimeError("revision_plan has not been built")
        return ctx.revision_plan

    def _schema_mapping(self, ctx: AnalysisRunContext) -> SchemaMapping:
        if ctx.schema_mapping is None:
            raise RuntimeError("schema_mapping has not been loaded")
        return ctx.schema_mapping

    def _section_priority(self, ctx: AnalysisRunContext) -> dict[str, object]:
        if ctx.section_priority is None:
            raise RuntimeError("section_priority has not been built")
        return ctx.section_priority
