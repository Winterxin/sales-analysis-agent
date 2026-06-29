from __future__ import annotations

from pathlib import Path
import re

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
import pandas as pd
from sqlalchemy.orm import Session

from app.application.analysis_run_service import AnalysisRunService
from app.db.models import AnalysisTask
from app.db.session import get_session
from app.schemas.schema_mapping import SchemaMapping
from app.schemas.tasks import (
    RunResponse,
    TaskCreateResponse,
    TaskDetailResponse,
    UploadResponse,
)
from app.services.analysis_planner import build_analysis_plan
from app.services.analysis_results import build_analysis_results_payload
from app.services.artifact_store import ArtifactStore
from app.services.csv_ingestion import ingest_csv
from app.services.dataset_profile import build_dataset_profile, select_analysis_focuses
from app.services.llm_client import get_default_llm_client
from app.services.llm_trace_utils import with_llm_trace_summary
from app.services.output_language import DEFAULT_OUTPUT_LANGUAGE, normalize_output_language
from app.services.schema_mapper import map_schema_with_trace

router = APIRouter(tags=["analysis"])

ARTIFACT_ROUTE_MAP = {
    "business-review": ("business_review_md", "text/markdown; charset=utf-8"),
    "notebook": ("analysis_notebook", "application/x-ipynb+json"),
    "source-notebook": ("analysis_source_notebook", "application/x-ipynb+json"),
    "llm-trace": ("llm_trace_json", "application/json"),
    "agent-loop-state": ("notebook_agent_state_json", "application/json"),
    "revision-decisions": ("revision_decisions_json", "application/json"),
    "client-report-html": ("client_report_html", "text/html; charset=utf-8"),
    "client-report-json": ("client_report_json", "application/json"),
    "report-html": ("report_html", "text/html; charset=utf-8"),
    "report-json": ("report_json", "application/json"),
    "raw-csv": ("raw_csv", "text/csv; charset=utf-8"),
}


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


def _get_task_or_404(session: Session, task_id: str) -> AnalysisTask:
    task = session.get(AnalysisTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


def _resolve_artifact_path(task_id: str, artifact_name: str) -> tuple[Path, str]:
    manifest = ArtifactStore().load_manifest(task_id)
    try:
        manifest_key, media_type = ARTIFACT_ROUTE_MAP[artifact_name]
        artifact_path = Path(str(manifest["files"][manifest_key]))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Artifact not found") from exc

    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return artifact_path, media_type


def _read_csv_for_profile(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _artifact_stem_from_filename(filename: str | None) -> str:
    stem = Path(filename or "analysis").stem.strip().lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "analysis"


def _derive_sales_amount_if_needed(raw_path: Path, schema_mapping: SchemaMapping) -> None:
    mapped = set(schema_mapping.field_mapping.values())
    if "sales_amount" in mapped or not {"quantity", "unit_price"} <= mapped:
        return
    frame = _read_csv_for_profile(raw_path)
    canonical_to_original = {
        canonical: original for original, canonical in schema_mapping.field_mapping.items()
    }
    quantity_col = canonical_to_original["quantity"]
    unit_price_col = canonical_to_original["unit_price"]
    frame["__sales_amount"] = (
        pd.to_numeric(frame[quantity_col], errors="coerce").fillna(0)
        * pd.to_numeric(frame[unit_price_col], errors="coerce").fillna(0)
    )
    frame.to_csv(raw_path, index=False, encoding="utf-8")
    schema_mapping.field_mapping["__sales_amount"] = "sales_amount"
    schema_mapping.missing_required_fields = [
        field for field in schema_mapping.missing_required_fields if field != "sales_amount"
    ]
    if {"order_datetime", "quantity", "sales_amount"} <= set(schema_mapping.field_mapping.values()):
        schema_mapping.dataset_type = "sales_transaction"


def _normalize_raw_csv_encoding(raw_path: Path) -> None:
    frame = _read_csv_for_profile(raw_path)
    frame.to_csv(raw_path, index=False, encoding="utf-8")


@router.post("/analysis/tasks", response_model=TaskCreateResponse, status_code=201)
def create_task(session: Session = Depends(get_session)) -> TaskCreateResponse:
    store = ArtifactStore()
    task = AnalysisTask(status="created")
    session.add(task)
    session.commit()
    session.refresh(task)

    manifest = {
        "task_id": task.task_id,
        "status": task.status,
        "output_language": DEFAULT_OUTPUT_LANGUAGE,
        "files": {},
    }
    manifest_path = store.save_manifest(task.task_id, manifest)
    task.artifact_manifest_path = str(manifest_path)
    session.add(task)
    session.commit()

    return TaskCreateResponse(task_id=task.task_id, status=task.status)


@router.get("/analysis/tasks/{task_id}", response_model=TaskDetailResponse)
def get_task(task_id: str, session: Session = Depends(get_session)) -> TaskDetailResponse:
    task = _get_task_or_404(session, task_id)
    manifest = ArtifactStore().load_manifest(task_id)
    return TaskDetailResponse(
        task_id=task.task_id,
        status=task.status,
        dataset_type=task.dataset_type,
        artifact_manifest=manifest,
        error_message=task.error_message,
    )


@router.get("/analysis/tasks/{task_id}/artifacts/{artifact_name}")
def download_artifact(
    task_id: str, artifact_name: str, session: Session = Depends(get_session)
) -> FileResponse:
    _get_task_or_404(session, task_id)
    artifact_path, media_type = _resolve_artifact_path(task_id, artifact_name)
    content_disposition_type = "inline" if media_type.startswith("text/html") else "attachment"
    return FileResponse(
        artifact_path,
        filename=artifact_path.name,
        media_type=media_type,
        content_disposition_type=content_disposition_type,
    )


@router.get("/analysis/tasks/{task_id}/results")
def get_analysis_results(task_id: str, session: Session = Depends(get_session)) -> dict[str, object]:
    _get_task_or_404(session, task_id)
    manifest = ArtifactStore().load_manifest(task_id)
    try:
        return build_analysis_results_payload(task_id, manifest)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analysis results are not available") from exc


@router.post("/analysis/tasks/{task_id}/upload", response_model=UploadResponse)
def upload_csv(
    task_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> UploadResponse:
    task = _get_task_or_404(session, task_id)
    llm_client = get_default_llm_client()
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported")

    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    store = ArtifactStore()
    raw_path = store.save_bytes(task_id, "raw.csv", content)
    ingestion = ingest_csv(raw_path)
    schema_mapping, schema_trace = map_schema_with_trace(ingestion.columns, llm_client=llm_client)
    _normalize_raw_csv_encoding(raw_path)
    _derive_sales_amount_if_needed(raw_path, schema_mapping)
    ingestion = ingest_csv(raw_path)
    profile_frame = _read_csv_for_profile(raw_path)
    dataset_profile = build_dataset_profile(profile_frame, schema_mapping)
    analysis_focus = select_analysis_focuses(dataset_profile)
    analysis_plan = build_analysis_plan(schema_mapping, ingestion)
    llm_trace = {"schema_mapping": schema_trace.model_dump()}
    manifest = store.load_manifest(task_id)
    output_language = normalize_output_language(manifest.get("output_language"))
    llm_trace_path = store.save_json(
        task_id,
        "llm_trace.json",
        _llm_trace_artifact_payload(llm_trace, output_language=output_language),
    )
    dataset_profile_path = store.save_json(task_id, "dataset_profile.json", dataset_profile)
    analysis_focus_path = store.save_json(task_id, "analysis_focus.json", analysis_focus)

    manifest.update(
        {
            "status": "uploaded",
            "output_language": output_language,
            "uploaded_filename": file.filename,
            "artifact_name_stem": _artifact_stem_from_filename(file.filename),
            "ingestion": ingestion.model_dump(),
            "schema_mapping": schema_mapping.model_dump(),
            "dataset_profile": dataset_profile,
            "analysis_focus": analysis_focus,
            "analysis_plan": analysis_plan.model_dump(),
            "llm_trace": _llm_trace_artifact_payload(llm_trace, output_language=output_language),
            "files": {
                **manifest.get("files", {}),
                "llm_trace_json": str(llm_trace_path),
                "dataset_profile_json": str(dataset_profile_path),
                "analysis_focus_json": str(analysis_focus_path),
                "raw_csv": str(raw_path),
            },
        }
    )
    manifest_path = store.save_manifest(task_id, manifest)

    task.status = "uploaded"
    task.dataset_type = schema_mapping.dataset_type
    task.artifact_manifest_path = str(manifest_path)
    session.add(task)
    session.commit()

    return UploadResponse(
        task_id=task_id,
        status="uploaded",
        ingestion=ingestion,
        schema_mapping=schema_mapping,
        analysis_plan=analysis_plan,
        llm_trace={"schema_mapping": schema_trace},
    )


@router.post(
    "/analysis/tasks/{task_id}/run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_task(
    task_id: str,
    llm_profile: str | None = Query(default=None),
    output_language: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> RunResponse:
    _get_task_or_404(session, task_id)
    return AnalysisRunService(session).run(
        task_id,
        llm_profile=llm_profile,
        output_language=output_language,
    )
