from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.application.runtime_runner import get_runtime_runner
from app.application.runtime_state import (
    RUNNING_STATUSES,
    TERMINAL_STATUSES,
    RuntimeStateStore,
    runtime_response_from_task,
)
from app.application.tasks.task_service import (
    AnalysisResultsNotAvailableError,
    ArtifactFileNotFoundError,
    ArtifactNotFoundError,
    InvalidUploadError,
    TaskNotFoundError,
    TaskService,
)
from app.db.session import get_session
from app.schemas.tasks import (
    RunResponse,
    TaskCreateResponse,
    TaskDetailResponse,
    UploadResponse,
)

router = APIRouter(tags=["analysis"])

def _task_service(session: Session) -> TaskService:
    return TaskService(session)


def _raise_http_error(exc: Exception) -> None:
    if isinstance(exc, TaskNotFoundError):
        raise HTTPException(status_code=404, detail="Task not found") from exc
    if isinstance(exc, ArtifactNotFoundError):
        raise HTTPException(status_code=404, detail="Artifact not found") from exc
    if isinstance(exc, ArtifactFileNotFoundError):
        raise HTTPException(status_code=404, detail="Artifact file not found") from exc
    if isinstance(exc, AnalysisResultsNotAvailableError):
        raise HTTPException(
            status_code=404, detail="Analysis results are not available"
        ) from exc
    if isinstance(exc, InvalidUploadError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise exc


@router.post("/analysis/tasks", response_model=TaskCreateResponse, status_code=201)
def create_task(session: Session = Depends(get_session)) -> TaskCreateResponse:
    return _task_service(session).create_task()


@router.get("/analysis/tasks/{task_id}", response_model=TaskDetailResponse)
def get_task(task_id: str, session: Session = Depends(get_session)) -> TaskDetailResponse:
    try:
        return _task_service(session).get_task_detail(task_id)
    except TaskNotFoundError as exc:
        _raise_http_error(exc)


@router.get("/analysis/tasks/{task_id}/artifacts/{artifact_name}")
def download_artifact(
    task_id: str, artifact_name: str, session: Session = Depends(get_session)
) -> FileResponse:
    try:
        artifact = _task_service(session).resolve_artifact(task_id, artifact_name)
    except (TaskNotFoundError, ArtifactNotFoundError, ArtifactFileNotFoundError) as exc:
        _raise_http_error(exc)

    artifact_path = artifact.path
    media_type = artifact.media_type
    content_disposition_type = "inline" if media_type.startswith("text/html") else "attachment"
    return FileResponse(
        artifact_path,
        filename=artifact_path.name,
        media_type=media_type,
        content_disposition_type=content_disposition_type,
    )


@router.get("/analysis/tasks/{task_id}/results")
def get_analysis_results(task_id: str, session: Session = Depends(get_session)) -> dict[str, object]:
    try:
        return _task_service(session).get_analysis_results(task_id)
    except (TaskNotFoundError, AnalysisResultsNotAvailableError) as exc:
        _raise_http_error(exc)


@router.post("/analysis/tasks/{task_id}/upload", response_model=UploadResponse)
def upload_csv(
    task_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
) -> UploadResponse:
    try:
        return _task_service(session).upload_csv(
            task_id, filename=file.filename, content=file.file.read()
        )
    except (TaskNotFoundError, InvalidUploadError) as exc:
        _raise_http_error(exc)


@router.post(
    "/analysis/tasks/{task_id}/run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def run_task(
    task_id: str,
    llm_profile: str | None = Query(default=None),
    output_language: str | None = Query(default=None),
    user_goal: str | None = Query(default=None, max_length=1000),
    session: Session = Depends(get_session),
) -> RunResponse:
    try:
        task = _task_service(session).get_task_model(task_id)
    except TaskNotFoundError as exc:
        _raise_http_error(exc)
    resolved_output_language = output_language or "en"
    if task.status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="Task already finished")
    if task.status in RUNNING_STATUSES:
        return runtime_response_from_task(
            task,
            output_language=resolved_output_language,
        )
    if task.status != "uploaded":
        raise HTTPException(status_code=400, detail="Task has not been uploaded")

    queued_task = RuntimeStateStore().mark_queued(task_id)
    get_runtime_runner().submit(
        task_id,
        llm_profile=llm_profile,
        output_language=output_language,
        user_goal=user_goal,
    )
    return runtime_response_from_task(
        queued_task,
        output_language=resolved_output_language,
    )


@router.post("/analysis/tasks/{task_id}/cancel", response_model=RunResponse)
def cancel_task(
    task_id: str,
    session: Session = Depends(get_session),
) -> RunResponse:
    try:
        task = _task_service(session).get_task_model(task_id)
    except TaskNotFoundError as exc:
        _raise_http_error(exc)
    if task.status in TERMINAL_STATUSES:
        raise HTTPException(status_code=409, detail="Task already finished")
    if task.status == "cancelling":
        return runtime_response_from_task(task)
    if task.status not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Task is not running")

    cancelling_task = RuntimeStateStore().request_cancel(task_id)
    return runtime_response_from_task(cancelling_task)
