from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import AnalysisTask, utc_now
from app.db.session import get_engine, init_db
from app.schemas.tasks import RunResponse, TaskDetailResponse


RUNNING_STATUSES = {"queued", "running", "cancelling"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
LLM_STATUSES = {"unknown", "off", "pending", "called", "failed", "fallback"}
LLM_STAGES = {
    "schema_mapping",
    "evidence_and_chart_planning",
    "notebook_planning",
    "summary_and_modeling_interpretation",
    "final_synthesis",
    "notebook_revision",
    "postrun_reflection",
    "client_report",
}
REPORT_BUILD_STAGES = {
    "client_report",
    "agent_state",
    "completed_artifact_persistence",
    "task_completion",
}


class AnalysisRunCancelled(RuntimeError):
    def __init__(self, task_id: str) -> None:
        super().__init__("Analysis run cancelled")
        self.task_id = task_id


def _iso_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def safe_public_error_message(exc: Exception, limit: int = 180) -> str:
    raw = " ".join(str(exc).split())
    raw = re.sub(r"sk-[A-Za-z0-9_\-]+", "[redacted]", raw)
    raw = re.sub(r"https?://\S+", "[redacted-url]", raw)
    raw = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", raw, flags=re.IGNORECASE)
    message = f"{type(exc).__name__}: {raw}" if raw else type(exc).__name__
    if len(message) > limit:
        message = message[: limit - 3] + "..."
    return message


def derive_llm_runtime_status(
    llm_trace: dict[str, object],
    *,
    current_status: str = "unknown",
) -> tuple[str, str]:
    payloads = [
        payload
        for stage, payload in llm_trace.items()
        if stage != "summary" and isinstance(payload, dict)
    ]
    if not payloads:
        return current_status if current_status in LLM_STATUSES else "unknown", ""

    if not any(bool(payload.get("llm_enabled")) for payload in payloads):
        return "off", "LLM is disabled."

    def _has_success(payload: dict[str, object]) -> bool:
        status = str(payload.get("status") or "")
        return bool(payload.get("attempted")) and (
            bool(payload.get("applied"))
            or status in {"llm_applied", "llm_partial", "llm_success"}
        )

    if any(_has_success(payload) for payload in payloads):
        return "called", "LLM response received."

    fallback_statuses = {
        "fallback",
        "fallback_on_error",
        "fallback_invalid_payload",
        "llm_no_change",
    }
    if any(
        bool(payload.get("attempted"))
        and (
            str(payload.get("status") or "") in fallback_statuses
            or bool(payload.get("fallback_type"))
        )
        for payload in payloads
    ):
        return "fallback", "LLM unavailable. Continuing with deterministic fallback."

    if any(bool(payload.get("attempted")) and payload.get("error_type") for payload in payloads):
        return "failed", "LLM call failed."

    if current_status == "pending":
        return "pending", "Waiting for the first LLM call."
    return "unknown", ""


def runtime_response_from_task(
    task: AnalysisTask,
    *,
    output_language: str = "en",
) -> RunResponse:
    return RunResponse(
        task_id=task.task_id,
        status=task.status,
        output_language=output_language,
        current_stage=task.current_stage,
        current_stage_label=task.current_stage_label,
        status_message=task.status_message,
        cancel_requested=bool(task.cancel_requested),
        llm_status=task.llm_status or "unknown",
        llm_message=task.llm_message,
        started_at=_iso_or_none(task.started_at),
        heartbeat_at=_iso_or_none(task.heartbeat_at),
        finished_at=_iso_or_none(task.finished_at),
        error_message=task.error_message,
    )


def task_detail_from_model(
    task: AnalysisTask,
    *,
    artifact_manifest: dict[str, object],
) -> TaskDetailResponse:
    return TaskDetailResponse(
        task_id=task.task_id,
        status=task.status,
        dataset_type=task.dataset_type,
        current_stage=task.current_stage,
        current_stage_label=task.current_stage_label,
        status_message=task.status_message,
        cancel_requested=bool(task.cancel_requested),
        llm_status=task.llm_status or "unknown",
        llm_message=task.llm_message,
        started_at=_iso_or_none(task.started_at),
        heartbeat_at=_iso_or_none(task.heartbeat_at),
        finished_at=_iso_or_none(task.finished_at),
        artifact_manifest=artifact_manifest,
        error_message=task.error_message,
    )


class RuntimeStateStore:
    def __init__(
        self,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self._session_factory = session_factory

    def _new_session(self) -> Session:
        if self._session_factory is not None:
            return self._session_factory()
        init_db()
        factory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
        return factory()

    def _mutate(self, task_id: str, mutator: Callable[[AnalysisTask], None]) -> AnalysisTask:
        session = self._new_session()
        try:
            task = session.get(AnalysisTask, task_id)
            if task is None:
                raise KeyError(task_id)
            mutator(task)
            session.add(task)
            session.commit()
            session.refresh(task)
            session.expunge(task)
            return task
        finally:
            session.close()

    def get_task(self, task_id: str) -> AnalysisTask:
        session = self._new_session()
        try:
            task = session.get(AnalysisTask, task_id)
            if task is None:
                raise KeyError(task_id)
            session.expunge(task)
            return task
        finally:
            session.close()

    def mark_uploaded(self, task_id: str, *, dataset_type: str | None = None) -> AnalysisTask:
        return self._mutate(
            task_id,
            lambda task: _set_uploaded(task, dataset_type=dataset_type),
        )

    def mark_queued(self, task_id: str) -> AnalysisTask:
        now = utc_now()

        def apply(task: AnalysisTask) -> None:
            task.status = "queued"
            task.cancel_requested = False
            task.current_stage = "queued"
            task.current_stage_label = "Queued"
            task.status_message = "Queued for background analysis."
            task.llm_status = task.llm_status or "unknown"
            task.started_at = None
            task.heartbeat_at = now
            task.finished_at = None
            task.error_message = None

        return self._mutate(task_id, apply)

    def mark_stage_started(
        self,
        task_id: str,
        *,
        stage_id: str,
        stage_label: str,
        llm_enabled: bool,
    ) -> AnalysisTask:
        now = utc_now()

        def apply(task: AnalysisTask) -> None:
            if task.cancel_requested and stage_id != "task_completion":
                task.status = "cancelling"
                task.status_message = (
                    "Stop requested. The current stage will finish before cancellation."
                )
                raise AnalysisRunCancelled(task_id)
            task.status = "running"
            task.current_stage = stage_id
            task.current_stage_label = stage_label
            task.status_message = _stage_message(stage_id, stage_label)
            task.started_at = task.started_at or now
            task.heartbeat_at = now
            if stage_id in LLM_STAGES:
                if llm_enabled:
                    task.llm_status = "pending"
                    task.llm_message = "Waiting for the first LLM call."
                elif task.llm_status in {"unknown", "pending"}:
                    task.llm_status = "off"
                    task.llm_message = "LLM is disabled."

        return self._mutate(task_id, apply)

    def mark_stage_finished(
        self,
        task_id: str,
        *,
        llm_trace: dict[str, object],
    ) -> AnalysisTask:
        def apply(task: AnalysisTask) -> None:
            status, message = derive_llm_runtime_status(
                llm_trace, current_status=task.llm_status or "unknown"
            )
            task.llm_status = status
            task.llm_message = message or task.llm_message
            task.heartbeat_at = utc_now()

        return self._mutate(task_id, apply)

    def heartbeat(self, task_id: str) -> AnalysisTask:
        return self._mutate(task_id, lambda task: setattr(task, "heartbeat_at", utc_now()))

    def request_cancel(self, task_id: str) -> AnalysisTask:
        def apply(task: AnalysisTask) -> None:
            if task.status in TERMINAL_STATUSES:
                raise ValueError("Task already finished")
            task.cancel_requested = True
            task.status = "cancelling"
            task.status_message = (
                "Stop requested. The current stage will finish before cancellation."
            )
            task.heartbeat_at = utc_now()

        return self._mutate(task_id, apply)

    def check_cancel_requested(self, task_id: str) -> None:
        task = self.get_task(task_id)
        if task.cancel_requested or task.status == "cancelling":
            raise AnalysisRunCancelled(task_id)

    def mark_completed(self, task_id: str) -> AnalysisTask:
        def apply(task: AnalysisTask) -> None:
            now = utc_now()
            task.status = "completed"
            task.current_stage = "task_completion"
            task.current_stage_label = "Mark task completed"
            task.status_message = "Analysis completed."
            task.cancel_requested = False
            task.heartbeat_at = now
            task.finished_at = now
            task.error_message = None
            if task.llm_status in {"unknown", "pending"}:
                task.llm_status = "off"
                task.llm_message = "LLM is disabled."

        return self._mutate(task_id, apply)

    def mark_cancelled(self, task_id: str) -> AnalysisTask:
        def apply(task: AnalysisTask) -> None:
            now = utc_now()
            task.status = "cancelled"
            task.status_message = "Analysis cancelled after the current stage finished."
            task.cancel_requested = True
            task.heartbeat_at = now
            task.finished_at = now

        return self._mutate(task_id, apply)

    def mark_failed(self, task_id: str, exc: Exception) -> AnalysisTask:
        def apply(task: AnalysisTask) -> None:
            now = utc_now()
            task.status = "failed"
            task.status_message = "Analysis failed."
            task.error_message = safe_public_error_message(exc)
            task.heartbeat_at = now
            task.finished_at = now
            if task.llm_status == "pending":
                task.llm_status = "failed"
                task.llm_message = "LLM call failed."

        return self._mutate(task_id, apply)


def _set_uploaded(task: AnalysisTask, *, dataset_type: str | None) -> None:
    task.status = "uploaded"
    task.dataset_type = dataset_type
    task.current_stage = "uploaded"
    task.current_stage_label = "Uploaded"
    task.status_message = "Upload completed."
    task.heartbeat_at = utc_now()


def _stage_message(stage_id: str, stage_label: str) -> str:
    if stage_id in REPORT_BUILD_STAGES:
        return "Building final reports and artifacts."
    return "Working on the current analysis stage."
