from __future__ import annotations

from types import SimpleNamespace
import time

from fastapi.testclient import TestClient


def _create_uploaded_task(client: TestClient, sample_csv_path) -> str:
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]
    with sample_csv_path.open("rb") as csv_file:
        upload_response = client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )
    assert upload_response.status_code == 200
    return task_id


def _wait_for_status(client: TestClient, task_id: str, status: str) -> dict[str, object]:
    deadline = time.monotonic() + 5.0
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/analysis/tasks/{task_id}")
        assert response.status_code == 200
        last = response.json()
        if last.get("status") == status:
            return last
        time.sleep(0.05)
    raise AssertionError(f"Task did not reach {status}; last={last}")


def test_cancel_running_task_sets_cancelling_without_claiming_immediate_stop(
    client: TestClient, sample_csv_path, monkeypatch
) -> None:
    from app.application.analysis_run_service import AnalysisRunService
    from app.application.runtime_state import AnalysisRunCancelled

    def slow_run(self, task_id: str, **_: object) -> None:
        time.sleep(0.8)
        raise AnalysisRunCancelled(task_id)

    monkeypatch.setattr(AnalysisRunService, "run", slow_run)
    task_id = _create_uploaded_task(client, sample_csv_path)
    assert client.post(f"/api/v1/analysis/tasks/{task_id}/run").status_code == 202

    cancel_response = client.post(f"/api/v1/analysis/tasks/{task_id}/cancel")

    assert cancel_response.status_code == 200
    body = cancel_response.json()
    assert body["status"] == "cancelling"
    assert body["cancel_requested"] is True
    assert "current stage will finish" in body["status_message"]
    _wait_for_status(client, task_id, "cancelled")


def test_cancel_rejects_terminal_task(client: TestClient) -> None:
    create_response = client.post("/api/v1/analysis/tasks")
    assert create_response.status_code == 201
    task_id = create_response.json()["task_id"]

    from app.application.runtime_state import RuntimeStateStore

    RuntimeStateStore().mark_completed(task_id)

    cancel_response = client.post(f"/api/v1/analysis/tasks/{task_id}/cancel")

    assert cancel_response.status_code == 409
    assert "already finished" in cancel_response.json()["detail"]


def test_cancel_requested_between_stages_stops_later_stage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SALES_AGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'cancel.db'}")

    from app.application.runtime_state import (
        AnalysisRunCancelled,
        RuntimeStateStore,
    )
    from app.db.models import AnalysisTask
    from app.db.session import get_engine, init_db
    from sqlalchemy.orm import sessionmaker

    init_db()
    session = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)()
    task = AnalysisTask(status="running", cancel_requested=True)
    session.add(task)
    session.commit()
    task_id = task.task_id
    session.close()

    store = RuntimeStateStore()

    try:
        store.check_cancel_requested(task_id)
    except AnalysisRunCancelled as exc:
        assert exc.task_id == task_id
    else:
        raise AssertionError("Expected AnalysisRunCancelled")


def test_stage_finished_does_not_apply_cancellation_until_next_stage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SALES_AGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'finish.db'}")

    from app.application.runtime_state import RuntimeStateStore
    from app.db.models import AnalysisTask
    from app.db.session import get_engine, init_db
    from sqlalchemy.orm import sessionmaker

    init_db()
    session = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)()
    task = AnalysisTask(status="cancelling", cancel_requested=True)
    session.add(task)
    session.commit()
    task_id = task.task_id
    session.close()

    store = RuntimeStateStore()

    finished_task = store.mark_stage_finished(task_id, llm_trace={})

    assert finished_task.status == "cancelling"
    assert finished_task.cancel_requested is True


def test_real_stage_loop_cancels_between_stages_without_running_later_stage(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SALES_AGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'loop.db'}")
    monkeypatch.setenv("SALES_AGENT_RUNTIME_DIR", str(tmp_path / "runtime"))

    from app.application.analysis_run_service import AnalysisRunService
    from app.application.run_stages import RunStage
    from app.application.runtime_runner import RuntimeRunner
    from app.application.runtime_state import RuntimeStateStore
    from app.db.models import AnalysisTask
    from app.db.session import get_engine, init_db
    from app.services.artifact_store import ArtifactStore
    from sqlalchemy.orm import sessionmaker

    init_db()
    session_factory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    session = session_factory()
    task = AnalysisTask(status="running")
    session.add(task)
    session.commit()
    task_id = task.task_id
    session.close()

    store = RuntimeStateStore()
    executed: list[str] = []

    def prepare_context(self, task_id: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            task_id=task_id,
            llm_trace={},
            llm_client=SimpleNamespace(enabled=False),
        )

    def build_stages(self, ctx: SimpleNamespace) -> list[RunStage]:
        def first_stage() -> None:
            executed.append("stage_1")
            store.request_cancel(ctx.task_id)

        def second_stage() -> None:
            executed.append("stage_2")
            ArtifactStore().save_manifest(
                ctx.task_id,
                {"task_id": ctx.task_id, "status": "completed", "files": {}},
            )

        return [
            RunStage("deterministic_analysis", "Stage 1", first_stage),
            RunStage("notebook_planning", "Stage 2", second_stage),
        ]

    monkeypatch.setattr(AnalysisRunService, "_prepare_context", prepare_context)
    monkeypatch.setattr(AnalysisRunService, "_build_stages", build_stages)

    runner = RuntimeRunner(
        session_factory=session_factory,
        heartbeat_seconds=0.01,
        state_store=store,
    )
    runner._run_worker(task_id, llm_profile=None, output_language=None)

    final_task = store.get_task(task_id)
    manifest = ArtifactStore().load_manifest(task_id)
    assert executed == ["stage_1"]
    assert final_task.status == "cancelled"
    assert final_task.finished_at is not None
    assert manifest.get("status") != "completed"


def test_cancel_during_completed_artifact_persistence_allows_final_completion(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SALES_AGENT_DATABASE_URL", f"sqlite:///{tmp_path / 'final.db'}")
    monkeypatch.setenv("SALES_AGENT_RUNTIME_DIR", str(tmp_path / "runtime"))

    from app.application.analysis_run_service import AnalysisRunService
    from app.application.run_stages import RunStage
    from app.application.runtime_runner import RuntimeRunner
    from app.application.runtime_state import RuntimeStateStore
    from app.db.models import AnalysisTask
    from app.db.session import get_engine, init_db
    from app.services.artifact_store import ArtifactStore
    from sqlalchemy.orm import sessionmaker

    init_db()
    session_factory = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    session = session_factory()
    task = AnalysisTask(status="running")
    session.add(task)
    session.commit()
    task_id = task.task_id
    session.close()

    store = RuntimeStateStore()
    executed: list[str] = []

    def prepare_context(self, task_id: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            task_id=task_id,
            llm_trace={},
            llm_client=SimpleNamespace(enabled=False),
        )

    def build_stages(self, ctx: SimpleNamespace) -> list[RunStage]:
        def persist_artifacts() -> None:
            executed.append("completed_artifact_persistence")
            ArtifactStore().save_manifest(
                ctx.task_id,
                {"task_id": ctx.task_id, "status": "completed", "files": {}},
            )
            store.request_cancel(ctx.task_id)

        def complete_task() -> None:
            executed.append("task_completion")

        return [
            RunStage(
                "completed_artifact_persistence",
                "Persist completed artifacts",
                persist_artifacts,
            ),
            RunStage("task_completion", "Mark task completed", complete_task),
        ]

    monkeypatch.setattr(AnalysisRunService, "_prepare_context", prepare_context)
    monkeypatch.setattr(AnalysisRunService, "_build_stages", build_stages)
    monkeypatch.setattr(
        AnalysisRunService,
        "_build_run_response",
        lambda self, ctx: SimpleNamespace(task_id=ctx.task_id, status="completed"),
    )

    runner = RuntimeRunner(
        session_factory=session_factory,
        heartbeat_seconds=0.01,
        state_store=store,
    )
    runner._run_worker(task_id, llm_profile=None, output_language=None)

    final_task = store.get_task(task_id)
    manifest = ArtifactStore().load_manifest(task_id)
    assert executed == ["completed_artifact_persistence", "task_completion"]
    assert final_task.status == "completed"
    assert manifest.get("status") == "completed"
