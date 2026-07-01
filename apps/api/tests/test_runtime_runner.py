from __future__ import annotations

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


def test_run_returns_runtime_payload_without_waiting_for_worker_completion(
    client: TestClient, sample_csv_path, monkeypatch
) -> None:
    from app.application.analysis_run_service import AnalysisRunService
    from app.application.runtime_state import RuntimeStateStore

    def slow_run(self, task_id: str, **_: object) -> None:
        time.sleep(0.4)
        RuntimeStateStore().mark_completed(task_id)

    monkeypatch.setattr(AnalysisRunService, "run", slow_run)
    task_id = _create_uploaded_task(client, sample_csv_path)

    started_at = time.monotonic()
    response = client.post(f"/api/v1/analysis/tasks/{task_id}/run")
    elapsed = time.monotonic() - started_at

    assert response.status_code == 202
    assert elapsed < 0.25
    body = response.json()
    assert body["task_id"] == task_id
    assert body["status"] in {"queued", "running"}
    assert "report" not in body
    _wait_for_status(client, task_id, "completed")


def test_runtime_runner_uses_a_new_database_session_for_worker() -> None:
    from app.application.runtime_runner import RuntimeRunner

    created_sessions: list[object] = []

    class DummySession:
        def close(self) -> None:
            pass

    def session_factory() -> DummySession:
        session = DummySession()
        created_sessions.append(session)
        return session

    runner = RuntimeRunner(session_factory=session_factory)
    worker_session = runner._create_worker_session()

    assert worker_session is created_sessions[0]
    assert len(created_sessions) == 1


def test_repeated_run_does_not_submit_duplicate_worker(
    client: TestClient, sample_csv_path, monkeypatch
) -> None:
    from app.application.analysis_run_service import AnalysisRunService

    run_calls: list[str] = []

    def slow_run(self, task_id: str, **_: object) -> None:
        run_calls.append(task_id)
        time.sleep(0.5)

    monkeypatch.setattr(AnalysisRunService, "run", slow_run)
    task_id = _create_uploaded_task(client, sample_csv_path)

    first = client.post(f"/api/v1/analysis/tasks/{task_id}/run")
    second = client.post(f"/api/v1/analysis/tasks/{task_id}/run")

    assert first.status_code == 202
    assert second.status_code == 202
    assert second.json()["status"] in {"queued", "running"}
    time.sleep(0.1)
    assert run_calls == [task_id]


def test_worker_top_level_exception_marks_task_failed(
    client: TestClient, sample_csv_path, monkeypatch
) -> None:
    from app.application.analysis_run_service import AnalysisRunService

    def failing_run(self, task_id: str, **_: object) -> None:
        raise RuntimeError("boom sk-secret-token traceback line 1\nhttp://private")

    monkeypatch.setattr(AnalysisRunService, "run", failing_run)
    task_id = _create_uploaded_task(client, sample_csv_path)

    response = client.post(f"/api/v1/analysis/tasks/{task_id}/run")

    assert response.status_code == 202
    body = _wait_for_status(client, task_id, "failed")
    assert body["status_message"] == "Analysis failed."
    assert "sk-secret" not in body["error_message"]
    assert "http://private" not in body["error_message"]
    assert body["finished_at"] is not None
