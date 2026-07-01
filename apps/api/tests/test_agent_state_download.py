from __future__ import annotations

import time

from fastapi.testclient import TestClient


def _wait_for_completed(client: TestClient, task_id: str) -> None:
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/analysis/tasks/{task_id}")
        assert response.status_code == 200
        if response.json()["status"] == "completed":
            return
        time.sleep(0.25)
    raise AssertionError(f"Task {task_id} did not complete")


def test_artifact_download_endpoint_serves_agent_loop_state(
    client: TestClient, sample_csv_path
) -> None:
    task_id = client.post("/api/v1/analysis/tasks").json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )

    client.post(f"/api/v1/analysis/tasks/{task_id}/run")
    _wait_for_completed(client, task_id)

    response = client.get(
        f"/api/v1/analysis/tasks/{task_id}/artifacts/agent-loop-state"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task_id
    assert payload["stages"][-1]["stage"] == "finalize"
