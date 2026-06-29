from __future__ import annotations

from fastapi.testclient import TestClient


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

    response = client.get(
        f"/api/v1/analysis/tasks/{task_id}/artifacts/agent-loop-state"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == task_id
    assert payload["stages"][-1]["stage"] == "finalize"
