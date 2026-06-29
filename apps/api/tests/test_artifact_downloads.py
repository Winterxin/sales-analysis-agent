from __future__ import annotations

from fastapi.testclient import TestClient


def test_artifact_download_endpoint_serves_business_review(
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
        f"/api/v1/analysis/tasks/{task_id}/artifacts/business-review"
    )

    assert response.status_code == 200
    assert "Business Review" in response.text


def test_artifact_download_endpoint_serves_client_report_html(
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
        f"/api/v1/analysis/tasks/{task_id}/artifacts/client-report-html"
    )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "inline" in response.headers["content-disposition"]
    assert "Client-ready Sales Analysis Report" in response.text


def test_artifact_download_endpoint_serves_client_report_json(
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
        f"/api/v1/analysis/tasks/{task_id}/artifacts/client-report-json"
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Client-ready Sales Analysis Report"


def test_artifact_downloads_preserve_chinese_when_requested(
    client: TestClient, sample_csv_path
) -> None:
    task_id = client.post("/api/v1/analysis/tasks").json()["task_id"]

    with sample_csv_path.open("rb") as csv_file:
        client.post(
            f"/api/v1/analysis/tasks/{task_id}/upload",
            files={"file": ("sales_orders.csv", csv_file, "text/csv")},
        )

    client.post(f"/api/v1/analysis/tasks/{task_id}/run?output_language=zh-CN")

    business_review = client.get(
        f"/api/v1/analysis/tasks/{task_id}/artifacts/business-review"
    )
    client_report = client.get(
        f"/api/v1/analysis/tasks/{task_id}/artifacts/client-report-json"
    )

    assert business_review.status_code == 200
    assert "销售经营复盘报告" in business_review.text
    assert client_report.status_code == 200
    assert client_report.json()["title"] == "客户经营分析简报"
