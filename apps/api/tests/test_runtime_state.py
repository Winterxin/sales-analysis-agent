from __future__ import annotations

from datetime import datetime

from sqlalchemy import inspect, text


def test_init_db_migrates_existing_task_table_with_runtime_columns(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "legacy.db"
    monkeypatch.setenv("SALES_AGENT_DATABASE_URL", f"sqlite:///{db_path}")

    from app.db.session import get_engine, init_db

    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE analysis_tasks (
                    task_id VARCHAR(36) PRIMARY KEY,
                    status VARCHAR(32) NOT NULL,
                    dataset_type VARCHAR(64),
                    artifact_manifest_path TEXT,
                    error_message TEXT,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
                """
            )
        )

    init_db()

    columns = {column["name"] for column in inspect(engine).get_columns("analysis_tasks")}
    assert {
        "current_stage",
        "current_stage_label",
        "status_message",
        "cancel_requested",
        "llm_status",
        "llm_message",
        "started_at",
        "heartbeat_at",
        "finished_at",
    } <= columns


def test_llm_runtime_status_prefers_any_successful_call_over_later_fallback() -> None:
    from app.application.runtime_state import derive_llm_runtime_status

    status, message = derive_llm_runtime_status(
        {
            "notebook_outline": {
                "llm_enabled": True,
                "attempted": True,
                "applied": True,
                "status": "llm_success",
                "remote_elapsed_ms": 120.0,
            },
            "client_report": {
                "llm_enabled": True,
                "attempted": True,
                "applied": False,
                "status": "fallback_on_error",
                "fallback_type": "error",
                "error_type": "HTTPStatusError",
            },
        }
    )

    assert status == "called"
    assert message == "LLM response received."


def test_llm_runtime_status_reports_fallback_without_leaking_error_detail() -> None:
    from app.application.runtime_state import derive_llm_runtime_status

    status, message = derive_llm_runtime_status(
        {
            "notebook_content": {
                "llm_enabled": True,
                "attempted": True,
                "applied": False,
                "status": "fallback_on_error",
                "fallback_type": "error",
                "error_type": "HTTPStatusError",
                "reason": "Bearer sk-secret failed against http://private.example",
            }
        }
    )

    assert status == "fallback"
    assert message == "LLM unavailable. Continuing with deterministic fallback."
    assert "secret" not in message.lower()
    assert "http" not in message.lower()


def test_public_error_message_is_short_and_sanitized() -> None:
    from app.application.runtime_state import safe_public_error_message

    message = safe_public_error_message(
        RuntimeError("boom sk-secret-token traceback line 1\nline 2 http://private")
    )

    assert "sk-secret" not in message
    assert "http://private" not in message
    assert "\n" not in message
    assert len(message) <= 180


def test_runtime_response_treats_naive_database_datetimes_as_utc() -> None:
    from app.application.runtime_state import runtime_response_from_task
    from app.db.models import AnalysisTask

    task = AnalysisTask(
        task_id="task-1",
        status="running",
        heartbeat_at=datetime(2026, 6, 30, 8, 0, 0),
    )

    response = runtime_response_from_task(task)

    assert response.heartbeat_at == "2026-06-30T08:00:00+00:00"
