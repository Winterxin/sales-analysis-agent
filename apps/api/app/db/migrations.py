from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


RUNTIME_STATE_COLUMNS: dict[str, str] = {
    "current_stage": "VARCHAR(64)",
    "current_stage_label": "VARCHAR(128)",
    "status_message": "TEXT",
    "cancel_requested": "BOOLEAN NOT NULL DEFAULT 0",
    "llm_status": "VARCHAR(32) NOT NULL DEFAULT 'unknown'",
    "llm_message": "TEXT",
    "started_at": "DATETIME",
    "heartbeat_at": "DATETIME",
    "finished_at": "DATETIME",
}


def migrate_runtime_state_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    if not inspector.has_table("analysis_tasks"):
        return

    existing_columns = {
        column["name"] for column in inspector.get_columns("analysis_tasks")
    }
    missing_columns = [
        (name, ddl)
        for name, ddl in RUNTIME_STATE_COLUMNS.items()
        if name not in existing_columns
    ]
    if not missing_columns:
        return

    with engine.begin() as connection:
        for name, ddl in missing_columns:
            connection.execute(
                text(f"ALTER TABLE analysis_tasks ADD COLUMN {name} {ddl}")
            )
        connection.execute(
            text(
                """
                UPDATE analysis_tasks
                SET
                  cancel_requested = COALESCE(cancel_requested, 0),
                  llm_status = COALESCE(llm_status, 'unknown')
                """
            )
        )
