from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def sample_csv_path() -> Path:
    return Path(__file__).parent / "fixtures" / "sales_orders.csv"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    runtime_dir = tmp_path / "runtime"
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("SALES_AGENT_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("SALES_AGENT_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SALES_AGENT_LLM_ENABLED", "false")

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
