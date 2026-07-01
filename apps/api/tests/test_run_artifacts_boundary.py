from __future__ import annotations

import ast
from pathlib import Path


def test_analysis_run_service_delegates_run_artifact_persistence() -> None:
    app_dir = Path(__file__).parents[1] / "app"
    service_path = app_dir / "application" / "analysis_run_service.py"
    artifacts_path = app_dir / "application" / "run_artifacts.py"

    assert artifacts_path.exists()

    tree = ast.parse(service_path.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    imported_modules: set[str] = set()
    method_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.FunctionDef):
            method_names.add(node.name)

    assert "app.application.run_artifacts" in imported_modules
    assert "RunArtifacts" in imported_names
    assert "_save_llm_trace" not in method_names

    source = service_path.read_text(encoding="utf-8")
    assert ".save_manifest(" not in source
    assert 'task.status = "completed"' not in source
