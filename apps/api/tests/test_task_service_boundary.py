from __future__ import annotations

import ast
from pathlib import Path


def test_analysis_routes_delegate_task_lifecycle_to_task_service() -> None:
    route_path = Path(__file__).parents[1] / "app" / "api" / "routes" / "analysis.py"
    source = route_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported_names: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
            imported_names.update(alias.name for alias in node.names)

    assert "TaskService" in imported_names
    assert "app.application.tasks.task_service" in imported_modules

    route_forbidden_imports = {
        "pandas",
        "app.db.models",
        "app.schemas.schema_mapping",
        "app.services.analysis_planner",
        "app.services.analysis_results",
        "app.services.artifact_store",
        "app.services.csv_ingestion",
        "app.services.dataset_profile",
        "app.services.llm_client",
        "app.services.llm_trace_utils",
        "app.services.output_language",
        "app.services.schema_mapper",
    }
    assert route_forbidden_imports.isdisjoint(imported_modules)
