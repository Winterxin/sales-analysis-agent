from __future__ import annotations

import ast
from pathlib import Path


EXPECTED_STAGE_IDS = [
    "load_uploaded_inputs",
    "deterministic_analysis",
    "evidence_and_chart_planning",
    "modeling_opportunity_planning",
    "notebook_planning",
    "summary_and_modeling_interpretation",
    "final_synthesis",
    "notebook_build_and_execution",
    "notebook_revision",
    "postrun_reflection",
    "client_report",
    "agent_state",
    "completed_artifact_persistence",
    "task_completion",
]

LEGACY_DIRECT_RUN_CALLS = [
    "_load_uploaded_inputs",
    "_run_analysis_modules",
    "_build_evidence_and_chart_plan",
    "_build_modeling_opportunity_plan",
    "_build_notebook_plans",
    "_build_report_summary_and_modeling_interpretation",
    "_build_final_synthesis",
    "_build_and_execute_notebook",
    "_apply_revision_if_needed",
    "_apply_postrun_reflections",
    "_build_client_report",
    "_build_agent_state",
    "_persist_completed_artifacts",
    "_mark_task_completed",
]


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _class_def(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found")


def _method_def(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"{method_name} not found")


def test_run_stage_boundary_types_exist_and_stay_behavior_free() -> None:
    app_dir = Path(__file__).parents[1] / "app"
    stages_path = app_dir / "application" / "run_stages.py"

    assert stages_path.exists()
    tree = _module_tree(stages_path)

    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {"RunStage", "RunStageExecutor"} <= class_names

    source = stages_path.read_text(encoding="utf-8")
    assert ".save_manifest(" not in source
    assert "session." not in source
    assert "llm_client" not in source
    assert "get_default_llm_client" not in source


def test_analysis_run_service_uses_stage_executor_boundary() -> None:
    service_path = (
        Path(__file__).parents[1] / "app" / "application" / "analysis_run_service.py"
    )
    tree = _module_tree(service_path)
    imported_names: set[str] = set()
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
            imported_names.update(alias.name for alias in node.names)

    assert "app.application.run_stages" in imported_modules
    assert {"RunStage", "RunStageExecutor"} <= imported_names

    service_class = _class_def(tree, "AnalysisRunService")
    init_method = _method_def(service_class, "__init__")
    run_method = _method_def(service_class, "run")
    build_stages_method = _method_def(service_class, "_build_stages")

    init_source = ast.unparse(init_method)
    assert "RunStageExecutor" in init_source
    assert "stage_executor" in init_source

    run_source = ast.unparse(run_method)
    assert "self.stage_executor.run_stage" in run_source
    assert "self._build_stages(ctx)" in run_source
    assert "self._build_run_response(ctx)" in run_source
    assert all(f"self.{name}(ctx)" not in run_source for name in LEGACY_DIRECT_RUN_CALLS)

    build_stages_source = ast.unparse(build_stages_method)
    for stage_id in EXPECTED_STAGE_IDS:
        assert stage_id in build_stages_source

    stage_ids_in_order = [
        node.value
        for node in ast.walk(build_stages_method)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        if node.value in EXPECTED_STAGE_IDS
    ]
    assert stage_ids_in_order == EXPECTED_STAGE_IDS
