from __future__ import annotations

from pathlib import Path

from app.services.notebook_toolset import execute_notebook_document, load_notebook, save_notebook


def execute_notebook(
    notebook_path: Path,
    output_path: Path,
    working_dir: Path,
    *,
    timeout: int = 300,
) -> Path:
    notebook = load_notebook(notebook_path)
    executed = execute_notebook_document(
        notebook,
        working_dir=working_dir,
        timeout=timeout,
    )
    return save_notebook(executed, output_path)
