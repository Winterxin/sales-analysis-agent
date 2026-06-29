from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import jupyter_core.paths as jupyter_paths
import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def new_analysis_notebook(title: str | None = None, *, task_id: str | None = None) -> NotebookNode:
    notebook = new_notebook()
    if title:
        heading = f"# {title}"
        if task_id:
            heading += f"\n\nTask ID: `{task_id}`"
        notebook.cells.append(new_markdown_cell(heading))
    return notebook


def load_notebook(path: Path) -> NotebookNode:
    return nbformat.read(path, as_version=4)


def save_notebook(notebook: NotebookNode, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


def append_markdown_cell(notebook: NotebookNode, source: str) -> NotebookNode:
    notebook.cells.append(new_markdown_cell(source))
    return notebook


def append_code_cell(notebook: NotebookNode, source: str) -> NotebookNode:
    notebook.cells.append(new_code_cell(source))
    return notebook


def insert_markdown_cell_at_index(
    notebook: NotebookNode,
    index: int,
    source: str,
) -> NotebookNode:
    notebook.cells.insert(index, new_markdown_cell(source))
    return notebook


def insert_code_cell_at_index(
    notebook: NotebookNode,
    index: int,
    source: str,
) -> NotebookNode:
    notebook.cells.insert(index, new_code_cell(source))
    return notebook


def insert_markdown_cell_after_index(
    notebook: NotebookNode,
    index: int,
    source: str,
) -> NotebookNode:
    notebook.cells.insert(index + 1, new_markdown_cell(source))
    return notebook


@contextmanager
def notebook_execution_environment(working_dir: Path) -> Iterator[Path]:
    runtime_dir = working_dir / ".jupyter_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    env_keys = ("JUPYTER_RUNTIME_DIR", "JUPYTER_ALLOW_INSECURE_WRITES")
    previous_env = {key: os.environ.get(key) for key in env_keys}
    previous_insecure_write_flag = jupyter_paths.allow_insecure_writes

    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"
    jupyter_paths.allow_insecure_writes = True
    try:
        yield runtime_dir
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        jupyter_paths.allow_insecure_writes = previous_insecure_write_flag


def find_markdown_heading_index(
    notebook: NotebookNode,
    heading: str,
) -> int | None:
    for index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            normalized = "".join(source)
        else:
            normalized = str(source)
        if heading in normalized:
            return index
    return None


def execute_notebook_document(
    notebook: NotebookNode,
    *,
    working_dir: Path,
    timeout: int = 300,
) -> NotebookNode:
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_dir)}},
    )
    with notebook_execution_environment(working_dir):
        return client.execute()
