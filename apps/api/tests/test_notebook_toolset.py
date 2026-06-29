from __future__ import annotations

import os
import shutil
from pathlib import Path

import jupyter_core.paths as jupyter_paths

from app.services.notebook_toolset import (
    append_code_cell,
    append_markdown_cell,
    insert_markdown_cell_after_index,
    load_notebook,
    new_analysis_notebook,
    notebook_execution_environment,
    save_notebook,
)


def test_notebook_toolset_appends_and_inserts_cells_in_stable_order(tmp_path: Path) -> None:
    notebook = new_analysis_notebook("销售数据分析 Notebook", task_id="task-1")

    append_markdown_cell(notebook, "## 商品与类目分析")
    append_code_cell(notebook, "top_products = clean_df.head()")
    insert_markdown_cell_after_index(notebook, 2, "### 图后分析\n\n头部商品销售额明显集中。")

    sources = []
    for cell in notebook.cells:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        sources.append(source)

    assert sources[0].startswith("# 销售数据分析 Notebook")
    assert sources[1] == "## 商品与类目分析"
    assert sources[2] == "top_products = clean_df.head()"
    assert sources[3].startswith("### 图后分析")


def test_notebook_toolset_loads_and_saves_notebook_round_trip(tmp_path: Path) -> None:
    notebook = new_analysis_notebook("销售数据分析 Notebook", task_id="task-2")
    append_markdown_cell(notebook, "## 销售趋势分析")
    append_code_cell(notebook, "monthly_sales = clean_df.groupby('Order Date').size()")

    notebook_path = tmp_path / "analysis.ipynb"
    save_notebook(notebook, notebook_path)

    loaded = load_notebook(notebook_path)

    assert len(loaded.cells) == 3
    source = loaded.cells[2].get("source", "")
    if isinstance(source, list):
        source = "".join(source)
    assert "monthly_sales" in source


def test_notebook_execution_environment_uses_workspace_runtime_and_restores_state(
    monkeypatch,
) -> None:
    working_dir = Path.cwd() / ".pytest-runtime" / "notebook-toolset-env"
    runtime_dir = working_dir / ".jupyter_runtime"
    shutil.rmtree(working_dir, ignore_errors=True)
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", "original-runtime")
    monkeypatch.setenv("JUPYTER_ALLOW_INSECURE_WRITES", "false")
    previous_insecure_write_flag = jupyter_paths.allow_insecure_writes
    jupyter_paths.allow_insecure_writes = False

    try:
        with notebook_execution_environment(working_dir) as active_runtime_dir:
            assert active_runtime_dir == runtime_dir
            assert runtime_dir.is_dir()
            assert os.environ["JUPYTER_RUNTIME_DIR"] == str(runtime_dir)
            assert os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] == "1"
            assert jupyter_paths.allow_insecure_writes is True

        assert os.environ["JUPYTER_RUNTIME_DIR"] == "original-runtime"
        assert os.environ["JUPYTER_ALLOW_INSECURE_WRITES"] == "false"
        assert jupyter_paths.allow_insecure_writes is False
    finally:
        jupyter_paths.allow_insecure_writes = previous_insecure_write_flag
        shutil.rmtree(working_dir, ignore_errors=True)
