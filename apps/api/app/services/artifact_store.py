from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings


class ArtifactStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_dir = Path(settings.runtime_dir) / "artifacts"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def task_dir(self, task_id: str) -> Path:
        path = self.base_dir / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(self, task_id: str) -> Path:
        return self.task_dir(task_id) / "artifact_manifest.json"

    def save_manifest(self, task_id: str, manifest: dict[str, object]) -> Path:
        path = self.manifest_path(task_id)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_manifest(self, task_id: str) -> dict[str, object]:
        path = self.manifest_path(task_id)
        if not path.exists():
            return {"task_id": task_id, "files": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_bytes(self, task_id: str, filename: str, content: bytes) -> Path:
        path = self.task_dir(task_id) / filename
        path.write_bytes(content)
        return path

    def save_text(self, task_id: str, filename: str, content: str) -> Path:
        path = self.task_dir(task_id) / filename
        path.write_text(content, encoding="utf-8")
        return path

    def save_json(self, task_id: str, filename: str, payload: object) -> Path:
        path = self.task_dir(task_id) / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
