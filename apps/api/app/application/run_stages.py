from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RunStage:
    stage_id: str
    label: str
    handler: Callable[[], None]


class RunStageExecutor:
    def run_stage(self, stage: RunStage) -> None:
        stage.handler()
