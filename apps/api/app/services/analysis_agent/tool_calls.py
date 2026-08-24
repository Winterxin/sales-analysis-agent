from __future__ import annotations

import hashlib
import json
from pydantic import BaseModel, ConfigDict, Field


class AnalysisToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)


def normalized_call_signature(tool_name: str, arguments: dict[str, object]) -> str:
    payload = json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
