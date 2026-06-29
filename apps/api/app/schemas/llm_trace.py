from __future__ import annotations

from pydantic import BaseModel


class LLMStageTrace(BaseModel):
    stage: str
    status: str
    reason: str
    llm_enabled: bool
    attempted: bool = False
    applied: bool = False
    source: str = "disabled"
    model: str | None = None
    elapsed_ms: float | None = None
    prompt_chars: int | None = None
    response_chars: int | None = None
    attempt_count: int | None = None
    fallback_type: str | None = None
    error_type: str | None = None
    budget_remaining_ms: float | None = None
    cache_enabled: bool | None = None
    cache_status: str | None = None
    cache_key: str | None = None
    remote_elapsed_ms: float | None = None
    saved_ms: float = 0.0
    subcall_count: int | None = None
    subcalls: list[dict] | None = None
    aggregate_prompt_chars: int | None = None
    aggregate_response_chars: int | None = None
    aggregate_remote_elapsed_ms: float | None = None
    aggregate_saved_ms: float | None = None
    aggregate_cache_hit_count: int | None = None
    aggregate_cache_miss_count: int | None = None
    aggregate_cache_disabled_count: int | None = None
    aggregate_cache_bypassed_remote_count: int | None = None
