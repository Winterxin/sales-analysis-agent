from __future__ import annotations

import httpx

from app.core.config import get_settings
from app.schemas.llm_trace import LLMStageTrace

EVIDENCE_PACK_LLM_STAGES = {
    "report_summary",
    "notebook_narrative",
    "notebook_content",
    "notebook_revision_decision",
    "client_report",
}


SKIPPED_WITHOUT_COMPLETION_STATUSES = {
    "skipped",
    "skipped_by_profile",
    "skipped_by_budget",
    "skipped_by_demo_safe",
    "disabled",
}


def _should_clear_metrics(status: str, attempted: bool) -> bool:
    return not attempted and status in SKIPPED_WITHOUT_COMPLETION_STATUSES


def describe_llm_error(exc: Exception, limit: int = 220) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        response = exc.response
        detail = ""
        try:
            payload = response.json()
        except Exception:
            detail = response.text
        else:
            error = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("code") or "")
            elif isinstance(payload, dict):
                detail = str(payload.get("message") or payload)
            else:
                detail = str(payload)

        detail = " ".join(detail.split())
        prefix = f"HTTPStatusError {response.status_code}"
        message = f"{prefix}: {detail}" if detail else prefix
        if len(message) > limit:
            message = message[: limit - 3] + "..."
        return message

    detail = " ".join(str(exc).split())
    if detail and len(detail) > limit:
        detail = detail[: limit - 3] + "..."
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def build_llm_stage_trace(
    *,
    stage: str,
    llm_client,
    status: str,
    reason: str,
    attempted: bool = False,
    applied: bool = False,
) -> LLMStageTrace:
    enabled = bool(llm_client is not None and getattr(llm_client, "enabled", False))
    metrics = (
        {}
        if _should_clear_metrics(status, attempted)
        else getattr(llm_client, "last_completion_metrics", {}) or {}
    )
    cache_enabled = metrics.get("cache_enabled")
    if cache_enabled is None:
        cache_enabled = bool(get_settings().llm_cache_enabled)
    cache_status = metrics.get("cache_status")
    if not cache_status:
        cache_status = "disabled" if not enabled else "bypassed"
    fallback_type = None
    if status == "fallback_on_error":
        fallback_type = "error"
    elif status == "fallback_invalid_payload":
        fallback_type = "invalid_payload"
    elif status == "disabled":
        fallback_type = "disabled"
    elif status == "fallback":
        fallback_type = "deterministic"
    elif status == "skipped":
        fallback_type = "skipped"
    elif status == "skipped_by_budget":
        fallback_type = "budget"
    elif status == "skipped_by_demo_safe":
        fallback_type = "demo_safe"
    elif status == "skipped_by_profile":
        fallback_type = "profile"
    return LLMStageTrace(
        stage=stage,
        status=status,
        reason=reason,
        llm_enabled=enabled,
        attempted=attempted,
        applied=applied,
        source=str(getattr(llm_client, "source", "disabled" if not enabled else "unknown")),
        model=getattr(llm_client, "configured_model", None),
        elapsed_ms=metrics.get("elapsed_ms"),
        prompt_chars=metrics.get("prompt_chars"),
        response_chars=metrics.get("response_chars"),
        attempt_count=metrics.get("attempt_count"),
        fallback_type=fallback_type,
        error_type=metrics.get("error_type"),
        cache_enabled=bool(cache_enabled),
        cache_status=str(cache_status),
        cache_key=metrics.get("cache_key"),
        remote_elapsed_ms=metrics.get("remote_elapsed_ms"),
        saved_ms=float(metrics.get("saved_ms") or 0.0),
    )


def _numeric(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _integer(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _prefer_aggregate(payload: dict[str, object], aggregate_key: str, fallback_key: str) -> object:
    value = payload.get(aggregate_key)
    return value if value is not None else payload.get(fallback_key)


def _compact_subcall(subcall: dict[str, object]) -> dict[str, object]:
    return {
        "stage": subcall.get("stage"),
        "cache_status": subcall.get("cache_status"),
        "prompt_chars": subcall.get("prompt_chars"),
        "response_chars": subcall.get("response_chars"),
        "remote_elapsed_ms": subcall.get("remote_elapsed_ms"),
        "saved_ms": subcall.get("saved_ms"),
        "cache_key": subcall.get("cache_key"),
        "error_type": subcall.get("error_type"),
    }


def _cache_status_count(subcalls: list[dict[str, object]], status: str) -> int:
    return sum(1 for subcall in subcalls if str(subcall.get("cache_status") or "") == status)


def build_aggregated_llm_stage_trace(
    *,
    stage: str,
    llm_client,
    status: str,
    reason: str,
    subcalls: list[dict],
    attempted: bool = False,
    applied: bool = False,
) -> LLMStageTrace:
    if not subcalls:
        return build_llm_stage_trace(
            stage=stage,
            llm_client=llm_client,
            status=status,
            reason=reason,
            attempted=attempted,
            applied=applied,
        )

    compacted = [_compact_subcall(dict(subcall)) for subcall in subcalls]
    prompt_chars = sum(_integer(subcall.get("prompt_chars")) for subcall in compacted)
    response_chars = sum(_integer(subcall.get("response_chars")) for subcall in compacted)
    remote_elapsed_ms = round(sum(_numeric(subcall.get("remote_elapsed_ms")) for subcall in compacted), 3)
    saved_ms = round(sum(_numeric(subcall.get("saved_ms")) for subcall in compacted), 3)
    statuses = {str(subcall.get("cache_status") or "") for subcall in compacted}
    cache_status = statuses.pop() if len(statuses) == 1 else "mixed"
    cache_keys = {
        str(subcall.get("cache_key"))
        for subcall in compacted
        if subcall.get("cache_key")
    }
    fallback_type = None
    if status == "fallback_on_error":
        fallback_type = "error"
    elif status == "fallback_invalid_payload":
        fallback_type = "invalid_payload"
    elif status == "disabled":
        fallback_type = "disabled"
    elif status == "fallback":
        fallback_type = "deterministic"
    elif status == "skipped":
        fallback_type = "skipped"
    elif status == "skipped_by_budget":
        fallback_type = "budget"
    elif status == "skipped_by_demo_safe":
        fallback_type = "demo_safe"
    elif status == "skipped_by_profile":
        fallback_type = "profile"

    enabled = bool(llm_client is not None and getattr(llm_client, "enabled", False))
    cache_enabled_values = [
        subcall.get("cache_enabled")
        for subcall in subcalls
        if subcall.get("cache_enabled") is not None
    ]
    cache_enabled = bool(cache_enabled_values[0]) if cache_enabled_values else bool(get_settings().llm_cache_enabled)
    first_error = next(
        (subcall.get("error_type") for subcall in compacted if subcall.get("error_type")),
        None,
    )
    attempt_count = sum(_integer(subcall.get("attempt_count")) for subcall in subcalls)
    return LLMStageTrace(
        stage=stage,
        status=status,
        reason=reason,
        llm_enabled=enabled,
        attempted=attempted,
        applied=applied,
        source=str(getattr(llm_client, "source", "disabled" if not enabled else "unknown")),
        model=getattr(llm_client, "configured_model", None),
        elapsed_ms=round(sum(_numeric(subcall.get("elapsed_ms")) for subcall in subcalls), 3),
        prompt_chars=prompt_chars,
        response_chars=response_chars,
        attempt_count=attempt_count or None,
        fallback_type=fallback_type,
        error_type=str(first_error) if first_error else None,
        cache_enabled=cache_enabled,
        cache_status=cache_status,
        cache_key=next(iter(cache_keys)) if len(cache_keys) == 1 else None,
        remote_elapsed_ms=remote_elapsed_ms,
        saved_ms=saved_ms,
        subcall_count=len(compacted),
        subcalls=compacted,
        aggregate_prompt_chars=prompt_chars,
        aggregate_response_chars=response_chars,
        aggregate_remote_elapsed_ms=remote_elapsed_ms,
        aggregate_saved_ms=saved_ms,
        aggregate_cache_hit_count=_cache_status_count(compacted, "hit"),
        aggregate_cache_miss_count=_cache_status_count(compacted, "miss"),
        aggregate_cache_disabled_count=_cache_status_count(compacted, "disabled"),
        aggregate_cache_bypassed_remote_count=sum(
            1
            for subcall in compacted
            if str(subcall.get("cache_status") or "") == "bypassed"
            and _numeric(subcall.get("remote_elapsed_ms")) > 0
        ),
    )


def build_llm_trace_summary(
    llm_trace: dict[str, object],
    *,
    llm_profile: str | None = None,
    output_language: str | None = None,
    llm_profile_policy: dict[str, object] | None = None,
    profile_limited_stage_count: int | None = None,
    max_llm_postrun_reflections_resolved: int | None = None,
    allow_postrun_fallback_reflections: bool | None = None,
) -> dict[str, object]:
    stage_payloads = [
        payload
        for stage, payload in llm_trace.items()
        if stage != "summary" and isinstance(payload, dict)
    ]
    cache_statuses = [str(payload.get("cache_status") or "") for payload in stage_payloads]
    remote_elapsed_values = [
        _numeric(_prefer_aggregate(payload, "aggregate_remote_elapsed_ms", "remote_elapsed_ms"))
        for payload in stage_payloads
    ]
    saved_values = [
        _numeric(_prefer_aggregate(payload, "aggregate_saved_ms", "saved_ms"))
        for payload in stage_payloads
    ]
    prompt_values = [
        _integer(_prefer_aggregate(payload, "aggregate_prompt_chars", "prompt_chars"))
        for payload in stage_payloads
    ]
    subcall_payloads = [
        subcall
        for payload in stage_payloads
        for subcall in (payload.get("subcalls") if isinstance(payload.get("subcalls"), list) else [])
        if isinstance(subcall, dict)
    ]
    def stage_cache_count(aggregate_key: str, status: str, *, remote_only: bool = False) -> int:
        total = 0
        for payload in stage_payloads:
            if payload.get(aggregate_key) is not None:
                total += _integer(payload.get(aggregate_key))
                continue
            if str(payload.get("cache_status") or "") != status:
                continue
            if remote_only and _numeric(payload.get("remote_elapsed_ms")) <= 0:
                continue
            total += 1
        return total

    cache_hit_count = stage_cache_count("aggregate_cache_hit_count", "hit")
    cache_miss_count = stage_cache_count("aggregate_cache_miss_count", "miss")
    cache_disabled_count = stage_cache_count("aggregate_cache_disabled_count", "disabled")
    cache_bypassed_remote_count = stage_cache_count(
        "aggregate_cache_bypassed_remote_count",
        "bypassed",
        remote_only=True,
    )
    remote_call_count = (
        sum(1 for subcall in subcall_payloads if _numeric(subcall.get("remote_elapsed_ms")) > 0)
        + sum(
            1
            for payload in stage_payloads
            if not isinstance(payload.get("subcalls"), list)
            and _numeric(payload.get("remote_elapsed_ms")) > 0
        )
    )
    summary = {
        "cache_enabled": bool(get_settings().llm_cache_enabled),
        "evidence_pack_enabled": True,
        "llm_stage_count": len(stage_payloads),
        "evidence_pack_stage_count": sum(
            1
            for payload in stage_payloads
            if str(payload.get("stage") or "") in EVIDENCE_PACK_LLM_STAGES
            and _integer(_prefer_aggregate(payload, "aggregate_prompt_chars", "prompt_chars")) > 0
        ),
        "remote_call_count": remote_call_count,
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "cache_disabled_count": cache_disabled_count,
        "cache_bypassed_remote_count": cache_bypassed_remote_count,
        "total_prompt_chars": sum(prompt_values),
        "remote_prompt_chars": sum(
            _integer(subcall.get("prompt_chars"))
            for subcall in subcall_payloads
            if _numeric(subcall.get("remote_elapsed_ms")) > 0
        )
        + sum(
            _integer(payload.get("prompt_chars"))
            for payload in stage_payloads
            if not isinstance(payload.get("subcalls"), list)
            and _numeric(payload.get("remote_elapsed_ms")) > 0
        ),
        "total_llm_remote_elapsed_ms": round(sum(remote_elapsed_values), 3),
        "total_llm_saved_ms_by_cache": round(sum(saved_values), 3),
    }
    if llm_profile:
        summary["llm_profile"] = llm_profile
    if output_language:
        summary["output_language"] = output_language
    if llm_profile_policy is not None:
        summary["llm_profile_policy"] = llm_profile_policy
    summary["skipped_by_profile_count"] = sum(
        1 for payload in stage_payloads if str(payload.get("status") or "") == "skipped_by_profile"
    )
    if profile_limited_stage_count is not None:
        summary["profile_limited_stage_count"] = int(profile_limited_stage_count)
    if max_llm_postrun_reflections_resolved is not None:
        summary["max_llm_postrun_reflections_resolved"] = int(max_llm_postrun_reflections_resolved)
    if allow_postrun_fallback_reflections is not None:
        summary["allow_postrun_fallback_reflections"] = bool(allow_postrun_fallback_reflections)
    return summary


def with_llm_trace_summary(
    llm_trace: dict[str, object],
    *,
    llm_profile: str | None = None,
    output_language: str | None = None,
    llm_profile_policy: dict[str, object] | None = None,
    profile_limited_stage_count: int | None = None,
    max_llm_postrun_reflections_resolved: int | None = None,
    allow_postrun_fallback_reflections: bool | None = None,
) -> dict[str, object]:
    payload = {
        stage: trace
        for stage, trace in llm_trace.items()
        if stage != "summary"
    }
    payload["summary"] = build_llm_trace_summary(
        payload,
        llm_profile=llm_profile,
        output_language=output_language,
        llm_profile_policy=llm_profile_policy,
        profile_limited_stage_count=profile_limited_stage_count,
        max_llm_postrun_reflections_resolved=max_llm_postrun_reflections_resolved,
        allow_postrun_fallback_reflections=allow_postrun_fallback_reflections,
    )
    return payload
