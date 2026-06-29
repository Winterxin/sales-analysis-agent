from __future__ import annotations

import json
import queue
import re
import threading
import time
import hashlib
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from app.core.config import get_settings
from app.schemas.analysis_plan import AnalysisPlan
from app.schemas.notebook_content import NotebookContentPlan
from app.schemas.notebook_narrative import NotebookSectionNarrative
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport
from app.schemas.schema_mapping import SchemaMapping
from app.services.llm_evidence_pack import build_llm_evidence_pack
from app.services.prompt_loader import load_prompt

NOTEBOOK_PREFERRED_MODELS = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.2", "gpt-5.1"]

LLM_CACHE_VOLATILE_KEYS = {
    "task_id",
    "run_id",
    "request_id",
    "created_at",
    "updated_at",
    "timestamp",
}

SCHEMA_MAPPING_ALLOWED_FIELDS = {
    "order_id",
    "order_datetime",
    "product_name",
    "sku",
    "quantity",
    "sales_amount",
    "unit_price",
    "category",
    "sub_category",
    "channel",
    "store",
    "customer_id",
    "region",
    "segment",
    "city",
    "state",
    "country",
    "discount",
    "profit",
    "productline",
    "deal_size",
    "order_status",
}


class LLMCompletionUnavailable(RuntimeError):
    """Raised after the active LLM backend has already proven unavailable."""


class LLMCompletionTimedOut(TimeoutError):
    """Raised when an LLM completion exceeds the per-call wall-clock budget."""


@dataclass(slots=True)
class LLMClientConfig:
    base_url: str
    api_key: str
    model: str | None = None
    source: str = "manual"


def extract_stream_text(lines: list[str]) -> str:
    content_parts: list[str] = []
    for line in lines:
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        data = json.loads(payload)
        for choice in data.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if isinstance(content, str):
                content_parts.append(content)
    return "".join(content_parts)


def extract_nonstream_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("No choices returned from LLM response")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "".join(text_parts)
    raise ValueError("Unsupported non-stream response payload")


def parse_json_text(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    snippets = [candidate]
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippets.append(candidate[start : end + 1])

    for snippet in snippets:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            cleaned = re.sub(r",(\s*[}\]])", r"\1", snippet)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("Unable to parse JSON text", candidate, 0)


def _cache_stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _cache_stable_payload(item)
            for key, item in value.items()
            if str(key) not in LLM_CACHE_VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [_cache_stable_payload(item) for item in value]
    return value


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(
        _cache_stable_payload(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_llm_cache_key(
    *,
    stage: str,
    model: str | None,
    source: str,
    prompt_version: str,
    system_prompt: str,
    user_payload: dict[str, Any],
) -> str:
    key_payload = {
        "stage": stage,
        "model": model or "",
        "source": source,
        "prompt_version": prompt_version,
        "system_prompt": system_prompt,
        "user_payload": _cache_stable_payload(user_payload),
    }
    return _sha256_text(_stable_json_dumps(key_payload))


def _method_accepts_keyword(method: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def complete_json_for_stage(
    llm_client: Any,
    *,
    system_prompt: str,
    user_payload: dict[str, Any],
    cache_stage: str,
    preferred_models: list[str] | None = None,
) -> dict[str, Any]:
    method = getattr(llm_client, "complete_json")
    if _method_accepts_keyword(method, "cache_stage"):
        return method(
            system_prompt=system_prompt,
            user_payload=user_payload,
            preferred_models=preferred_models,
            cache_stage=cache_stage,
        )
    if preferred_models is not None and _method_accepts_keyword(method, "preferred_models"):
        return method(
            system_prompt=system_prompt,
            user_payload=user_payload,
            preferred_models=preferred_models,
        )
    return method(system_prompt=system_prompt, user_payload=user_payload)


def _sanitize_schema_mapping_payload(
    result: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    allowed_columns = set(columns)
    raw_mapping = result.get("field_mapping")
    cleaned_mapping: dict[str, str] = {}
    if isinstance(raw_mapping, dict):
        for original_name, canonical_name in raw_mapping.items():
            if original_name not in allowed_columns:
                continue
            if not isinstance(canonical_name, str):
                continue
            canonical_name = canonical_name.strip()
            if canonical_name in SCHEMA_MAPPING_ALLOWED_FIELDS:
                cleaned_mapping[str(original_name)] = canonical_name
    cleaned = dict(result)
    cleaned["field_mapping"] = cleaned_mapping
    for key in ("missing_required_fields", "uncertain_fields"):
        raw_values = result.get(key, [])
        cleaned[key] = [str(item) for item in raw_values if isinstance(item, str) and item.strip()]
    return cleaned


def _message_content_for_user_payload(
    user_payload: dict[str, Any],
    image_urls: list[str] | None = None,
) -> str | list[dict[str, Any]]:
    payload_text = json.dumps(user_payload, ensure_ascii=False)
    if not image_urls:
        return payload_text
    content: list[dict[str, Any]] = [{"type": "text", "text": payload_text}]
    for image_url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def _preview_tables(
    tables: dict[str, list[dict[str, Any]]],
    *,
    max_tables: int = 8,
    max_rows: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    preview: dict[str, list[dict[str, Any]]] = {}
    for table_name, rows in list(tables.items())[:max_tables]:
        if isinstance(rows, list):
            preview[str(table_name)] = [
                dict(row)
                for row in rows[:max_rows]
                if isinstance(row, dict)
            ]
    return preview


def _compact_report_for_prompt(report: AnalysisReport) -> dict[str, Any]:
    return {
        "task_id": report.task_id,
        "dataset_type": report.dataset_type,
        "module_count": report.module_count,
        "summary": report.summary[:8],
        "modules": [
            {
                "module_id": module.module_id,
                "title": module.title,
                "chart_type": module.chart_type,
                "summary_metrics": module.summary_metrics,
                "tables": _preview_tables(module.tables),
                "findings": module.findings[:8],
                "warnings": module.warnings[:5],
            }
            for module in report.modules
        ],
    }


def discover_cliproxy_config_path(config_path: Path) -> Path | None:
    if config_path.exists():
        return config_path
    return None


def load_cliproxy_config(config_path: Path) -> LLMClientConfig | None:
    resolved_config_path = discover_cliproxy_config_path(config_path)
    if resolved_config_path is None:
        return None

    payload = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    api_keys = payload.get("api-keys") or []
    if not api_keys:
        return None

    host = payload.get("host") or "127.0.0.1"
    if host in {"", "0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = int(payload.get("port") or 8317)
    return LLMClientConfig(
        base_url=f"http://{host}:{port}/v1",
        api_key=str(api_keys[0]),
        model=None,
        source="cliproxy",
    )


class LLMClient:
    def __init__(self, config: LLMClientConfig | None) -> None:
        self._config = config
        self._resolved_model: str | None = config.model if config else None
        self._model_validated = False
        self._last_request_model: str | None = config.model if config else None
        self._completion_unavailable_reason: str | None = None
        self._last_completion_metrics: dict[str, Any] = {}
        self._completion_metrics_history: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self._config is not None

    @property
    def source(self) -> str:
        if self._config is None:
            return "disabled"
        return self._config.source

    @property
    def configured_model(self) -> str | None:
        if self._config is None:
            return None
        return self._last_request_model or self._resolved_model or self._config.model

    @property
    def completion_unavailable_reason(self) -> str | None:
        return self._completion_unavailable_reason

    @property
    def last_completion_metrics(self) -> dict[str, Any]:
        return dict(self._last_completion_metrics)

    @property
    def completion_metrics_history(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._completion_metrics_history]

    def snapshot_completion_metrics(self) -> int:
        return len(self._completion_metrics_history)

    def collect_completion_metrics_since(self, snapshot_index: int) -> list[dict[str, Any]]:
        start = max(0, int(snapshot_index))
        return [dict(item) for item in self._completion_metrics_history[start:]]

    def _cache_model_for_key(self, preferred_models: list[str] | None = None) -> str | None:
        if self._config is None:
            return None
        if self._last_request_model:
            return self._last_request_model
        if self._resolved_model:
            return self._resolved_model
        if self._config.model:
            return self._config.model
        if preferred_models:
            return preferred_models[0]
        if self.source == "cliproxy":
            return NOTEBOOK_PREFERRED_MODELS[0]
        return None

    def _completion_prompt_chars(self, system_prompt: str, user_payload: dict[str, Any]) -> int:
        return len(system_prompt) + len(json.dumps(user_payload, ensure_ascii=False))

    def _record_completion_metrics(
        self,
        *,
        started_at: float,
        prompt_chars: int,
        attempt_count: int,
        result: dict[str, Any] | None = None,
        exc: Exception | None = None,
        cache_enabled: bool | None = None,
        cache_status: str = "bypassed",
        cache_key: str | None = None,
        cache_stage: str | None = None,
        remote_elapsed_ms: float | None = None,
        saved_ms: float = 0.0,
    ) -> None:
        elapsed_ms = round((time.monotonic() - started_at) * 1000, 3)
        if remote_elapsed_ms is None:
            remote_elapsed_ms = elapsed_ms if attempt_count > 0 else 0.0
        metrics: dict[str, Any] = {
            "stage": cache_stage,
            "elapsed_ms": elapsed_ms,
            "prompt_chars": prompt_chars,
            "response_chars": None,
            "attempt_count": attempt_count,
            "error_type": None,
            "cache_enabled": cache_enabled,
            "cache_status": cache_status,
            "cache_key": cache_key[:16] if cache_key else None,
            "remote_elapsed_ms": round(float(remote_elapsed_ms), 3),
            "saved_ms": round(float(saved_ms or 0.0), 3),
            "model": self.configured_model,
            "source": self.source,
        }
        if result is not None:
            metrics["response_chars"] = len(json.dumps(result, ensure_ascii=False))
        if exc is not None:
            metrics["error_type"] = type(exc).__name__
        self._last_completion_metrics = metrics
        self._completion_metrics_history.append(dict(metrics))

    def _cache_file_path(self, cache_key: str) -> Path:
        return get_settings().llm_cache_dir / f"{cache_key}.json"

    def _read_cached_completion(self, cache_key: str) -> dict[str, Any] | None:
        cache_path = self._cache_file_path(cache_key)
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if payload.get("cache_key") != cache_key:
            return None
        response = payload.get("response")
        return payload if isinstance(response, dict) else None

    def _write_cached_completion(
        self,
        *,
        cache_key: str,
        stage: str,
        model: str | None,
        system_prompt: str,
        user_payload: dict[str, Any],
        response: dict[str, Any],
        remote_elapsed_ms: float,
        prompt_chars: int,
    ) -> None:
        settings = get_settings()
        cache_path = settings.llm_cache_dir / f"{cache_key}.json"
        payload_json = _stable_json_dumps(user_payload)
        response_json = json.dumps(response, ensure_ascii=False)
        cache_payload = {
            "cache_key": cache_key,
            "stage": stage,
            "model": model,
            "source": self.source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "system_prompt_hash": _sha256_text(system_prompt),
            "user_payload_hash": _sha256_text(payload_json),
            "response": response,
            "remote_elapsed_ms": round(float(remote_elapsed_ms), 3),
            "prompt_chars": prompt_chars,
            "response_chars": len(response_json),
        }
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(cache_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temp_path.replace(cache_path)
        except Exception:
            return

    def _headers(self) -> dict[str, str]:
        if self._config is None:
            return {}
        return {"Authorization": f"Bearer {self._config.api_key}"}

    def list_models(self) -> list[str]:
        if self._config is None:
            return []
        response = httpx.get(
            f"{self._config.base_url}/models",
            headers=self._headers(),
            timeout=20.0,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        return [str(item["id"]) for item in data if "id" in item]

    def resolve_model(self, preferred_models: list[str] | None = None) -> str:
        if self._config is None:
            raise RuntimeError("LLM client is disabled")
        if self._resolved_model and self._model_validated and not preferred_models:
            return self._resolved_model

        effective_preferred_models = preferred_models
        if effective_preferred_models is None and self.source == "cliproxy":
            effective_preferred_models = NOTEBOOK_PREFERRED_MODELS
        try:
            models = self.list_models()
        except Exception:
            fallback_candidates: list[str] = []
            if self._resolved_model:
                fallback_candidates.append(self._resolved_model)
            if self._config.model and self._config.model not in fallback_candidates:
                fallback_candidates.append(self._config.model)
            if effective_preferred_models:
                for candidate in effective_preferred_models:
                    if candidate not in fallback_candidates:
                        fallback_candidates.append(candidate)
            if fallback_candidates:
                self._resolved_model = fallback_candidates[0]
                self._model_validated = True
                return self._resolved_model
            raise

        if effective_preferred_models and models:
            for candidate in effective_preferred_models:
                if candidate in models:
                    self._resolved_model = candidate
                    self._model_validated = True
                    return candidate
        if self._resolved_model and (not models or self._resolved_model in models):
            self._model_validated = True
            return self._resolved_model

        self._resolved_model = models[0] if models else (self._resolved_model or "gpt-5.4-mini")
        self._model_validated = True
        return self._resolved_model

    def _completion_timeout(self) -> float:
        configured_timeout = get_settings().llm_completion_timeout_seconds
        if configured_timeout is not None and configured_timeout > 0:
            return float(configured_timeout)
        if self.source == "cliproxy":
            return 90.0
        return 60.0

    def _completion_wall_timeout(self) -> float:
        configured_timeout = get_settings().llm_completion_wall_timeout_seconds
        if configured_timeout is not None and configured_timeout > 0:
            return float(configured_timeout)
        if self.source == "cliproxy":
            return 120.0
        return 90.0

    def _completion_max_attempts(self) -> int:
        return max(1, int(get_settings().llm_max_attempts or 1))

    def _http_error_detail(self, exc: httpx.HTTPStatusError, limit: int = 360) -> str:
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
        if len(detail) > limit:
            detail = detail[: limit - 3] + "..."
        return detail

    def _error_detail(self, exc: Exception, limit: int = 360) -> str:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            detail = self._http_error_detail(exc, limit=limit)
            if detail:
                return f"HTTP {exc.response.status_code}: {detail}"
            return f"HTTP {exc.response.status_code}"
        detail = " ".join(str(exc).split())
        if len(detail) > limit:
            detail = detail[: limit - 3] + "..."
        return detail or type(exc).__name__

    def _marks_backend_unavailable(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            return self._is_cliproxy_upstream_connection_failure(exc)
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
            ),
        )

    def _mark_completion_unavailable(self, exc: Exception) -> None:
        if self.source != "cliproxy" or not self._marks_backend_unavailable(exc):
            return
        self._completion_unavailable_reason = self._error_detail(exc)

    def _is_cliproxy_upstream_connection_failure(self, exc: Exception) -> bool:
        if self.source != "cliproxy":
            return False
        detail = self._error_detail(exc).lower()
        connection_markers = [
            "dial tcp",
            "connectex",
            "connection attempt failed",
            "connected host has failed to respond",
        ]
        return any(marker in detail for marker in connection_markers)

    def _should_use_compat_completion(self, exc: Exception) -> bool:
        if isinstance(exc, (json.JSONDecodeError, ValueError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            return exc.response.status_code == 400 or (
                self.source == "cliproxy" and exc.response.status_code >= 500
            )
        return False

    def _is_cliproxy_model_cooling_down(self, exc: Exception) -> bool:
        if self.source != "cliproxy":
            return False
        if not isinstance(exc, httpx.HTTPStatusError) or exc.response is None:
            return False
        if exc.response.status_code != 429:
            return False
        return "cooling down" in self._error_detail(exc).lower()

    def _retry_delay_seconds(self, exc: Exception, attempt_index: int) -> float:
        if self._is_cliproxy_model_cooling_down(exc):
            return min(2.0 * (2**attempt_index), 8.0)
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            if exc.response.status_code == 429:
                return min(1.0 * (attempt_index + 1), 3.0)
        return 0.0

    def _sleep_before_retry(self, delay_seconds: float) -> None:
        if delay_seconds <= 0:
            return
        time.sleep(delay_seconds)

    def _run_with_wall_timeout(self, operation, *, description: str) -> dict[str, Any]:
        timeout_seconds = self._completion_wall_timeout()
        if timeout_seconds <= 0:
            return operation()

        result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

        def run_operation() -> None:
            try:
                result_queue.put(("result", operation()))
            except Exception as exc:  # pragma: no cover - exercised through caller paths
                result_queue.put(("error", exc))

        thread = threading.Thread(
            target=run_operation,
            name=f"llm-completion-{self.source}",
            daemon=True,
        )
        thread.start()
        thread.join(timeout_seconds)
        if thread.is_alive():
            raise LLMCompletionTimedOut(
                f"{description} timed out after {timeout_seconds:.1f}s"
            )

        kind, payload = result_queue.get_nowait()
        if kind == "error":
            raise payload
        return payload

    def _request_payload(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        *,
        stream: bool = True,
        response_format: bool = True,
        preferred_models: list[str] | None = None,
        image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        model = self.resolve_model(preferred_models)
        self._last_request_model = model
        payload = {
            "model": model,
            "temperature": 0.1,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _message_content_for_user_payload(user_payload, image_urls=image_urls),
                },
            ],
        }
        if response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code if exc.response is not None else None
            return status_code in {408, 409, 429} or (status_code is not None and status_code >= 500)
        return isinstance(
            exc,
            (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                httpx.TransportError,
            ),
        )

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        preferred_models: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._config is None:
            raise RuntimeError("LLM client is disabled")

        with httpx.stream(
            "POST",
            f"{self._config.base_url}/chat/completions",
            headers=self._headers(),
            timeout=self._completion_timeout(),
            json=self._request_payload(
                system_prompt,
                user_payload,
                stream=True,
                response_format=True,
                preferred_models=preferred_models,
                image_urls=None,
            ),
        ) as response:
            if response.is_error:
                response.read()
                response.raise_for_status()
            lines = [line for line in response.iter_lines() if line]
        content = extract_stream_text(lines)
        return parse_json_text(content)

    def _complete_json_compat_once(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        preferred_models: list[str] | None = None,
    ) -> dict[str, Any]:
        if self._config is None:
            raise RuntimeError("LLM client is disabled")

        response = httpx.post(
            f"{self._config.base_url}/chat/completions",
            headers=self._headers(),
            timeout=self._completion_timeout(),
            json=self._request_payload(
                system_prompt,
                user_payload,
                stream=False,
                response_format=False,
                preferred_models=preferred_models,
                image_urls=None,
            ),
        )
        response.raise_for_status()
        content = extract_nonstream_text(response.json())
        return parse_json_text(content)

    def complete_json(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        preferred_models: list[str] | None = None,
        cache_stage: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        prompt_chars = self._completion_prompt_chars(system_prompt, user_payload)
        completion_attempt_count = 0
        settings = get_settings()
        cache_enabled = bool(settings.llm_cache_enabled)
        cache_status = "bypassed" if cache_stage is None else ("miss" if cache_enabled else "disabled")
        cache_key: str | None = None
        cache_model = self._cache_model_for_key(preferred_models)
        if cache_stage is not None and cache_enabled:
            cache_key = build_llm_cache_key(
                stage=cache_stage,
                model=cache_model,
                source=self.source,
                prompt_version=settings.llm_cache_version,
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
            cached_payload = self._read_cached_completion(cache_key)
            if cached_payload is not None:
                response = dict(cached_payload["response"])
                self._record_completion_metrics(
                    started_at=started_at,
                    prompt_chars=prompt_chars,
                    attempt_count=0,
                    result=response,
                    cache_enabled=cache_enabled,
                    cache_status="hit",
                    cache_key=cache_key,
                    cache_stage=cache_stage,
                    remote_elapsed_ms=0.0,
                    saved_ms=float(cached_payload.get("remote_elapsed_ms") or 0.0),
                )
                return response

        def record_success(result: dict[str, Any]) -> None:
            self._record_completion_metrics(
                started_at=started_at,
                prompt_chars=prompt_chars,
                attempt_count=completion_attempt_count,
                result=result,
                cache_enabled=cache_enabled,
                cache_status=cache_status,
                cache_key=cache_key,
                cache_stage=cache_stage,
                saved_ms=0.0,
            )
            if cache_key is not None and cache_stage is not None and cache_status == "miss":
                self._write_cached_completion(
                    cache_key=cache_key,
                    stage=cache_stage,
                    model=cache_model,
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    response=result,
                    remote_elapsed_ms=float(self._last_completion_metrics.get("remote_elapsed_ms") or 0.0),
                    prompt_chars=prompt_chars,
                )

        if self._completion_unavailable_reason is not None:
            exc = LLMCompletionUnavailable(
                "LLM completion backend is unavailable for this analysis run. "
                f"Previous failure: {self._completion_unavailable_reason}"
            )
            self._record_completion_metrics(
                started_at=started_at,
                prompt_chars=prompt_chars,
                attempt_count=completion_attempt_count,
                exc=exc,
                cache_enabled=cache_enabled,
                cache_status=cache_status,
                cache_key=cache_key,
                cache_stage=cache_stage,
                remote_elapsed_ms=0.0,
            )
            raise exc

        last_error: Exception | None = None
        attempt = 0
        max_attempts = self._completion_max_attempts()
        while attempt < max_attempts:
            try:
                completion_attempt_count += 1
                result = self._run_with_wall_timeout(
                    lambda: self._complete_json_once(
                        system_prompt,
                        user_payload,
                        preferred_models=preferred_models,
                    ),
                    description="LLM completion",
                )
                record_success(result)
                return result
            except Exception as exc:
                if self._is_cliproxy_model_cooling_down(exc):
                    max_attempts = max(max_attempts, 5)
                if self._is_cliproxy_upstream_connection_failure(exc):
                    self._mark_completion_unavailable(exc)
                    self._record_completion_metrics(
                        started_at=started_at,
                        prompt_chars=prompt_chars,
                        attempt_count=completion_attempt_count,
                        exc=exc,
                        cache_enabled=cache_enabled,
                        cache_status=cache_status,
                        cache_key=cache_key,
                        cache_stage=cache_stage,
                    )
                    raise
                if self._should_use_compat_completion(exc):
                    try:
                        completion_attempt_count += 1
                        result = self._run_with_wall_timeout(
                            lambda: self._complete_json_compat_once(
                                system_prompt,
                                user_payload,
                                preferred_models=preferred_models,
                            ),
                            description="LLM compat completion",
                        )
                        record_success(result)
                        return result
                    except Exception as compat_exc:
                        self._mark_completion_unavailable(compat_exc)
                        self._record_completion_metrics(
                            started_at=started_at,
                            prompt_chars=prompt_chars,
                            attempt_count=completion_attempt_count,
                            exc=compat_exc,
                            cache_enabled=cache_enabled,
                            cache_status=cache_status,
                            cache_key=cache_key,
                            cache_stage=cache_stage,
                        )
                        raise
                last_error = exc
                if attempt == max_attempts - 1 or not self._should_retry(exc):
                    self._mark_completion_unavailable(exc)
                    self._record_completion_metrics(
                        started_at=started_at,
                        prompt_chars=prompt_chars,
                        attempt_count=completion_attempt_count,
                        exc=exc,
                        cache_enabled=cache_enabled,
                        cache_status=cache_status,
                        cache_key=cache_key,
                        cache_stage=cache_stage,
                    )
                    raise
                self._sleep_before_retry(self._retry_delay_seconds(exc, attempt))
                attempt += 1

        if last_error is not None:
            self._mark_completion_unavailable(last_error)
            self._record_completion_metrics(
                started_at=started_at,
                prompt_chars=prompt_chars,
                attempt_count=completion_attempt_count,
                exc=last_error,
                cache_enabled=cache_enabled,
                cache_status=cache_status,
                cache_key=cache_key,
                cache_stage=cache_stage,
            )
            raise last_error
        exc = RuntimeError("LLM completion failed without an exception")
        self._record_completion_metrics(
            started_at=started_at,
            prompt_chars=prompt_chars,
            attempt_count=completion_attempt_count,
            exc=exc,
            cache_enabled=cache_enabled,
            cache_status=cache_status,
            cache_key=cache_key,
            cache_stage=cache_stage,
        )
        raise exc

    def complete_json_with_images(
        self,
        system_prompt: str,
        user_payload: dict[str, Any],
        image_urls: list[str],
        preferred_models: list[str] | None = None,
        cache_stage: str | None = None,
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        prompt_chars = self._completion_prompt_chars(system_prompt, user_payload)
        attempt_count = 0
        cache_enabled = bool(get_settings().llm_cache_enabled)
        if self._config is None:
            exc = RuntimeError("LLM client is disabled")
            self._record_completion_metrics(
                started_at=started_at,
                prompt_chars=prompt_chars,
                attempt_count=attempt_count,
                exc=exc,
                cache_enabled=cache_enabled,
                cache_status="bypassed",
                cache_stage=cache_stage,
                remote_elapsed_ms=0.0,
            )
            raise exc

        try:
            attempt_count += 1
            with httpx.stream(
                "POST",
                f"{self._config.base_url}/chat/completions",
                headers=self._headers(),
                timeout=self._completion_timeout(),
                json=self._request_payload(
                    system_prompt,
                    user_payload,
                    stream=True,
                    response_format=True,
                    preferred_models=preferred_models,
                    image_urls=image_urls,
                ),
            ) as response:
                if response.is_error:
                    response.read()
                    response.raise_for_status()
                lines = [line for line in response.iter_lines() if line]
            content = extract_stream_text(lines)
            result = parse_json_text(content)
            self._record_completion_metrics(
                started_at=started_at,
                prompt_chars=prompt_chars,
                attempt_count=attempt_count,
                result=result,
                cache_enabled=cache_enabled,
                cache_status="bypassed",
                cache_stage=cache_stage,
            )
            return result
        except Exception:
            attempt_count += 1
            try:
                response = httpx.post(
                    f"{self._config.base_url}/chat/completions",
                    headers=self._headers(),
                    timeout=self._completion_timeout(),
                    json=self._request_payload(
                        system_prompt,
                        user_payload,
                        stream=False,
                        response_format=False,
                        preferred_models=preferred_models,
                        image_urls=image_urls,
                    ),
                )
                response.raise_for_status()
                content = extract_nonstream_text(response.json())
                result = parse_json_text(content)
                self._record_completion_metrics(
                    started_at=started_at,
                    prompt_chars=prompt_chars,
                    attempt_count=attempt_count,
                    result=result,
                    cache_enabled=cache_enabled,
                    cache_status="bypassed",
                    cache_stage=cache_stage,
                )
                return result
            except Exception as exc:
                self._record_completion_metrics(
                    started_at=started_at,
                    prompt_chars=prompt_chars,
                    attempt_count=attempt_count,
                    exc=exc,
                    cache_enabled=cache_enabled,
                    cache_status="bypassed",
                    cache_stage=cache_stage,
                )
                raise

    def suggest_schema_mapping(
        self, columns: list[str], rule_mapping: SchemaMapping
    ) -> SchemaMapping:
        payload = {
            "task": "Map sales dataset columns to canonical business fields.",
            "columns": columns,
            "rule_mapping": rule_mapping.model_dump(),
            "allowed_fields": sorted(SCHEMA_MAPPING_ALLOWED_FIELDS),
            "dataset_types": ["sales_transaction", "unknown"],
        }
        result = complete_json_for_stage(
            self,
            system_prompt=(
                "You map sales CSV columns to canonical field names. "
                "Return valid JSON with keys: dataset_type, field_mapping, confidence, "
                "missing_required_fields, uncertain_fields. Do not invent columns."
            ),
            user_payload=payload,
            cache_stage="schema_mapping",
        )
        result = _sanitize_schema_mapping_payload(result, columns)
        return SchemaMapping.model_validate(result)

    def summarize_report(
        self,
        report: AnalysisReport,
        evidence_pack: dict[str, Any] | None = None,
        language_instruction: str | None = None,
    ) -> list[str]:
        llm_evidence_pack = evidence_pack or build_llm_evidence_pack(
            report,
            SchemaMapping(dataset_type=report.dataset_type),
        )
        payload = {
            "task": "Summarize sales analysis report from evidence pack.",
            "evidence_pack": llm_evidence_pack,
            "constraints": [
                "Only use facts in evidence_pack.",
                "Must mention sales/profit KPI when available.",
                "Must mention discount threshold or what-if when available.",
                "Must mention modeling role when available.",
                "Do not invent causal claims.",
                "Avoid strong causal, attribution, or certainty wording unless evidence_pack contains approval records, experiments, causal identification, or explicit rules.",
                "Do not write 审批失控、核心原因、必然导致、证明 unless directly evidenced; prefer 风险信号、复核重点、可能加剧、需要结合明细验证.",
                "If evidence is missing, write a data limitation note instead of guessing.",
                "Return JSON with a single key named summary.",
                "Keep summary items concise.",
                *( [language_instruction] if language_instruction else [] ),
            ],
        }
        result = complete_json_for_stage(
            self,
            system_prompt=(
                "You are a careful analytics narrator. Return JSON only. "
                "The JSON must contain a 'summary' array of concise bullet strings."
            ),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="report_summary",
        )
        summary = result.get("summary", [])
        return [str(item) for item in summary if str(item).strip()]

    def suggest_notebook_outline(
        self,
        schema_mapping: SchemaMapping,
        analysis_plan: AnalysisPlan,
        fallback_outline: NotebookOutline,
        language_instruction: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "task": "Design a rich sales analysis notebook outline.",
            "schema_mapping": schema_mapping.model_dump(),
            "analysis_plan": analysis_plan.model_dump(),
            "fallback_outline": fallback_outline.model_dump(),
            "constraints": [
                "Return JSON only.",
                "Use only valid section ids from the fallback outline registry.",
                "Prefer a rich but coherent notebook structure.",
                *( [language_instruction] if language_instruction else [] ),
            ],
        }
        return complete_json_for_stage(
            self,
            system_prompt=(
                "You are planning a Kaggle-style sales analysis notebook. "
                "Return JSON with keys: title, sections. "
                "Each section must contain section_id, title, purpose."
            ),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="notebook_outline",
        )

    def suggest_notebook_narrative(
        self,
        outline: NotebookOutline,
        report: AnalysisReport,
        schema_mapping: SchemaMapping,
        fallback_narrative: NotebookNarrative,
        evidence_pack: dict[str, Any] | None = None,
        language_instruction: str | None = None,
    ) -> dict[str, Any]:
        llm_evidence_pack = evidence_pack or build_llm_evidence_pack(report, schema_mapping)
        payload = {
            "task": "Write notebook section narratives for a sales analysis notebook.",
            "outline": outline.model_dump(),
            "evidence_pack": llm_evidence_pack,
            "schema_mapping": schema_mapping.model_dump(),
            "fallback_narrative": fallback_narrative.model_dump(),
            "constraints": [
                "Return JSON only.",
                "Use only section ids already present in the outline.",
                "Ground all observations in evidence_pack.",
                "Use concrete values, categories, thresholds, what-if, and model metrics from evidence_pack when available.",
                "Do not invent causal claims or unsupported metrics.",
                "Avoid strong causal, attribution, or certainty wording unless evidence_pack contains approval records, experiments, causal identification, or explicit rules.",
                "Prefer 风险信号、复核重点、可能加剧、需要结合明细验证 over 审批失控、核心原因、必然导致、证明.",
                *( [language_instruction] if language_instruction else [] ),
            ],
        }
        return complete_json_for_stage(
            self,
            system_prompt=(
                "You are writing a Kaggle-style sales analysis notebook narrative. "
                "Return JSON with keys: sections, suggested_followups. "
                "Each section must contain section_id, intro, key_observations, "
                "business_takeaway, followup_question."
            ),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="notebook_narrative",
        )

    def suggest_metric_distribution_strategy(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("metric_distribution_strategy.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="metric_distribution_strategy",
        )

    def suggest_sales_trend_strategy(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("sales_trend_strategy.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="sales_trend_strategy",
        )

    def suggest_discount_profit_strategy(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("discount_profit_strategy.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="discount_profit_strategy",
        )

    def suggest_modeling_interpretation(
        self,
        payload: dict[str, Any],
        language_instruction: str | None = None,
    ) -> dict[str, Any]:
        if language_instruction:
            payload = {**payload, "language_instruction": language_instruction}
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("modeling_interpretation.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="modeling_interpretation",
        )

    def suggest_modeling_opportunity_decision(
        self,
        payload: dict[str, Any],
        language_instruction: str | None = None,
    ) -> dict[str, Any]:
        if language_instruction:
            payload = {**payload, "language_instruction": language_instruction}
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("modeling_opportunity_decision.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="modeling_opportunity_decision",
        )

    def suggest_modeling_outcome_interpretation(
        self,
        payload: dict[str, Any],
        language_instruction: str | None = None,
    ) -> dict[str, Any]:
        if language_instruction:
            payload = {**payload, "language_instruction": language_instruction}
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("modeling_outcome_interpretation.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="modeling_outcome_interpretation",
        )

    def suggest_product_category_strategy(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("product_category_strategy.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="product_category_strategy",
        )

    def suggest_segment_region_strategy(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("segment_region_strategy.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="segment_region_strategy",
        )

    def suggest_notebook_section_content(
        self,
        section: NotebookSection,
        narrative_section: NotebookSectionNarrative | None,
        report_context: dict[str, Any],
        schema_mapping: SchemaMapping,
        analysis_plan: AnalysisPlan,
        fallback_section: dict[str, Any],
        language_instruction: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "task": "Generate richer executable notebook content for a single section.",
            "section": section.model_dump(),
            "narrative_section": (
                narrative_section.model_dump() if narrative_section is not None else None
            ),
            "report_context": report_context,
            "schema_mapping": schema_mapping.model_dump(),
            "analysis_plan": analysis_plan.model_dump(),
            "fallback_section": fallback_section,
            "language_instruction": language_instruction,
        }
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("notebook_section_content.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="notebook_content",
        )

    def suggest_postrun_chart_reflection(
        self,
        chart_context: dict[str, Any],
        fallback_markdown: str,
        language_instruction: str | None = None,
    ) -> dict[str, Any]:
        prompt_chart_context = dict(chart_context)
        english = bool(
            language_instruction
            and (
                "English" in language_instruction
                or "Do not include Chinese" in language_instruction
            )
        )
        if english:
            constraints = [
                "Return JSON only.",
                "Write one compact English paragraph with 3-5 complete sentences.",
                "Use English only in all user-facing prose.",
                "Do not output Chinese characters or CJK narrative.",
                "Keep raw dataset field names unchanged when needed.",
                "Ground the paragraph in chart_context, table_preview, module_summary_metrics, and module_findings.",
                "Refer to at least one concrete number, date, category, product, country, or share when present.",
                "Include at least one comparison such as peak/trough, head/tail, growth/decline, or sales/profit mismatch.",
                "Build an observation -> business interpretation -> next action chain from executed evidence.",
                "When section_peer_charts are present, connect the current chart to at least one peer chart or section-level finding.",
                "Prioritize the current section and its module findings; use global report summary only when directly relevant.",
                "Do not use boilerplate opening phrases.",
                "Do not invent causal claims or unsupported metrics.",
                *( [language_instruction] if language_instruction else [] ),
            ]
        else:
            constraints = [
                "Return JSON only.",
                "Use one compact Chinese paragraph with 3-5 sentences, about 180-380 Chinese characters total.",
                "Ground the paragraph in chart_context, table_preview, module_summary_metrics, and module_findings.",
                "Refer to at least one concrete number, date, category, product, country, or share when present.",
                "Include at least one comparison such as peak/trough, head/tail, growth/decline, or sales/profit mismatch.",
                "Do not use headings like 下一步追问 or 关键观察.",
                "Focus on analysis, not just chart description.",
                "Do not use boilerplate like '如果图表显示', '这张图主要用于', or '当前图表用于观察'.",
                "Build an observation -> business interpretation -> next action chain from executed evidence.",
                "When section_peer_charts are present, connect the current chart to at least one peer chart or section-level finding.",
                "Prioritize the current section and its module findings; use global report summary only when directly relevant.",
                "Avoid repeating a generic opening sentence across chart kinds.",
                "Avoid strong causal, attribution, or certainty wording unless evidence contains approval records, experiments, causal identification, or explicit rules.",
                "Prefer 风险信号、复核重点、可能加剧、需要结合明细验证 over 审批失控、核心原因、必然导致、证明.",
                *( [language_instruction] if language_instruction else [] ),
            ]
        payload = {
            "task": "Write a compact post-run chart analysis based on executed notebook outputs.",
            "chart_context": prompt_chart_context,
            "constraints": constraints,
        }
        image_urls = [str(item) for item in chart_context.get("image_urls", []) if str(item).strip()]
        if self.source in {"cliproxy", "openai_compatible"}:
            image_urls = []
            prompt_chart_context.pop("image_urls", None)
        system_prompt = load_prompt("postrun_chart_reflection.md")
        if image_urls:
            return self.complete_json_with_images(
                system_prompt=system_prompt,
                user_payload=payload,
                image_urls=image_urls,
                preferred_models=NOTEBOOK_PREFERRED_MODELS,
                cache_stage="postrun_chart_reflection",
            )
        return complete_json_for_stage(
            self,
            system_prompt=system_prompt,
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="postrun_chart_reflection",
        )

    def suggest_notebook_revision_decisions(
        self,
        report_context: dict[str, Any],
        chart_contexts: list[dict[str, Any]],
        allowed_revisions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "task": "Decide whether the executed sales notebook needs one more safe revision pass.",
            "report_context": report_context,
            "chart_contexts": chart_contexts,
            "allowed_revisions": allowed_revisions,
            "constraints": [
                "Return JSON only.",
                "Choose at most two revision_key values.",
                "Only choose from allowed_revisions.",
                "Do not generate code.",
                "Prefer revisions that turn visible risks into more actionable analysis.",
            ],
        }
        return complete_json_for_stage(
            self,
            system_prompt=load_prompt("notebook_revision_decision.md"),
            user_payload=payload,
            preferred_models=NOTEBOOK_PREFERRED_MODELS,
            cache_stage="notebook_revision_decision",
        )


def get_default_llm_client() -> LLMClient:
    settings = get_settings()
    if not settings.llm_enabled:
        return LLMClient(None)

    provider = settings.llm_provider.strip().lower()

    if settings.llm_base_url and settings.llm_api_key:
        return LLMClient(
            LLMClientConfig(
                base_url=settings.llm_base_url.rstrip("/"),
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                source="openai_compatible",
            )
        )

    if provider != "cliproxy" or settings.cliproxy_config_path is None:
        return LLMClient(None)

    cliproxy_config = load_cliproxy_config(settings.cliproxy_config_path)
    if cliproxy_config is None:
        return LLMClient(None)
    if settings.llm_model:
        cliproxy_config.model = settings.llm_model
    return LLMClient(cliproxy_config)
