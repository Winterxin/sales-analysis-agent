from __future__ import annotations

import json
from pathlib import Path
import time

import httpx
import pytest

from app.services.llm_client import (
    LLMClient,
    LLMCompletionUnavailable,
    LLMCompletionTimedOut,
    LLMClientConfig,
    build_llm_cache_key,
    _message_content_for_user_payload,
    extract_nonstream_text,
    extract_stream_text,
    get_default_llm_client,
    load_cliproxy_config,
    parse_json_text,
)
from app.services.llm_trace_utils import describe_llm_error
from app.services.llm_trace_utils import build_llm_stage_trace
from app.schemas.notebook_narrative import NotebookNarrative
from app.schemas.notebook_outline import NotebookOutline, NotebookSection
from app.schemas.report import AnalysisReport, ModuleReport
from app.schemas.schema_mapping import SchemaMapping


def test_load_cliproxy_config_reads_local_proxy_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'host: "127.0.0.1"',
                "port: 8317",
                "api-keys:",
                '  - "test-client-key"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_cliproxy_config(config_path)

    assert config is not None
    assert config.base_url == "http://127.0.0.1:8317/v1"
    assert config.api_key == "test-client-key"


def test_load_cliproxy_config_does_not_discover_neighboring_folders(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing" / "config.yaml"
    actual_config = tmp_path / "neighbor" / "config.yaml"
    actual_config.parent.mkdir()
    actual_config.write_text(
        "\n".join(
            [
                'host: "127.0.0.1"',
                "port: 8318",
                "api-keys:",
                '  - "neighbor-client-key"',
            ]
        ),
        encoding="utf-8",
    )

    assert load_cliproxy_config(missing_config) is None


def test_default_llm_client_is_disabled_without_complete_openai_compatible_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SALES_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SALES_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("SALES_AGENT_CLIPROXY_CONFIG_PATH", raising=False)

    client = get_default_llm_client()

    assert not client.enabled
    assert client.source == "disabled"


def test_default_llm_client_uses_openai_compatible_env_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SALES_AGENT_LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setenv("SALES_AGENT_LLM_API_KEY", "test-openai-compatible-key")
    monkeypatch.setenv("SALES_AGENT_LLM_MODEL", "example-model")

    client = get_default_llm_client()

    assert client.enabled
    assert client.source == "openai_compatible"
    assert client.configured_model == "example-model"


def test_default_llm_client_can_still_use_cliproxy_when_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'host: "127.0.0.1"',
                "port: 8317",
                "api-keys:",
                '  - "test-client-key"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SALES_AGENT_LLM_PROVIDER", "cliproxy")
    monkeypatch.setenv("SALES_AGENT_CLIPROXY_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("SALES_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SALES_AGENT_LLM_API_KEY", raising=False)

    client = get_default_llm_client()

    assert client.enabled
    assert client.source == "cliproxy"
    assert client.configured_model is None


class FakeModelListingClient(LLMClient):
    def list_models(self) -> list[str]:
        return ["gpt-5.4-mini", "gpt-5.4"]


def test_llm_client_falls_back_when_preferred_model_is_missing() -> None:
    client = FakeModelListingClient(
        LLMClientConfig(
            base_url="http://127.0.0.1:8317/v1",
            api_key="test-client-key",
            model="gpt-5-mini",
        )
    )

    assert client.resolve_model() == "gpt-5.4-mini"


def test_llm_client_prefers_requested_model_when_available() -> None:
    client = FakeModelListingClient(
        LLMClientConfig(
            base_url="http://127.0.0.1:8317/v1",
            api_key="test-client-key",
            model="gpt-5-mini",
        )
    )

    assert client.resolve_model(["gpt-5.4"]) == "gpt-5.4"


class NullSchemaMappingLLMClient(LLMClient):
    def complete_json(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return {
            "dataset_type": "sales_transaction",
            "field_mapping": {
                "Row ID": None,
                "Order ID": "order_id",
                "Sales": "sales_amount",
                "Profit": "profit",
            },
            "confidence": 0.82,
            "missing_required_fields": [None, "quantity"],
            "uncertain_fields": [None, "Row ID"],
        }


def test_suggest_schema_mapping_ignores_null_field_mapping_values() -> None:
    client = NullSchemaMappingLLMClient(
        LLMClientConfig(
            base_url="http://127.0.0.1:8317/v1",
            api_key="test-client-key",
            source="test",
        )
    )
    rule_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Order ID": "order_id"},
        confidence=0.7,
    )

    mapping = client.suggest_schema_mapping(["Row ID", "Order ID", "Sales", "Profit"], rule_mapping)

    assert mapping.field_mapping == {
        "Order ID": "order_id",
        "Sales": "sales_amount",
        "Profit": "profit",
    }
    assert mapping.missing_required_fields == ["quantity"]
    assert mapping.uncertain_fields == ["Row ID"]


class FakeCliproxyModelListingClient(LLMClient):
    def list_models(self) -> list[str]:
        return ["gpt-5.3-codex", "gpt-5.4-mini", "gpt-5.4"]


def test_cliproxy_defaults_to_notebook_preferred_model_instead_of_first_model() -> None:
    client = FakeCliproxyModelListingClient(
        LLMClientConfig(
            base_url="http://127.0.0.1:8317/v1",
            api_key="test-client-key",
            source="cliproxy",
        )
    )

    assert client.resolve_model() == "gpt-5.4-mini"


class BrokenCliproxyModelListingClient(LLMClient):
    def list_models(self) -> list[str]:
        request = httpx.Request("GET", "http://127.0.0.1:8317/v1/models")
        response = httpx.Response(502, request=request, text="")
        raise httpx.HTTPStatusError("bad gateway", request=request, response=response)


def test_cliproxy_resolve_model_falls_back_to_fixed_default_when_models_endpoint_fails() -> None:
    client = BrokenCliproxyModelListingClient(
        LLMClientConfig(
            base_url="http://127.0.0.1:8317/v1",
            api_key="test-client-key",
            source="cliproxy",
        )
    )

    assert client.resolve_model() == "gpt-5.4-mini"


def test_extract_stream_text_reassembles_delta_content() -> None:
    text = extract_stream_text(
        [
            'data: {"choices":[{"delta":{"content":"{\\"ok\\""}}]}',
            'data: {"choices":[{"delta":{"content":": true}"}}]}',
            "data: [DONE]",
        ]
    )

    assert text == '{"ok": true}'


def test_extract_nonstream_text_reassembles_message_content() -> None:
    text = extract_nonstream_text(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": '{"ok":'},
                            {"type": "text", "text": ' true}'},
                        ]
                    }
                }
            ]
        }
    )

    assert text == '{"ok": true}'


def test_parse_json_text_handles_code_fences_and_trailing_commas() -> None:
    payload = parse_json_text(
        """```json
        {
          "sections": [
            {"section_id": "sales_trends",},
          ],
        }
        ```"""
    )

    assert payload == {"sections": [{"section_id": "sales_trends"}]}


def test_message_content_for_user_payload_supports_image_inputs() -> None:
    content = _message_content_for_user_payload(
        {"task": "reflect chart"},
        image_urls=["data:image/png;base64,abc123"],
    )

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


class RetryableLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
            )
        )
        self.calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.calls += 1
        if self.calls < 3:
            request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
            response = httpx.Response(
                500,
                request=request,
                json={"error": {"message": "upstream unavailable"}},
            )
            raise httpx.HTTPStatusError("boom", request=request, response=response)
        return {"ok": True}


def test_llm_client_retries_retryable_failures_before_succeeding() -> None:
    client = RetryableLLMClient()

    result = client.complete_json("system", {"task": "retry"})

    assert result == {"ok": True}
    assert client.calls == 3


class OpenAICompatibleTextOnlyPostrunClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="https://llm.example.test/v1",
                api_key="test-client-key",
                model="example-model",
                source="openai_compatible",
            )
        )
        self.used_text_completion = False

    def complete_json_with_images(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("postrun reflection should use text-only completion")

    def complete_json(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.used_text_completion = True
        return {"reflection_markdown": "折扣 30% 区间亏损率 75%。这说明利润风险集中。后续应复盘高折扣订单。"}


def test_openai_compatible_postrun_reflection_uses_text_only_context() -> None:
    client = OpenAICompatibleTextOnlyPostrunClient()

    result = client.suggest_postrun_chart_reflection(
        {
            "chart_title": "折扣与利润",
            "image_urls": ["data:image/png;base64,abc123"],
            "table_preview": "loss_rate=75%",
        },
        "",
    )

    assert client.used_text_completion
    assert "75%" in result["reflection_markdown"]


class CompatFallbackLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
            )
        )
        self.primary_calls = 0
        self.compat_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
        response = httpx.Response(
            400,
            request=request,
            json={"error": {"message": "streaming json mode unsupported"}},
        )
        raise httpx.HTTPStatusError("bad request", request=request, response=response)

    def _complete_json_compat_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.compat_calls += 1
        return {"ok": True}


def test_llm_client_falls_back_to_compat_mode_on_bad_request() -> None:
    client = CompatFallbackLLMClient()

    result = client.complete_json("system", {"task": "compat"})

    assert result == {"ok": True}
    assert client.primary_calls == 1
    assert client.compat_calls == 1


class CliproxyServerErrorCompatClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0
        self.compat_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
        response = httpx.Response(
            500,
            request=request,
            json={"error": {"message": "proxy upstream crashed"}},
        )
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    def _complete_json_compat_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.compat_calls += 1
        return {"ok": True}


def test_llm_client_falls_back_to_compat_mode_on_cliproxy_server_error() -> None:
    client = CliproxyServerErrorCompatClient()

    result = client.complete_json("system", {"task": "compat-500"})

    assert result == {"ok": True}
    assert client.primary_calls == 1
    assert client.compat_calls == 1


class JsonDecodeCompatClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0
        self.compat_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        raise ValueError("stream response did not contain parseable JSON")

    def _complete_json_compat_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.compat_calls += 1
        return {"ok": True}


def test_llm_client_falls_back_to_compat_mode_on_stream_parse_failure() -> None:
    client = JsonDecodeCompatClient()

    result = client.complete_json("system", {"task": "parse-fallback"})

    assert result == {"ok": True}
    assert client.primary_calls == 1
    assert client.compat_calls == 1


class CliproxyGenericEofClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0
        self.compat_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        if self.primary_calls == 1:
            request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
            response = httpx.Response(
                500,
                request=request,
                json={"error": {"message": "EOF"}},
            )
            raise httpx.HTTPStatusError("server error", request=request, response=response)
        return {"ok": True}

    def _complete_json_compat_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.compat_calls += 1
        request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
        response = httpx.Response(
            500,
            request=request,
            json={"error": {"message": "EOF"}},
        )
        raise httpx.HTTPStatusError("server error", request=request, response=response)


def test_llm_client_does_not_fail_fast_after_generic_cliproxy_eof() -> None:
    client = CliproxyGenericEofClient()

    try:
        client.complete_json("system", {"task": "generic-eof"})
    except httpx.HTTPStatusError:
        pass

    assert client.completion_unavailable_reason is None

    result = client.complete_json("system", {"task": "second-call"})

    assert result == {"ok": True}
    assert client.primary_calls == 2
    assert client.compat_calls == 1


class CliproxyBackendApiEofClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0
        self.compat_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        if self.primary_calls == 1:
            request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
            response = httpx.Response(
                500,
                request=request,
                json={
                    "error": {
                        "message": "Post https://chatgpt.com/backend-api/codex/responses: EOF"
                    }
                },
            )
            raise httpx.HTTPStatusError("server error", request=request, response=response)
        return {"ok": True}

    def _complete_json_compat_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.compat_calls += 1
        request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
        response = httpx.Response(
            500,
            request=request,
            json={
                "error": {
                    "message": "Post https://chatgpt.com/backend-api/codex/responses: EOF"
                }
            },
        )
        raise httpx.HTTPStatusError("server error", request=request, response=response)


def test_llm_client_does_not_poison_run_after_cliproxy_backend_api_eof() -> None:
    client = CliproxyBackendApiEofClient()

    with pytest.raises(httpx.HTTPStatusError):
        client.complete_json("system", {"task": "backend-api-eof"})

    assert client.completion_unavailable_reason is None

    result = client.complete_json("system", {"task": "next-stage"})

    assert result == {"ok": True}
    assert client.primary_calls == 2
    assert client.compat_calls == 1


class CliproxyReadTimeoutClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        if self.primary_calls <= 3:
            request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
            raise httpx.ReadTimeout("timed out", request=request)
        return {"ok": True}


def test_llm_client_keeps_cliproxy_available_after_repeated_read_timeouts() -> None:
    client = CliproxyReadTimeoutClient()

    with pytest.raises(httpx.ReadTimeout):
        client.complete_json("system", {"task": "slow-section"})

    assert client.primary_calls == 3
    assert client.completion_unavailable_reason is None

    result = client.complete_json("system", {"task": "next-stage"})

    assert result == {"ok": True}
    assert client.primary_calls == 4


class CliproxyStageIsolatedReadTimeoutClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        if self.primary_calls <= 3:
            request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
            raise httpx.ReadTimeout("slow section timed out", request=request)
        return {"ok": True, "stage": user_payload["task"]}


def test_llm_client_keeps_later_cliproxy_stages_available_after_read_timeouts() -> None:
    client = CliproxyStageIsolatedReadTimeoutClient()

    with pytest.raises(httpx.ReadTimeout):
        client.complete_json("system", {"task": "slow-section"})

    assert client.primary_calls == 3
    assert client.completion_unavailable_reason is None

    result = client.complete_json("system", {"task": "next-stage"})

    assert result == {"ok": True, "stage": "next-stage"}
    assert client.primary_calls == 4


class HangingCliproxyClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0

    def _completion_wall_timeout(self) -> float:
        return 0.01

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        time.sleep(1.0)
        return {"ok": True}


def test_llm_client_keeps_cliproxy_available_after_completion_wall_timeout() -> None:
    client = HangingCliproxyClient()
    started_at = time.monotonic()

    with pytest.raises(LLMCompletionTimedOut):
        client.complete_json("system", {"task": "hanging-stream"})

    elapsed = time.monotonic() - started_at
    assert elapsed < 0.5
    assert client.primary_calls == 1
    assert client.completion_unavailable_reason is None

    with pytest.raises(LLMCompletionTimedOut):
        client.complete_json("system", {"task": "next-stage"})

    assert client.primary_calls == 2


class CliproxyCoolingDownBackoffClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0
        self.sleeps: list[float] = []

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        if self.primary_calls < 3:
            request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
            response = httpx.Response(
                429,
                request=request,
                json={"error": {"message": "All credentials for model gpt-5.4-mini are cooling down"}},
            )
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return {"ok": True}

    def _sleep_before_retry(self, delay_seconds: float) -> None:
        self.sleeps.append(delay_seconds)


def test_llm_client_retries_with_backoff_when_cliproxy_model_is_cooling_down() -> None:
    client = CliproxyCoolingDownBackoffClient()

    result = client.complete_json(
        "system",
        {"task": "cooling-down"},
        preferred_models=["gpt-5.4-mini"],
    )

    assert result == {"ok": True}
    assert client.primary_calls == 3
    assert client.sleeps == [2.0, 4.0]


def test_llm_completion_metrics_history_and_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SALES_AGENT_LLM_CACHE_ENABLED", "false")
    client = CacheableLLMClient()

    first_snapshot = client.snapshot_completion_metrics()
    client.complete_json("system", {"task": "first"}, cache_stage="notebook_content")
    middle_snapshot = client.snapshot_completion_metrics()
    client.complete_json("system", {"task": "second"})

    first_window = client.collect_completion_metrics_since(first_snapshot)
    second_window = client.collect_completion_metrics_since(middle_snapshot)

    assert len(first_window) == 2
    assert len(second_window) == 1
    assert first_window[0]["stage"] == "notebook_content"
    assert first_window[0]["cache_status"] in {"miss", "disabled"}
    assert first_window[1]["cache_status"] == "bypassed"
    assert first_window[1]["stage"] is None
    assert client.completion_metrics_history == first_window
    assert client.last_completion_metrics == first_window[-1]


class CompletionMetricsClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        return {"ok": True, "stage": user_payload["task"]}


def test_llm_stage_trace_includes_completion_metrics_after_success() -> None:
    client = CompletionMetricsClient()

    result = client.complete_json("system prompt", {"task": "metrics-stage"})
    trace = build_llm_stage_trace(
        stage="notebook_content",
        llm_client=client,
        status="llm_applied",
        reason="LLM content applied.",
        attempted=True,
        applied=True,
    )

    assert result == {"ok": True, "stage": "metrics-stage"}
    assert trace.elapsed_ms is not None
    assert trace.elapsed_ms >= 0
    assert trace.prompt_chars is not None
    assert trace.prompt_chars >= len("system prompt")
    assert trace.response_chars is not None
    assert trace.response_chars > 0
    assert trace.attempt_count == 1
    assert trace.error_type is None
    assert trace.fallback_type is None


def test_llm_stage_trace_includes_fallback_metrics_after_error() -> None:
    client = CliproxyReadTimeoutClient()

    with pytest.raises(httpx.ReadTimeout):
        client.complete_json("system prompt", {"task": "slow-stage"})
    trace = build_llm_stage_trace(
        stage="notebook_narrative",
        llm_client=client,
        status="fallback_on_error",
        reason="LLM stage failed.",
        attempted=True,
        applied=False,
    )

    assert trace.elapsed_ms is not None
    assert trace.prompt_chars is not None
    assert trace.attempt_count == 3
    assert trace.fallback_type == "error"
    assert trace.error_type == "ReadTimeout"


class CliproxyExtendedCoolingDownClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0
        self.sleeps: list[float] = []

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        if self.primary_calls < 5:
            request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
            response = httpx.Response(
                429,
                request=request,
                json={"error": {"message": "All credentials for model gpt-5.4-mini are cooling down"}},
            )
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return {"ok": True}

    def _sleep_before_retry(self, delay_seconds: float) -> None:
        self.sleeps.append(delay_seconds)


def test_llm_client_allows_extra_attempts_for_extended_cliproxy_cooling_down() -> None:
    client = CliproxyExtendedCoolingDownClient()

    result = client.complete_json(
        "system",
        {"task": "extended-cooling"},
        preferred_models=["gpt-5.4-mini"],
    )

    assert result == {"ok": True}
    assert client.primary_calls == 5
    assert client.sleeps == [2.0, 4.0, 8.0, 8.0]


class AlwaysReadTimeoutClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
        raise httpx.ReadTimeout("timed out", request=request)


def test_llm_client_respects_configured_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SALES_AGENT_LLM_MAX_ATTEMPTS", "1")
    client = AlwaysReadTimeoutClient()

    with pytest.raises(httpx.ReadTimeout):
        client.complete_json("system", {"task": "single-attempt"})

    assert client.primary_calls == 1
    assert client.last_completion_metrics["attempt_count"] == 1


def test_llm_cache_key_is_stable_for_payload_order_and_changes_with_version() -> None:
    first = build_llm_cache_key(
        stage="notebook_content",
        model="example-model",
        source="openai_compatible",
        prompt_version="g4_llm_cache_v1",
        system_prompt="system",
        user_payload={"b": 2, "a": {"z": 1, "y": 0}},
    )
    second = build_llm_cache_key(
        stage="notebook_content",
        model="example-model",
        source="openai_compatible",
        prompt_version="g4_llm_cache_v1",
        system_prompt="system",
        user_payload={"a": {"y": 0, "z": 1}, "b": 2},
    )
    changed_version = build_llm_cache_key(
        stage="notebook_content",
        model="example-model",
        source="openai_compatible",
        prompt_version="g4_llm_cache_v2",
        system_prompt="system",
        user_payload={"b": 2, "a": {"z": 1, "y": 0}},
    )

    assert first == second
    assert first != changed_version


class CacheableLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        return {"ok": True, "call": self.primary_calls, "payload": user_payload}


def test_llm_cache_miss_then_hit_reuses_response_without_remote_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SALES_AGENT_LLM_CACHE_ENABLED", "true")
    monkeypatch.setenv("SALES_AGENT_LLM_CACHE_DIR", str(tmp_path / "llm_cache"))
    client = CacheableLLMClient()

    first = client.complete_json("system", {"b": 2, "a": 1}, cache_stage="notebook_content")
    second = client.complete_json("system", {"a": 1, "b": 2}, cache_stage="notebook_content")

    assert first == second
    assert client.primary_calls == 1
    assert client.last_completion_metrics["cache_status"] == "hit"
    assert client.last_completion_metrics["saved_ms"] > 0
    assert [item["cache_status"] for item in client.completion_metrics_history] == ["miss", "hit"]
    assert client.completion_metrics_history[0]["stage"] == "notebook_content"
    assert client.completion_metrics_history[1]["stage"] == "notebook_content"
    assert len(list((tmp_path / "llm_cache").glob("*.json"))) == 1


def test_llm_stage_trace_and_summary_include_cache_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SALES_AGENT_LLM_CACHE_ENABLED", "true")
    monkeypatch.setenv("SALES_AGENT_LLM_CACHE_DIR", str(tmp_path / "llm_cache"))
    client = CacheableLLMClient()

    client.complete_json("system", {"task": "summary"}, cache_stage="report_summary")
    miss_trace = build_llm_stage_trace(
        stage="report_summary",
        llm_client=client,
        status="llm_applied",
        reason="summary generated",
        attempted=True,
        applied=True,
    )
    client.complete_json("system", {"task": "summary"}, cache_stage="report_summary")
    hit_trace = build_llm_stage_trace(
        stage="report_summary",
        llm_client=client,
        status="llm_applied",
        reason="summary generated",
        attempted=True,
        applied=True,
    )

    from app.services.llm_trace_utils import build_llm_trace_summary

    summary = build_llm_trace_summary(
        {
            "report_summary_first": miss_trace.model_dump(),
            "report_summary_second": hit_trace.model_dump(),
            "chart_reflection_bypassed": {
                "stage": "postrun_chart_reflection",
                "cache_status": "bypassed",
                "remote_elapsed_ms": 12.5,
                "saved_ms": 0,
            },
        }
    )

    assert miss_trace.cache_status == "miss"
    assert miss_trace.remote_elapsed_ms is not None
    assert hit_trace.cache_status == "hit"
    assert hit_trace.saved_ms > 0
    assert summary["remote_call_count"] == 2
    assert summary["cache_hit_count"] == 1
    assert summary["cache_miss_count"] == 1
    assert summary["cache_bypassed_remote_count"] == 1
    assert summary["evidence_pack_enabled"] is True
    assert "evidence_pack_stage_count" in summary
    assert summary["total_prompt_chars"] > 0
    assert summary["remote_prompt_chars"] > 0
    assert summary["total_llm_remote_elapsed_ms"] > 0
    assert summary["total_llm_saved_ms_by_cache"] > 0


class CapturingPayloadLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.user_payload: dict[str, object] | None = None

    def complete_json(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.user_payload = user_payload
        return {"sections": [], "suggested_followups": []}


def test_notebook_narrative_prompt_uses_llm_evidence_pack() -> None:
    client = CapturingPayloadLLMClient()
    outline = NotebookOutline(
        title="Sales Notebook",
        sections=[NotebookSection(section_id="sales_trends", title="Sales Trends", purpose="Explain trend.")],
    )
    report = AnalysisReport(
        task_id="task-1",
        dataset_type="sales_transaction",
        module_count=1,
        summary=["summary"],
        modules=[
            ModuleReport(
                module_id="sales_trend_analysis",
                title="Sales Trend",
                chart_type="line",
                summary_metrics={"total_sales": 1000},
                tables={"daily_totals": [{"day": index, "sales": index * 10} for index in range(25)]},
                chart_payload={"large": "x" * 10000},
                findings=[f"finding {index}" for index in range(20)],
            )
        ],
    )
    schema_mapping = SchemaMapping(
        dataset_type="sales_transaction",
        field_mapping={"Sales": "sales_amount"},
        confidence=0.9,
        missing_required_fields=[],
        uncertain_fields=[],
    )
    fallback = NotebookNarrative(sections=[], suggested_followups=[])

    client.suggest_notebook_narrative(outline, report, schema_mapping, fallback)

    assert client.user_payload is not None
    assert "report" not in client.user_payload
    evidence_pack = client.user_payload["evidence_pack"]
    module = evidence_pack["module_evidence"][0]
    serialized = json.dumps(evidence_pack, ensure_ascii=False)
    assert "chart_payload" not in serialized
    assert len(module["key_tables"]["daily_totals"]) == 3
    assert len(module["top_findings"]) == 5


def test_report_summary_prompt_uses_llm_evidence_pack_instead_of_full_report() -> None:
    client = CapturingPayloadLLMClient()
    report = AnalysisReport(
        task_id="tatest-api-key",
        dataset_type="sales_transaction",
        module_count=2,
        summary=["deterministic summary"],
        modules=[
            ModuleReport(
                module_id="discount_profit_analysis",
                title="Discount",
                chart_type="dual_axis_bar_line",
                summary_metrics={"negative_profit_rate": 0.1872},
                tables={
                    "discount_cap_what_if": [
                        {"high_risk_bucket": "30%+", "estimated_profit_delta": 21200.0}
                    ]
                },
                chart_payload={"raw": "must-not-leak"},
                findings=["高折扣区间利润质量偏弱。"],
            ),
            ModuleReport(
                module_id="loss_risk_modeling",
                title="Modeling",
                chart_type="model",
                summary_metrics={"best_model": "LogisticRegression", "best_recall": 0.91},
                tables={
                    "threshold_analysis": [{"threshold": 0.4, "recall": 0.95}],
                    "feature_importance_grouped": [{"feature_group": "discount", "total_importance": 0.42}],
                },
            ),
        ],
    )
    evidence_pack = {
        "discount_evidence": {
            "what_if": [{"high_risk_bucket": "30%+", "estimated_profit_delta": 21200.0}]
        },
        "modeling_evidence": {
            "best_model": "LogisticRegression",
            "threshold_analysis": [{"threshold": 0.4, "recall": 0.95}],
        },
    }

    client.summarize_report(report, evidence_pack=evidence_pack)

    assert client.user_payload is not None
    assert "report" not in client.user_payload
    assert client.user_payload["evidence_pack"] == evidence_pack
    serialized = json.dumps(client.user_payload, ensure_ascii=False)
    assert "chart_payload" not in serialized
    assert "estimated_profit_delta" in serialized
    assert "LogisticRegression" in serialized


class CliproxyUnavailableClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(
            LLMClientConfig(
                base_url="http://127.0.0.1:8317/v1",
                api_key="test-client-key",
                model="gpt-5.4-mini",
                source="cliproxy",
            )
        )
        self.primary_calls = 0
        self.compat_calls = 0

    def _complete_json_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.primary_calls += 1
        request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
        response = httpx.Response(
            500,
            request=request,
            json={
                "error": {
                    "message": "Post https://chatgpt.com/backend-api/codex/responses: connectex timeout"
                }
            },
        )
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    def _complete_json_compat_once(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        preferred_models: list[str] | None = None,
    ) -> dict[str, object]:
        self.compat_calls += 1
        request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
        response = httpx.Response(
            500,
            request=request,
            json={
                "error": {
                    "message": "Post https://chatgpt.com/backend-api/codex/responses: connectex timeout"
                }
            },
        )
        raise httpx.HTTPStatusError("server error", request=request, response=response)


def test_llm_client_marks_cliproxy_completion_unavailable_after_backend_outage() -> None:
    client = CliproxyUnavailableClient()

    try:
        client.complete_json("system", {"task": "outage"})
    except httpx.HTTPStatusError:
        pass

    assert client.completion_unavailable_reason is not None
    assert "connectex timeout" in client.completion_unavailable_reason

    try:
        client.complete_json("system", {"task": "second-call"})
    except LLMCompletionUnavailable as exc:
        assert "connectex timeout" in str(exc)
    else:  # pragma: no cover - makes assertion failure easier to read
        raise AssertionError("Expected the second completion call to fail fast")

    assert client.primary_calls == 1
    assert client.compat_calls == 0


def test_describe_llm_error_includes_http_response_body() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8317/v1/chat/completions")
    response = httpx.Response(
        500,
        request=request,
        json={"error": {"message": "upstream connectex timeout"}},
    )
    exc = httpx.HTTPStatusError("server error", request=request, response=response)

    description = describe_llm_error(exc)

    assert "500" in description
    assert "upstream connectex timeout" in description
