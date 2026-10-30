from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot_plugin_astrbot_enhance_mode import main as main_module
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    PluginConfig,
    WebSearchConfig,
)

from astrbot.core.provider.sources.chatgpt_plan_source import ProviderChatGPTPlan


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_subscription_search_reuses_provider_without_http_fallback(
    monkeypatch, fails
):
    """Use OAuth provider transport and never retry with its placeholder API key."""
    plugin = _build_plugin()
    provider = ProviderChatGPTPlan(
        {"id": "plan", "type": "chatgpt_plan", "model": "test"}, {}
    )
    raw = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Search result",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://example.com/source",
                                "title": "Source",
                            }
                        ],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }
    provider._query = AsyncMock(
        return_value=SimpleNamespace(
            raw_completion=SimpleNamespace(model_dump=lambda **kwargs: raw)
        )
    )
    if fails:
        provider._query.side_effect = RuntimeError("OAuth refresh failed")
    plugin._resolve_web_search_provider = lambda cfg: provider
    plugin._build_web_search_http_requests = Mock(
        side_effect=AssertionError("Must not extract placeholder key")
    )
    direct_http = Mock(side_effect=AssertionError("Must not bypass provider auth"))
    monkeypatch.setattr(main_module.aiohttp, "ClientSession", direct_http)
    try:
        result = await plugin._run_web_search(
            _DummyEvent(),
            "search",
            PluginConfig(web_search=WebSearchConfig(enable=True)),
        )
        assert result["ok"] is not fails
        if fails:
            assert "OAuth refresh failed" in result["error"]
        else:
            assert result["content"] == "Search result"
            assert result["sources"][0]["url"] == "https://example.com/source"
            assert result["usage"]["total_tokens"] == 5
        payload = provider._query.call_args.args[0]
        assert payload["input"] == [{"role": "user", "content": "search"}]
        assert payload["tools"] == [{"type": "web_search"}]
        assert payload["tool_choice"] == "required"
        direct_http.assert_not_called()
        plugin._build_web_search_http_requests.assert_not_called()
    finally:
        await provider.client.close()


class _DummyEvent:
    def __init__(self, origin: str = "origin-test") -> None:
        self.unified_msg_origin = origin


class _FakeProvider:
    def __init__(self) -> None:
        self.id = "provider-test"
        self.provider_config = {
            "api_base": "https://example.com/v1",
            "model": "grok-4-1-fast-reasoning",
            "key": ["sk-test"],
            "custom_headers": {"X-Test": "1", "Authorization": "override"},
            "custom_extra_body": {
                "tools": [{"type": "web_search"}],
                "tool_choice": "auto",
                "messages": [{"role": "user", "content": "bad"}],
            },
        }

    def get_model(self) -> str:
        return "grok-4-1-fast-reasoning"

    def get_current_key(self) -> str:
        return "sk-test"

    def get_keys(self) -> list[str]:
        return ["sk-test"]


class _FakeResponse:
    def __init__(
        self, status: int, text: str, headers: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self._text = text
        self.headers = headers or {"Content-Type": "application/json"}

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_post_kwargs: dict[str, Any] = {}

    def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_post_kwargs = {"args": args, "kwargs": kwargs}
        return self._response

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def _build_plugin() -> Main:
    plugin = Main.__new__(Main)
    return plugin


def test_build_web_search_http_requests_uses_provider_config() -> None:
    plugin = _build_plugin()
    provider = _FakeProvider()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="system prompt",
        )
    )

    request_specs, provider_label = plugin._build_web_search_http_requests(
        provider=provider,
        query="latest world model vla",
        cfg=cfg,
    )

    assert len(request_specs) == 2
    assert request_specs[0]["mode"] == "responses"
    assert request_specs[0]["url"] == "https://example.com/v1/responses"
    assert request_specs[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert request_specs[0]["headers"]["X-Test"] == "1"
    assert request_specs[0]["body"]["model"] == "grok-4-1-fast-reasoning"
    assert request_specs[0]["body"]["input"] == "latest world model vla"
    assert request_specs[0]["body"]["tools"] == [{"type": "web_search"}]
    assert request_specs[0]["body"]["tool_choice"] == "auto"

    assert request_specs[1]["mode"] == "chat_completions"
    assert request_specs[1]["url"] == "https://example.com/v1/chat/completions"
    assert request_specs[1]["body"]["messages"][0]["content"] == "system prompt"
    assert (
        request_specs[1]["body"]["messages"][1]["content"] == "latest world model vla"
    )
    assert provider_label == "provider-test"


@pytest.mark.asyncio
async def test_run_web_search_success_with_json_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="sys",
            timeout_sec=10,
        )
    )

    plugin._resolve_web_search_provider = lambda _cfg: object()
    plugin._build_web_search_http_requests = lambda provider, query, cfg: (
        [
            {
                "mode": "responses",
                "url": "https://example.com/v1/responses",
                "headers": {
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                },
                "body": {"model": "grok", "input": query},
            }
        ],
        "provider-test",
    )

    raw_json = (
        '{"output":[{"type":"message","content":[{"type":"output_text","text":"DreamZero is a recent VLA direction",'
        '"annotations":[{"type":"url_citation","url":"https://example.com/dreamzero","title":"DreamZero"}]}]}],'
        '"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}'
    )
    fake_response = _FakeResponse(status=200, text=raw_json)
    fake_session = _FakeSession(fake_response)
    monkeypatch.setattr(
        main_module.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: fake_session,
    )

    result = await plugin._run_web_search(event, "latest world model vla", cfg)
    assert result["ok"] is True
    assert "DreamZero" in str(result["content"])
    assert isinstance(result["sources"], list)
    assert result["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_run_web_search_http_error_returns_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="sys",
            timeout_sec=10,
        )
    )

    plugin._resolve_web_search_provider = lambda _cfg: object()
    plugin._build_web_search_http_requests = lambda provider, query, cfg: (
        [
            {
                "mode": "responses",
                "url": "https://example.com/v1/responses",
                "headers": {
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                },
                "body": {"model": "grok", "input": query},
            }
        ],
        "provider-test",
    )

    fake_response = _FakeResponse(status=422, text='{"error":"openai_error"}')
    fake_session = _FakeSession(fake_response)
    monkeypatch.setattr(
        main_module.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: fake_session,
    )

    result = await plugin._run_web_search(event, "latest world model vla", cfg)
    assert result["ok"] is False
    assert "HTTP 422" in str(result["error"])
    assert "openai_error" in str(result["raw"])
