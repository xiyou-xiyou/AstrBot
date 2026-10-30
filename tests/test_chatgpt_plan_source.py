import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from astrbot.core.provider.entities import LLMResponse
from astrbot.core.provider.sources import chatgpt_plan_source, request_retry
from astrbot.core.provider.sources.chatgpt_plan_source import ProviderChatGPTPlan


@pytest.mark.asyncio
async def test_subscription_preserves_native_web_search_tools():
    """Keep search tools on the authenticated streaming Responses request."""
    provider = ProviderChatGPTPlan({"type": "chatgpt_plan", "model": "test"}, {})
    provider._authorize = AsyncMock()
    provider._parse_response = AsyncMock(return_value=LLMResponse("assistant", "found"))
    stream = AsyncMock()
    stream.__aiter__.return_value = [
        SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[])),
    ]
    provider.client.responses.create = AsyncMock(return_value=stream)
    try:
        await provider._query(
            {
                "model": "test",
                "input": [{"role": "user", "content": "search"}],
                "tools": [{"type": "web_search"}],
                "tool_choice": "required",
            },
            None,
        )
        provider._authorize.assert_awaited_once()
        request = provider.client.responses.create.call_args.kwargs
        assert request["tools"] == [{"type": "web_search"}]
        assert request["tool_choice"] == "required"
        assert request["store"] is False and request["stream"] is True
        stream.close.assert_awaited_once()
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_subscription_collects_stream_and_filters_unsupported_parameters():
    provider = ProviderChatGPTPlan({"type": "chatgpt_plan", "model": "test"}, {})
    provider._authorize = AsyncMock()
    provider._parse_response = AsyncMock(return_value=LLMResponse("assistant", "hello"))
    stream = AsyncMock()
    stream.__aiter__.return_value = [
        SimpleNamespace(type="response.output_text.delta", delta="hello"),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[])),
    ]
    provider.client.responses.create = AsyncMock(return_value=stream)
    tools = SimpleNamespace(
        openai_schema=lambda: [{"type": "function", "function": {"name": "lookup"}}]
    )
    result = await provider._query(
        {
            "model": "test",
            "input": [{"role": "system", "content": "instructions"}],
            "temperature": 0.7,
            "max_output_tokens": 100,
            "previous_response_id": "old-response",
            "store": True,
        },
        tools,
    )
    assert result.completion_text == "hello"
    request = provider.client.responses.create.call_args.kwargs
    assert request["store"] is False and request["stream"] is True
    assert request["input"][0]["role"] == "developer"
    assert request["tools"][0]["type"] == "namespace"
    assert "temperature" not in request and "max_output_tokens" not in request
    assert "previous_response_id" not in request
    stream.close.assert_awaited_once()
    await provider.client.close()


@pytest.mark.asyncio
async def test_subscription_restores_output_omitted_from_terminal_event():
    provider = ProviderChatGPTPlan({"type": "chatgpt_plan", "model": "test"}, {})
    provider._authorize = AsyncMock()
    terminal = SimpleNamespace(output=[])
    item = {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
    provider._parse_response = AsyncMock(return_value=LLMResponse("assistant", "ok"))
    stream = AsyncMock()
    stream.__aiter__.return_value = [
        SimpleNamespace(type="response.output_item.done", output_index=0, item=item),
        SimpleNamespace(type="response.completed", response=terminal),
    ]
    provider.client.responses.create = AsyncMock(return_value=stream)
    result = await provider._query({"model": "test", "input": []}, None)
    assert result.completion_text == "ok"
    assert terminal.output == [item]
    stream.close.assert_awaited_once()
    await provider.client.close()


@pytest.mark.asyncio
async def test_subscription_retries_overloaded_error_while_reading_stream(monkeypatch):
    """Retry a stream that fails after its HTTP request succeeds."""
    monkeypatch.setattr(request_retry, "REQUEST_RETRY_WAIT_MIN_S", 0)
    monkeypatch.setattr(request_retry, "REQUEST_RETRY_WAIT_MAX_S", 0)

    class APIError(Exception):
        pass

    provider = ProviderChatGPTPlan({"type": "chatgpt_plan", "model": "test"}, {})
    provider._authorize = AsyncMock()
    provider._parse_response = AsyncMock(return_value=LLMResponse("assistant", "ok"))
    failing_stream = AsyncMock()
    failing_stream.__aiter__.side_effect = APIError(
        "Our servers are currently overloaded. Please try again later."
    )
    successful_stream = AsyncMock()
    successful_stream.__aiter__.return_value = [
        SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[])),
    ]
    provider.client.responses.create = AsyncMock(
        side_effect=[failing_stream, successful_stream]
    )

    response = await provider._query(
        {"model": "test", "input": []}, None, request_max_retries=2
    )

    assert response.completion_text == "ok"
    assert provider.client.responses.create.await_count == 2
    failing_stream.close.assert_awaited_once()
    successful_stream.close.assert_awaited_once()
    await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["response.failed", "response.incomplete", "error"])
async def test_subscription_never_accepts_unsuccessful_terminal_events(status):
    provider = ProviderChatGPTPlan({"type": "chatgpt_plan", "model": "test"}, {})
    provider._authorize = AsyncMock()
    stream = AsyncMock()
    stream.__aiter__.return_value = [SimpleNamespace(type=status)]
    provider.client.responses.create = AsyncMock(return_value=stream)
    with pytest.raises(RuntimeError, match="did not complete"):
        await provider._query({"model": "test", "input": []}, None)
    stream.close.assert_awaited_once()
    await provider.client.close()


@pytest.mark.asyncio
async def test_refresh_is_serialized_across_provider_instances(tmp_path, monkeypatch):
    record = {
        "client_id": "issued-client",
        "scopes": ["chatgpt.tokens.use.direct"],
        "refresh_token": "old-refresh",
        "access_token": "old-access",
        "expires_at": 0,
    }
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps(record))
    requests = []

    async def refresh(request):
        requests.append(request)
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
                "scope": "chatgpt.tokens.use.direct",
            },
        )

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        chatgpt_plan_source.httpx,
        "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(refresh), **kwargs),
    )
    providers = [object.__new__(ProviderChatGPTPlan) for _ in range(2)]
    for provider in providers:
        provider.credential_path = path
        provider.client = SimpleNamespace(api_key="")
    await asyncio.gather(*(provider._authorize() for provider in providers))
    assert len(requests) == 1
    assert all(provider.client.api_key == "new-access" for provider in providers)
    updated = json.loads(path.read_text())
    assert updated["refresh_token"] == "new-refresh"
    assert updated["expires_at"] > time.time()


@pytest.mark.asyncio
async def test_missing_plan_permission_never_uses_access_token(tmp_path):
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"scopes": [], "access_token": "unapproved"}))
    provider = object.__new__(ProviderChatGPTPlan)
    provider.credential_path = path
    provider.client = SimpleNamespace(api_key="")
    with pytest.raises(RuntimeError, match="permission"):
        await provider._authorize()
    assert provider.client.api_key == ""
