"""ChatGPT subscription provider using the official OAuth Responses endpoint."""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
from filelock import FileLock

from astrbot.core.agent.tool import ToolSet
from astrbot.core.exceptions import EmptyModelOutputError
from astrbot.core.provider.entities import LLMResponse
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from ..register import register_provider_adapter
from .openai_responses_source import ProviderOpenAIResponses
from .request_retry import _build_retrying, retry_provider_request


@register_provider_adapter("chatgpt_plan", "ChatGPT subscription OAuth provider")
class ProviderChatGPTPlan(ProviderOpenAIResponses):
    """Use a locally authorized ChatGPT plan and renew rotating credentials."""

    def __init__(self, provider_config: dict, provider_settings: dict) -> None:
        """Initialize the provider without exposing OAuth tokens to the dashboard.

        Args:
            provider_config: Provider and model configuration.
            provider_settings: Global provider settings.
        """
        config = dict(provider_config)
        config["api_base"] = "https://api.openai.com/v1"
        config["key"] = ["local-oauth"]
        config["custom_headers"] = {}
        config["custom_extra_body"] = {}
        super().__init__(config, provider_settings)
        self.credential_path = (
            Path(get_astrbot_data_path()) / "chatgpt" / "credentials.json"
        )

    async def _authorize(self) -> None:
        """Load the grant and serialize refreshes across model instances/processes.

        Raises:
            RuntimeError: If login is missing, expired, or lacks plan permission.
        """
        if not self.credential_path.exists():
            raise RuntimeError("Run python scripts/chatgpt_login.py to sign in first.")
        lock = FileLock(
            str(self.credential_path.with_suffix(".lock")),
            timeout=30,
            thread_local=False,
        )
        # Shield acquisition so cancellation cannot leave an acquired lock behind.
        acquisition = asyncio.create_task(asyncio.to_thread(lock.acquire))
        try:
            await asyncio.shield(acquisition)
        except asyncio.CancelledError:
            await acquisition
            lock.release()
            raise
        try:
            record = json.loads(self.credential_path.read_text(encoding="utf-8"))
            if "chatgpt.tokens.use.direct" not in record.get("scopes", []):
                raise RuntimeError("ChatGPT plan usage permission is missing.")
            if record["expires_at"] <= time.time() + 30:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        "https://auth.openai.com/api/accounts/oauth/token",
                        data={
                            "grant_type": "refresh_token",
                            "client_id": record["client_id"],
                            "refresh_token": record["refresh_token"],
                            "resource": "https://api.openai.com/v1",
                        },
                    )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"ChatGPT token refresh failed (HTTP {response.status_code}). "
                        "Run python scripts/chatgpt_login.py to sign in again."
                    )
                tokens = response.json()
                scopes = tokens.get("scope", " ".join(record["scopes"])).split()
                if "chatgpt.tokens.use.direct" not in scopes:
                    raise RuntimeError("ChatGPT plan permission was revoked.")
                record.update(tokens)
                record["scopes"] = scopes
                record["expires_at"] = time.time() + tokens["expires_in"]
                temporary = self.credential_path.with_suffix(".tmp")
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(record, output)
                temporary.replace(self.credential_path)
            self.client.api_key = record["access_token"]
        finally:
            lock.release()

    async def get_models(self) -> list[str]:
        """Fetch model slugs visible to the authorized subscription account.

        Returns:
            Model identifiers in the account catalog's original order.
        """
        await self._authorize()
        response = await self.client.get("/models", cast_to=dict[str, object])
        return [
            model["slug"]
            for model in response.get("models", [])
            if model.get("visibility") == "list"
        ]

    async def _query(
        self, payloads: dict, tools: ToolSet | None, **kwargs
    ) -> LLMResponse:
        """Collect the required server stream for a non-streaming AstrBot caller.

        Args:
            payloads: Prepared Responses input.
            tools: AstrBot functions available to the model.
            **kwargs: Transport retry and conversation settings.

        Returns:
            The completed response.

        Raises:
            EmptyModelOutputError: If the stream does not complete.
        """
        retrying = _build_retrying(
            "ChatGPT subscription",
            retry_rate_limits=True,
            max_attempts=kwargs.get("request_max_retries"),
        )
        async for attempt in retrying:
            with attempt:
                stream = self._query_stream(
                    payloads, tools, **{**kwargs, "request_max_retries": 1}
                )
                try:
                    async for response in stream:
                        if not response.is_chunk:
                            return response
                finally:
                    await stream.aclose()
        raise EmptyModelOutputError(
            "ChatGPT stream ended without a completed response."
        )

    async def _query_stream(
        self,
        payloads: dict,
        tools: ToolSet | None,
        *,
        request_max_retries: int | None = None,
        conversation_id: str | None = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Send only parameters supported by the subscription Responses route.

        Args:
            payloads: Prepared Responses input and instructions.
            tools: AstrBot functions available to the model.
            request_max_retries: Maximum transport attempts.
            conversation_id: Local conversation identifier.

        Yields:
            Text deltas followed by a completed normalized response.

        Raises:
            RuntimeError: If the provider fails or the response is incomplete.
            EmptyModelOutputError: If the stream ends without completion.
        """
        await self._authorize()
        request = {
            key: value
            for key, value in payloads.items()
            if key
            in {
                "model",
                "input",
                "instructions",
                "reasoning",
                "include",
                "tools",
                "tool_choice",
            }
        }
        # Subscription requests reject explicit system messages.
        request["input"] = [
            {**item, "role": "developer"} if item.get("role") == "system" else item
            for item in request.get("input", [])
        ]
        if tools:
            request["tools"] = [
                {
                    "type": "namespace",
                    "name": "astrbot",
                    "description": "Functions provided by AstrBot",
                    "tools": [
                        {"type": "function", **tool["function"]}
                        for tool in tools.openai_schema()
                    ],
                }
            ]
            request["tool_choice"] = payloads.get("tool_choice", "auto")
        stream = await retry_provider_request(
            "ChatGPT subscription",
            lambda: self.client.responses.create(**request, store=False, stream=True),
            max_attempts=request_max_retries,
        )
        completed_items = {}
        try:
            async for event in stream:
                if event.type == "response.output_item.done":
                    completed_items[event.output_index] = event.item
                elif event.type == "response.output_text.delta":
                    yield LLMResponse(
                        "assistant", completion_text=event.delta, is_chunk=True
                    )
                elif event.type == "response.completed":
                    # This route can omit output from the terminal event; the
                    # completed output-item events remain authoritative.
                    if completed_items and not event.response.output:
                        event.response.output = [
                            completed_items[index] for index in sorted(completed_items)
                        ]
                    yield await self._parse_response(event.response, tools)
                    return
                elif event.type in {"response.failed", "response.incomplete", "error"}:
                    raise RuntimeError(
                        f"ChatGPT inference did not complete: {event.type}"
                    )
        finally:
            await stream.close()
        raise EmptyModelOutputError("ChatGPT stream ended without a completion event.")
