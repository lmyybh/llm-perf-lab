"""OpenAI-compatible backend implementation."""

import json
import time
from typing import Callable, Optional

import aiohttp

from llmperf.core.models import LLMRequest, LLMResponse, ChatCompletionOutput
from .base import LLMBackend


def _create_client_session(timeout: float = 300.0) -> aiohttp.ClientSession:
    """Create an aiohttp client session with project defaults.

    Args:
        timeout (float): Total request timeout in seconds.

    Returns:
        aiohttp.ClientSession: A configured ``aiohttp.ClientSession`` instance.
    """
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        read_bufsize=10 * 1024**2,
    )


def remove_prefix(text: str, prefix: str) -> str:
    """Remove a prefix from text when it is present.

    Args:
        text (str): Source text to inspect.
        prefix (str): Prefix to remove.

    Returns:
        str: ``text`` without the prefix when it matches, otherwise the
        original string unchanged.
    """
    return text[len(prefix) :] if text.startswith(prefix) else text


class OpenAIChatBackend(LLMBackend):
    """Backend for OpenAI-compatible chat completion APIs.

    Attributes:
        name (str): Backend name used for identification.
        timeout (float): Total request timeout in seconds.
    """

    name: str = "openai-chat"

    def __init__(self, timeout: float = 300.0) -> None:
        """Initialize the backend.

        Args:
            timeout (float): Total request timeout in seconds.
        """
        self.timeout = timeout

    def _validate_request(self, request: LLMRequest) -> None:
        """Validate a request before sending it.

        Args:
            request (LLMRequest): Request to validate.

        Returns:
            None: This method currently performs no validation.
        """
        pass

    def _build_payload(self, request: LLMRequest) -> dict[str, object]:
        """Build the OpenAI-compatible JSON payload.

        Args:
            request (LLMRequest): Internal request model.

        Returns:
            dict[str, object]: JSON payload sent to the backend.
        """
        payload: dict[str, object] = {
            "model": request.model,
            "stream": request.stream,
        }

        payload.update(request.input.model_dump(exclude_none=True))
        payload.update(request.sampling_params.model_dump())

        if "enable_thinking" in request.extra:
            payload["chat_template_kwargs"] = {
                "enable_thinking": request.extra["enable_thinking"]
            }

        return payload

    async def _parse_no_stream_response(
        self, response: aiohttp.ClientResponse, result: LLMResponse
    ) -> None:
        """Parse a non-streaming response payload.

        Args:
            response (aiohttp.ClientResponse): HTTP response object.
            result (LLMResponse): Response model to populate.

        Returns:
            None: The result object is updated in place.
        """
        response_json = await response.json()

        message = response_json.get("choices", [{}])[0].get("message")

        if message:
            result.output = ChatCompletionOutput(message=message)

    async def _parse_stream_response(
        self,
        response: aiohttp.ClientResponse,
        result: LLMResponse,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Parse a streaming response payload.

        Args:
            response (aiohttp.ClientResponse): HTTP response object.
            result (LLMResponse): Response model to populate.
            on_chunk (Optional[Callable[[str], None]]): Optional callback for
                streamed text chunks.

        Returns:
            None: The result object is updated in place.
        """

        last_chunk_time = result.start_time

        async for chunk_bytes in response.content:
            chunk_bytes = chunk_bytes.strip()
            if not chunk_bytes:
                continue

            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
            if chunk != "[DONE]":
                data = json.loads(chunk)

                content = (
                    data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                )

                if content:
                    if on_chunk is not None:
                        on_chunk(content)

                    now = time.perf_counter()
                    if result.ttft == 0.0:
                        result.ttft = now - result.start_time
                    else:
                        result.itls.append(now - last_chunk_time)
                    last_chunk_time = now

    async def send(
        self,
        url: str,
        request: LLMRequest,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """Send one request to an OpenAI-compatible endpoint.

        Args:
            url (str): Target endpoint URL.
            request (LLMRequest): Internal request model.
            on_chunk (Optional[Callable[[str], None]]): Optional callback for
                streamed text chunks.

        Returns:
            LLMResponse: Aggregated backend response.
        """
        self._validate_request(request)

        result = LLMResponse()

        headers = {"Content-Type": "application/json"}
        payload = self._build_payload(request)

        try:
            async with _create_client_session(timeout=self.timeout) as session:

                result.start_time = time.perf_counter()

                async with session.post(
                    url=url, json=payload, headers=headers
                ) as response:
                    result.status_code = response.status
                    response.raise_for_status()

                    if not request.stream:
                        await self._parse_no_stream_response(response, result)
                    else:
                        await self._parse_stream_response(response, result, on_chunk)

        finally:
            result.finish_time = time.perf_counter()
            result.latency = result.finish_time - result.start_time

        return result
