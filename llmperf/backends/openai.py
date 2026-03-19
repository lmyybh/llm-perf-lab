"""OpenAI-compatible backend implementation."""

import json
import time
import aiohttp
from typing import Callable, Optional

from llmperf.core.models import LLMRequest, LLMResponse
from llmperf.backends.base import LLMBackend


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


async def _read_error_response(response: aiohttp.ClientResponse) -> str:
    """Read and normalize an HTTP error response body.

    Args:
        response (aiohttp.ClientResponse): HTTP response with an error status.

    Returns:
        str: A readable error message containing status and response body.
    """
    try:
        body = await response.text()
    except Exception as exc:
        return f"HTTP {response.status}: failed to read error body: {exc}"

    body = body.strip()
    if not body:
        return f"HTTP {response.status}: empty error response"

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return f"HTTP {response.status}: {body}"

    return f"HTTP {response.status}: {json.dumps(parsed, ensure_ascii=False)}"


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
        payload: dict[str, object] = {"stream": request.stream}
        if request.model is not None:
            payload["model"] = request.model

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

        choice = response_json.get("choices", [{}])[0]
        usage = response_json.get("usage", {})

        # TODO: 做显式校验
        result.output = result.output.model_copy(update=choice.get("message", {}))

        for key, value in usage.items():
            if key in LLMResponse.model_fields:
                setattr(result, key, value)

        result.finish_reason = choice.get("finish_reason")

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
                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                role = delta.get("role")
                if role is not None and result.output.role is None:
                    result.output.role = role

                # TODO: 处理 tool_calls
                reasoning_content = delta.get("reasoning_content")
                content = delta.get("content")

                text = reasoning_content or content or ""
                if text:
                    if on_chunk is not None:
                        on_chunk(text)

                    result.completion_tokens += 1

                    if reasoning_content:
                        if result.output.reasoning_content is None:
                            result.output.reasoning_content = ""
                        result.output.reasoning_content += text
                    else:
                        if result.output.content is None:
                            result.output.content = ""
                        result.output.content += text

                    now = time.perf_counter()
                    if result.ttft == 0.0:
                        result.ttft = now - result.start_time
                    else:
                        result.itls.append(now - last_chunk_time)
                    last_chunk_time = now
                else:
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        result.finish_reason = finish_reason

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

        result = LLMResponse(model=request.model, stream=request.stream)

        headers = {"Content-Type": "application/json"}
        payload = self._build_payload(request)

        result.start_time = time.perf_counter()
        try:
            async with _create_client_session(timeout=self.timeout) as session:
                async with session.post(
                    url=url, json=payload, headers=headers
                ) as response:
                    result.status_code = response.status

                    if response.status >= 400:
                        result.error = await _read_error_response(response)
                        return result

                    if not request.stream:
                        await self._parse_no_stream_response(response, result)
                    else:
                        await self._parse_stream_response(response, result, on_chunk)
        except aiohttp.ClientResponseError as exc:
            result.status_code = exc.status
            result.error = str(exc)
        except aiohttp.ClientError as exc:
            result.error = f"Network error: {exc}"
        except Exception as exc:
            result.error = f"Unexpected error: {type(exc).__name__}: {exc}"
        finally:
            result.finish_time = time.perf_counter()
            result.latency = result.finish_time - result.start_time

            if result.stream and result.completion_tokens > 1:
                result.tpot = (result.latency - result.ttft) / (
                    result.completion_tokens - 1
                )

        return result
