"""OpenAI-compatible backend implementation."""

import json
import time
import aiohttp
from typing import Callable, Optional, List, Dict, Literal, TypeAlias, TypedDict

from llmperf.core import LLMRequest, LLMResponse, ChatCompletionOutput, ToolCall
from llmperf.backends.base import LLMBackend, StreamEvent

StreamRole: TypeAlias = Literal["system", "user", "assistant", "tool"]


class StreamFunctionDelta(TypedDict, total=False):
    """Function payload contained in a streaming tool call delta."""

    name: str
    arguments: str


class StreamToolCallDelta(TypedDict, total=False):
    """Single tool call delta emitted by the streaming API."""

    id: str
    index: int
    type: Literal["function"]
    function: StreamFunctionDelta


class StreamChoiceDelta(TypedDict, total=False):
    """Delta payload for one streamed choice item."""

    role: StreamRole
    content: str
    reasoning_content: str
    tool_calls: List[StreamToolCallDelta]


class StreamChoice(TypedDict, total=False):
    """One streamed choice entry returned by the backend."""

    delta: StreamChoiceDelta
    finish_reason: Optional[str]


class StreamChunk(TypedDict, total=False):
    """Top-level streaming response chunk."""

    choices: List[StreamChoice]


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
        payload: dict[str, object] = {
            "stream": request.stream,
            "rid": request.rid,
            "chat_template_kwargs": request.chat_template_kwargs,
        }
        if request.model is not None:
            payload["model"] = request.model

        payload.update(request.input.model_dump(exclude_none=True))
        payload.update(request.sampling_params.model_dump())

        return payload

    async def send(
        self,
        url: str,
        request: LLMRequest,
        on_chunk: Optional[Callable[[StreamEvent], None]] = None,
    ) -> LLMResponse:
        """Send one request to an OpenAI-compatible endpoint.

        Args:
            url (str): Target endpoint URL.
            request (LLMRequest): Internal request model.
            on_chunk (Optional[Callable[[StreamEvent], None]]): Optional
                callback for streamed events.

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

        result.output = ChatCompletionOutput.model_validate(choice.get("message", {}))

        usage = response_json.get("usage", {})
        for key, value in usage.items():
            if key in LLMResponse.model_fields:
                setattr(result, key, value)

        result.finish_reason = choice.get("finish_reason")

    async def _parse_stream_response(
        self,
        response: aiohttp.ClientResponse,
        result: LLMResponse,
        on_chunk: Optional[Callable[[StreamEvent], None]] = None,
    ) -> None:
        """Parse a streaming response payload.

        Args:
            response (aiohttp.ClientResponse): HTTP response object.
            result (LLMResponse): Response model to populate.
            on_chunk (Optional[Callable[[StreamEvent], None]]): Optional
                callback for streamed events.

        Returns:
            None: The result object is updated in place.
        """

        tool_calls_list: List[ToolCall] = []
        last_chunk_time = result.start_time

        async for chunk_bytes in response.content:
            data = self._parse_stream_chunk(chunk_bytes)
            if data is None:
                continue

            choice = self._get_first_choice(data)
            delta = self._get_choice_delta(choice)

            self._apply_role_delta(result, delta)

            produced_output = False
            produced_output |= self._apply_tool_call_delta(
                tool_calls_list, delta, on_chunk
            )
            produced_output |= self._apply_text_delta(result, delta, on_chunk)

            self._apply_finish_reason(result, choice)

            if produced_output:
                last_chunk_time = self._record_stream_progress(result, last_chunk_time)

        if tool_calls_list:
            result.output.tool_calls = tool_calls_list

    def _parse_stream_chunk(self, chunk_bytes: bytes) -> Optional[StreamChunk]:
        """Decode one SSE line into a structured stream chunk.

        Args:
            chunk_bytes (bytes): Raw bytes read from the HTTP stream.

        Returns:
            Optional[StreamChunk]: Parsed JSON chunk, or ``None`` for keepalive
                and ``[DONE]`` markers.
        """
        chunk_text = remove_prefix(chunk_bytes.strip().decode("utf-8"), "data: ")
        if not chunk_text or chunk_text == "[DONE]":
            return None

        return json.loads(chunk_text)

    def _get_first_choice(self, data: StreamChunk) -> StreamChoice:
        """Return the first choice from a parsed stream chunk.

        Args:
            data (StreamChunk): Parsed chunk payload.

        Returns:
            StreamChoice: The first choice object, or an empty mapping when the
                chunk has no choices.
        """
        choices = data.get("choices", [{}])

        return choices[0] if isinstance(choices, list) and choices else {}

    def _get_choice_delta(self, choice: StreamChoice) -> StreamChoiceDelta:
        """Extract the delta payload from one streamed choice.

        Args:
            choice (StreamChoice): Choice object from the stream payload.

        Returns:
            StreamChoiceDelta: Delta mapping, or an empty mapping when absent.
        """
        delta = choice.get("delta", {})
        return delta if isinstance(delta, dict) else {}

    def _apply_role_delta(self, result: LLMResponse, delta: StreamChoiceDelta) -> None:
        """Persist an emitted role onto the accumulated response."""
        role = delta.get("role")
        if role is not None and result.output.role is None:
            result.output.role = role

    def _apply_tool_call_delta(
        self,
        tool_calls_list: List[ToolCall],
        delta: StreamChoiceDelta,
        on_chunk: Optional[Callable[[StreamEvent], None]],
    ) -> bool:
        """Merge streamed tool call updates into the accumulated response.

        Args:
            tool_calls_list (List[ToolCall]): Accumulated tool call objects.
            delta (StreamChoiceDelta): Current streaming delta payload.
            on_chunk (Optional[Callable[[StreamEvent], None]]): Optional
                callback for surfaced stream events.

        Returns:
            bool: ``True`` when the delta produced user-visible output.
        """
        tool_calls = delta.get("tool_calls")
        if tool_calls is None:
            return False

        tool_call = tool_calls[0]
        tool_call_id = tool_call.get("id")
        function_delta = tool_call.get("function", {})
        tool_name = function_delta.get("name")
        arguments_delta = function_delta.get("arguments", "")

        if tool_call_id is not None:
            tool_calls_list.append(ToolCall.model_validate(tool_call))

        existing_arguments = tool_calls_list[-1].function.arguments

        if isinstance(existing_arguments, str):
            tool_calls_list[-1].function.arguments = (
                existing_arguments + arguments_delta
            )
        else:
            tool_calls_list[-1].function.arguments = arguments_delta

        if on_chunk is not None:
            on_chunk(
                StreamEvent(
                    type="tool_call",
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    tool_arguments_delta=arguments_delta,
                )
            )

        return True

    def _apply_text_delta(
        self,
        result: LLMResponse,
        delta: StreamChoiceDelta,
        on_chunk: Optional[Callable[[StreamEvent], None]],
    ) -> bool:
        """Merge streamed text deltas into the accumulated response.

        Args:
            result (LLMResponse): Response object being updated in place.
            delta (StreamChoiceDelta): Current streaming delta payload.
            on_chunk (Optional[Callable[[StreamEvent], None]]): Optional
                callback for surfaced stream events.

        Returns:
            bool: ``True`` when content or reasoning text was emitted.
        """

        def _parse_content(
            content_key: Literal["content", "reasoning_content"],
        ) -> bool:
            """Apply one text field from a streaming delta.

            Args:
                content_key (Literal["content", "reasoning_content"]): Text
                    field to read from the delta.

            Returns:
                bool: ``True`` when the field contained non-empty text.
            """
            content = delta.get(content_key) or ""
            if not content:
                return False

            if on_chunk is not None:
                on_chunk(StreamEvent(type=content_key, text=content))

            text = getattr(result.output, content_key)
            if text is None:
                setattr(result.output, content_key, content)
            else:
                setattr(result.output, content_key, text + content)

            return True

        return _parse_content("content") | _parse_content("reasoning_content")

    def _apply_finish_reason(self, result: LLMResponse, choice: StreamChoice) -> None:
        """Store the finish reason reported by the current choice, if any."""
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            result.finish_reason = finish_reason

    def _record_stream_progress(
        self, result: LLMResponse, last_chunk_time: float
    ) -> float:
        """Update streaming latency counters after one visible event.

        Args:
            result (LLMResponse): Response object being updated in place.
            last_chunk_time (float): Timestamp of the previous visible chunk.

        Returns:
            float: Timestamp recorded for the current visible chunk.
        """
        result.completion_tokens += 1

        now = time.perf_counter()
        if result.ttft == 0.0:
            result.ttft = now - result.start_time
        else:
            result.itls.append(now - last_chunk_time)

        return now
