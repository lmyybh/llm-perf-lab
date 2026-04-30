import json
import time
import aiohttp

from llmperf.backends.base import LLMBackend, StreamEvent
from llmperf.common import LLMRequest, LLMResponse, GenerateOutput
from llmperf.backends.openai import (
    _create_client_session,
    _read_error_response,
    remove_prefix,
)


class GenerateBackend(LLMBackend):

    name: str = "generate"

    def __init__(self, timeout: float = 300.0) -> None:
        self.timeout = timeout

    def _build_payload(self, request: LLMRequest) -> dict[str, object]:
        payload: dict[str, object] = {
            "stream": request.stream,
            "rid": request.rid,
        }

        payload.update(request.input.model_dump(exclude_none=True))

        sampling_params = request.sampling_params.model_dump()
        sampling_params["max_new_tokens"] = sampling_params.pop("max_completion_tokens")
        sampling_params["sampling_seed"] = sampling_params.pop("seed")
        payload["sampling_params"] = sampling_params

        return payload

    async def send(
        self,
        url: str,
        request: LLMRequest,
        on_chunk=None,
    ) -> LLMResponse:

        result = LLMResponse(
            model=request.model, stream=request.stream, output=GenerateOutput()
        )

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
        response_json = await response.json()

        result.output.text = response_json.get("text", None)

        meta_info = response_json.get("meta_info", {})
        for key, value in meta_info.items():
            if key in LLMResponse.model_fields:
                setattr(result, key, value)

        if isinstance(result.finish_reason, dict):
            result.finish_reason = result.finish_reason.get("type", None)

    async def _parse_stream_response(
        self,
        response: aiohttp.ClientResponse,
        result: LLMResponse,
        on_chunk=None,
    ) -> None:

        last_chunk_time = result.start_time
        last_text = ""

        async for chunk_bytes in response.content:
            data = self._parse_stream_chunk(chunk_bytes)
            if data is None:
                continue

            current_text = data.get("text")
            if current_text:
                last_chunk_time = self._record_stream_progress(result, last_chunk_time)

                if on_chunk is not None:
                    delta_text = current_text[len(last_text) :]
                    on_chunk(StreamEvent(type="text", text=delta_text))

                last_text = current_text

            finish_reason = data.get("meta_info", {}).get("finish_reason", None)
            if finish_reason:
                result.finish_reason = finish_reason.get("type", None)

    def _parse_stream_chunk(self, chunk_bytes: bytes):
        chunk_text = remove_prefix(chunk_bytes.strip().decode("utf-8"), "data: ")
        if not chunk_text or chunk_text == "[DONE]":
            return None

        return json.loads(chunk_text)

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
