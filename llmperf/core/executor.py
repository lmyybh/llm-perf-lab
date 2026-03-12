"""HTTP 执行层。

这里同时保留同步请求路径和异步请求路径：
- `request` 单次命令继续复用同步实现，代码改动最小。
- `replay` 批量压测场景使用异步实现，避免大量线程带来的调度开销。
"""

import codecs
import json
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from llmperf.core.errors import HttpError, NetworkError, ProtocolError, TimeoutError
from llmperf.core.models import RequestConfig


@dataclass
class ResponseEnvelope:
    """统一封装 HTTP 响应，供不同适配器做后处理。"""

    status_code: int
    latency_ms: int
    ttft_ms: int | None
    body_text: str
    body_json: dict | None
    streamed_output: str | None


def _read_stream(resp) -> tuple[str, int | None]:
    """同步读取整条响应流，并记录首个 chunk 到达时间。"""
    chunks: list[bytes] = []
    first_chunk_ts = None
    start = time.perf_counter()

    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        if first_chunk_ts is None:
            first_chunk_ts = time.perf_counter()
        chunks.append(chunk)

    ttft_ms = None
    if first_chunk_ts is not None:
        ttft_ms = int((first_chunk_ts - start) * 1000)
    return b"".join(chunks).decode("utf-8", errors="replace"), ttft_ms


def _read_stream_with_parser(
    resp,
    parse_stream_line: Callable[[str], str] | None,
    on_stream_text: Callable[[str], None] | None,
) -> tuple[str, int | None, str]:
    """同步读取流式响应，并用适配器解析增量文本。"""
    raw_chunks: list[bytes] = []
    output_chunks: list[str] = []
    emitted_text = ""
    first_chunk_ts = None
    start = time.perf_counter()
    buffer = b""
    plain_text_decoder = codecs.getincrementaldecoder("utf-8")()

    while True:
        chunk = resp.read(1024)
        if not chunk:
            break
        if first_chunk_ts is None:
            first_chunk_ts = time.perf_counter()
        raw_chunks.append(chunk)
        buffer += chunk

        if parse_stream_line is None:
            chunk_text = plain_text_decoder.decode(chunk)
            output_chunks.append(chunk_text)
            if on_stream_text:
                on_stream_text(chunk_text)
            continue

        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            parsed_text = parse_stream_line(line)
            if parsed_text:
                delta = parsed_text
                if emitted_text and parsed_text.startswith(emitted_text):
                    delta = parsed_text[len(emitted_text) :]
                if delta:
                    output_chunks.append(delta)
                    emitted_text += delta
                    if on_stream_text:
                        on_stream_text(delta)

    if parse_stream_line is None:
        tail = plain_text_decoder.decode(b"", final=True)
        if tail:
            output_chunks.append(tail)
            if on_stream_text:
                on_stream_text(tail)
    elif buffer:
        parsed_text = parse_stream_line(
            buffer.decode("utf-8", errors="replace").rstrip("\r")
        )
        if parsed_text:
            delta = parsed_text
            if emitted_text and parsed_text.startswith(emitted_text):
                delta = parsed_text[len(emitted_text) :]
            if delta:
                output_chunks.append(delta)
                emitted_text += delta
                if on_stream_text:
                    on_stream_text(delta)

    ttft_ms = None
    if first_chunk_ts is not None:
        ttft_ms = int((first_chunk_ts - start) * 1000)
    return (
        b"".join(raw_chunks).decode("utf-8", errors="replace"),
        ttft_ms,
        "".join(output_chunks),
    )


async def _read_async_stream(resp) -> tuple[str, int | None]:
    """异步读取整条响应流，并记录 TTFT。"""
    chunks: list[bytes] = []
    first_chunk_ts = None
    start = time.perf_counter()

    async for chunk in resp.aiter_bytes(chunk_size=4096):
        if not chunk:
            continue
        if first_chunk_ts is None:
            first_chunk_ts = time.perf_counter()
        chunks.append(chunk)

    ttft_ms = None
    if first_chunk_ts is not None:
        ttft_ms = int((first_chunk_ts - start) * 1000)
    return b"".join(chunks).decode("utf-8", errors="replace"), ttft_ms


async def _read_async_stream_with_parser(
    resp,
    parse_stream_line: Callable[[str], str] | None,
    on_stream_text: Callable[[str], None] | None,
) -> tuple[str, int | None, str]:
    """异步读取流式响应，并在接收过程中产出可显示的文本增量。"""
    raw_chunks: list[bytes] = []
    output_chunks: list[str] = []
    emitted_text = ""
    first_chunk_ts = None
    start = time.perf_counter()
    buffer = b""
    plain_text_decoder = codecs.getincrementaldecoder("utf-8")()

    async for chunk in resp.aiter_bytes(chunk_size=1024):
        if not chunk:
            continue
        if first_chunk_ts is None:
            first_chunk_ts = time.perf_counter()
        raw_chunks.append(chunk)
        buffer += chunk

        if parse_stream_line is None:
            chunk_text = plain_text_decoder.decode(chunk)
            output_chunks.append(chunk_text)
            if on_stream_text:
                on_stream_text(chunk_text)
            continue

        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            parsed_text = parse_stream_line(line)
            if parsed_text:
                delta = parsed_text
                if emitted_text and parsed_text.startswith(emitted_text):
                    delta = parsed_text[len(emitted_text) :]
                if delta:
                    output_chunks.append(delta)
                    emitted_text += delta
                    if on_stream_text:
                        on_stream_text(delta)

    if parse_stream_line is None:
        tail = plain_text_decoder.decode(b"", final=True)
        if tail:
            output_chunks.append(tail)
            if on_stream_text:
                on_stream_text(tail)
    elif buffer:
        parsed_text = parse_stream_line(
            buffer.decode("utf-8", errors="replace").rstrip("\r")
        )
        if parsed_text:
            delta = parsed_text
            if emitted_text and parsed_text.startswith(emitted_text):
                delta = parsed_text[len(emitted_text) :]
            if delta:
                output_chunks.append(delta)
                emitted_text += delta
                if on_stream_text:
                    on_stream_text(delta)

    ttft_ms = None
    if first_chunk_ts is not None:
        ttft_ms = int((first_chunk_ts - start) * 1000)
    return (
        b"".join(raw_chunks).decode("utf-8", errors="replace"),
        ttft_ms,
        "".join(output_chunks),
    )


def _parse_body_json(body_text: str) -> dict | None:
    """尽力把响应文本解析成 JSON 对象；失败时返回 None。"""
    body_json = None
    if body_text:
        try:
            parsed = json.loads(body_text)
            if isinstance(parsed, dict):
                body_json = parsed
        except json.JSONDecodeError:
            body_json = None
    return body_json


def execute_request(
    config: RequestConfig,
    payload: dict,
    api_key: str | None,
    parse_stream_line: Callable[[str], str] | None = None,
    on_stream_text: Callable[[str], None] | None = None,
) -> ResponseEnvelope:
    """供单次 request 命令使用的同步执行入口。"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return execute_payload_request(
        endpoint=config.endpoint,
        timeout_ms=config.timeout_ms,
        payload=payload,
        api_key=api_key,
        stream=config.stream,
        parse_stream_line=parse_stream_line,
        on_stream_text=on_stream_text,
    )


def execute_payload_request(
    endpoint: str,
    timeout_ms: int,
    payload: dict,
    api_key: str | None,
    stream: bool,
    parse_stream_line: Callable[[str], str] | None = None,
    on_stream_text: Callable[[str], None] | None = None,
) -> ResponseEnvelope:
    """同步发送任意 payload，并返回统一响应结构。"""
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=timeout_ms / 1000.0) as resp:
            streamed_output = None
            if stream:
                body_text, ttft_ms, streamed_output = _read_stream_with_parser(
                    resp=resp,
                    parse_stream_line=parse_stream_line,
                    on_stream_text=on_stream_text,
                )
            else:
                body_text, ttft_ms = _read_stream(resp)
            latency_ms = int((time.perf_counter() - start) * 1000)
            status_code = int(resp.status)
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise HttpError(
            f"HTTP {exc.code}: {message}", status_code=int(exc.code)
        ) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            raise TimeoutError("request timed out") from exc
        raise NetworkError(f"network error: {reason}") from exc
    except TimeoutError:
        raise
    except socket.timeout as exc:
        raise TimeoutError("request timed out") from exc
    except Exception as exc:  # pragma: no cover
        raise NetworkError(f"request failed: {exc}") from exc

    body_json = _parse_body_json(body_text)

    if status_code < 200 or status_code >= 300:
        raise HttpError(f"HTTP {status_code}: {body_text}", status_code=status_code)

    if not body_text:
        raise ProtocolError("empty response body")

    return ResponseEnvelope(
        status_code=status_code,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms if stream else None,
        body_text=body_text,
        body_json=body_json,
        streamed_output=streamed_output if stream else None,
    )


async def execute_payload_request_async(
    endpoint: str,
    timeout_ms: int,
    payload: dict,
    api_key: str | None,
    stream: bool,
    parse_stream_line: Callable[[str], str] | None = None,
    on_stream_text: Callable[[str], None] | None = None,
) -> ResponseEnvelope:
    """异步发送任意 payload。

    replay 场景需要高并发与速率控制，因此这里使用 httpx.AsyncClient
    来替代同步 urllib 实现。
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout = httpx.Timeout(timeout_ms / 1000.0)
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if stream:
                async with client.stream(
                    "POST", endpoint, json=payload, headers=headers
                ) as resp:
                    streamed_output = None
                    body_text, ttft_ms, streamed_output = (
                        await _read_async_stream_with_parser(
                            resp=resp,
                            parse_stream_line=parse_stream_line,
                            on_stream_text=on_stream_text,
                        )
                    )
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    status_code = int(resp.status_code)
            else:
                resp = await client.post(endpoint, json=payload, headers=headers)
                body_text, ttft_ms = await _read_async_stream(resp)
                streamed_output = None
                latency_ms = int((time.perf_counter() - start) * 1000)
                status_code = int(resp.status_code)
    except httpx.TimeoutException as exc:
        raise TimeoutError("request timed out") from exc
    except httpx.NetworkError as exc:
        raise NetworkError(f"network error: {exc}") from exc
    except httpx.HTTPError as exc:
        raise NetworkError(f"request failed: {exc}") from exc
    except Exception as exc:  # pragma: no cover
        raise NetworkError(f"request failed: {exc}") from exc

    body_json = _parse_body_json(body_text)

    if status_code < 200 or status_code >= 300:
        raise HttpError(f"HTTP {status_code}: {body_text}", status_code=status_code)

    if not body_text:
        raise ProtocolError("empty response body")

    return ResponseEnvelope(
        status_code=status_code,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms if stream else None,
        body_text=body_text,
        body_json=body_json,
        streamed_output=streamed_output if stream else None,
    )
