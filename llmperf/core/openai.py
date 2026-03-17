"""Async helpers for interacting with OpenAI-compatible APIs."""

import asyncio
import json
import time
from typing import Any

import aiohttp

from .data import (
    OpenAIRequestContext,
    OpenAIRequestOutput,
    RequestHeaders,
    RequestPayload,
)
from .listener import OpenAIBaseListener, notify


def _create_client_session(timeout: float = 300) -> aiohttp.ClientSession:
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


async def openai_chat_request(
    url: str,
    payload: RequestPayload,
    headers: RequestHeaders,
    timeout: float = 300,
    listeners: list[OpenAIBaseListener] | None = None,
) -> OpenAIRequestOutput:
    """Send a chat completion request and collect response metrics.

    Args:
        url (str): Target OpenAI-compatible endpoint URL.
        payload (RequestPayload): JSON request body sent to the endpoint.
        headers (RequestHeaders): HTTP headers included with the request.
        timeout (float): Total request timeout in seconds.
        listeners (list[OpenAIBaseListener] | None): Optional request lifecycle
            listeners.

    Returns:
        OpenAIRequestOutput: The populated request output including content,
        error state, and timing metrics.
    """
    ctx = OpenAIRequestContext(
        url=url,
        payload=payload,
        headers=headers,
        result=OpenAIRequestOutput(stream=bool(payload.get("stream", False))),
    )

    # start
    ctx.result.start_time = time.perf_counter()
    last_chunk_time = ctx.result.start_time
    await notify(listeners, "on_request_start", ctx)

    try:
        async with _create_client_session(timeout) as session:
            async with session.post(url=url, json=payload, headers=headers) as response:
                ctx.result.status = response.status
                response.raise_for_status()

                # 非流式
                if not ctx.result.stream:
                    response_json: dict[str, Any] = await response.json()

                    ctx.result.prompt_tokens = response_json.get("usage", {}).get(
                        "prompt_tokens", 0
                    )
                    ctx.result.completion_tokens = response_json.get("usage", {}).get(
                        "completion_tokens", 0
                    )
                    ctx.result.generated_text = (
                        response_json.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                # 流式
                else:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # TODO: 处理 reasoning_content 情况
                            content = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )

                            if content:
                                ctx.result.generated_text += content
                                ctx.result.completion_tokens += 1

                                # ttft and itls
                                now = time.perf_counter()
                                if ctx.result.ttft == 0.0:
                                    ctx.result.ttft = now - ctx.result.start_time
                                else:
                                    ctx.result.itls.append(now - last_chunk_time)
                                last_chunk_time = now

                                await notify(listeners, "on_chunk", ctx, content)

                await notify(listeners, "on_request_success", ctx)
    except asyncio.CancelledError:
        ctx.result.error = "cancelled"
        await notify(listeners, "on_request_cancel", ctx)
    except Exception as exc:
        ctx.result.error = str(exc)
        await notify(listeners, "on_request_error", ctx, exc)
    finally:
        ctx.result.finish_time = time.perf_counter()
        ctx.result.latency = ctx.result.finish_time - ctx.result.start_time
        if ctx.result.completion_tokens > 1:
            ctx.result.tpot = (ctx.result.latency - ctx.result.ttft) / (
                ctx.result.completion_tokens - 1
            )

        await notify(listeners, "on_request_finish", ctx)

    return ctx.result
