import aiohttp
import asyncio
import json
import time
from typing import List

from .data import OpenAIRequestOutput, OpenAIRequestContext
from .listener import notify, OpenAIBaseListener


def _create_client_session(timeout: float = 300):
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        read_bufsize=10 * 1024**2,
    )


def remove_prefix(text: str, prefix: str):
    return text[len(prefix) :] if text.startswith(prefix) else text


async def send_request(
    url,
    payload,
    headers,
    timeout: float = 300,
    listeners: List[OpenAIBaseListener] | None = None,
):
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
                    response_json = await response.json()

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
