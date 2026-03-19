import asyncio
from tqdm.asyncio import tqdm
from typing import List
from dataclasses import asdict

from llmperf.core.data import OpenAIRequestInput, OpenAIRequestOutput
from llmperf.core.dataset import read_zss_file
from llmperf.core.openai import openai_chat_request


async def bench_openai_chat_requests(
    url: str,
    requests: List[OpenAIRequestInput],
    qps: float | None = None,
    max_concurrency: int | None = None,
    timeout: float = 300,
):
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency is not None and max_concurrency > 0
        else None
    )
    results: List[OpenAIRequestOutput | None] = [None] * len(requests)

    async def _send_one_request(index: int, request: OpenAIRequestInput, pbar: tqdm):
        headers = {"Content-Type": "application/json"}
        payload = asdict(request)

        if semaphore is not None:
            async with semaphore:
                result = await openai_chat_request(url, payload, headers, timeout)
        else:
            result = await openai_chat_request(url, payload, headers, timeout)

        results[index] = result
        pbar.update(1)

    interval = 1.0 / qps if qps is not None and qps > 0 else 0.0

    with tqdm(total=len(requests)) as pbar:
        tasks = []
        for i, request in enumerate(requests):
            tasks.append(asyncio.create_task(_send_one_request(i, request, pbar)))

            if interval > 0:
                await asyncio.sleep(interval)

        await asyncio.gather(*tasks)

    return results


async def bench_requests(url, file, num_requests, qps, max_concurrency):
    requests = read_zss_file(file, num_requests)

    results = await bench_openai_chat_requests(
        url, requests, qps=qps, max_concurrency=max_concurrency, timeout=10 * 60
    )

    return results
