"""Replay 批量回放执行器。

本模块负责三类事情：
1. 以异步方式按 QPS 或并发限制发送请求。
2. 从响应里提取 e2e、TTFT、TPOT、accept 等指标。
3. 汇总单请求结果，生成最终报表所需的聚合统计。
"""

import asyncio
import json
import logging
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from llmperf.adapters.generate import GenerateAdapter
from llmperf.core.errors import HttpError, LLMPerfError
from llmperf.core.executor import ResponseEnvelope, execute_payload_request_async
from llmperf.core.models import ReplayItemResult, ReplayRequest, RequestConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


@dataclass
class MetricStats:
    """描述一个指标在样本集合上的聚合结果。"""

    mean: float | None
    p50: float | None
    p99: float | None
    std: float | None


@dataclass
class ReplaySummary:
    """描述整批 replay 的汇总统计。"""

    requests_total: int
    requests_succeeded: int
    requests_failed: int
    success_rate: float
    duration_s: float
    actual_qps: float | None
    request_throughput: float
    concurrency: float
    token_throughput: float | None
    accept_len: float | None
    accept_rate: float | None
    e2e_latency_ms: MetricStats
    ttft_ms: MetricStats
    tpot_ms: MetricStats
    failures: dict[str, int]


def _percentile(values: list[float], percentile: int) -> float | None:
    """计算简单分位数。

    当前实现按排序后的索引近似取值，足够满足 CLI 统计展示需求。
    """
    if not values:
        return None
    ordered = sorted(values)
    index = max(
        0, min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1))))
    )
    return ordered[index]


def _build_metric(values: list[float]) -> MetricStats:
    """把原始样本列表聚合成 mean/p50/p99/std。"""
    if not values:
        return MetricStats(mean=None, p50=None, p99=None, std=None)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return MetricStats(
        mean=sum(values) / len(values),
        p50=_percentile(values, 50),
        p99=_percentile(values, 99),
        std=std,
    )


def _extract_output_tokens(response: ResponseEnvelope) -> int | None:
    """从 generate 响应中提取输出 token 数。

    优先读取非流式响应里的 `meta_info.completion_tokens`；
    若是流式响应，则尝试从最后一个 chunk 的 meta_info 或 output_ids 推断。
    """
    if response.body_json:
        meta_info = response.body_json.get("meta_info")
        if isinstance(meta_info, dict):
            completion_tokens = meta_info.get("completion_tokens")
            if isinstance(completion_tokens, int):
                return completion_tokens
        output_ids = response.body_json.get("output_ids")
        if isinstance(output_ids, list):
            return len(output_ids)

    completion_tokens = None
    output_ids_len = None
    body_text = response.body_text.strip()
    for line in body_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        meta_info = payload.get("meta_info")
        if isinstance(meta_info, dict) and isinstance(
            meta_info.get("completion_tokens"), int
        ):
            completion_tokens = meta_info["completion_tokens"]
        output_ids = payload.get("output_ids")
        if isinstance(output_ids, list):
            output_ids_len = len(output_ids)

    return completion_tokens if completion_tokens is not None else output_ids_len


def _extract_meta_metric(response: ResponseEnvelope, key: str) -> float | None:
    """从响应中提取某个 meta_info 指标。

    该函数同时兼容：
    - 非流式：直接从 body_json 读取
    - 流式：遍历每一行，取最后一个可用值
    """
    if response.body_json:
        meta_info = response.body_json.get("meta_info")
        if isinstance(meta_info, dict):
            value = meta_info.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    latest_value = None
    body_text = response.body_text.strip()
    for line in body_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        meta_info = payload.get("meta_info")
        if isinstance(meta_info, dict):
            value = meta_info.get(key)
            if isinstance(value, (int, float)):
                latest_value = float(value)
    return latest_value


def _build_item_from_response(
    request: ReplayRequest,
    response: ResponseEnvelope,
    output_text: str,
    request_start_time: float | None,
) -> ReplayItemResult:
    """把 HTTP 响应转换成可聚合的单请求结果。"""
    output_tokens = _extract_output_tokens(response)
    tpot_ms = None
    if (
        output_tokens is not None
        and output_tokens > 1
        and response.ttft_ms is not None
        and response.latency_ms >= response.ttft_ms
    ):
        tpot_ms = (response.latency_ms - response.ttft_ms) / (output_tokens - 1)
    return ReplayItemResult(
        source_file=request.source_file,
        source_index=request.source_index,
        status="ok",
        latency_ms=response.latency_ms,
        request_start_time=request_start_time,
        ttft_ms=response.ttft_ms,
        output_text=output_text,
        output_tokens=output_tokens,
        tpot_ms=tpot_ms,
        accept_len=_extract_meta_metric(response, "spec_accept_length"),
        accept_rate=_extract_meta_metric(response, "spec_accept_rate"),
        status_code=response.status_code,
    )


async def execute_replay_request_async(
    request: ReplayRequest,
    endpoint: str,
    timeout_ms: int,
    api_key: str | None,
    request_start_time: float | None = None,
) -> ReplayItemResult:
    """异步执行单个 replay 请求。"""
    adapter = GenerateAdapter()
    config = RequestConfig(
        endpoint=endpoint,
        api_key=api_key,
        timeout_ms=timeout_ms,
        model=None,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        stream=request.stream,
        messages=None,
        prompt=None,
    )
    try:
        response = await execute_payload_request_async(
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            payload=request.payload,
            api_key=api_key,
            stream=request.stream,
            parse_stream_line=adapter.parse_stream_line if request.stream else None,
        )
        result = adapter.parse_response(response, config)
        return _build_item_from_response(
            request,
            response,
            result.output_text,
            request_start_time=request_start_time,
        )
    except LLMPerfError as exc:
        status_code = exc.status_code if isinstance(exc, HttpError) else None
        return ReplayItemResult(
            source_file=request.source_file,
            source_index=request.source_index,
            status="error",
            latency_ms=0,
            request_start_time=request_start_time,
            ttft_ms=None,
            output_text="",
            status_code=status_code,
            error_type=exc.error_type,
            error_message=exc.message,
        )


def execute_replay(
    requests: list[ReplayRequest],
    endpoint: str,
    timeout_ms: int,
    api_key: str | None,
    qps: float | None,
    max_concurrency: int | None,
) -> tuple[list[ReplayItemResult], ReplaySummary]:
    """同步外观的 replay 入口。

    CLI 层仍然按同步函数调用，本函数内部再切进 asyncio 事件循环。
    """
    started = time.perf_counter()
    results = asyncio.run(
        _execute_replay_async(
            requests=requests,
            endpoint=endpoint,
            timeout_ms=timeout_ms,
            api_key=api_key,
            qps=qps,
            max_concurrency=max_concurrency,
        )
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return results, summarize_replay(results, elapsed_ms)


async def _execute_replay_async(
    requests: list[ReplayRequest],
    endpoint: str,
    timeout_ms: int,
    api_key: str | None,
    qps: float | None,
    max_concurrency: int | None,
) -> list[ReplayItemResult]:
    """异步调度入口。

    - 当传入 qps 时，按照固定发起节拍送出请求。
    - 当只传并发时，用 semaphore 控制在途请求数。
    - 当两者都不传时，默认并发发出全部请求。
    """
    results: list[ReplayItemResult | None] = [None] * len(requests)
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency and max_concurrency > 0
        else None
    )

    async def _run_one(index: int, request: ReplayRequest, progress) -> None:
        """执行单个请求，并在完成后刷新进度条。"""
        request_start_time = time.perf_counter()
        if semaphore is None:
            result = await execute_replay_request_async(
                request,
                endpoint,
                timeout_ms,
                api_key,
                request_start_time=request_start_time,
            )
        else:
            async with semaphore:
                result = await execute_replay_request_async(
                    request,
                    endpoint,
                    timeout_ms,
                    api_key,
                    request_start_time=request_start_time,
                )
        results[index] = result
        progress.update(1)

    async with tqdm_async(total=len(requests)) as progress:
        if qps is not None:
            await _run_with_qps(
                requests, endpoint, timeout_ms, api_key, qps, progress, results
            )
        else:
            tasks = [
                asyncio.create_task(_run_one(i, request, progress))
                for i, request in enumerate(requests)
            ]
            await asyncio.gather(*tasks)
    return [item for item in results if item is not None]


class tqdm_async:
    """把普通 tqdm 包装成 async with 可用的上下文管理器。"""

    def __init__(self, total: int):
        self._bar = tqdm(total=total, desc="Replay progress", unit="req")

    async def __aenter__(self):
        return self._bar

    async def __aexit__(self, exc_type, exc, tb):
        self._bar.close()


async def _run_with_qps(
    requests: list[ReplayRequest],
    endpoint: str,
    timeout_ms: int,
    api_key: str | None,
    qps: float,
    progress,
    results: list[ReplayItemResult | None],
) -> None:
    """按 QPS 发起请求。

    这里控制的是“发起速率”，而不是“在途请求数”。
    因此任务一旦到达发起时刻就创建为 asyncio task，后续完成时间不影响继续发起。
    """
    interval = 1.0 / qps
    tasks: list[asyncio.Task[None]] = []

    async def _run_one(index: int, request: ReplayRequest) -> None:
        """QPS 路径下的单请求执行函数。"""
        request_start_time = time.perf_counter()
        result = await execute_replay_request_async(
            request,
            endpoint,
            timeout_ms,
            api_key,
            request_start_time=request_start_time,
        )
        results[index] = result
        progress.update(1)

    next_deadline = time.perf_counter()
    for index, request in enumerate(requests):
        now = time.perf_counter()
        if next_deadline > now:
            await asyncio.sleep(next_deadline - now)
        tasks.append(asyncio.create_task(_run_one(index, request)))
        next_deadline += interval

    if tasks:
        await asyncio.gather(*tasks)


def summarize_replay(results: list[ReplayItemResult], elapsed_ms: int) -> ReplaySummary:
    """把所有单请求结果聚合成最终报表需要的 summary。"""
    succeeded = [item for item in results if item.status == "ok"]
    failed = [item for item in results if item.status != "ok"]
    duration_s = elapsed_ms / 1000.0 if elapsed_ms > 0 else 0.0
    total_latency_s = sum(item.latency_ms for item in succeeded) / 1000.0
    total_output_tokens = sum(
        item.output_tokens for item in succeeded if item.output_tokens is not None
    )
    request_throughput = (len(results) / duration_s) if duration_s > 0 else 0.0
    start_times = sorted(
        item.request_start_time
        for item in results
        if item.request_start_time is not None
    )
    actual_qps = None
    if len(start_times) >= 2:
        send_span_s = start_times[-1] - start_times[0]
        if send_span_s > 0:
            actual_qps = (len(start_times) - 1) / send_span_s
    concurrency = (total_latency_s / duration_s) if duration_s > 0 else 0.0
    token_throughput = (total_output_tokens / duration_s) if duration_s > 0 else None
    accept_lens = [item.accept_len for item in succeeded if item.accept_len is not None]
    accept_rates = [
        item.accept_rate for item in succeeded if item.accept_rate is not None
    ]

    failure_counts: Counter[str] = Counter()
    for item in failed:
        if item.status_code is not None:
            failure_counts[f"http_{item.status_code}"] += 1
        elif item.error_type:
            failure_counts[item.error_type] += 1
        else:
            failure_counts["unknown_error"] += 1

    return ReplaySummary(
        requests_total=len(results),
        requests_succeeded=len(succeeded),
        requests_failed=len(failed),
        success_rate=((len(succeeded) / len(results)) * 100.0) if results else 0.0,
        duration_s=duration_s,
        actual_qps=actual_qps,
        request_throughput=request_throughput,
        concurrency=concurrency,
        token_throughput=token_throughput,
        accept_len=(sum(accept_lens) / len(accept_lens)) if accept_lens else None,
        accept_rate=(sum(accept_rates) / len(accept_rates)) if accept_rates else None,
        e2e_latency_ms=_build_metric([float(item.latency_ms) for item in succeeded]),
        ttft_ms=_build_metric(
            [float(item.ttft_ms) for item in succeeded if item.ttft_ms is not None]
        ),
        tpot_ms=_build_metric(
            [item.tpot_ms for item in succeeded if item.tpot_ms is not None]
        ),
        failures=dict(sorted(failure_counts.items())),
    )


def save_replay_results(path: str, results: list[ReplayItemResult]) -> None:
    """把逐请求结果保存成 JSONL，便于后续离线分析。"""
    output_path = Path(path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(item.to_dict(), ensure_ascii=False) for item in results
    )
    output_path.write_text(f"{content}\n" if content else "", encoding="utf-8")
