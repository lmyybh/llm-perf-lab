"""Benchmark summary models and aggregation helpers."""

from typing import Callable, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from llmperf.common import LLMResponse


class MetricStats(BaseModel):
    """Percentile summary for one benchmark metric.

    Attributes:
        mean (float): Arithmetic mean of the metric values.
        p50 (float): 50th percentile value.
        p90 (float): 90th percentile value.
        p95 (float): 95th percentile value.
        p99 (float): 99th percentile value.
    """

    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class BenchSummary(BaseModel):
    """Aggregated benchmark summary for a full benchmark run.

    Attributes:
        total_requests (int): Total number of attempted requests.
        succeeded_requests (int): Number of successful requests.
        failed_requests (int): Number of failed requests.
        success_rate (float): Success ratio across all requests.
        total_duration (float): Total benchmark wall-clock duration in seconds.
        request_throughput (float): Successful requests per second.
        concurrency (float): Average concurrent request load.
        qps (Optional[float]): Configured target QPS, when provided.
        max_concurrency (Optional[int]): Configured concurrency limit, when
            provided.
        total_prompt_tokens (int): Total input tokens across successful
            requests.
        total_completion_tokens (int): Total output tokens across successful
            requests.
        mean_prompt_tokens (float): Mean input tokens across successful
            requests.
        mean_completion_tokens (float): Mean output tokens across successful
            requests.
        metrics (Dict[str, Optional[MetricStats]]): Named latency summaries.
    """

    total_requests: int
    succeeded_requests: int
    failed_requests: int
    success_rate: float

    total_duration: float
    request_throughput: float
    concurrency: float
    qps: Optional[float] = None
    max_concurrency: Optional[int] = None
    total_prompt_tokens: int
    total_completion_tokens: int
    mean_prompt_tokens: int
    mean_completion_tokens: int

    metrics: Dict[str, Optional[MetricStats]] = Field(default_factory=dict)


def _build_metric_stats(values: List[float]) -> Optional[MetricStats]:
    if len(values) == 0:
        return None

    return MetricStats(
        mean=float(np.mean(values)),
        p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
    )


def _summarize_metric(
    responses: List[LLMResponse], value_getter: Callable[[LLMResponse], float]
) -> Optional[MetricStats]:
    values = []
    for response in responses:
        value = value_getter(response)
        if value > 0.0:
            values.append(value)

    return _build_metric_stats(values)


def summarize_bench_results(
    responses: List[LLMResponse],
    start_time: float,
    finish_time: float,
    qps: Optional[float] = None,
    max_concurrency: Optional[int] = None,
) -> BenchSummary:
    """Aggregate benchmark responses into a terminal-friendly summary.

    Args:
        responses (List[LLMResponse]): Responses returned by the benchmark.
        start_time (float): Benchmark start time from ``time.perf_counter``.
        finish_time (float): Benchmark finish time from ``time.perf_counter``.
        qps (Optional[float]): Configured target QPS, when provided.
        max_concurrency (Optional[int]): Configured concurrency limit, when
            provided.

    Returns:
        BenchSummary: Aggregated summary derived from the benchmark results.
    """
    succeeded_responses = [
        response for response in responses if response.status_code == 200
    ]

    total_requests = len(responses)
    succeeded_requests = len(succeeded_responses)
    failed_requests = total_requests - succeeded_requests
    success_rate = succeeded_requests / total_requests if total_requests > 0 else 0.0

    total_duration = finish_time - start_time
    request_throughput = (
        succeeded_requests / total_duration if total_duration > 0 else 0.0
    )
    concurrency = (
        sum(response.latency for response in succeeded_responses) / total_duration
        if total_duration > 0
        else 0.0
    )
    total_prompt_tokens = sum(
        response.prompt_tokens for response in succeeded_responses
    )
    total_completion_tokens = sum(
        response.completion_tokens for response in succeeded_responses
    )
    mean_prompt_tokens = (
        int(total_prompt_tokens / succeeded_requests) if succeeded_requests > 0 else 0
    )
    mean_completion_tokens = (
        int(total_completion_tokens / succeeded_requests)
        if succeeded_requests > 0
        else 0
    )

    metrics = {
        "latency": _summarize_metric(
            succeeded_responses, lambda response: response.latency
        ),
        "ttft": _summarize_metric(succeeded_responses, lambda response: response.ttft),
        "tpot": _summarize_metric(succeeded_responses, lambda response: response.tpot),
    }

    return BenchSummary(
        total_requests=total_requests,
        succeeded_requests=succeeded_requests,
        failed_requests=failed_requests,
        success_rate=success_rate,
        total_duration=total_duration,
        request_throughput=request_throughput,
        concurrency=concurrency,
        qps=qps,
        max_concurrency=max_concurrency,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        mean_prompt_tokens=mean_prompt_tokens,
        mean_completion_tokens=mean_completion_tokens,
        metrics=metrics,
    )
