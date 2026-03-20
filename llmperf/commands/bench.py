"""Helpers for building and running the benchmark command."""

import asyncio
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple, Literal
import time
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from llmperf.backends.base import LLMBackend
from llmperf.backends.openai import OpenAIChatBackend
from llmperf.core.models import BenchConfig, LLMRequest, LLMResponse
from llmperf.datasets.base import Dataset
from llmperf.datasets.file import FileDataset
from llmperf.benchmarks.summary import (
    MetricStats,
    BenchSummary,
    summarize_bench_results,
)

console = Console()


class BenchCommandArgs(BaseModel):
    """Arguments required to run the benchmark command.

    Attributes:
        url (str): Target OpenAI-compatible chat completions endpoint.
        file (Optional[Path]): Optional dataset file path.
        mode (str): Dataset mode name used by the reader registry.
        num_requests (Optional[int]): Optional maximum number of requests.
        qps (Optional[float]): Optional request rate limit.
        max_concurrency (Optional[int]): Optional in-flight request limit.
        timeout (float): Request timeout in seconds.
        model (Optional[str]): Optional model name override.
        temperature (Optional[float]): Optional sampling temperature override.
        max_completion_tokens (Optional[int]): Optional completion token limit.
        ignore_eos (Optional[bool]): Optional EOS handling override.
        enable_thinking (Optional[bool]): Optional thinking flag override.
    """

    url: str

    file: Optional[Path] = None
    mode: str

    num_requests: Optional[int] = Field(default=None, gt=0)
    qps: Optional[float] = Field(default=None, gt=0)
    max_concurrency: Optional[int] = Field(default=None, gt=0)
    timeout: float = Field(default=300.0, gt=0)

    model: Optional[str] = None
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    ignore_eos: Optional[bool] = None
    enable_thinking: Optional[bool] = None


def override_request(request: LLMRequest, config: BenchConfig) -> LLMRequest:
    """Apply benchmark-level overrides to one dataset request.

    Args:
        request (LLMRequest): Original request produced by the dataset.
        config (BenchConfig): Benchmark configuration containing overrides.

    Returns:
        LLMRequest: Request copy with benchmark overrides applied.
    """
    sampling_update = {}
    if config.temperature is not None:
        sampling_update["temperature"] = config.temperature
    if config.max_completion_tokens is not None:
        sampling_update["max_completion_tokens"] = config.max_completion_tokens
    if config.ignore_eos is not None:
        sampling_update["ignore_eos"] = config.ignore_eos

    sampling_params = request.sampling_params
    if sampling_update:
        sampling_params = request.sampling_params.model_copy(update=sampling_update)

    chat_template_kwargs = request.chat_template_kwargs
    if config.enable_thinking is not None:
        chat_template_kwargs = {
            **(request.chat_template_kwargs or {}),
            "enable_thinking": config.enable_thinking,
        }

    update_data = {
        "sampling_params": sampling_params,
        "chat_template_kwargs": chat_template_kwargs,
        "stream": True,  # always set stream=True
    }
    if config.model is not None:
        update_data["model"] = config.model

    return request.model_copy(update=update_data)


async def bench_requests(
    backend: LLMBackend, url: str, config: BenchConfig, requests: Iterable[LLMRequest]
) -> List[LLMResponse]:
    """Send benchmark requests and preserve input order in the results.

    Args:
        backend (LLMBackend): Backend used to send requests.
        url (str): Target endpoint URL.
        config (BenchConfig): Benchmark runtime configuration.
        requests (Iterable[LLMRequest]): Requests yielded by the dataset.

    Returns:
        List[LLMResponse]: Responses ordered to match the input sequence.
    """
    semaphore = (
        asyncio.Semaphore(config.max_concurrency)
        if config.max_concurrency is not None and config.max_concurrency > 0
        else None
    )

    async def _send_one(index: int, request: LLMRequest) -> Tuple[int, LLMResponse]:
        """Send one request and retain its original index.

        Args:
            index (int): Input position of the request.
            request (LLMRequest): Request payload to send.

        Returns:
            Tuple[int, LLMResponse]: Input index paired with its response.
        """
        if semaphore is not None:
            async with semaphore:
                result = await backend.send(url, request)
        else:
            result = await backend.send(url, request)

        return index, result

    interval = 1.0 / config.qps if config.qps is not None and config.qps > 0 else 0.0
    tasks: List[asyncio.Task[Tuple[int, LLMResponse]]] = []

    for index, request in enumerate(requests):
        request = override_request(request, config)
        tasks.append(asyncio.create_task(_send_one(index, request)))
        if interval > 0:
            await asyncio.sleep(interval)

    results: List[Tuple[int, LLMResponse]] = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await task)

    results.sort(key=lambda x: x[0])

    return [r[1] for r in results]


def create_backend(args: BenchCommandArgs) -> LLMBackend:
    """Create the backend instance for the benchmark command.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        LLMBackend: Configured backend instance.
    """
    return OpenAIChatBackend(timeout=args.timeout)


def build_config(args: BenchCommandArgs) -> BenchConfig:
    """Build runtime benchmark configuration from CLI arguments.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        BenchConfig: Normalized benchmark configuration.
    """
    return BenchConfig(
        num_requests=args.num_requests,
        qps=args.qps,
        max_concurrency=args.max_concurrency,
        timeout=args.timeout,
        model=args.model,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        ignore_eos=args.ignore_eos,
        enable_thinking=args.enable_thinking,
    )


def build_dataset(args: BenchCommandArgs) -> Dataset:
    """Create the dataset adapter used by the benchmark command.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        Dataset: Dataset capable of yielding benchmark requests.

    Raises:
        AssertionError: Raised when ``args.file`` is missing.
    """
    assert args.file is not None
    return FileDataset(file=args.file, mode=args.mode, num_requests=args.num_requests)


def _format_seconds(value: float, unit: Literal["s", "ms"] = "s") -> str:
    """Format a duration in seconds or milliseconds.

    Args:
        value (float): Duration value expressed in seconds.
        unit (Literal["s", "ms"]): Display unit for the formatted string.

    Returns:
        str: Human-readable duration string.
    """
    if unit == "s":
        return f"{value:.2f} s"
    else:
        return f"{(value*1000):.2f} ms"


def build_overview_table(summary: BenchSummary) -> Table:
    """Build the high-level benchmark overview table.

    Args:
        summary (BenchSummary): Aggregated benchmark summary.

    Returns:
        Table: Rich table containing overview metrics.
    """
    table = Table(show_header=False, box=None, pad_edge=False, expand=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white", no_wrap=True)

    table.add_row(
        "Requests (total/success/failure)",
        f"{summary.total_requests}/{summary.succeeded_requests}/{summary.failed_requests}",
    )
    table.add_row("Success Rate", f"{summary.success_rate:.2%}")
    table.add_row("Total Duration", _format_seconds(summary.total_duration))
    table.add_row("Request Throughput (req/s)", f"{summary.request_throughput:.2f}")
    table.add_row("Concurrency", f"{summary.concurrency:.2f}")

    return table


def build_metrics_table(metrics: Dict[str, Optional[MetricStats]]) -> Optional[Table]:
    """Build the detailed latency metric table when values are available.

    Args:
        metrics (Dict[str, Optional[MetricStats]]): Named metric summaries.

    Returns:
        Optional[Table]: Rich table with percentile statistics, or ``None``
            when no metric contains data.
    """
    rows = [
        (metric_name, stats)
        for metric_name, stats in metrics.items()
        if stats is not None
    ]
    if not rows:
        return None

    table = Table(box=None, pad_edge=False, expand=False)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Mean", style="white", justify="right", no_wrap=True)
    table.add_column("P50", style="white", justify="right", no_wrap=True)
    table.add_column("P90", style="white", justify="right", no_wrap=True)
    table.add_column("P95", style="white", justify="right", no_wrap=True)
    table.add_column("P99", style="white", justify="right", no_wrap=True)

    for metric_name, stats in rows:
        unit = "ms" if metric_name != "latency" else "s"
        table.add_row(
            metric_name,
            _format_seconds(stats.mean, unit),
            _format_seconds(stats.p50, unit),
            _format_seconds(stats.p90, unit),
            _format_seconds(stats.p95, unit),
            _format_seconds(stats.p99, unit),
        )

    return table


def render_bench_summary(summary: BenchSummary) -> None:
    """Render the benchmark summary panel to the terminal.

    Args:
        summary (BenchSummary): Aggregated benchmark summary.

    Returns:
        None: This function writes the summary to the terminal.
    """
    overview_table = build_overview_table(summary)
    metrics_table = build_metrics_table(summary.metrics)

    renderables: List[RenderableType] = [overview_table]
    if metrics_table is not None:
        renderables.append(Text(""))
        renderables.append(metrics_table)

    console.print(
        Panel.fit(
            Group(*renderables),
            title="Benchmark Summary",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def run_bench_command(args: BenchCommandArgs) -> List[LLMResponse]:
    """Run the full benchmark command synchronously.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        List[LLMResponse]: Aggregated responses returned by the backend.
    """
    backend = create_backend(args)
    config = build_config(args)
    dataset = build_dataset(args)

    start_time = time.perf_counter()
    responses = asyncio.run(
        bench_requests(backend, args.url, config, dataset.iter_requests())
    )
    finish_time = time.perf_counter()

    summary = summarize_bench_results(responses, start_time, finish_time)

    render_bench_summary(summary)

    return responses
