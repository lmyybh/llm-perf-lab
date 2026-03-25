"""Helpers for building and running the benchmark command."""

import asyncio
from enum import Enum
from pathlib import Path
import time
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import typer
from pydantic import BaseModel, Field
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from llmperf.backends import LLMBackend, OpenAIChatBackend
from llmperf.benchmarks import BenchSummary, MetricStats, summarize_bench_results
from llmperf.common import LLMRequest, LLMResponse
from llmperf.datasets import Dataset, FileDataset, RandomDataset
from llmperf.common import apply_prompt_token_fallback, load_tokenizer

console = Console()


class DatasetMode(str, Enum):
    """Supported dataset modes for the benchmark command."""

    openai = "openai-jsonl"
    zss = "zss-jsonl"
    random = "random"


class BenchCommonArgs(BaseModel):
    """Common arguments for benchmark execution.

    Attributes:
        url (str): Target endpoint URL.
        num_requests (Optional[int]): Optional request count limit.
        qps (Optional[float]): Optional request rate cap.
        max_concurrency (Optional[int]): Optional in-flight request cap.
        timeout (float): Request timeout in seconds.
        model (Optional[str]): Model name sent in benchmark requests.
        tokenizer_path (Optional[Path]): Optional local or remote tokenizer path
            used by the random dataset.
        temperature (Optional[float]): Optional temperature override.
        max_completion_tokens (Optional[int]): Optional max token override.
        ignore_eos (Optional[bool]): Optional EOS override.
        enable_thinking (Optional[bool]): Optional thinking override.
    """

    url: str

    num_requests: Optional[int] = Field(default=None, gt=0)
    qps: Optional[float] = Field(default=None, gt=0)
    max_concurrency: Optional[int] = Field(default=None, gt=0)
    timeout: float = Field(default=300.0, gt=0)

    model: Optional[str] = None
    tokenizer_path: Optional[Path] = None
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    ignore_eos: Optional[bool] = None
    enable_thinking: Optional[bool] = None


class FileDatasetArgs(BaseModel):
    """Arguments for file-backed benchmark datasets.

    Attributes:
        mode (Literal["openai-jsonl", "zss-jsonl"]): Dataset file mode.
        file (Path): Dataset file path.
    """

    mode: Literal["openai-jsonl", "zss-jsonl"]
    file: Path


class RandomDatasetArgs(BaseModel):
    """Arguments for randomly generated benchmark datasets.

    Attributes:
        mode (Literal["random"]): Random dataset mode discriminator.
        seed (int): Random seed for request generation.
        min_input_tokens (int): Minimum input token count per request.
        max_input_tokens (int): Maximum input token count per request.
        min_output_tokens (int): Minimum output token target per request.
        max_output_tokens (int): Maximum output token target per request.
    """

    mode: Literal["random"]
    seed: int = 0
    min_input_tokens: int = Field(default=64, gt=0)
    max_input_tokens: int = Field(default=256, gt=0)
    min_output_tokens: int = Field(default=64, gt=0)
    max_output_tokens: int = Field(default=256, gt=0)


DatasetArgs = FileDatasetArgs | RandomDatasetArgs


class BenchCommandArgs(BaseModel):
    """Arguments required to run the benchmark command.

    Attributes:
        common (BenchCommonArgs): Benchmark runtime configuration.
        dataset (DatasetArgs): Dataset-specific configuration.
    """

    common: BenchCommonArgs
    dataset: DatasetArgs


def build_bench_dataset_args(
    mode: DatasetMode,
    file: Optional[Path],
    seed: int = 0,
    min_input_tokens: int = 64,
    max_input_tokens: int = 256,
    min_output_tokens: int = 64,
    max_output_tokens: int = 256,
) -> DatasetArgs:
    """Build dataset arguments for the selected benchmark mode.

    Args:
        mode (DatasetMode): Dataset mode selected from the CLI.
        file (Optional[Path]): Optional dataset file path.
        seed (int): Random seed for random mode generation.
        min_input_tokens (int): Minimum input tokens for random mode.
        max_input_tokens (int): Maximum input tokens for random mode.
        min_output_tokens (int): Minimum output tokens for random mode.
        max_output_tokens (int): Maximum output tokens for random mode.

    Returns:
        DatasetArgs: Dataset configuration matching the selected mode.

    Raises:
        typer.BadParameter: Raised when required file input is missing.
    """
    if mode in {DatasetMode.openai, DatasetMode.zss}:
        if file is None or not file.is_file():
            raise typer.BadParameter("")

        return FileDatasetArgs(mode=mode.value, file=file)

    if mode is DatasetMode.random:
        return RandomDatasetArgs(
            mode=mode.value,
            seed=seed,
            min_input_tokens=min_input_tokens,
            max_input_tokens=max_input_tokens,
            min_output_tokens=min_output_tokens,
            max_output_tokens=max_output_tokens,
        )

    raise typer.BadParameter("unsupport")


def load_random_tokenizer(
    tokenizer_path: Optional[Path], model_name: Optional[str]
) -> PreTrainedTokenizerBase:
    """Load the tokenizer used by the random benchmark dataset.

    Args:
        tokenizer_path (Optional[Path]): Preferred tokenizer path or identifier.
        model_name (Optional[str]): Fallback model identifier accepted by
            ``transformers``.

    Returns:
        PreTrainedTokenizerBase: Loaded tokenizer instance.

    Raises:
        typer.BadParameter: Raised when no tokenizer source is available or the
            tokenizer cannot be loaded.
    """
    tokenizer = load_tokenizer(
        tokenizer_path=tokenizer_path,
        model_name=model_name,
        purpose="random dataset",
        required=True,
    )
    assert tokenizer is not None
    return tokenizer


def override_request(request: LLMRequest, config: BenchCommonArgs) -> LLMRequest:
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
    backend: LLMBackend,
    url: str,
    config: BenchCommonArgs,
    requests: Iterable[LLMRequest],
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

        result = apply_prompt_token_fallback(request, result)

        return index, result

    prepared_requests: List[Tuple[int, LLMRequest]] = [
        (index, override_request(request, config))
        for index, request in enumerate(requests)
    ]

    progress = tqdm(total=len(prepared_requests))
    tasks: List[asyncio.Task[Tuple[int, LLMResponse]]] = []

    def _update_progress(_task: asyncio.Task[Tuple[int, LLMResponse]]) -> None:
        progress.update(1)

    interval = 1.0 / config.qps if config.qps is not None and config.qps > 0 else 0.0
    for index, request in prepared_requests:
        task = asyncio.create_task(_send_one(index, request))
        task.add_done_callback(_update_progress)
        tasks.append(task)
        if interval > 0:
            await asyncio.sleep(interval)

    results: List[Tuple[int, LLMResponse]] = []
    try:
        for task in asyncio.as_completed(tasks):
            results.append(await task)
    finally:
        progress.close()

    results.sort(key=lambda x: x[0])

    return [r[1] for r in results]


def create_backend(args: BenchCommandArgs) -> LLMBackend:
    """Create the backend instance for the benchmark command.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        LLMBackend: Configured backend instance.
    """
    return OpenAIChatBackend(timeout=args.common.timeout)


def build_dataset(args: BenchCommandArgs) -> Dataset:
    """Create the dataset adapter used by the benchmark command.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        Dataset: Dataset capable of yielding benchmark requests.

    Raises:
        AssertionError: Raised when ``args.file`` is missing.
    """

    dataset_args = args.dataset

    if isinstance(dataset_args, FileDatasetArgs):
        return FileDataset(
            file=dataset_args.file,
            mode=dataset_args.mode,
            num_requests=args.common.num_requests,
        )

    if isinstance(dataset_args, RandomDatasetArgs):
        if args.common.num_requests is None:
            raise typer.BadParameter(
                "random dataset requires --num-requests to be specified"
            )

        return RandomDataset(
            tokenizer=load_random_tokenizer(
                tokenizer_path=args.common.tokenizer_path,
                model_name=args.common.model,
            ),
            num_requests=args.common.num_requests,
            min_input_tokens=dataset_args.min_input_tokens,
            max_input_tokens=dataset_args.max_input_tokens,
            min_output_tokens=dataset_args.min_output_tokens,
            max_output_tokens=dataset_args.max_output_tokens,
            seed=dataset_args.seed,
        )

    raise typer.BadParameter("unsupported dataset args")


def _format_seconds(value: float, unit: Literal["s", "ms"] = "s") -> str:
    if unit == "s":
        return f"{value:.2f} s"
    else:
        return f"{(value*1000):.2f} ms"


def _format_request_counts(summary: BenchSummary) -> Text:
    rendered = Text()
    rendered.append(f"{summary.total_requests} total", style="white")
    rendered.append(" / ", style="dim")
    rendered.append(f"{summary.succeeded_requests} ok", style="green")
    rendered.append(" / ", style="dim")
    failure_style = "red" if summary.failed_requests > 0 else "dim"
    rendered.append(f"{summary.failed_requests} fail", style=failure_style)
    return rendered


def build_run_table(summary: BenchSummary) -> Table:
    """Build the benchmark run overview table.

    Args:
        summary (BenchSummary): Aggregated benchmark summary.

    Returns:
        Table: Rich table containing run-level metrics.
    """
    table = Table(show_header=False, box=None, pad_edge=False, expand=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white", no_wrap=True)

    success_style = "green" if summary.failed_requests == 0 else "yellow"

    table.add_row("Requests", _format_request_counts(summary))
    table.add_row(
        "Success Rate", Text(f"{summary.success_rate:.2%}", style=success_style)
    )
    table.add_row("Total Duration", _format_seconds(summary.total_duration))
    qps_value = f"{summary.qps:.2f}" if summary.qps is not None else "inf"
    table.add_row("QPS", qps_value)
    max_concurrency_value = (
        str(summary.max_concurrency) if summary.max_concurrency is not None else "inf"
    )
    table.add_row("Max Concurrency", max_concurrency_value)
    table.add_row("Requests Throughput", f"{summary.request_throughput:.2f} req/s")
    table.add_row("Concurrency", f"{summary.concurrency:.2f}")

    return table


def build_token_table(summary: BenchSummary) -> Table:
    """Build the compact token summary section.

    Args:
        summary (BenchSummary): Aggregated benchmark summary.

    Returns:
        Table: Rich table containing token summary rows.
    """
    table = Table(show_header=False, box=None, pad_edge=False, expand=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white", no_wrap=True)

    total_width = max(
        len(str(summary.total_prompt_tokens)), len(str(summary.total_completion_tokens))
    )
    mean_width = max(
        len(str(summary.mean_prompt_tokens)), len(str(summary.mean_completion_tokens))
    )

    prompt_text = Text()
    prompt_text.append(f"{summary.total_prompt_tokens:>{total_width}}", style="white")
    prompt_text.append(" total / ", style="dim")
    prompt_text.append(f"{summary.mean_prompt_tokens:>{mean_width}}", style="white")
    prompt_text.append(" mean", style="dim")

    completion_text = Text()
    completion_text.append(
        f"{summary.total_completion_tokens:>{total_width}}", style="white"
    )
    completion_text.append(" total / ", style="dim")
    completion_text.append(
        f"{summary.mean_completion_tokens:>{mean_width}}", style="white"
    )
    completion_text.append(" mean", style="dim")

    table.add_row(
        "Prompt Tokens",
        prompt_text,
    )
    table.add_row(
        "Completion Tokens",
        completion_text,
    )

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
        if metric_name == "ttft":
            label = "TTFT"
        elif metric_name == "tpot":
            label = "TPOT"
        else:
            label = metric_name.capitalize()
        table.add_row(
            label,
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
    run_table = build_run_table(summary)
    token_table = build_token_table(summary)
    metrics_table = build_metrics_table(summary.metrics)

    renderables: List[RenderableType] = [run_table, Text(""), token_table]
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
    dataset = build_dataset(args)

    start_time = time.perf_counter()
    responses = asyncio.run(
        bench_requests(backend, args.common.url, args.common, dataset.iter_requests())
    )
    finish_time = time.perf_counter()

    summary = summarize_bench_results(
        responses,
        start_time,
        finish_time,
        qps=args.common.qps,
        max_concurrency=args.common.max_concurrency,
    )

    render_bench_summary(summary)

    return responses
