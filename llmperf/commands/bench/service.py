"""Execution logic for the benchmark command."""

import asyncio
import time
from pathlib import Path
from typing import Iterable, Optional

import typer
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from llmperf.backends import LLMBackend, OpenAIChatBackend
from llmperf.benchmarks import BenchSummary, summarize_bench_results
from llmperf.commands.bench.args import (
    BenchCommandArgs,
    BenchCommonArgs,
    FileDatasetArgs,
    RandomDatasetArgs,
)
from llmperf.common import (
    LLMRequest,
    LLMResponse,
    apply_prompt_token_fallback,
    load_tokenizer,
)
from llmperf.datasets import Dataset, FileDataset, RandomDataset

BenchResultWithIndex = tuple[int, LLMResponse]


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
        config (BenchCommonArgs): Benchmark configuration containing overrides.

    Returns:
        LLMRequest: Request copy with benchmark overrides applied.
    """
    sampling_update: dict[str, float | int | bool] = {}
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

    update_data: dict[str, object] = {
        "sampling_params": sampling_params,
        "chat_template_kwargs": chat_template_kwargs,
        "stream": True,
    }
    if config.model is not None:
        update_data["model"] = config.model

    return request.model_copy(update=update_data)


async def bench_requests(
    backend: LLMBackend,
    url: str,
    config: BenchCommonArgs,
    requests: Iterable[LLMRequest],
) -> list[LLMResponse]:
    """Send benchmark requests and preserve input order in the results.

    Args:
        backend (LLMBackend): Backend used to send requests.
        url (str): Target endpoint URL.
        config (BenchCommonArgs): Benchmark runtime configuration.
        requests (Iterable[LLMRequest]): Requests yielded by the dataset.

    Returns:
        list[LLMResponse]: Responses ordered to match the input sequence.
    """
    semaphore = (
        asyncio.Semaphore(config.max_concurrency)
        if config.max_concurrency is not None and config.max_concurrency > 0
        else None
    )

    async def _send_one(index: int, request: LLMRequest) -> BenchResultWithIndex:
        """Send one request and retain its original index.

        Args:
            index (int): Input position of the request.
            request (LLMRequest): Request payload to send.

        Returns:
            BenchResultWithIndex: Input index paired with its response.
        """
        if semaphore is not None:
            async with semaphore:
                result = await backend.send(url, request)
        else:
            result = await backend.send(url, request)

        result = apply_prompt_token_fallback(request, result)
        return index, result

    prepared_requests: list[tuple[int, LLMRequest]] = [
        (index, override_request(request, config))
        for index, request in enumerate(requests)
    ]

    progress = tqdm(total=len(prepared_requests))
    tasks: list[asyncio.Task[BenchResultWithIndex]] = []

    def _update_progress(_task: asyncio.Task[BenchResultWithIndex]) -> None:
        progress.update(1)

    interval = 1.0 / config.qps if config.qps is not None and config.qps > 0 else 0.0
    for index, request in prepared_requests:
        task = asyncio.create_task(_send_one(index, request))
        task.add_done_callback(_update_progress)
        tasks.append(task)
        if interval > 0:
            await asyncio.sleep(interval)

    results: list[BenchResultWithIndex] = []
    try:
        for task in asyncio.as_completed(tasks):
            results.append(await task)
    finally:
        progress.close()

    results.sort(key=lambda item: item[0])
    return [response for _, response in results]


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
        typer.BadParameter: Raised when random mode omits ``num_requests`` or the
            dataset arguments are unsupported.
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


def execute_bench(args: BenchCommandArgs) -> tuple[list[LLMResponse], BenchSummary]:
    """Run the full benchmark command and compute the summary.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        tuple[list[LLMResponse], BenchSummary]: Responses and aggregated summary.
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
    return responses, summary
