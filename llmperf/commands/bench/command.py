"""Entrypoint orchestration for the benchmark command."""

from llmperf.commands.bench.args import BenchCommandArgs
from llmperf.commands.bench.render import render_bench_summary
from llmperf.commands.bench.service import execute_bench
from llmperf.common import LLMResponse


def run_bench_command(args: BenchCommandArgs) -> list[LLMResponse]:
    """Run the full benchmark command synchronously.

    Args:
        args (BenchCommandArgs): Parsed benchmark command arguments.

    Returns:
        list[LLMResponse]: Aggregated responses returned by the backend.
    """
    responses, summary = execute_bench(args)
    render_bench_summary(summary)
    return responses
