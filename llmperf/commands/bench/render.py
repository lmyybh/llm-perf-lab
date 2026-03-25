"""Terminal rendering helpers for the benchmark command."""

from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmperf.benchmarks import BenchSummary, MetricStats

console = Console()


def _format_seconds(value: float, unit: str = "s") -> str:
    """Format one metric value in seconds or milliseconds.

    Args:
        value (float): Raw metric value in seconds.
        unit (str): Target display unit, either ``"s"`` or ``"ms"``.

    Returns:
        str: Formatted metric string.
    """
    if unit == "s":
        return f"{value:.2f} s"
    return f"{(value * 1000):.2f} ms"


def _format_request_counts(summary: BenchSummary) -> Text:
    """Build the request count line for the run overview.

    Args:
        summary (BenchSummary): Aggregated benchmark summary.

    Returns:
        Text: Styled request count line.
    """
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

    table.add_row("Prompt Tokens", prompt_text)
    table.add_row("Completion Tokens", completion_text)

    return table


def build_metrics_table(metrics: dict[str, Optional[MetricStats]]) -> Optional[Table]:
    """Build the detailed latency metric table when values are available.

    Args:
        metrics (dict[str, Optional[MetricStats]]): Named metric summaries.

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

    renderables: list[RenderableType] = [run_table, Text(""), token_table]
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
