"""CLI entrypoints for llmperf."""

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from llmperf.commands.bench import BenchCommandArgs, run_bench_command
from llmperf.commands.request import RequestCommandArgs, run_request_command

app = typer.Typer(add_completion=False)


@app.command()
def test() -> None:
    """Run a minimal CLI smoke test."""
    print("test ok")


class DataMode(str, Enum):
    """Supported dataset modes for the benchmark command."""

    openai = "openai-jsonl"
    zss = "zss-jsonl"
    random = "random"


@app.command("bench")
def bench(
    url: Annotated[
        str,
        typer.Option(help="Target OpenAI-compatible chat completions endpoint."),
    ] = "http://localhost:32110/v1/chat/completions",
    file: Annotated[
        Optional[Path],
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to a dataset file.",
        ),
    ] = None,
    mode: Annotated[
        DataMode,
        typer.Option(help="Dataset mode used to parse the input file."),
    ] = DataMode.openai,
    num_requests: Annotated[
        Optional[int],
        typer.Option(min=1, help="Maximum number of requests to read and send."),
    ] = None,
    qps: Annotated[
        Optional[float],
        typer.Option(
            min=0.01, help="Optional request rate limit in queries per second."
        ),
    ] = None,
    max_concurrency: Annotated[
        Optional[int],
        typer.Option(min=1, help="Optional maximum number of in-flight requests."),
    ] = None,
    timeout: Annotated[
        float, typer.Option(min=0.01, help="Set timeout for request.")
    ] = 300.0,
    model: Annotated[
        Optional[str],
        typer.Option(help="Model name sent in the request payload."),
    ] = None,
    temperature: Annotated[
        Optional[float],
        typer.Option(
            min=0.0, help="Sampling temperature for the chat completion request."
        ),
    ] = None,
    max_completion_tokens: Annotated[
        Optional[int],
        typer.Option(min=1, help="Maximum number of completion tokens to generate."),
    ] = None,
    enable_thinking: Annotated[
        Optional[bool],
        typer.Option(
            "--enable-thinking/--disable-thinking", help="Whether to enable thinking."
        ),
    ] = None,
    ignore_eos: Annotated[
        Optional[bool],
        typer.Option("--ignore-eos", help="Whether to ignore the EOS token."),
    ] = None,
) -> None:
    """Run benchmark requests against an OpenAI-compatible endpoint."""
    args = BenchCommandArgs(
        url=url,
        file=file,
        mode=mode.value,
        num_requests=num_requests,
        qps=qps,
        max_concurrency=max_concurrency,
        timeout=timeout,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        ignore_eos=ignore_eos,
        enable_thinking=enable_thinking,
    )

    run_bench_command(args)


@app.command("request")
def request(
    url: Annotated[
        str,
        typer.Option(help="Target OpenAI-compatible chat completions endpoint."),
    ] = "http://localhost:32110/v1/chat/completions",
    messages: Annotated[
        Optional[str],
        typer.Option(
            help="Chat messages as a JSON array. Mutually exclusive with --file and --user."
        ),
    ] = None,
    file: Annotated[
        Optional[Path],
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to a file containing chat messages. Mutually exclusive with --messages and --user.",
        ),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(
            help="User prompt text. Mutually exclusive with --messages and --file."
        ),
    ] = None,
    system: Annotated[
        Optional[str],
        typer.Option(
            help="System prompt text. Only valid when used together with --user."
        ),
    ] = None,
    tools: Annotated[
        Optional[str],
        typer.Option(
            help="tools",
        ),
    ] = None,
    tool_choice: Annotated[str, typer.Option()] = "auto",
    model: Annotated[
        Optional[str],
        typer.Option(help="Model name sent in the request payload."),
    ] = None,
    rid: Annotated[
        Optional[str],
        typer.Option(help="Optional request identifier included in the payload."),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            min=0.0, help="Sampling temperature for the chat completion request."
        ),
    ] = 1.0,
    presence_penalty: Annotated[
        float,
        typer.Option(
            min=-2.0, max=2.0, help="Penalty applied to tokens based on prior presence."
        ),
    ] = 0.0,
    frequency_penalty: Annotated[
        float,
        typer.Option(
            min=-2.0,
            max=2.0,
            help="Penalty applied to tokens based on prior frequency.",
        ),
    ] = 0.0,
    repetition_penalty: Annotated[
        Optional[float],
        typer.Option(
            min=0.0, max=2.0, help="Optional repetition penalty for generated tokens."
        ),
    ] = None,
    max_completion_tokens: Annotated[
        Optional[int],
        typer.Option(min=1, help="Maximum number of completion tokens to generate."),
    ] = None,
    ignore_eos: Annotated[
        bool,
        typer.Option("--ignore-eos", help="Whether to ignore the EOS token."),
    ] = False,
    seed: Annotated[
        Optional[int],
        typer.Option(help="Optional random seed for sampling."),
    ] = None,
    enable_thinking: Annotated[
        bool,
        typer.Option(
            "--enable-thinking/--disable-thinking", help="Whether to enable thinking."
        ),
    ] = True,
    stream: Annotated[
        bool,
        typer.Option(help="Whether to request and render a streaming response."),
    ] = True,
    timeout: Annotated[
        float, typer.Option(min=0.01, help="Set timeout for request.")
    ] = 300.0,
) -> None:
    """Send a single chat completion request from the command line."""
    args = RequestCommandArgs(
        url=url,
        messages_json=messages,
        file=file,
        user=user,
        system=system,
        tools=tools,
        tool_choice=tool_choice,
        model=model,
        rid=rid,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        max_completion_tokens=max_completion_tokens,
        ignore_eos=ignore_eos,
        seed=seed,
        enable_thinking=enable_thinking,
        stream=stream,
        timeout=timeout,
    )
    response = run_request_command(args)
    if response.error:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
