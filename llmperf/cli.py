"""CLI entrypoints for llmperf."""

import typer
import asyncio
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

# from llmperf.commands.bench import bench_requests
from llmperf.commands.request import RequestCommandArgs, run_request_command

app = typer.Typer(add_completion=False)


@app.command()
def test() -> None:
    """Run a minimal smoke test for the CLI."""
    print("test ok")


@app.command("bench")
def bench() -> None:
    """Run the built-in benchmarking command with a fixed dataset.

    Returns:
        None: This command does not return a value.
    """

    """
    asyncio.run(
        bench_requests(
            url="http://172.18.16.59:8000/v1/chat/completions",
            file=Path(
                "/data/cgl/download/datasets/boss/zhishanshan/qwen235-2507-fp8/raw_request/request_235b_20min.jsonl"
            ),
            num_requests=18000,
            qps=2,
            max_concurrency=None,
        )
    )
    """
    pass


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
            help="Path to a file containing chat messages. Mutually exclusive with --messages and --user."
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
        typer.Option(help="Sampling temperature for the chat completion request."),
    ] = 1.0,
    max_completion_tokens: Annotated[
        Optional[int],
        typer.Option(help="Maximum number of completion tokens to generate."),
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
        typer.Option(help="Whether to request a streaming response."),
    ] = True,
    timeout: Annotated[float, typer.Option(help="Set timeout for request.")] = 300.0,
) -> None:
    """Send a single chat completion request from the command line.

    Args:
        url (str): Target OpenAI-compatible chat completions endpoint.
        messages (Optional[str]): Chat messages as a JSON array string.
        file (Optional[Path]): Path to a JSON file containing chat messages.
        user (Optional[str]): User prompt used to build a simple message list.
        system (Optional[str]): Optional system prompt paired with ``user``.
        model (str): Model name sent in the request payload.
        rid (Optional[str]): Optional request identifier propagated downstream.
        temperature (float): Sampling temperature for the request.
        max_completion_tokens (Optional[int]): Maximum number of completion
            tokens to generate.
        ignore_eos (bool): Whether to ignore the EOS token during sampling.
        seed (Optional[int]): Optional sampling seed.
        enable_thinking (bool): Whether to enable server-side thinking behavior.
        stream (bool): Whether to request streaming output.
        timeout (float): Request timeout in seconds.

    Returns:
        None: This command does not return a value.

    Raises:
        typer.Exit: Raised with exit code ``1`` when the request fails.
    """

    args = RequestCommandArgs(
        url=url,
        messages_json=messages,
        file=file,
        user=user,
        system=system,
        model=model,
        rid=rid,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        ignore_eos=ignore_eos,
        sampling_seed=seed,
        enable_thinking=enable_thinking,
        stream=stream,
        timeout=timeout,
    )
    response = run_request_command(args)
    if response.error:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
