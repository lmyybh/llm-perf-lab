"""CLI entrypoints for llmperf."""

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from llmperf.commands.bench import (
    BenchCommandArgs,
    BenchCommonArgs,
    DatasetMode,
    build_bench_dataset_args,
    run_bench_command,
)
from llmperf.commands.request import RequestCommandArgs, run_request_command
from llmperf.errors import LLMPerfError

app = typer.Typer(add_completion=False)


def _raise_cli_error(exc: LLMPerfError) -> None:
    """Translate project-level errors into Typer-friendly CLI errors."""
    raise typer.BadParameter(str(exc)) from exc


@app.command("request")
def request(
    url: Annotated[
        str,
        typer.Option(help="Target OpenAI-compatible chat completions endpoint."),
    ] = "http://localhost:32110/v1/chat/completions",
    messages: Annotated[
        Optional[str],
        typer.Option(
            help="Chat messages as a JSON array. Mutually exclusive with --file, --user, and --target-input-tokens."
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
            help="Path to a file containing chat messages. Mutually exclusive with --messages, --user, and --target-input-tokens.",
        ),
    ] = None,
    user: Annotated[
        Optional[str],
        typer.Option(
            help="User prompt text. Mutually exclusive with --messages, --file, and --target-input-tokens."
        ),
    ] = None,
    system: Annotated[
        Optional[str],
        typer.Option(
            help="System prompt text. Only valid when used together with --user."
        ),
    ] = None,
    text: Annotated[Optional[str], typer.Option(help="Generate text")] = None,
    dataset_file: Annotated[
        Optional[Path],
        typer.Option(
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to a dataset file used with --target-input-tokens.",
        ),
    ] = None,
    dataset_mode: Annotated[
        DatasetMode,
        typer.Option(help="Dataset mode used to select one request sample."),
    ] = DatasetMode.openai,
    target_input_tokens: Annotated[
        Optional[int],
        typer.Option(
            min=1,
            help="Target prompt token count for selecting one dataset sample.",
        ),
    ] = None,
    input_token_tolerance: Annotated[
        int,
        typer.Option(
            min=0,
            help="Allowed absolute token tolerance around --target-input-tokens.",
        ),
    ] = 64,
    with_tools: Annotated[
        bool,
        typer.Option(
            "--with-tools/--without-tools",
            help="Select only dataset samples with non-empty tools and include those tools in the request.",
        ),
    ] = False,
    tools: Annotated[
        Optional[str],
        typer.Option(
            help="tools",
        ),
    ] = None,
    tool_choice: Annotated[
        Optional[str],
        typer.Option(
            help="Tool choice mode or JSON object. When omitted, file-provided tool_choice is preserved."
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option(help="Model name sent in the request payload."),
    ] = None,
    tokenizer_path: Annotated[
        Optional[Path],
        typer.Option(
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Optional tokenizer path or directory. Preferred over --model for local prompt token estimation.",
        ),
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
        float,
        typer.Option(min=0.0, max=2.0, help="Repetition penalty for generated tokens."),
    ] = 1.0,
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
        text=text,
        dataset_file=dataset_file,
        dataset_mode=dataset_mode,
        target_input_tokens=target_input_tokens,
        input_token_tolerance=input_token_tolerance,
        with_tools=with_tools,
        tools=tools,
        tool_choice=tool_choice,
        model=model,
        tokenizer_path=tokenizer_path,
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
    try:
        response = run_request_command(args)
    except LLMPerfError as exc:
        _raise_cli_error(exc)
    if response.error:
        raise typer.Exit(code=1)


@app.command("bench")
def bench(
    url: Annotated[
        str,
        typer.Option(help="Target OpenAI-compatible chat completions endpoint."),
    ] = "http://localhost:32110/v1/chat/completions",
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
    tokenizer_path: Annotated[
        Optional[Path],
        typer.Option(
            file_okay=True,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Optional tokenizer path or directory. Random mode prefers this over --model.",
        ),
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
    seed: Annotated[
        int,
        typer.Option(help="Random seed used by the random dataset mode."),
    ] = 0,
    min_input_tokens: Annotated[
        int,
        typer.Option(min=1, help="Minimum prompt token count for random mode."),
    ] = 64,
    max_input_tokens: Annotated[
        int,
        typer.Option(min=1, help="Maximum prompt token count for random mode."),
    ] = 256,
    min_output_tokens: Annotated[
        int,
        typer.Option(min=1, help="Minimum completion token count for random mode."),
    ] = 64,
    max_output_tokens: Annotated[
        int,
        typer.Option(min=1, help="Maximum completion token count for random mode."),
    ] = 256,
    file: Annotated[
        Optional[Path],
        typer.Option(
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to a dataset file. Required for file-backed dataset modes.",
        ),
    ] = None,
    mode: Annotated[
        DatasetMode,
        typer.Option(help="Dataset mode used to parse the input file."),
    ] = DatasetMode.openai,
) -> None:
    """Run benchmark requests against an OpenAI-compatible endpoint."""

    common_args = BenchCommonArgs(
        url=url,
        num_requests=num_requests,
        qps=qps,
        max_concurrency=max_concurrency,
        timeout=timeout,
        model=model,
        tokenizer_path=tokenizer_path,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        ignore_eos=ignore_eos,
        enable_thinking=enable_thinking,
    )
    dataset_args = build_bench_dataset_args(
        mode,
        file=file,
        seed=seed,
        min_input_tokens=min_input_tokens,
        max_input_tokens=max_input_tokens,
        min_output_tokens=min_output_tokens,
        max_output_tokens=max_output_tokens,
    )

    args = BenchCommandArgs(common=common_args, dataset=dataset_args)

    try:
        run_bench_command(args)
    except LLMPerfError as exc:
        _raise_cli_error(exc)


if __name__ == "__main__":
    app()
