"""CLI entrypoints for llmperf."""

import typer
import asyncio
from pathlib import Path
from typing_extensions import Annotated

from llmperf.commands.request import openai_chat
from llmperf.commands.bench import send_by_qps

app = typer.Typer(add_completion=False)


@app.command()
def test() -> None:
    """Run a minimal smoke test for the CLI."""
    print("test ok")


@app.command("request")
def request(
    messages: Annotated[
        str | None,
        typer.Option(
            help="Chat messages as a JSON array. Mutually exclusive with --file and --user."
        ),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            help="Path to a file containing chat messages. Mutually exclusive with --messages and --user."
        ),
    ] = None,
    user: Annotated[
        str | None,
        typer.Option(
            help="User prompt text. Mutually exclusive with --messages and --file."
        ),
    ] = None,
    system: Annotated[
        str | None,
        typer.Option(
            help="System prompt text. Only valid when used together with --user."
        ),
    ] = None,
    url: Annotated[
        str,
        typer.Option(help="Target OpenAI-compatible chat completions endpoint."),
    ] = "http://localhost:32110/v1/chat/completions",
    model: Annotated[
        str,
        typer.Option(help="Model name sent in the request payload."),
    ] = "unknown",
    temperature: Annotated[
        float,
        typer.Option(help="Sampling temperature for the chat completion request."),
    ] = 1.0,
    max_tokens: Annotated[
        int | None,
        typer.Option(help="Maximum number of tokens to generate."),
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
        messages (str | None): Chat messages encoded as a JSON array string.
        file (Path | None): Path to a JSON file containing chat messages.
        user (str | None): User prompt used to build a simple chat message list.
        system (str | None): Optional system prompt paired with ``user``.
        url (str): Target OpenAI-compatible chat completions endpoint.
        model (str): Model name sent in the request payload.
        temperature (float): Sampling temperature for the request.
        max_tokens (int | None): Optional maximum number of output tokens.
        enable_thinking (bool): Whether to enable server-side thinking behavior.
        stream (bool): Whether to request streaming output.
        timeout (float): Set timeout for request.

    Raises:
        typer.Exit: Raised with exit code ``1`` when the request fails.
    """
    result = asyncio.run(
        openai_chat(
            messages,
            file,
            user,
            system,
            url,
            model,
            temperature,
            max_tokens,
            enable_thinking,
            stream,
            timeout,
        )
    )

    if result.error:
        raise typer.Exit(code=1)


@app.command("bench")
def bench():
    asyncio.run(send_by_qps(2))


if __name__ == "__main__":
    app()
