import typer
import asyncio
from pathlib import Path
from typing_extensions import Annotated

from llmperf.commands.request import request_v1_chat

app = typer.Typer(add_completion=False)


@app.command()
def test():
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
            "--enable-thinking/--disable-thinking", help="Whether to enable think."
        ),
    ] = True,
    stream: Annotated[
        bool,
        typer.Option(help="Whether to request a streaming response."),
    ] = True,
):
    result = asyncio.run(
        request_v1_chat(
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
        )
    )

    if result.error:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
