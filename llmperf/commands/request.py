"""Helpers for building and running the request command."""

import asyncio
import json
from pathlib import Path
from typing import Callable, Literal, Optional

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from llmperf.core.models import (
    ChatCompletionInput,
    ChatCompletionMessage,
    LLMRequest,
    LLMResponse,
    SamplingParams,
)
from llmperf.backends.base import LLMBackend
from llmperf.backends.openai import OpenAIChatBackend

DEFAULT_SYSTEM_PROMPT = "你是一个专业的助手"
InputMode = Literal["messages", "file", "user"]
console = Console()


def handle_messages_mode(messages_json: str) -> ChatCompletionInput:
    """Parse chat messages from a JSON string.

    Args:
        messages_json (str): JSON-encoded message list from the CLI.

    Returns:
        ChatCompletionInput: Parsed chat completion input.

    Raises:
        typer.BadParameter: Raised when ``messages_json`` is not valid JSON.
    """
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON for --messages: {exc.msg}") from exc

    return ChatCompletionInput(messages=messages)


def handle_file_mode(file: Path) -> ChatCompletionInput:
    """Load chat messages from a JSON file.

    Args:
        file (Path): Path to a JSON file containing chat messages.

    Returns:
        ChatCompletionInput: Parsed chat completion input.

    Raises:
        typer.BadParameter: Raised when the file does not exist, is not a
            regular file, or contains invalid JSON.
    """

    if not file.exists():
        raise typer.BadParameter(f"message file not found: {file}")

    if not file.is_file():
        raise typer.BadParameter(f"message file is not a regular file: {file}")

    try:
        with file.open() as f:
            messages = json.load(f)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON for --file: {exc.msg}") from exc

    return ChatCompletionInput(messages=messages)


def handle_prompt_mode(user: str, system: Optional[str] = None) -> ChatCompletionInput:
    """Build chat messages from plain prompt text.

    Args:
        user (str): User prompt text.
        system (Optional[str]): Optional system prompt text.

    Returns:
        ChatCompletionInput: Chat completion input with system and user messages.
    """
    if system is None:
        system = DEFAULT_SYSTEM_PROMPT

    return ChatCompletionInput(
        messages=[
            ChatCompletionMessage(role="system", content=system),
            ChatCompletionMessage(role="user", content=user),
        ]
    )


class RequestCommandArgs(BaseModel):
    """Arguments required to run the request command.

    Attributes:
        url (str): Target OpenAI-compatible chat completions endpoint.
        messages_json (Optional[str]): JSON-encoded message list from the CLI.
        file (Optional[Path]): Optional file containing JSON chat messages.
        user (Optional[str]): Optional user prompt text.
        system (Optional[str]): Optional system prompt text.
        model (Optional[str]): Optional model name to send downstream.
        rid (Optional[str]): Optional request identifier.
        temperature (float): Sampling temperature for the request.
        presence_penalty (float): Penalty applied to tokens based on prior
            presence.
        frequency_penalty (float): Penalty applied to tokens based on prior
            frequency.
        repetition_penalty (Optional[float]): Optional repetition penalty for
            generated tokens.
        max_completion_tokens (Optional[int]): Maximum number of completion
            tokens to generate.
        ignore_eos (bool): Whether to ignore EOS during sampling.
        seed (Optional[int]): Optional sampling seed.
        enable_thinking (bool): Whether to enable server-side thinking behavior.
        stream (bool): Whether to request a streaming response.
        timeout (float): Request timeout in seconds.
    """

    url: str

    messages_json: Optional[str] = None
    file: Optional[Path] = None
    user: Optional[str] = None
    system: Optional[str] = None

    model: Optional[str] = None
    rid: Optional[str] = None

    temperature: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: Optional[float] = None
    max_completion_tokens: Optional[int] = 128
    ignore_eos: bool = False
    seed: Optional[int] = None

    enable_thinking: bool = True
    stream: bool = True

    timeout: float = 300

    def detect_input_mode(self) -> InputMode:
        """Determine which mutually exclusive input mode is active.

        Returns:
            InputMode: The selected input mode.

        Raises:
            typer.BadParameter: Raised when zero or multiple input modes are
                selected, or when ``system`` is used without ``user``.
        """
        has_messages = (
            self.messages_json is not None and self.messages_json.strip() != ""
        )
        has_file = self.file is not None
        has_user = self.user is not None and self.user.strip() != ""

        selected: list[tuple[InputMode, bool]] = [
            ("messages", has_messages),
            ("file", has_file),
            ("user", has_user),
        ]
        enabled_modes: list[InputMode] = [name for name, ok in selected if ok]

        if len(enabled_modes) == 0:
            raise typer.BadParameter(
                "must provide exactly one of --messages, --file, or --user"
            )

        if len(enabled_modes) > 1:
            raise typer.BadParameter(
                "options --messages, --file, and --user are mutually exclusive"
            )

        if self.system is not None and not has_user:
            raise typer.BadParameter("--system can only be used with --user")

        return enabled_modes[0]

    def parse_input(self) -> ChatCompletionInput:
        """Parse command arguments into chat completion input.

        Returns:
            ChatCompletionInput: Parsed chat completion input.
        """
        mode = self.detect_input_mode()

        if mode == "messages":
            assert self.messages_json is not None
            messages = handle_messages_mode(self.messages_json)
        elif mode == "file":
            assert self.file is not None
            messages = handle_file_mode(self.file)
        else:
            assert self.user is not None
            messages = handle_prompt_mode(self.user, self.system)

        return messages

    def parse_sampling_params(self) -> SamplingParams:
        """Build sampling parameters for the outgoing request.

        Returns:
            SamplingParams: Sampling parameters derived from command arguments.
        """
        return SamplingParams(
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            ignore_eos=self.ignore_eos,
            seed=self.seed,
        )

    def build_llm_request(self) -> LLMRequest:
        """Build the internal request model for the backend.

        Returns:
            LLMRequest: Internal request object passed to the backend.
        """
        return LLMRequest(
            input=self.parse_input(),
            sampling_params=self.parse_sampling_params(),
            model=self.model,
            stream=self.stream,
            rid=self.rid,
            chat_template_kwargs={"enable_thinking": self.enable_thinking},
            extra={},
        )


def create_backend(args: RequestCommandArgs) -> LLMBackend:
    """Create the backend instance for the request command.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.

    Returns:
        LLMBackend: Configured backend instance.
    """
    return OpenAIChatBackend(timeout=args.timeout)


def create_chunk_printer(stream: bool) -> Optional[Callable[[str], None]]:
    """Create a chunk printer for streaming output.

    Args:
        stream (bool): Whether the request uses streaming output.

    Returns:
        Optional[Callable[[str], None]]: A chunk printer for streamed chunks,
        otherwise ``None``.
    """
    if not stream:
        return None

    def _print_chunk(chunk: str) -> None:
        """Print one streamed response chunk.

        Args:
            chunk (str): Incremental text returned by the backend.

        Returns:
            None: This function writes one chunk to the terminal.
        """
        console.print(chunk, end="")

    return _print_chunk


def render_header(stream: bool) -> None:
    """Render the request mode header.

    Args:
        stream (bool): Whether the request uses streaming output.

    Returns:
        None: This function writes the header to the terminal.
    """
    console.rule("Stream") if stream else console.rule("No-Stream")


def build_response_summary(response: LLMResponse) -> Table:
    """Build a summary table for the aggregated response.

    Args:
        response (LLMResponse): Aggregated backend response.

    Returns:
        Table: A terminal table containing key response metadata.
    """
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Status", str(response.status_code))

    if response.model:
        table.add_row("Model", response.model)

    table.add_row("Latency", f"{response.latency:.2f} s")

    if response.ttft > 0:
        table.add_row("TTFT", f"{(response.ttft*1000):.2f} ms")

    if response.tpot > 0:
        table.add_row("TPOT", f"{(response.tpot*1000):.2f} ms")

    if response.prompt_tokens > 0:
        table.add_row("Prompt Tokens", str(response.prompt_tokens))

    if response.completion_tokens > 0:
        table.add_row("Completion Tokens", str(response.completion_tokens))

    if response.finish_reason:
        table.add_row("Finish Reason", response.finish_reason)

    return table


def get_response_text(response: LLMResponse) -> str:
    """Build the visible response text for non-stream output.

    Args:
        response (LLMResponse): Aggregated backend response.

    Returns:
        str: Reasoning text followed by visible content when present.
    """
    text = response.output.reasoning_content or ""
    if text:
        text += "\n"
    text += response.output.content or ""
    return text


def render_response(response: LLMResponse, stream: bool) -> None:
    """Render the response body, errors, and summary sections.

    Args:
        response (LLMResponse): Aggregated backend response.
        stream (bool): Whether the response body was streamed.

    Returns:
        None: This function writes the response to the terminal.
    """
    if stream:
        console.print()
    else:
        text = get_response_text(response)
        if text:
            console.print(text)

    console.print()

    if response.error:
        console.rule("Error")
        console.print(response.error, style="bold red")

    console.rule("Response")
    console.print(build_response_summary(response))
    console.rule()


def run_request_command(args: RequestCommandArgs) -> LLMResponse:
    """Run the request command and return the aggregated response.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.

    Returns:
        LLMResponse: Aggregated backend response.
    """
    backend = create_backend(args)

    render_header(args.stream)

    response = asyncio.run(
        backend.send(
            url=args.url,
            request=args.build_llm_request(),
            on_chunk=create_chunk_printer(args.stream),
        )
    )

    render_response(response, stream=args.stream)

    return response
