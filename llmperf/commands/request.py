"""Helpers for building and running the request command."""

import asyncio
import typer
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from llmperf.core.models import (
    ChatCompletionInput,
    ChatCompletionMessage,
    LLMRequest,
    LLMResponse,
    SamplingParams,
)
from llmperf.backends.openai import OpenAIChatBackend

DEFAULT_SYSTEM_PROMPT = "你是一个专业的助手"
InputMode = Literal["messages", "file", "user"]


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
        max_completion_tokens (Optional[int]): Maximum number of completion
            tokens to generate.
        ignore_eos (bool): Whether to ignore EOS during sampling.
        sampling_seed (Optional[int]): Optional sampling seed.
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
    max_completion_tokens: Optional[int] = 128
    ignore_eos: bool = False
    sampling_seed: Optional[int] = None

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

        selected = [
            ("messages", has_messages),
            ("file", has_file),
            ("user", has_user),
        ]
        enabled_modes = [name for name, ok in selected if ok]

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
            ignore_eos=self.ignore_eos,
            sampling_seed=self.sampling_seed,
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
            extra={"enable_thinking": self.enable_thinking},
        )


def create_backend(args: RequestCommandArgs) -> OpenAIChatBackend:
    """Create the backend instance for the request command.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.

    Returns:
        OpenAIChatBackend: Configured backend instance.
    """
    return OpenAIChatBackend(timeout=args.timeout)


def run_request_command(args: RequestCommandArgs) -> LLMResponse:
    """Run the request command and return the aggregated response.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.

    Returns:
        LLMResponse: Aggregated backend response.
    """
    backend = create_backend(args)

    response = asyncio.run(
        backend.send(
            url=args.url,
            request=args.build_llm_request(),
            on_chunk=lambda x: print(x, end="", flush=True),
        )
    )

    print()

    return response
