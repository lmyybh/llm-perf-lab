"""Argument models and input parsing for the request command."""

import json
from pathlib import Path
from typing import Literal, Optional, cast

import typer
from pydantic import BaseModel

from llmperf.common import (
    ChatCompletionInput,
    ChatCompletionMessage,
    SamplingParams,
    Tool,
    ToolChoice,
)

DEFAULT_SYSTEM_PROMPT = "你是一个专业的助手"
InputMode = Literal["messages", "file", "user"]
ToolChoiceMode = Literal["auto", "required", "none"]


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
    """Load chat completion input from a JSON file.

    Args:
        file (Path): Path to a JSON file containing either a chat message list
            or a chat completion input object.

    Returns:
        ChatCompletionInput: Parsed chat completion input.

    Raises:
        typer.BadParameter: Raised when the file does not exist, is not a
            regular file, contains invalid JSON, or does not match a supported
            request payload shape.
    """
    if not file.exists():
        raise typer.BadParameter(f"message file not found: {file}")

    if not file.is_file():
        raise typer.BadParameter(f"message file is not a regular file: {file}")

    try:
        with file.open() as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON for --file: {exc.msg}") from exc

    if isinstance(payload, list):
        return ChatCompletionInput(messages=payload)

    if isinstance(payload, dict):
        try:
            return ChatCompletionInput.model_validate(payload)
        except ValueError as exc:
            raise typer.BadParameter(
                "invalid request payload for --file: expected an object with "
                "'messages' and optional 'tools' / 'tool_choice'"
            ) from exc

    raise typer.BadParameter(
        "invalid request payload for --file: expected either a message list "
        "or an object with 'messages' and optional 'tools' / 'tool_choice'"
    )


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
        tools (Optional[str]): Optional JSON-encoded tool declarations.
        tool_choice (Optional[str]): Tool selection mode or JSON-encoded tool
            choice. When omitted, file-provided tool choice is preserved.
        model (Optional[str]): Optional model name to send downstream.
        tokenizer_path (Optional[Path]): Optional tokenizer path or identifier
            used to estimate prompt tokens locally.
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

    tools: Optional[str] = None
    tool_choice: Optional[str] = None

    model: Optional[str] = None
    tokenizer_path: Optional[Path] = None
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

        Raises:
            typer.BadParameter: Raised when tool declarations or tool choice are
                not valid JSON or do not satisfy the response schema.
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

        if self.tools is not None:
            try:
                tools_payload = json.loads(self.tools)
            except json.JSONDecodeError as exc:
                raise typer.BadParameter(
                    f"invalid JSON for --tools: {exc.msg}"
                ) from exc

            messages.tools = [Tool.model_validate(tool) for tool in tools_payload]

        if self.tool_choice is not None:
            if self.tool_choice not in {"auto", "required", "none"}:
                try:
                    tool_choice = json.loads(self.tool_choice)
                except json.JSONDecodeError as exc:
                    raise typer.BadParameter(
                        f"invalid JSON for --tool-choice: {exc.msg}"
                    ) from exc

                messages.tool_choice = ToolChoice.model_validate(tool_choice)
            else:
                messages.tool_choice = cast(ToolChoiceMode, self.tool_choice)
        elif messages.tool_choice is None:
            messages.tool_choice = "auto"

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
