"""Argument models and input parsing for the request command."""

import json
from pathlib import Path
from typing import Literal, Optional, Union, cast

import typer
from pydantic import BaseModel

from llmperf.commands.bench.args import DatasetMode
from llmperf.common import (
    ChatCompletionInput,
    ChatCompletionMessage,
    GenerateInput,
    SamplingParams,
    Tool,
    ToolChoice,
)

DEFAULT_SYSTEM_PROMPT = "你是一个专业的助手"
UrlMode = Literal["openai", "generate"]
OpenaiInputMode = Literal["messages", "file", "user", "dataset"]
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
        dataset_file (Optional[Path]): Optional dataset file used when
            selecting one request by target prompt length.
        dataset_mode (DatasetMode): Dataset parser or generator mode used for
            target-length request selection.
        target_input_tokens (Optional[int]): Target prompt token count that
            enables dataset selection mode.
        input_token_tolerance (int): Absolute prompt token tolerance around the
            target length.
        with_tools (bool): Whether to select only dataset samples with
            non-empty tools and include those tools in the request.
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

    # chat mode
    messages_json: Optional[str] = None
    file: Optional[Path] = None
    user: Optional[str] = None
    system: Optional[str] = None

    # generate mode
    text: Optional[str] = None

    # dataset selection mode
    dataset_file: Optional[Path] = None
    dataset_mode: DatasetMode = DatasetMode.openai
    target_input_tokens: Optional[int] = None
    input_token_tolerance: int = 64
    with_tools: bool = False

    tools: Optional[str] = None
    tool_choice: Optional[str] = None

    model: Optional[str] = None
    tokenizer_path: Optional[Path] = None
    rid: Optional[str] = None

    temperature: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_completion_tokens: Optional[int] = 128
    ignore_eos: bool = False
    seed: Optional[int] = None

    enable_thinking: bool = True
    stream: bool = True

    timeout: float = 300

    def detect_url_mode(self) -> UrlMode:
        if self.url.endswith("/generate"):
            return "generate"

        return "openai"

    def detect_openai_input_mode(self) -> OpenaiInputMode:
        """Determine which mutually exclusive input mode is active.

        Returns:
            OpenaiInputMode: The selected input mode.

        Raises:
            typer.BadParameter: Raised when zero or multiple input modes are
                selected, or when ``system`` is used without ``user``.
        """
        if self.text is not None:
            raise typer.BadParameter("--text can only be used with a /generate URL")

        has_dataset = self.target_input_tokens is not None
        has_messages = (
            self.messages_json is not None and self.messages_json.strip() != ""
        )
        has_file = self.file is not None
        has_user = self.user is not None and self.user.strip() != ""

        if not has_dataset:
            if self.dataset_file is not None:
                raise typer.BadParameter(
                    "--dataset-file requires --target-input-tokens"
                )
            if self.with_tools:
                raise typer.BadParameter("--with-tools requires --target-input-tokens")

        selected: list[tuple[OpenaiInputMode, bool]] = [
            ("dataset", has_dataset),
            ("messages", has_messages),
            ("file", has_file),
            ("user", has_user),
        ]
        enabled_modes: list[OpenaiInputMode] = [name for name, ok in selected if ok]

        if len(enabled_modes) == 0:
            raise typer.BadParameter(
                "must provide exactly one of --target-input-tokens, --messages, "
                "--file, or --user"
            )

        if len(enabled_modes) > 1:
            raise typer.BadParameter(
                "options --target-input-tokens, --messages, --file, and --user "
                "are mutually exclusive"
            )

        if self.system is not None and not has_user:
            raise typer.BadParameter("--system can only be used with --user")

        if has_dataset:
            self.validate_dataset_input_mode()

        return enabled_modes[0]

    def validate_dataset_input_mode(self) -> None:
        """Validate options that are only meaningful for dataset selection.

        Raises:
            typer.BadParameter: Raised when dataset selection options conflict
                with manual request inputs or unsupported URL modes.
        """
        if self.target_input_tokens is None:
            return

        if self.system is not None:
            raise typer.BadParameter(
                "--system cannot be used with --target-input-tokens"
            )

        if self.tools is not None or self.tool_choice is not None:
            raise typer.BadParameter(
                "--tools and --tool-choice cannot be used with "
                "--target-input-tokens; use --with-tools to select dataset "
                "tool samples"
            )

        if self.dataset_mode == DatasetMode.random:
            if self.dataset_file is not None:
                raise typer.BadParameter(
                    "--dataset-file cannot be used with --dataset-mode random"
                )
            if self.with_tools:
                raise typer.BadParameter(
                    "--with-tools cannot be used with --dataset-mode random"
                )
            return

        if self.dataset_file is None:
            raise typer.BadParameter("file-backed dataset modes require --dataset-file")
        if not self.dataset_file.is_file():
            raise typer.BadParameter(
                "file-backed dataset modes require a valid --dataset-file path"
            )

    def parse_generate_input(self) -> GenerateInput:
        """Parse command arguments into a generate input.

        Returns:
            GenerateInput: Parsed text generation input.

        Raises:
            typer.BadParameter: Raised when chat-only input options are used or
                when ``--text`` is omitted.
        """
        forbidden_options = []
        if self.messages_json is not None:
            forbidden_options.append("--messages")
        if self.file is not None:
            forbidden_options.append("--file")
        if self.user is not None:
            forbidden_options.append("--user")
        if self.system is not None:
            forbidden_options.append("--system")
        if self.target_input_tokens is not None:
            forbidden_options.append("--target-input-tokens")
        if self.dataset_file is not None:
            forbidden_options.append("--dataset-file")
        if self.with_tools:
            forbidden_options.append("--with-tools")

        if forbidden_options:
            options = ", ".join(forbidden_options)
            raise typer.BadParameter(
                f"/generate only accepts --text; unsupported options: {options}"
            )

        if self.text is None or self.text.strip() == "":
            raise typer.BadParameter("/generate requires non-empty --text")

        return GenerateInput(text=self.text)

    def parse_openai_input(self) -> ChatCompletionInput:
        """Parse command arguments into chat completion input.

        Returns:
            ChatCompletionInput: Parsed chat completion input.

        Raises:
            typer.BadParameter: Raised when tool declarations or tool choice are
                not valid JSON or do not satisfy the response schema.
        """
        mode = self.detect_openai_input_mode()

        if mode == "dataset":
            raise typer.BadParameter(
                "dataset input mode must be parsed by the request service"
            )
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

    def parse_input(self) -> Union[ChatCompletionInput, GenerateInput]:
        if self.detect_url_mode() == "generate":
            return self.parse_generate_input()
        else:
            return self.parse_openai_input()

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
