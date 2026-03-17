"""Helpers for building and sending request command payloads."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import typer

from llmperf.core.data import ChatMessages, OpenAIRequestOutput, RequestHeaders
from llmperf.core.listener import OpenAILogListener
from llmperf.core.openai import openai_chat_request

ALLOWED_ROLES = {"system", "user", "assistant"}
DEFAULT_SYSTEM_PROMPT = "你是一个知识丰富的助手"
MessagesMode = Literal["messages", "file", "user"]


@dataclass
class MessagesInputOptions:
    """Represents the mutually exclusive message input options.

    Attributes:
        messages_json (str | None): JSON-encoded chat messages from the CLI.
        file (Path | None): Path to a JSON file containing chat messages.
        user (str | None): User prompt text supplied from the CLI.
        system (str | None): Optional system prompt paired with ``user``.
    """

    messages_json: str | None = None
    file: Path | None = None
    user: str | None = None
    system: str | None = None


def detect_messages_mode(opts: MessagesInputOptions) -> MessagesMode:
    """Determine which message input mode is active.

    Args:
        opts (MessagesInputOptions): Parsed message input options from the CLI.

    Returns:
        MessagesMode: The active mode name, one of ``messages``, ``file``, or
        ``user``.

    Raises:
        typer.BadParameter: If no mode or more than one mode is selected, or if
            ``system`` is used without ``user``.
    """
    has_messages = opts.messages_json is not None and opts.messages_json.strip() != ""
    has_file = opts.file is not None
    has_user = opts.user is not None and opts.user.strip() != ""

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

    if opts.system is not None and not has_user:
        raise typer.BadParameter("--system can only be used with --user")

    return enabled_modes[0]


def handle_messages_mode(messages_json: str) -> ChatMessages:
    """Parse chat messages from a JSON string.

    Args:
        messages_json (str): JSON-encoded message list.

    Returns:
        ChatMessages: The parsed JSON value.

    Raises:
        typer.BadParameter: If the JSON cannot be parsed.
    """
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON for --messages: {exc.msg}") from exc

    return messages


def handle_file_mode(file: Path | None) -> ChatMessages:
    """Load chat messages from a JSON file.

    Args:
        file (Path | None): Path to a file containing JSON chat messages.

    Returns:
        ChatMessages: The parsed JSON value.

    Raises:
        typer.BadParameter: If the file does not exist, is not a regular file,
            or contains invalid JSON.
    """
    assert file is not None

    if not file.exists():
        raise typer.BadParameter(f"message file not found: {file}")

    if not file.is_file():
        raise typer.BadParameter(f"message file is not a regular file: {file}")

    try:
        with file.open() as f:
            messages = json.load(f)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON for --file: {exc.msg}") from exc

    return messages


def handle_prompt_mode(user: str, system: str | None) -> ChatMessages:
    """Build chat messages from user and system prompt text.

    Args:
        user (str): User prompt text.
        system (str | None): Optional system prompt text.

    Returns:
        ChatMessages: A two-message chat payload with system and user roles.
    """
    if system is None:
        system = DEFAULT_SYSTEM_PROMPT

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_messages(messages: ChatMessages) -> None:
    """Validate the structure of a chat message list.

    Args:
        messages (ChatMessages): Parsed message payload to validate.

    Raises:
        typer.BadParameter: If the payload is not a non-empty list of chat
            messages with supported roles and non-empty string content.
    """
    if not isinstance(messages, list):
        raise typer.BadParameter("messages must be a list")

    if not messages:
        raise typer.BadParameter("messages must not be empty")

    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise typer.BadParameter(f"messages[{i}] must be an dict")

        role = message.get("role")
        content = message.get("content")

        if role not in ALLOWED_ROLES:
            raise typer.BadParameter(
                f"messages[{i}].role must be one of: {ALLOWED_ROLES}"
            )

        if not isinstance(content, str):
            raise typer.BadParameter(f"messages[{i}].content must be a string")

        if content is None or content.strip() == "":
            raise typer.BadParameter(f"messages[{i}].content must not be empty")


def resolve_messages(opts: MessagesInputOptions) -> ChatMessages:
    """Resolve and validate chat messages from CLI input options.

    Args:
        opts (MessagesInputOptions): Parsed message input options from the CLI.

    Returns:
        ChatMessages: A validated list of chat messages.
    """
    mode = detect_messages_mode(opts)
    if mode == "messages":
        messages = handle_messages_mode(opts.messages_json)
    elif mode == "file":
        messages = handle_file_mode(opts.file)
    else:
        messages = handle_prompt_mode(opts.user, opts.system)

    validate_messages(messages)

    return messages


async def openai_chat(
    messages_json: str | None,
    file: Path | None,
    user: str | None,
    system: str | None,
    url: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    enable_thinking: bool = True,
    stream: bool = True,
    timeout: float = 300,
) -> OpenAIRequestOutput:
    """Send one OpenAI-compatible chat completion request.

    Args:
        messages_json (str | None): JSON-encoded chat messages provided via CLI.
        file (Path | None): Path to a file containing chat messages.
        user (str | None): User prompt used to build messages when no JSON is
            supplied.
        system (str | None): Optional system prompt paired with ``user``.
        url (str): Target chat completions endpoint.
        model (str): Model name sent in the request body.
        temperature (float): Sampling temperature for the request.
        max_tokens (int | None): Optional maximum number of output tokens.
        enable_thinking (bool): Whether to pass thinking configuration to the
            server.
        stream (bool): Whether to request a streaming response.
        timeout (float): Request timeout in seconds.

    Returns:
        OpenAIRequestOutput: The aggregated request result from the
        OpenAI-compatible endpoint.
    """

    opts = MessagesInputOptions(
        messages_json=messages_json, file=file, user=user, system=system
    )
    messages = resolve_messages(opts)

    headers: RequestHeaders = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    result = await openai_chat_request(
        url=url,
        payload=payload,
        headers=headers,
        timeout=timeout,
        listeners=[OpenAILogListener()],
    )

    return result
