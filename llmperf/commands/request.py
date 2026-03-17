import json
import typer
from pathlib import Path
from dataclasses import dataclass

from llmperf.core.listener import OpenAILogListener
from llmperf.core.openai import send_request

ALLOWED_ROLES = {"system", "user", "assistant"}
DEFAULT_SYSTEM_PROMPT = "你是一个知识丰富的助手"


@dataclass
class MessagesInputOptions:
    messages_json: str | None = None
    file: Path | None = None
    user: str | None = None
    system: str | None = None


def detect_messages_mode(opts: MessagesInputOptions):
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


def handle_messages_mode(messages_json):
    try:
        messages = json.loads(messages_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid JSON for --messages: {exc.msg}") from exc

    return messages


def handle_file_mode(file: Path | None):
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


def handle_prompt_mode(user, system):
    if system is None:
        system = DEFAULT_SYSTEM_PROMPT

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def validate_messages(messages):
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


def resolve_messages(opts: MessagesInputOptions):
    mode = detect_messages_mode(opts)
    if mode == "messages":
        messages = handle_messages_mode(opts.messages_json)
    elif mode == "file":
        messages = handle_file_mode(opts.file)
    else:
        messages = handle_prompt_mode(opts.user, opts.system)

    validate_messages(messages)

    return messages


async def request_v1_chat(
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
):

    opts = MessagesInputOptions(
        messages_json=messages_json, file=file, user=user, system=system
    )
    messages = resolve_messages(opts)

    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    result = await send_request(
        url=url,
        payload=payload,
        headers=headers,
        timeout=timeout,
        listeners=[OpenAILogListener()],
    )

    return result
