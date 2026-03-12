"""参数校验与 endpoint 识别逻辑。"""

import json
from urllib.parse import urlparse

from llmperf.core.errors import ConfigError, InputError
from llmperf.core.models import RequestConfig

CHAT_SUFFIX = "/v1/chat/completions"
GENERATE_SUFFIX = "/generate"


def parse_messages(messages_text: str) -> list[dict]:
    """把 CLI 传入的 JSON 字符串解析成 chat messages，并做基本结构校验。"""
    try:
        messages = json.loads(messages_text)
    except json.JSONDecodeError as exc:
        raise InputError(f"--messages is not valid JSON: {exc}") from exc

    if not isinstance(messages, list):
        raise InputError("--messages must be a JSON array")

    has_user = False
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise InputError(f"--messages[{i}] must be an object")
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise InputError(
                f"--messages[{i}].role must be one of system|user|assistant|tool"
            )
        if not isinstance(content, str):
            raise InputError(f"--messages[{i}].content must be a string")
        if role == "user":
            has_user = True

    if not has_user:
        raise InputError("--messages must contain at least one user message")
    return messages


def detect_endpoint_type(endpoint: str) -> str:
    """根据 URL 后缀判断请求应走哪一种适配器。"""
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"}:
        raise ConfigError("--endpoint must be a full http/https URL")
    if endpoint.endswith(CHAT_SUFFIX):
        return "chat_completions"
    if endpoint.endswith(GENERATE_SUFFIX):
        return "generate"
    raise ConfigError("--endpoint must end with /v1/chat/completions or /generate")


def build_request_config(args) -> tuple[str, RequestConfig]:
    """把 argparse Namespace 转成内部统一配置对象。"""
    endpoint_type = detect_endpoint_type(args.endpoint)
    messages = None
    prompt = None

    if endpoint_type == "chat_completions":
        if not args.messages:
            raise InputError("--messages is required for /v1/chat/completions")
        if args.prompt is not None:
            raise InputError("--prompt is not allowed for /v1/chat/completions")
        messages = parse_messages(args.messages)
    else:
        if args.prompt is None:
            raise InputError("--prompt is required for /generate")
        if args.messages is not None:
            raise InputError("--messages is not allowed for /generate")
        prompt = args.prompt

    if args.timeout_ms <= 0:
        raise InputError("--timeout-ms must be > 0")

    return endpoint_type, RequestConfig(
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout_ms=args.timeout_ms,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=args.stream,
        messages=messages,
        prompt=prompt,
    )
