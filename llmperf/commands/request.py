"""`llmperf request` 命令实现。"""

import sys

from llmperf.adapters.registry import get_adapter
from llmperf.commands.common import resolve_api_key
from llmperf.core.errors import ConfigError, InputError, LLMPerfError
from llmperf.core.executor import execute_request
from llmperf.core.models import NormalizedResult
from llmperf.core.validator import build_request_config
from llmperf.output.formatter import (
    render_input_section,
    render_output_section,
    render_status_section,
    render_terminal,
    save_output,
)


def _error_result(error: LLMPerfError) -> NormalizedResult:
    """把异常对象转换成统一的错误结果结构。"""
    return NormalizedResult(
        status="error",
        latency_ms=0,
        ttft_ms=None,
        output_text="",
        error_type=error.error_type,
        error_message=error.message,
    )


def _build_input_view(
    endpoint_type: str, config, payload: dict, args, api_key: str | None
) -> dict[str, object]:
    """构造最终终端输出里的 INPUT 区块内容。"""
    api_key_source = "none"
    if args.api_key:
        api_key_source = "provided"
    elif api_key is not None:
        api_key_source = "env:API_KEY"

    input_view: dict[str, object] = {
        "endpoint": config.endpoint,
        "endpoint_type": endpoint_type,
        "timeout_ms": config.timeout_ms,
        "stream": config.stream,
        "api_key": api_key_source,
    }
    if config.model:
        input_view["model"] = config.model
    if config.prompt is not None:
        input_view["prompt"] = config.prompt
    if config.messages is not None:
        input_view["messages"] = config.messages
    input_view["payload"] = payload
    return input_view


def run_request_command(args) -> int:
    """执行单次请求命令。"""
    input_view: dict[str, object] = {
        "endpoint": getattr(args, "endpoint", None),
        "prompt": getattr(args, "prompt", None),
        "messages": getattr(args, "messages", None),
        "stream": getattr(args, "stream", False),
    }
    try:
        endpoint_type, config = build_request_config(args)
        api_key = resolve_api_key(args.api_key)
        adapter = get_adapter(endpoint_type)
        payload = adapter.build_payload(config)
        input_view = _build_input_view(endpoint_type, config, payload, args, api_key)
        stream_sink = sys.stdout

        def _on_stream_text(text: str) -> None:
            """流式输出时把增量文本立即刷到终端。"""
            for ch in text:
                stream_sink.write(ch)
                stream_sink.flush()

        if config.stream:
            print(render_input_section(input_view))
            print(render_output_section("", streamed=True))

        response = execute_request(
            config,
            payload,
            api_key,
            parse_stream_line=adapter.parse_stream_line if config.stream else None,
            on_stream_text=_on_stream_text if config.stream else None,
        )
        if config.stream:
            stream_sink.write("\n")
            stream_sink.flush()
        result = adapter.parse_response(response, config)
        if config.stream:
            print(render_status_section(result))
        else:
            print(render_terminal(input_view, result, show_output_text=True))
        if args.save_output:
            save_output(args.save_output, result)
        return 0
    except (ConfigError, InputError) as exc:
        err_result = _error_result(exc)
        print(
            render_terminal(input_view, err_result, show_output_text=True),
            file=sys.stderr,
        )
        return exc.exit_code
    except LLMPerfError as exc:
        err_result = _error_result(exc)
        print(
            render_terminal(input_view, err_result, show_output_text=True),
            file=sys.stderr,
        )
        if args.save_output:
            save_output(args.save_output, err_result)
        return exc.exit_code
