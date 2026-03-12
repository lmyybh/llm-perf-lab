"""终端与文件输出格式化工具。

这里集中维护所有文本报表格式，避免命令层散落字符串拼接逻辑。
"""

import json
from pathlib import Path

from llmperf.core.models import NormalizedResult


def _format_value(value) -> str:
    """把任意值渲染成适合终端展示的字符串。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "-"
    return str(value)


def _render_kv_lines(data: dict[str, object]) -> str:
    """把字典渲染成左对齐键值表。"""
    if not data:
        return "(empty)"
    width = max(len(k) for k in data.keys())
    lines = []
    for key, value in data.items():
        lines.append(f"{key:<{width}} : {_format_value(value)}")
    return "\n".join(lines)


def _render_section(title: str, body: str) -> str:
    """渲染传统三段式输出块。"""
    bar = "=" * 72
    return f"{bar}\n{title}\n{bar}\n{body}"


def _render_centered_line(title: str, fill: str = "-", width: int = 72) -> str:
    """渲染固定宽度、标题居中的分割线。"""
    if len(title) + 2 >= width:
        return title
    return f" {title} ".center(width, fill)


def _format_metric_value(value) -> str:
    """统一统计数值的显示格式。"""
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.1f}"


def render_input_section(input_data: dict[str, object]) -> str:
    """渲染 request 命令的输入块。"""
    return _render_section("1) INPUT", _render_kv_lines(input_data))


def render_output_section(output_text: str, streamed: bool = False) -> str:
    """渲染 request 命令的输出块。"""
    if not output_text and streamed:
        body = "(streaming content below)"
    elif not output_text:
        body = "(empty)"
    else:
        body = output_text
    return _render_section("2) OUTPUT", body)


def render_status_section(result: NormalizedResult) -> str:
    """渲染 request 命令的状态块。"""
    data = result.to_dict()
    status_data: dict[str, object] = {
        "status": data.get("status"),
        "latency_ms": data.get("latency_ms"),
    }
    if "ttft_ms" in data:
        status_data["ttft_ms"] = data["ttft_ms"]
    if "usage" in data:
        status_data["usage"] = data["usage"]
    if "token_stats" in data:
        status_data["token_stats"] = data["token_stats"]
    if "error_type" in data:
        status_data["error_type"] = data["error_type"]
    if "error_message" in data:
        status_data["error_message"] = data["error_message"]
    return _render_section("3) STATUS", _render_kv_lines(status_data))


def render_terminal(
    input_data: dict[str, object],
    result: NormalizedResult,
    show_output_text: bool = True,
) -> str:
    """拼接 request 命令完整的终端输出。"""
    output_text = result.output_text if show_output_text else ""
    return "\n".join(
        [
            render_input_section(input_data),
            render_output_section(output_text, streamed=not show_output_text),
            render_status_section(result),
        ]
    )


def render_replay_summary(summary: dict[str, object]) -> str:
    """兼容旧接口的 replay summary 输出。"""
    return _render_section("REPLAY SUMMARY", _render_kv_lines(summary))


def render_replay_notice(message: str) -> str:
    """渲染 replay 的提示信息，例如参数优先级冲突。"""
    return _render_section("REPLAY NOTICE", message)


def render_replay_report(
    config: dict[str, object],
    summary: dict[str, object],
    metrics: dict[str, dict[str, object]],
    failures: dict[str, int],
) -> str:
    """渲染 replay 命令最终的整页报表。"""
    lines = [_render_centered_line("LLMPERF REPLAY REPORT", fill="=")]
    lines.extend(_render_kv_lines(config).splitlines())
    lines.append(_render_centered_line("REPLAY SUMMARY"))
    lines.extend(_render_kv_lines(summary).splitlines())
    lines.append(_render_centered_line("LATENCY METRICS (ms)"))
    metric_lines = [
        f"{'metric':<20} {'mean':>10} {'p50':>10} {'p99':>10} {'std':>10}",
    ]
    for name, values in metrics.items():
        metric_lines.append(
            f"{name:<20} "
            f"{_format_metric_value(values.get('mean')):>10} "
            f"{_format_metric_value(values.get('p50')):>10} "
            f"{_format_metric_value(values.get('p99')):>10} "
            f"{_format_metric_value(values.get('std')):>10}"
        )
    lines.extend(metric_lines)
    if failures:
        lines.append(_render_centered_line("FAILURES"))
        lines.extend(_render_kv_lines(failures).splitlines())
    lines.append("=" * 72)
    return "\n".join(lines)


def save_output(path: str, result: NormalizedResult) -> None:
    """把单次请求的归一化结果保存为 JSON 文件。"""
    output_path = Path(path)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
