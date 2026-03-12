"""命令行入口。

本模块只负责定义顶层参数结构，并把不同子命令分发给各自的处理器。
所有业务逻辑都下沉到 commands/core 层，避免 CLI 文件承担实现细节。
"""

import argparse

from llmperf.commands.replay import run_replay_command
from llmperf.commands.request import run_request_command


def build_parser() -> argparse.ArgumentParser:
    """构建 llmperf 顶层参数解析器。"""
    parser = argparse.ArgumentParser(prog="llmperf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    request = subparsers.add_parser("request", help="Send a single inference request")
    request.add_argument(
        "--endpoint",
        required=True,
        help="Full request URL. Must end with /generate or /v1/chat/completions.",
    )
    request.add_argument(
        "--messages",
        help="Chat messages as a JSON array string. Only valid for /v1/chat/completions.",
    )
    request.add_argument(
        "--prompt",
        help="Plain text prompt. Only valid for /generate.",
    )
    request.add_argument(
        "--api-key",
        default=None,
        help="API key passed as a bearer token. Falls back to API_KEY if unset.",
    )
    request.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Request timeout in milliseconds.",
    )
    request.add_argument(
        "--save-output",
        default=None,
        help="Optional path to save the normalized result as JSON.",
    )
    request.add_argument(
        "--model",
        default="chat-model",
        help="Model name to include in the request body. Default: chat-model.",
    )
    request.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum number of generated tokens.",
    )
    request.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature. Default: 0.6.",
    )
    request.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling threshold.",
    )
    request.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=False,
        help="Enable streaming response mode.",
    )
    request.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Disable streaming response mode.",
    )

    replay = subparsers.add_parser(
        "replay", help="Replay requests from SGLang dump data"
    )
    replay.add_argument(
        "--endpoint",
        required=True,
        help="Replay target URL. Currently must end with /generate.",
    )
    replay.add_argument(
        "--dump-path",
        required=True,
        help="Path to a replay .pkl file or a directory containing multiple .pkl files.",
    )
    replay.add_argument(
        "--api-key",
        default=None,
        help="API key passed as a bearer token. Falls back to API_KEY if unset.",
    )
    replay.add_argument(
        "--timeout-ms",
        type=int,
        default=30000,
        help="Per-request timeout in milliseconds.",
    )
    replay.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Optional path to save per-request replay results as JSONL.",
    )
    replay.add_argument(
        "--num-requests",
        type=int,
        help="Maximum number of requests to replay after loading dump files.",
    )
    replay.add_argument(
        "--qps",
        type=float,
        help="Target request send rate in requests per second.",
    )
    replay.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum number of in-flight requests when QPS is not set.",
    )
    replay.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Send replay requests in streaming mode.",
    )
    replay.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Send replay requests in non-streaming mode.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数并执行对应子命令。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "request":
        return run_request_command(args)
    if args.command == "replay":
        return run_replay_command(args)

    parser.error(f"unknown command: {args.command}")
    return 2
