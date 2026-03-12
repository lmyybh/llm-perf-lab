"""`llmperf replay` 命令实现。"""

import sys

from llmperf.commands.common import resolve_api_key
from llmperf.core.errors import ConfigError, InputError, LLMPerfError
from llmperf.core.replay_executor import execute_replay, save_replay_results
from llmperf.core.replay_loader import load_replay_requests, validate_replay_endpoint
from llmperf.output.formatter import render_replay_notice, render_replay_report


def run_replay_command(args) -> int:
    """执行 replay 命令，并把统计结果渲染成最终报表。"""
    try:
        validate_replay_endpoint(args.endpoint)
        if args.timeout_ms <= 0:
            raise InputError("--timeout-ms must be > 0")
        if args.num_requests is not None and args.num_requests <= 0:
            raise InputError("--num-requests must be > 0")
        if args.qps is not None and args.qps <= 0:
            raise InputError("--qps must be > 0")
        if args.max_concurrency is not None and args.max_concurrency <= 0:
            raise InputError("--max-concurrency must be > 0")

        api_key = resolve_api_key(args.api_key, args.api_key_env)
        requests = load_replay_requests(args.dump_path, limit=args.num_requests)
        # replay 统一以命令行指定的 stream 开关发送，避免混用 dump 内历史配置。
        for request in requests:
            request.stream = args.stream
            request.payload["stream"] = args.stream
        effective_qps = args.qps
        effective_concurrency = args.max_concurrency
        # 当前语义下两者不能同时生效；若同时传入，优先保留 QPS。
        if effective_qps is not None and effective_concurrency is not None:
            print(
                render_replay_notice(
                    "--qps and --max-concurrency were both provided; using QPS and ignoring max concurrency."
                )
            )
            effective_concurrency = None

        results, summary = execute_replay(
            requests=requests,
            endpoint=args.endpoint,
            timeout_ms=args.timeout_ms,
            api_key=api_key,
            qps=effective_qps,
            max_concurrency=effective_concurrency,
        )
        config_view: dict[str, object] = {
            "endpoint": args.endpoint,
            "dump_path": args.dump_path,
            "num_requests": len(requests),
            "target_qps": f"{effective_qps:.2f}" if effective_qps is not None else "-",
            "target_concurrency": (
                effective_concurrency if effective_concurrency is not None else "-"
            ),
            "timeout_ms": args.timeout_ms,
            "stream": str(args.stream).lower(),
        }
        summary_view: dict[str, object] = {
            "requests_total": summary.requests_total,
            "requests_succeeded": summary.requests_succeeded,
            "requests_failed": summary.requests_failed,
            "success_rate": f"{summary.success_rate:.2f}%",
            "duration_s": f"{summary.duration_s:.2f}",
            "actual_qps": (
                f"{summary.actual_qps:.2f}" if summary.actual_qps is not None else "-"
            ),
            "request_throughput": f"{summary.request_throughput:.2f} req/s",
            "concurrency": f"{summary.concurrency:.2f}",
            "token_throughput": (
                f"{summary.token_throughput:.1f} tok/s"
                if summary.token_throughput is not None
                else "-"
            ),
            "accept_len": (
                f"{summary.accept_len:.2f}" if summary.accept_len is not None else "-"
            ),
            "accept_rate": (
                f"{summary.accept_rate:.2f}" if summary.accept_rate is not None else "-"
            ),
        }
        metric_view = {
            "e2e_latency_ms": summary.e2e_latency_ms.__dict__,
            "ttft_ms": summary.ttft_ms.__dict__,
            "tpot_ms": summary.tpot_ms.__dict__,
        }
        print(
            render_replay_report(
                config_view, summary_view, metric_view, summary.failures
            )
        )
        if args.save_output:
            save_replay_results(args.save_output, results)
        return 0 if summary.requests_failed == 0 else 3
    except (ConfigError, InputError, LLMPerfError) as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
