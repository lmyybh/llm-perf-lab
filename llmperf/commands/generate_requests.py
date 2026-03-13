"""`llmperf random` 命令实现。"""

import random
import sys

from llmperf.commands.common import resolve_api_key
from llmperf.core.errors import ConfigError, InputError, LLMPerfError
from llmperf.core.models import ReplayRequest
from llmperf.core.replay_executor import execute_replay, save_replay_results
from llmperf.core.validator import detect_endpoint_type
from llmperf.output.formatter import render_replay_notice, render_replay_report


def _build_random_token_ids(rng: random.Random, input_length: int) -> list[int]:
    """生成一组随机 token id。

    这里不依赖具体 tokenizer，只构造稳定的随机整数序列：
    - 避开过小的特殊 token 区间
    - 控制在常见词表大小范围内，便于大多数后端接受
    """
    return [rng.randint(100, 50000) for _ in range(input_length)]


def _render_chat_content(token_ids: list[int]) -> str:
    """把随机 token id 序列渲染成 chat 接口可接受的短文本序列。

    不能直接把大整数 token id 作为文本发送给 chat 接口，否则 tokenizer 会把
    每个数字再切成多个 token，导致上下文长度远超用户指定值。这里改为把随机
    token id 映射成单字符片段，尽量让“一个片段约等于一个 token”，同时保持
    序列内容是随机的。
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return " ".join(alphabet[token_id % len(alphabet)] for token_id in token_ids)


def _build_generate_payload(
    args, token_ids: list[int], output_length: int
) -> dict[str, object]:
    """构造一条随机 `/generate` 请求体。"""
    sampling_params: dict[str, object] = {
        "max_new_tokens": output_length,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        sampling_params["top_p"] = args.top_p
    if args.top_k is not None:
        sampling_params["top_k"] = args.top_k
    if args.min_p is not None:
        sampling_params["min_p"] = args.min_p
    if args.presence_penalty is not None:
        sampling_params["presence_penalty"] = args.presence_penalty
    if args.frequency_penalty is not None:
        sampling_params["frequency_penalty"] = args.frequency_penalty
    if args.repetition_penalty is not None:
        sampling_params["repetition_penalty"] = args.repetition_penalty

    return {
        "input_ids": token_ids,
        "stream": args.stream,
        "sampling_params": sampling_params,
    }


def _build_chat_payload(
    args, token_ids: list[int], output_length: int
) -> dict[str, object]:
    """构造一条随机 `/v1/chat/completions` 请求体。"""
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": _render_chat_content(token_ids)}],
        "stream": args.stream,
        "model": args.model,
        "max_completion_tokens": output_length,
        "temperature": args.temperature,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.min_p is not None:
        payload["min_p"] = args.min_p
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty
    return payload


def _validate_args(args) -> str:
    """校验 random 命令参数。"""
    endpoint_type = detect_endpoint_type(args.endpoint)
    if args.timeout_ms <= 0:
        raise InputError("--timeout-ms must be > 0")
    if args.num_requests <= 0:
        raise InputError("--num-requests must be > 0")
    if args.min_input_length <= 0 or args.max_input_length <= 0:
        raise InputError("--min-input-length and --max-input-length must be > 0")
    if args.min_output_length <= 0 or args.max_output_length <= 0:
        raise InputError("--min-output-length and --max-output-length must be > 0")
    if args.min_input_length > args.max_input_length:
        raise InputError("--min-input-length must be <= --max-input-length")
    if args.min_output_length > args.max_output_length:
        raise InputError("--min-output-length must be <= --max-output-length")
    if args.qps is not None and args.qps <= 0:
        raise InputError("--qps must be > 0")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        raise InputError("--max-concurrency must be > 0")
    if args.temperature < 0:
        raise InputError("--temperature must be >= 0")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise InputError("--top-p must be in (0, 1]")
    if args.top_k is not None and args.top_k <= 0:
        raise InputError("--top-k must be > 0")
    if args.min_p is not None and not 0 <= args.min_p <= 1:
        raise InputError("--min-p must be in [0, 1]")
    return endpoint_type


def _build_random_requests(args, endpoint_type: str) -> list[ReplayRequest]:
    """构造随机请求列表，交给 replay 执行器复用发送与统计逻辑。"""
    rng = random.Random(args.seed)
    requests: list[ReplayRequest] = []
    for index in range(args.num_requests):
        input_length = rng.randint(args.min_input_length, args.max_input_length)
        output_length = rng.randint(args.min_output_length, args.max_output_length)
        token_ids = _build_random_token_ids(rng, input_length)
        if endpoint_type == "chat_completions":
            payload = _build_chat_payload(args, token_ids, output_length)
        else:
            payload = _build_generate_payload(args, token_ids, output_length)
        requests.append(
            ReplayRequest(
                source_file="random",
                source_index=index,
                endpoint_type=endpoint_type,
                payload=payload,
                stream=args.stream,
            )
        )
    return requests


def run_generate_requests_command(args) -> int:
    """执行 random 命令。"""
    try:
        endpoint_type = _validate_args(args)
        api_key = resolve_api_key(args.api_key)
        requests = _build_random_requests(args, endpoint_type)
        effective_qps = args.qps
        effective_concurrency = args.max_concurrency
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
            "endpoint_type": endpoint_type,
            "num_requests": len(requests),
            "input_length": f"[{args.min_input_length}, {args.max_input_length}]",
            "output_length": f"[{args.min_output_length}, {args.max_output_length}]",
            "target_qps": f"{effective_qps:.2f}" if effective_qps is not None else "-",
            "target_concurrency": (
                effective_concurrency if effective_concurrency is not None else "-"
            ),
            "timeout_ms": args.timeout_ms,
            "stream": str(args.stream).lower(),
            "seed": args.seed,
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
                config_view,
                summary_view,
                metric_view,
                summary.failures,
            )
        )
        if args.save_output:
            save_replay_results(args.save_output, results)
        return 0 if summary.requests_failed == 0 else 3
    except (ConfigError, InputError, LLMPerfError) as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
