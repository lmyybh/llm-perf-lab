"""random 命令测试。"""

import io
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from llmperf.commands.generate_requests import run_generate_requests_command
from llmperf.core.replay_executor import MetricStats, ReplaySummary


class GenerateRequestsCommandTest(unittest.TestCase):
    def test_random_command_executes_replay_flow(self):
        """random 命令应构造 generate 请求并复用 replay 的发送与统计链路。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            api_key=None,
            timeout_ms=1000,
            model="chat-model",
            save_output=None,
            num_requests=2,
            min_input_length=3,
            max_input_length=3,
            min_output_length=5,
            max_output_length=5,
            seed=7,
            temperature=0.6,
            top_p=0.9,
            top_k=20,
            min_p=0.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            repetition_penalty=1.1,
            qps=4.0,
            max_concurrency=2,
            stream=True,
        )
        summary = ReplaySummary(
            requests_total=2,
            requests_succeeded=2,
            requests_failed=0,
            success_rate=100.0,
            duration_s=0.5,
            actual_qps=3.8,
            request_throughput=4.0,
            concurrency=1.2,
            token_throughput=100.0,
            accept_len=None,
            accept_rate=None,
            e2e_latency_ms=MetricStats(mean=12.0, p50=12.0, p99=12.0, std=0.0),
            ttft_ms=MetricStats(mean=3.0, p50=3.0, p99=3.0, std=0.0),
            tpot_ms=MetricStats(mean=1.0, p50=1.0, p99=1.0, std=0.0),
            failures={},
        )
        with patch(
            "llmperf.commands.generate_requests.execute_replay",
            return_value=([], summary),
        ) as exec_mock:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = run_generate_requests_command(args)

        self.assertEqual(code, 0)
        self.assertIn("LLMPERF REPLAY REPORT", stdout.getvalue())
        self.assertIn("input_length", stdout.getvalue())
        self.assertIn("seed", stdout.getvalue())
        self.assertIn("request_throughput", stdout.getvalue())
        requests = exec_mock.call_args.kwargs["requests"]
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(requests[0].payload["input_ids"]), 3)
        self.assertNotIn("text", requests[0].payload)
        self.assertEqual(requests[0].payload["sampling_params"]["max_new_tokens"], 5)
        self.assertEqual(requests[0].payload["sampling_params"]["temperature"], 0.6)
        self.assertEqual(requests[0].payload["stream"], True)
        self.assertEqual(exec_mock.call_args.kwargs["qps"], 4.0)
        self.assertIsNone(exec_mock.call_args.kwargs["max_concurrency"])

    def test_random_command_supports_chat_endpoint(self):
        """random 命令传入 chat endpoint 时应生成 messages 形态的请求体。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            api_key=None,
            timeout_ms=1000,
            model="chat-model",
            save_output=None,
            num_requests=1,
            min_input_length=3,
            max_input_length=3,
            min_output_length=5,
            max_output_length=5,
            seed=7,
            temperature=0.6,
            top_p=0.9,
            top_k=20,
            min_p=0.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            repetition_penalty=1.1,
            qps=None,
            max_concurrency=2,
            stream=True,
        )
        summary = ReplaySummary(
            requests_total=1,
            requests_succeeded=1,
            requests_failed=0,
            success_rate=100.0,
            duration_s=0.5,
            actual_qps=None,
            request_throughput=2.0,
            concurrency=1.0,
            token_throughput=100.0,
            accept_len=None,
            accept_rate=None,
            e2e_latency_ms=MetricStats(mean=12.0, p50=12.0, p99=12.0, std=0.0),
            ttft_ms=MetricStats(mean=3.0, p50=3.0, p99=3.0, std=0.0),
            tpot_ms=MetricStats(mean=1.0, p50=1.0, p99=1.0, std=0.0),
            failures={},
        )
        with patch(
            "llmperf.commands.generate_requests.execute_replay",
            return_value=([], summary),
        ) as exec_mock:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = run_generate_requests_command(args)

        self.assertEqual(code, 0)
        self.assertIn("endpoint_type", stdout.getvalue())
        requests = exec_mock.call_args.kwargs["requests"]
        self.assertEqual(requests[0].payload["model"], "chat-model")
        self.assertEqual(
            requests[0].payload["messages"][0]["role"],
            "user",
        )
        self.assertEqual(len(requests[0].payload["messages"][0]["content"].split()), 3)
        self.assertEqual(requests[0].payload["max_completion_tokens"], 5)

    def test_invalid_length_range_returns_error(self):
        """长度范围非法时应返回输入错误退出码。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            api_key=None,
            timeout_ms=1000,
            model="chat-model",
            save_output=None,
            num_requests=2,
            min_input_length=5,
            max_input_length=3,
            min_output_length=1,
            max_output_length=2,
            seed=1,
            temperature=0.6,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
            frequency_penalty=None,
            repetition_penalty=None,
            qps=None,
            max_concurrency=None,
            stream=False,
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = run_generate_requests_command(args)
        self.assertEqual(code, 2)
        self.assertIn(
            "--min-input-length must be <= --max-input-length", stderr.getvalue()
        )


if __name__ == "__main__":
    unittest.main()
