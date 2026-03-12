"""replay 命令层测试。

这些测试关注参数优先级、报告输出和错误返回码。
"""

import io
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from llmperf.commands.replay import run_replay_command
from llmperf.core.errors import ConfigError
from llmperf.core.models import ReplayItemResult
from llmperf.core.replay_executor import MetricStats, ReplaySummary


class ReplayCommandTest(unittest.TestCase):
    def test_success_with_qps_priority_and_save_output(self):
        """同时给出 QPS 与并发时，应优先采用 QPS，并正常输出报告。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            dump_path="/tmp/fake.pkl",
            api_key=None,
            api_key_env=None,
            timeout_ms=1000,
            save_output=None,
            num_requests=None,
            qps=10.0,
            max_concurrency=4,
            stream=True,
        )
        summary = ReplaySummary(
            requests_total=1,
            requests_succeeded=1,
            requests_failed=0,
            success_rate=100.0,
            duration_s=0.12,
            actual_qps=9.50,
            request_throughput=8.13,
            concurrency=0.10,
            token_throughput=80.0,
            accept_len=2.15,
            accept_rate=0.2885,
            e2e_latency_ms=MetricStats(mean=12.0, p50=12.0, p99=12.0, std=0.0),
            ttft_ms=MetricStats(mean=3.0, p50=3.0, p99=3.0, std=0.0),
            tpot_ms=MetricStats(mean=1.0, p50=1.0, p99=1.0, std=0.0),
            failures={},
        )
        results = [
            ReplayItemResult(
                source_file="a.pkl",
                source_index=0,
                status="ok",
                latency_ms=12,
                ttft_ms=3,
                output_text="ok",
                output_tokens=10,
                tpot_ms=1.0,
                status_code=200,
            )
        ]
        replay_request = type(
            "ReplayRequestStub", (), {"stream": False, "payload": {}}
        )()
        with tempfile.TemporaryDirectory() as tmpdir:
            args.save_output = str(Path(tmpdir) / "results.jsonl")
            with patch(
                "llmperf.commands.replay.load_replay_requests",
                return_value=[replay_request],
            ), patch(
                "llmperf.commands.replay.execute_replay",
                return_value=(results, summary),
            ) as exec_mock:
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    code = run_replay_command(args)

            self.assertEqual(code, 0)
            self.assertIn("LLMPERF REPLAY REPORT", stdout.getvalue())
            self.assertIn("actual_qps", stdout.getvalue())
            self.assertIn("request_throughput", stdout.getvalue())
            self.assertIn("accept_len", stdout.getvalue())
            self.assertIn("accept_rate", stdout.getvalue())
            self.assertIn("LATENCY METRICS (ms)", stdout.getvalue())
            self.assertNotIn("FAILURES", stdout.getvalue())
            exec_mock.assert_called_once_with(
                requests=[replay_request],
                endpoint="http://127.0.0.1:8000/generate",
                timeout_ms=1000,
                api_key=None,
                qps=10.0,
                max_concurrency=None,
            )
            self.assertTrue(replay_request.stream)
            self.assertTrue(replay_request.payload["stream"])
            saved = Path(args.save_output).read_text(encoding="utf-8")
            self.assertIn('"status": "ok"', saved)

    def test_invalid_timeout_returns_input_error_exit_code(self):
        """timeout 非法时应返回参数错误退出码。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            dump_path="/tmp/fake.pkl",
            api_key=None,
            api_key_env=None,
            timeout_ms=0,
            save_output=None,
            num_requests=None,
            qps=None,
            max_concurrency=None,
            stream=True,
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = run_replay_command(args)
        self.assertEqual(code, 2)
        self.assertIn("--timeout-ms must be > 0", stderr.getvalue())

    def test_loader_error_is_reported(self):
        """底层加载失败时，命令层应把错误打印到 stderr。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            dump_path="/tmp/fake.pkl",
            api_key=None,
            api_key_env=None,
            timeout_ms=1000,
            save_output=None,
            num_requests=None,
            qps=None,
            max_concurrency=None,
            stream=True,
        )
        with patch(
            "llmperf.commands.replay.load_replay_requests",
            side_effect=ConfigError("boom"),
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = run_replay_command(args)
        self.assertEqual(code, 2)
        self.assertIn("boom", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
