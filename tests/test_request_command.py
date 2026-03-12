"""request 命令测试。"""

import io
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from llmperf.commands.request import run_request_command
from llmperf.core.errors import NetworkError
from llmperf.core.executor import ResponseEnvelope


class RequestCommandTest(unittest.TestCase):
    def test_success(self):
        """单次请求成功时应输出完整三段式终端结果。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            messages=None,
            prompt="hello",
            api_key=None,
            api_key_env=None,
            timeout_ms=1000,
            save_output=None,
            model=None,
            max_new_tokens=16,
            temperature=None,
            top_p=None,
            stream=False,
        )
        response = ResponseEnvelope(
            status_code=200,
            latency_ms=12,
            ttft_ms=None,
            body_text='{"generated_text":"ok"}',
            body_json={"generated_text": "ok"},
            streamed_output=None,
        )
        with patch("llmperf.commands.request.execute_request", return_value=response):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = run_request_command(args)
        self.assertEqual(code, 0)
        self.assertIn("1) INPUT", stdout.getvalue())
        self.assertIn("2) OUTPUT", stdout.getvalue())
        self.assertIn("3) STATUS", stdout.getvalue())
        self.assertIn("status     : ok", stdout.getvalue())

    def test_failure_exit_code(self):
        """网络错误应转成约定的退出码与错误标签。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            messages=None,
            prompt="hello",
            api_key=None,
            api_key_env=None,
            timeout_ms=1000,
            save_output=None,
            model=None,
            max_new_tokens=16,
            temperature=None,
            top_p=None,
            stream=False,
        )
        with patch(
            "llmperf.commands.request.execute_request", side_effect=NetworkError("x")
        ):
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = run_request_command(args)
        self.assertEqual(code, 3)
        self.assertIn("3) STATUS", stderr.getvalue())
        self.assertIn("error_type", stderr.getvalue())
        self.assertIn("network_error", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
