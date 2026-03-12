"""request 命令测试。"""

import io
import os
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
            timeout_ms=1000,
            save_output=None,
            model="chat-model",
            max_new_tokens=16,
            temperature=0.6,
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
            timeout_ms=1000,
            save_output=None,
            model="chat-model",
            max_new_tokens=16,
            temperature=0.6,
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

    def test_uses_fixed_api_key_environment_variable(self):
        """未显式传入 API key 时，应从固定的 API_KEY 环境变量读取。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            messages=None,
            prompt="hello",
            api_key=None,
            timeout_ms=1000,
            save_output=None,
            model="chat-model",
            max_new_tokens=16,
            temperature=0.6,
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
        with patch.dict(os.environ, {"API_KEY": "secret"}, clear=False), patch(
            "llmperf.commands.request.execute_request", return_value=response
        ) as exec_mock:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = run_request_command(args)
        self.assertEqual(code, 0)
        self.assertEqual(exec_mock.call_args.args[2], "secret")


if __name__ == "__main__":
    unittest.main()
