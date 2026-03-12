"""参数校验逻辑测试。"""

import unittest
from argparse import Namespace

from llmperf.core.errors import ConfigError, InputError
from llmperf.core.validator import build_request_config


class ValidatorTest(unittest.TestCase):
    def test_chat_requires_messages(self):
        """chat 接口缺少 messages 时必须报错。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            messages=None,
            prompt=None,
            api_key=None,
            timeout_ms=1000,
            model=None,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            stream=False,
        )
        with self.assertRaises(InputError):
            build_request_config(args)

    def test_chat_disallow_prompt(self):
        """chat 接口不允许同时提供 prompt。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            messages='[{"role":"user","content":"hi"}]',
            prompt="x",
            api_key=None,
            timeout_ms=1000,
            model=None,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            stream=False,
        )
        with self.assertRaises(InputError):
            build_request_config(args)

    def test_generate_requires_prompt(self):
        """generate 接口缺少 prompt 时必须报错。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/generate",
            messages=None,
            prompt=None,
            api_key=None,
            timeout_ms=1000,
            model=None,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            stream=False,
        )
        with self.assertRaises(InputError):
            build_request_config(args)

    def test_endpoint_suffix_validation(self):
        """不支持的 endpoint 后缀应被拒绝。"""
        args = Namespace(
            endpoint="http://127.0.0.1:8000/other",
            messages=None,
            prompt="x",
            api_key=None,
            timeout_ms=1000,
            model=None,
            max_new_tokens=None,
            temperature=None,
            top_p=None,
            stream=False,
        )
        with self.assertRaises(ConfigError):
            build_request_config(args)


if __name__ == "__main__":
    unittest.main()
