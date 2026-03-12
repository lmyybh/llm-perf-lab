"""适配器注册表测试。"""

import unittest

from llmperf.adapters.chat_completions import ChatCompletionsAdapter
from llmperf.adapters.generate import GenerateAdapter
from llmperf.adapters.registry import get_adapter
from llmperf.core.errors import ConfigError


class RegistryTest(unittest.TestCase):
    def test_chat_adapter(self):
        """chat 类型应返回 chat 适配器。"""
        adapter = get_adapter("chat_completions")
        self.assertIsInstance(adapter, ChatCompletionsAdapter)

    def test_generate_adapter(self):
        """generate 类型应返回 generate 适配器。"""
        adapter = get_adapter("generate")
        self.assertIsInstance(adapter, GenerateAdapter)

    def test_unknown_adapter(self):
        """未知类型应抛出配置错误。"""
        with self.assertRaises(ConfigError):
            get_adapter("not_exist")


if __name__ == "__main__":
    unittest.main()
