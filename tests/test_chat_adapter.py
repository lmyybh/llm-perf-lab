"""chat 适配器测试。"""

import unittest

from llmperf.adapters.chat_completions import ChatCompletionsAdapter
from llmperf.core.models import RequestConfig


class ChatAdapterTest(unittest.TestCase):
    def test_build_payload_uses_max_completion_tokens(self):
        """chat adapter 应使用 OpenAI 风格的 `max_completion_tokens` 字段。"""
        adapter = ChatCompletionsAdapter()
        config = RequestConfig(
            endpoint="http://127.0.0.1:32110/v1/chat/completions",
            api_key=None,
            timeout_ms=1000,
            model="test-model",
            max_new_tokens=64,
            temperature=0.3,
            top_p=0.95,
            stream=False,
            messages=[{"role": "user", "content": "hi"}],
            prompt=None,
        )

        payload = adapter.build_payload(config)

        self.assertEqual(payload["max_completion_tokens"], 64)
        self.assertNotIn("max_tokens", payload)


if __name__ == "__main__":
    unittest.main()
