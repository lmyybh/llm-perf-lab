"""generate 适配器测试。"""

import unittest

from llmperf.adapters.generate import GenerateAdapter
from llmperf.core.models import RequestConfig


class GenerateAdapterTest(unittest.TestCase):
    def test_build_payload_uses_text_and_sampling_params(self):
        """generate adapter 应把采样参数收敛到 `sampling_params` 下。"""
        adapter = GenerateAdapter()
        config = RequestConfig(
            endpoint="http://127.0.0.1:32110/generate",
            api_key=None,
            timeout_ms=1000,
            model=None,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
            stream=False,
            messages=None,
            prompt="hello",
        )

        payload = adapter.build_payload(config)

        self.assertEqual(payload["text"], "hello")
        self.assertIn("sampling_params", payload)
        self.assertEqual(
            payload["sampling_params"],
            {"max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9},
        )
        self.assertNotIn("prompt", payload)
        self.assertNotIn("max_new_tokens", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)


if __name__ == "__main__":
    unittest.main()
