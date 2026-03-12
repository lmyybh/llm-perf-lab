"""流式响应解析测试。"""

import unittest

from llmperf.adapters.chat_completions import ChatCompletionsAdapter
from llmperf.adapters.generate import GenerateAdapter


class StreamParsingTest(unittest.TestCase):
    def test_chat_sse_delta_content(self):
        """chat SSE 增量应能正确提取 content。"""
        adapter = ChatCompletionsAdapter()
        line = 'data: {"choices":[{"delta":{"content":"A"}}]}'
        self.assertEqual(adapter.parse_stream_line(line), "A")

    def test_generate_sse_text(self):
        """generate SSE 行中的 text 字段应被正确提取。"""
        adapter = GenerateAdapter()
        line = 'data: {"text":"abc"}'
        self.assertEqual(adapter.parse_stream_line(line), "abc")

    def test_generate_raw_text_fallback(self):
        """非 JSON 原文在 generate 场景下应回退为纯文本。"""
        adapter = GenerateAdapter()
        self.assertEqual(adapter.parse_stream_line("hello"), "hello")


if __name__ == "__main__":
    unittest.main()
