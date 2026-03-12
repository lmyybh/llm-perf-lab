"""OpenAI 兼容 `/v1/chat/completions` 适配器。"""

import json

from llmperf.core.errors import InputError, ProtocolError
from llmperf.core.executor import ResponseEnvelope
from llmperf.core.models import NormalizedResult, RequestConfig


class ChatCompletionsAdapter:
    """把内部配置映射为 chat-completions 请求，并解析 OpenAI 风格响应。"""

    endpoint_type = "chat_completions"

    def build_payload(self, config: RequestConfig) -> dict:
        """构造 chat-completions 请求体。"""
        if not config.messages:
            raise InputError("messages are required")
        payload: dict = {
            "messages": config.messages,
            "stream": config.stream,
        }
        if config.model:
            payload["model"] = config.model
        if config.max_new_tokens is not None:
            payload["max_completion_tokens"] = config.max_new_tokens
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        return payload

    def parse_response(
        self, response: ResponseEnvelope, config: RequestConfig
    ) -> NormalizedResult:
        """把 chat-completions 响应归一化成统一结果。"""
        if config.stream and response.streamed_output is not None:
            return NormalizedResult(
                status="ok",
                latency_ms=response.latency_ms,
                ttft_ms=response.ttft_ms,
                output_text=response.streamed_output,
                usage=None,
            )

        data = response.body_json
        if not data:
            raise ProtocolError("chat response is not valid JSON object")

        output_text = ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(
                    message.get("content"), str
                ):
                    output_text = message["content"]
                elif isinstance(first.get("text"), str):
                    output_text = first["text"]

        if not output_text:
            raise ProtocolError("chat response does not contain output text")

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return NormalizedResult(
            status="ok",
            latency_ms=response.latency_ms,
            ttft_ms=response.ttft_ms,
            output_text=output_text,
            usage=usage,
        )

    def parse_stream_line(self, line: str) -> str:
        """从 SSE 行中提取增量文本。"""
        text = line.strip()
        if not text:
            return ""
        if text.startswith("data:"):
            text = text[5:].strip()
        if text == "[DONE]":
            return ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""
        if not isinstance(payload, dict):
            return ""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        chunk_text = first.get("text")
        if isinstance(chunk_text, str):
            return chunk_text
        return ""
