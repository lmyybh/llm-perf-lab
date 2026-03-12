"""SGLang `/generate` 接口适配器。"""

import json

from llmperf.core.errors import InputError, ProtocolError
from llmperf.core.executor import ResponseEnvelope
from llmperf.core.models import NormalizedResult, RequestConfig


class GenerateAdapter:
    """把通用配置映射到 generate 接口，并负责解析其返回结构。"""

    endpoint_type = "generate"

    def build_payload(self, config: RequestConfig) -> dict:
        """构造 generate 请求体。"""
        if config.prompt is None:
            raise InputError("prompt is required")
        sampling_params: dict = {}
        if config.max_new_tokens is not None:
            sampling_params["max_new_tokens"] = config.max_new_tokens
        if config.temperature is not None:
            sampling_params["temperature"] = config.temperature
        if config.top_p is not None:
            sampling_params["top_p"] = config.top_p

        payload: dict = {
            "text": config.prompt,
            "stream": config.stream,
            "sampling_params": sampling_params,
        }
        return payload

    def parse_response(
        self, response: ResponseEnvelope, config: RequestConfig
    ) -> NormalizedResult:
        """把 generate 返回结构归一化成统一结果对象。"""
        if config.stream and response.streamed_output is not None:
            return NormalizedResult(
                status="ok",
                latency_ms=response.latency_ms,
                ttft_ms=response.ttft_ms,
                output_text=response.streamed_output,
                token_stats=None,
            )

        data = response.body_json
        output_text = ""
        token_stats = None

        if data:
            for key in ("text", "generated_text", "output_text", "response"):
                value = data.get(key)
                if isinstance(value, str):
                    output_text = value
                    break

            if not output_text:
                outputs = data.get("outputs")
                if (
                    isinstance(outputs, list)
                    and outputs
                    and isinstance(outputs[0], str)
                ):
                    output_text = outputs[0]

            if not output_text:
                choices = data.get("choices")
                if (
                    isinstance(choices, list)
                    and choices
                    and isinstance(choices[0], dict)
                ):
                    choice_text = choices[0].get("text")
                    if isinstance(choice_text, str):
                        output_text = choice_text

            token_stats_candidate = data.get("token_stats")
            if isinstance(token_stats_candidate, dict):
                token_stats = token_stats_candidate
        else:
            output_text = response.body_text.strip()

        if not output_text:
            raise ProtocolError("generate response does not contain output text")

        return NormalizedResult(
            status="ok",
            latency_ms=response.latency_ms,
            ttft_ms=response.ttft_ms,
            output_text=output_text,
            token_stats=token_stats,
        )

    def parse_stream_line(self, line: str) -> str:
        """从单行流式 chunk 中提取用户真正关心的文本增量。"""
        text = line.strip()
        if not text:
            return ""
        is_sse_data = False
        if text.startswith("data:"):
            is_sse_data = True
            text = text[5:].strip()
        if text == "[DONE]":
            return ""

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            if is_sse_data:
                return ""
            return text

        if not isinstance(payload, dict):
            return ""

        for key in ("text", "token", "generated_text", "output_text", "response"):
            value = payload.get(key)
            if isinstance(value, str):
                return value

        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return delta["content"]
            if isinstance(choices[0].get("text"), str):
                return choices[0]["text"]
        return ""
