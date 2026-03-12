"""适配器协议定义。

所有具体适配器都要遵守这个接口，命令层才能统一调用。
"""

from typing import Protocol

from llmperf.core.executor import ResponseEnvelope
from llmperf.core.models import NormalizedResult, RequestConfig


class Adapter(Protocol):
    """适配器抽象接口。"""

    endpoint_type: str

    def build_payload(self, config: RequestConfig) -> dict: ...

    def parse_response(
        self, response: ResponseEnvelope, config: RequestConfig
    ) -> NormalizedResult: ...

    def parse_stream_line(self, line: str) -> str: ...
