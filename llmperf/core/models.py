"""核心数据模型。"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RequestConfig:
    """单次 request 命令使用的内部配置。"""

    endpoint: str
    api_key: str | None
    timeout_ms: int
    model: str | None
    max_new_tokens: int | None
    temperature: float | None
    top_p: float | None
    disable_thinking: bool
    stream: bool
    messages: list[dict[str, Any]] | None
    prompt: str | None


@dataclass
class NormalizedResult:
    """把不同后端响应统一归一化后的结果。"""

    status: str
    latency_ms: int
    ttft_ms: int | None
    output_text: str
    usage: dict[str, Any] | None = None
    token_stats: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化时去掉值为 None 的字段，方便终端和 JSON 输出。"""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class ReplayRequest:
    """一条待回放请求。"""

    source_file: str
    source_index: int
    endpoint_type: str
    payload: dict[str, Any]
    stream: bool


@dataclass
class ReplayItemResult:
    """一条 replay 请求的最终观测结果。"""

    source_file: str
    source_index: int
    status: str
    latency_ms: int
    request_start_time: float | None = None
    ttft_ms: int | None = None
    output_text: str = ""
    output_tokens: int | None = None
    tpot_ms: float | None = None
    accept_len: float | None = None
    accept_rate: float | None = None
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """把逐请求结果转成紧凑字典，方便保存 JSONL。"""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}
