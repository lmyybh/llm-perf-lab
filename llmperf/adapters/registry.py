"""适配器注册表。

不同 endpoint 后缀对应不同的 payload 映射和响应解析逻辑。
"""

from llmperf.adapters.base import Adapter
from llmperf.adapters.chat_completions import ChatCompletionsAdapter
from llmperf.adapters.generate import GenerateAdapter
from llmperf.core.errors import ConfigError

_ADAPTERS: dict[str, Adapter] = {
    "chat_completions": ChatCompletionsAdapter(),
    "generate": GenerateAdapter(),
}


def get_adapter(endpoint_type: str) -> Adapter:
    """按 endpoint 类型返回适配器实例。"""
    adapter = _ADAPTERS.get(endpoint_type)
    if adapter is None:
        raise ConfigError(f"unsupported endpoint type: {endpoint_type}")
    return adapter
