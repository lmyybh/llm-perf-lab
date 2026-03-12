"""命令层共享辅助函数。"""

import os

from llmperf.core.errors import ConfigError


def resolve_api_key(cli_api_key: str | None, api_key_env: str | None) -> str | None:
    """按命令行优先、环境变量兜底的顺序解析 API Key。"""
    if cli_api_key:
        return cli_api_key
    if api_key_env:
        value = os.environ.get(api_key_env)
        if not value:
            raise ConfigError(f"environment variable {api_key_env} is empty or not set")
        return value
    return None
