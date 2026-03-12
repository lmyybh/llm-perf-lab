"""命令层共享辅助函数。"""

import os

DEFAULT_API_KEY_ENV = "API_KEY"


def resolve_api_key(cli_api_key: str | None) -> str | None:
    """按命令行优先、固定环境变量兜底的顺序解析 API Key。"""
    if cli_api_key:
        return cli_api_key
    value = os.environ.get(DEFAULT_API_KEY_ENV)
    if value:
        return value
    return None
