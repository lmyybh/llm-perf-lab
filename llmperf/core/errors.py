"""统一错误定义。

所有命令最终都依赖这些错误类型来决定：
- 终端上显示什么错误标签
- 进程退出码应该是多少
"""


class LLMPerfError(Exception):
    """项目内所有可预期业务错误的基类。"""

    error_type = "protocol_error"
    exit_code = 3

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigError(LLMPerfError):
    """配置层错误，例如 endpoint 或环境变量不合法。"""

    error_type = "config_error"
    exit_code = 2


class InputError(LLMPerfError):
    """用户输入参数本身不合法。"""

    error_type = "input_error"
    exit_code = 2


class NetworkError(LLMPerfError):
    """网络请求建立或传输阶段失败。"""

    error_type = "network_error"
    exit_code = 3


class HttpError(LLMPerfError):
    """HTTP 已返回，但状态码不在成功范围内。"""

    error_type = "http_error"
    exit_code = 3

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ProtocolError(LLMPerfError):
    """服务端返回格式不符合预期。"""

    error_type = "protocol_error"
    exit_code = 3


class TimeoutError(LLMPerfError):
    """请求超时。"""

    error_type = "timeout_error"
    exit_code = 3
