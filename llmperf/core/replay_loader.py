"""SGLang dump 加载器。

负责把 pkl 中的 `GenerateReqInput` 结构转换成 replay 可以直接发送的请求体。
"""

import os
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from llmperf.core.errors import ConfigError, InputError
from llmperf.core.models import ReplayRequest
from llmperf.core.validator import detect_endpoint_type

_EXCLUDED_FIELDS = {
    "validation_time",
    "received_time",
    "received_time_perf",
    "http_worker_ipc",
    "log_metrics",
    "rid",
}


def _iter_dump_paths(dump_path: str) -> Iterable[Path]:
    """把用户给定的路径展开成有序的 pkl 文件集合。"""
    path = Path(dump_path)
    if not path.exists():
        raise ConfigError(f"--dump-path does not exist: {dump_path}")
    if path.is_file():
        if path.suffix != ".pkl":
            raise ConfigError("--dump-path file must end with .pkl")
        yield path
        return

    paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".pkl")
    if not paths:
        raise ConfigError(f"no .pkl files found in {dump_path}")
    for item in paths:
        yield item


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """把 dataclass、对象实例或字典统一转成 dict。"""
    if isinstance(value, dict):
        return dict(value)
    fields = getattr(type(value), "__dataclass_fields__", None)
    if fields:
        return {name: getattr(value, name) for name in fields}
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return dict(data)
    raise InputError(f"unsupported replay request object type: {type(value).__name__}")


def _build_payload(raw_request: Any) -> tuple[dict[str, Any], bool]:
    """从原始请求对象中提取可回放的 payload 和 stream 配置。"""
    data = _coerce_mapping(raw_request)
    payload: dict[str, Any] = {}
    for key, value in data.items():
        if key in _EXCLUDED_FIELDS or value is None:
            continue
        payload[key] = value

    if "input_ids" not in payload and "text" not in payload:
        raise InputError("replay request is missing both input_ids and text")
    stream = bool(payload.get("stream", False))
    return payload, stream


def validate_replay_endpoint(endpoint: str) -> None:
    """当前 replay 仅支持回放到 SGLang `/generate`。"""
    endpoint_type = detect_endpoint_type(endpoint)
    if endpoint_type != "generate":
        raise ConfigError("--endpoint must end with /generate for replay")


def load_replay_requests(
    dump_path: str, limit: int | None = None
) -> list[ReplayRequest]:
    """读取一个或多个 dump 文件，生成用于 replay 的请求列表。"""
    requests: list[ReplayRequest] = []
    for path in _iter_dump_paths(dump_path):
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        entries = payload.get("requests")
        if not isinstance(entries, list):
            raise InputError(f"{path} does not contain a valid requests list")
        for index, item in enumerate(entries):
            if not isinstance(item, tuple) or not item:
                raise InputError(f"{path} request[{index}] is not a non-empty tuple")
            request_obj = item[0]
            replay_payload, stream = _build_payload(request_obj)
            requests.append(
                ReplayRequest(
                    source_file=os.path.basename(path),
                    source_index=index,
                    endpoint_type="generate",
                    payload=replay_payload,
                    stream=stream,
                )
            )
            if limit is not None and len(requests) >= limit:
                return requests

    if not requests:
        raise InputError("no replayable requests found in dump files")
    return requests
