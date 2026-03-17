from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import List, Dict, Any


class OpenAIRequestInput(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "unknown"
    temperature: float = 1.0
    max_tokens: int = 100
    stream: bool = True


@dataclass
class OpenAIRequestOutput:
    status: int = 200
    error: str = ""
    stream: bool = False

    generated_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    start_time: float = 0.0
    finish_time: float = 0.0
    latency: float = 0.0
    ttft: float = 0.0
    tpot: float = 0.0
    itls: List[float] = field(default_factory=list)


@dataclass
class OpenAIRequestContext:
    url: str
    payload: dict[str, Any]
    headers: dict[str, str]
    result: OpenAIRequestOutput
