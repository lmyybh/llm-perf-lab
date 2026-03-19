"""Core request and response data models."""

from dataclasses import dataclass, field
from typing import Any, TypeAlias

ChatMessage: TypeAlias = dict[str, str]
ChatMessages: TypeAlias = list[ChatMessage]
RequestPayload: TypeAlias = dict[str, Any]
RequestHeaders: TypeAlias = dict[str, str]


@dataclass
class OpenAIRequestInput:
    """Represents the request payload sent to a chat completions endpoint.

    Attributes:
        messages (ChatMessages): Chat message list included in the request.
        model (str): Model name sent to the endpoint.
        temperature (float): Sampling temperature for the request.
        max_tokens (int): Maximum number of tokens to generate.
        stream (bool): Whether to request streaming output.
    """

    messages: ChatMessages
    model: str = "unknown"
    stream: bool = True
    tools: list | None = None
    rid: str | None = None
    temperature: float = 1.0
    max_tokens: int = 100
    seed: int | None = None
    frequency_penalty: float = 0.1
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    chat_template_kwargs: dict | None = None


@dataclass
class OpenAIRequestOutput:
    """Stores response content and latency metrics for one request.

    Attributes:
        status (int): HTTP status code returned by the endpoint.
        error (str): Error message captured during request processing.
        stream (bool): Whether the request was made in streaming mode.
        generated_text (str): Aggregated generated text from the response.
        prompt_tokens (int): Prompt token count reported by the endpoint.
        completion_tokens (int): Completion token count or chunk count.
        start_time (float): Monotonic start timestamp.
        finish_time (float): Monotonic finish timestamp.
        latency (float): Total elapsed request latency in seconds.
        ttft (float): Time to first token in seconds.
        tpot (float): Time per output token in seconds.
        itls (list[float]): Inter-token latencies collected during streaming.
    """

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
    itls: list[float] = field(default_factory=list)


@dataclass
class OpenAIRequestContext:
    """Carries request metadata and the mutable request result state.

    Attributes:
        url (str): Target request URL.
        payload (RequestPayload): JSON payload sent to the endpoint.
        headers (RequestHeaders): HTTP headers sent with the request.
        result (OpenAIRequestOutput): Mutable request result and metrics.
    """

    url: str
    payload: RequestPayload
    headers: RequestHeaders
    result: OpenAIRequestOutput
