from abc import ABC, abstractmethod
from typing import Optional, Callable, Literal
from dataclasses import dataclass

from llmperf.core import LLMRequest, LLMResponse


@dataclass
class StreamEvent:
    """One logical item emitted while parsing a streamed response.

    Attributes:
        type (Literal["content", "reasoning_content", "tool_call"]): Event kind
            used by the CLI renderer.
        text (Optional[str]): Incremental text for content-like events.
        tool_call_id (Optional[str]): Tool call identifier when available.
        tool_name (Optional[str]): Tool function name when available.
        tool_arguments_delta (Optional[str]): Incremental tool argument payload.
    """

    type: Literal["content", "reasoning_content", "tool_call"]
    text: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_arguments_delta: Optional[str] = None


class LLMBackend(ABC):
    """Abstract base class for llmperf backend implementations.

    Attributes:
        name (str): Backend name used for identification.
    """

    name: str = "base"

    @abstractmethod
    async def send(
        self,
        url: str,
        request: LLMRequest,
        on_chunk: Optional[Callable[[StreamEvent], None]] = None,
    ) -> LLMResponse:
        """Send one request through the backend.

        Args:
            url (str): Target endpoint URL.
            request (LLMRequest): Internal request model.
            on_chunk (Optional[Callable[[StreamEvent], None]]): Optional
                callback for streamed events.

        Returns:
            LLMResponse: Backend response model.
        """
        raise NotImplementedError
