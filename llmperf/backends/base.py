from abc import ABC, abstractmethod
from typing import Optional, Callable

from llmperf.core import LLMRequest, LLMResponse


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
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """Send one request through the backend.

        Args:
            url (str): Target endpoint URL.
            request (LLMRequest): Internal request model.
            on_chunk (Optional[Callable[[str], None]]): Optional callback for
                streamed text chunks.

        Returns:
            LLMResponse: Backend response model.
        """
        raise NotImplementedError
