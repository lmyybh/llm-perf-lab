"""Shared tokenizer helpers for local prompt token estimation."""

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmperf.errors import ConfigError, ValidationError
from llmperf.common.models import (
    ChatCompletionInput,
    LLMRequest,
    LLMResponse,
)


def load_tokenizer(
    tokenizer_path: Optional[Path],
    model_name: Optional[str],
    purpose: str,
    required: bool = True,
) -> Optional[PreTrainedTokenizerBase]:
    """Load a tokenizer from an explicit path or a fallback model identifier.

    Args:
        tokenizer_path (Optional[Path]): Preferred tokenizer path or
            identifier.
        model_name (Optional[str]): Fallback model identifier.
        purpose (str): Human-readable purpose string used in error messages.
        required (bool): Whether missing tokenizer configuration should raise.

    Returns:
        Optional[PreTrainedTokenizerBase]: Loaded tokenizer instance, or
            ``None`` when not required and no source is configured.

    Raises:
        ConfigError: Raised when no tokenizer source is available while
            required, or when loading fails.
    """
    source = str(tokenizer_path) if tokenizer_path is not None else model_name
    if source is None or source.strip() == "":
        if required:
            raise ConfigError(
                f"{purpose} requires --tokenizer-path or --model to load a tokenizer"
            )
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
    except Exception as exc:
        raise ConfigError(
            f"failed to load tokenizer for {purpose} from {source!r}: {exc}"
        ) from exc

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise ConfigError(
            f"loaded tokenizer for {purpose} is invalid: {type(tokenizer).__name__}"
        )

    return tokenizer


def apply_prompt_token_fallback(
    request: LLMRequest, response: LLMResponse
) -> LLMResponse:
    """Fill prompt token usage from request metadata when the backend omits it.

    Args:
        request (LLMRequest): Request sent to the backend.
        response (LLMResponse): Response returned by the backend.

    Returns:
        LLMResponse: Response with prompt token fallback applied when
            available.
    """
    if response.prompt_tokens > 0:
        return response

    prompt_tokens = request.extra.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        return response

    return response.model_copy(update={"prompt_tokens": prompt_tokens})
