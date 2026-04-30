"""Execution logic for the request command."""

import asyncio
from typing import Callable, Optional

from llmperf.backends import LLMBackend, OpenAIChatBackend, GenerateBackend, StreamEvent
from llmperf.commands.request.args import RequestCommandArgs
from llmperf.common import (
    LLMRequest,
    LLMResponse,
    apply_prompt_token_fallback,
    load_tokenizer,
)


def create_backend(args: RequestCommandArgs) -> LLMBackend:
    """Create the backend instance for the request command.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.

    Returns:
        LLMBackend: Configured backend instance.
    """

    if args.url.endswith("/generate"):
        return GenerateBackend(timeout=args.timeout)

    return OpenAIChatBackend(timeout=args.timeout)


def build_llm_request(args: RequestCommandArgs) -> LLMRequest:
    """Build the internal request model for the backend.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.

    Returns:
        LLMRequest: Internal request object passed to the backend.
    """
    input = args.parse_input()
    tokenizer = load_tokenizer(
        tokenizer_path=args.tokenizer_path,
        model_name=args.model,
        purpose="request",
        required=False,
    )
    return LLMRequest(
        input=input,
        sampling_params=args.parse_sampling_params(),
        model=args.model,
        stream=args.stream,
        rid=args.rid,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        extra={
            "prompt_tokens": (
                input.estimate_prompt_tokens_length(tokenizer)
                if tokenizer is not None
                else None
            )
        },
    )


def execute_request(
    args: RequestCommandArgs,
    on_chunk: Optional[Callable[[StreamEvent], None]] = None,
) -> LLMResponse:
    """Run the request command and return the aggregated response.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.
        on_chunk (Optional[Callable[[StreamEvent], None]]): Optional callback for
            streamed output chunks.

    Returns:
        LLMResponse: Aggregated backend response.
    """
    backend = create_backend(args)
    request = build_llm_request(args)
    response = asyncio.run(
        backend.send(
            url=args.url,
            request=request,
            on_chunk=on_chunk,
        )
    )
    return apply_prompt_token_fallback(request, response)
