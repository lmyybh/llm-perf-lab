"""Execution logic for the request command."""

import asyncio
from typing import Callable, Optional

import typer

from llmperf.backends import GenerateBackend, LLMBackend, OpenAIChatBackend, StreamEvent
from llmperf.commands.request.args import RequestCommandArgs
from llmperf.commands.request.dataset import select_dataset_input
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


def build_dataset_llm_request(
    args: RequestCommandArgs,
    on_dataset_selection: Optional[Callable[[str], None]] = None,
) -> LLMRequest:
    """Build one request from a dataset-backed input selection."""
    if args.url.endswith("/generate"):
        raise typer.BadParameter(
            "--target-input-tokens can only be used with an OpenAI chat URL"
        )

    args.detect_openai_input_mode()
    assert args.target_input_tokens is not None

    tokenizer = load_tokenizer(
        tokenizer_path=args.tokenizer_path,
        model_name=args.model,
        purpose="dataset request selection",
        required=True,
    )
    assert tokenizer is not None

    selection = select_dataset_input(
        dataset_file=args.dataset_file,
        dataset_mode=args.dataset_mode,
        target_input_tokens=args.target_input_tokens,
        tolerance=args.input_token_tolerance,
        with_tools=args.with_tools,
        seed=args.seed,
        tokenizer=tokenizer,
    )

    if on_dataset_selection is not None:
        on_dataset_selection(selection.description)

    return LLMRequest(
        input=selection.input,
        sampling_params=args.parse_sampling_params(),
        model=args.model,
        stream=args.stream,
        rid=args.rid,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        extra={"prompt_tokens": selection.prompt_tokens},
    )


def execute_request(
    args: RequestCommandArgs,
    on_chunk: Optional[Callable[[StreamEvent], None]] = None,
    on_dataset_selection: Optional[Callable[[str], None]] = None,
) -> LLMResponse:
    """Run the request command and return the aggregated response.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.
        on_chunk (Optional[Callable[[StreamEvent], None]]): Optional callback for
            streamed output chunks.
        on_dataset_selection (Optional[Callable[[str], None]]): Optional callback
            for rendering selected dataset sample metadata.

    Returns:
        LLMResponse: Aggregated backend response.
    """
    backend = create_backend(args)
    request = (
        build_dataset_llm_request(args, on_dataset_selection=on_dataset_selection)
        if args.target_input_tokens is not None
        else build_llm_request(args)
    )
    response = asyncio.run(
        backend.send(
            url=args.url,
            request=request,
            on_chunk=on_chunk,
        )
    )
    return apply_prompt_token_fallback(request, response)
