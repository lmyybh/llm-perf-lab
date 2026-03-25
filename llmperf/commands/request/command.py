"""Entrypoint orchestration for the request command."""

from llmperf.commands.request.args import RequestCommandArgs
from llmperf.commands.request.render import (
    create_chunk_printer,
    render_header,
    render_response,
)
from llmperf.commands.request.service import execute_request
from llmperf.common import LLMResponse


def run_request_command(args: RequestCommandArgs) -> LLMResponse:
    """Run the request command and render the aggregated response.

    Args:
        args (RequestCommandArgs): Parsed request command arguments.

    Returns:
        LLMResponse: Aggregated backend response.
    """
    render_header(args.stream)
    response = execute_request(args, on_chunk=create_chunk_printer(args.stream))
    render_response(response, stream=args.stream)
    return response
