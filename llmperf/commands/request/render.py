"""Terminal rendering helpers for the request command."""

from typing import Callable, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from llmperf.backends import StreamEvent
from llmperf.common import LLMResponse, ChatCompletionOutput, GenerateOutput

console = Console()
REASONING_CONTENT_COLOR = "dim cyan"
CONTENT_COLOR = "white"
TOOL_CALL_TITLE_COLOR = "bold yellow"
TOOL_CALL_KEY_COLOR = "yellow"
TOOL_CALL_VALUE_COLOR = "white"


def build_kv_line(
    key: str,
    value: Optional[str],
    key_style: str = TOOL_CALL_KEY_COLOR,
    value_style: str = TOOL_CALL_VALUE_COLOR,
    width: int = 10,
) -> Text:
    """Build one aligned key-value line for terminal rendering.

    Args:
        key (str): Label rendered on the left.
        value (Optional[str]): Text rendered on the right.
        key_style (str): Rich style used for the key column.
        value_style (str): Rich style used for the value column.
        width (int): Display width reserved for the key column.

    Returns:
        Text: Styled line ready to print with Rich.
    """
    rendered = Text()
    rendered.append(f"{key:<{width}}", style=key_style)
    rendered.append(": ", style="dim")
    rendered.append("" if value is None else value, style=value_style)
    return rendered


def create_chunk_printer(stream: bool) -> Optional[Callable[[StreamEvent], None]]:
    """Create a chunk printer for streaming output.

    Args:
        stream (bool): Whether the request uses streaming output.

    Returns:
        Optional[Callable[[StreamEvent], None]]: A chunk printer for streamed
            events, otherwise ``None``.
    """
    if not stream:
        return None

    active_tool_call_id: Optional[str] = None

    def _on_stream_chunk(event: StreamEvent) -> None:
        nonlocal active_tool_call_id

        if event.type in {"content", "text"} and event.text is not None:
            console.out(event.text, end="", style=CONTENT_COLOR)
        elif event.type == "reasoning_content" and event.text is not None:
            console.out(event.text, end="", style=REASONING_CONTENT_COLOR)
        elif event.type == "tool_call":
            is_new_tool = (
                event.tool_call_id is not None
                and event.tool_call_id != active_tool_call_id
            )
            if is_new_tool:
                active_tool_call_id = event.tool_call_id
                console.print(f"[Tool Call]", style=TOOL_CALL_TITLE_COLOR)
                console.print(build_kv_line("id", event.tool_call_id))
                console.print(build_kv_line("name", event.tool_name))
                console.print(build_kv_line("arguments", ""), end="")

            if event.tool_arguments_delta:
                console.out(event.tool_arguments_delta, end="")

    return _on_stream_chunk


def render_header(stream: bool) -> None:
    """Render the request mode header.

    Args:
        stream (bool): Whether the request uses streaming output.

    Returns:
        None: This function writes the header to the terminal.
    """
    console.rule("Stream") if stream else console.rule("No-Stream")


def render_dataset_selection(description: str) -> None:
    """Render the dataset sample selected for this request."""
    console.print(description, style="cyan")


def build_response_summary(response: LLMResponse) -> Table:
    """Build a summary table for the aggregated response.

    Args:
        response (LLMResponse): Aggregated backend response.

    Returns:
        Table: A terminal table containing key response metadata.
    """
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Status", str(response.status_code))

    if response.model:
        table.add_row("Model", response.model)

    table.add_row("Latency", f"{response.latency:.2f} s")

    if response.ttft > 0:
        table.add_row("TTFT", f"{(response.ttft * 1000):.2f} ms")

    if response.tpot > 0:
        table.add_row("TPOT", f"{(response.tpot * 1000):.2f} ms")

    if response.prompt_tokens > 0:
        table.add_row("Prompt Tokens", str(response.prompt_tokens))

    if response.completion_tokens > 0:
        table.add_row("Completion Tokens", str(response.completion_tokens))

    if response.finish_reason:
        table.add_row("Finish Reason", response.finish_reason)

    return table


def render_no_stream_response(response: LLMResponse) -> None:
    """Render the visible response body for non-stream output.

    Args:
        response (LLMResponse): Aggregated backend response.

    Returns:
        None: This function writes the visible response to the terminal.
    """
    if isinstance(response.output, ChatCompletionOutput):
        if response.output.reasoning_content is not None:
            console.print(
                response.output.reasoning_content, style=REASONING_CONTENT_COLOR
            )

        if response.output.content is not None:
            console.print(response.output.content, style=CONTENT_COLOR)

        if response.output.tool_calls is not None:
            for tool_call in response.output.tool_calls:
                console.print(f"[Tool Call]", style=TOOL_CALL_TITLE_COLOR)
                console.print(build_kv_line("id", tool_call.id))
                console.print(build_kv_line("name", tool_call.function.name))
                console.print(
                    build_kv_line("arguments", str(tool_call.function.arguments))
                )
    elif isinstance(response.output, GenerateOutput):
        if response.output.text is not None:
            console.print(response.output.text, style=CONTENT_COLOR)


def render_response(response: LLMResponse, stream: bool) -> None:
    """Render the response body, errors, and summary sections.

    Args:
        response (LLMResponse): Aggregated backend response.
        stream (bool): Whether the response body was streamed.

    Returns:
        None: This function writes the response to the terminal.
    """
    if not stream:
        render_no_stream_response(response)

    console.print()

    if response.error:
        console.rule("Error")
        console.print(response.error, style="bold red")

    console.rule("Response")
    console.print(build_response_summary(response))
    console.rule()
