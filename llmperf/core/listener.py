"""Listener interfaces and helpers for request lifecycle events."""

from typing import Any

from llmperf.core.data import OpenAIRequestContext


class OpenAIBaseListener:
    """Base listener interface for observing request lifecycle events."""

    async def on_request_start(self, ctx: OpenAIRequestContext) -> None:
        """Handle the start of a request.

        Args:
            ctx (OpenAIRequestContext): Request context for the in-flight
                request.
        """
        pass

    async def on_request_finish(self, ctx: OpenAIRequestContext) -> None:
        """Handle request completion regardless of outcome.

        Args:
            ctx (OpenAIRequestContext): Request context for the completed
                request.
        """
        pass

    async def on_request_success(self, ctx: OpenAIRequestContext) -> None:
        """Handle a successfully completed request.

        Args:
            ctx (OpenAIRequestContext): Request context for the successful
                request.
        """
        pass

    async def on_request_error(self, ctx: OpenAIRequestContext, exc: Exception) -> None:
        """Handle a failed request.

        Args:
            ctx (OpenAIRequestContext): Request context for the failed request.
            exc (Exception): Exception raised while processing the request.
        """
        pass

    async def on_request_cancel(self, ctx: OpenAIRequestContext) -> None:
        """Handle request cancellation.

        Args:
            ctx (OpenAIRequestContext): Request context for the cancelled
                request.
        """
        pass

    async def on_chunk(self, ctx: OpenAIRequestContext, chunk_content: str) -> None:
        """Handle an incremental response chunk.

        Args:
            ctx (OpenAIRequestContext): Request context for the active request.
            chunk_content (str): Decoded content emitted by the stream.
        """
        pass


class OpenAILogListener(OpenAIBaseListener):
    """Print request lifecycle output to stdout."""

    async def on_request_success(self, ctx: OpenAIRequestContext) -> None:
        """Print the final response when a request succeeds.

        Args:
            ctx (OpenAIRequestContext): Request context for the successful
                request.
        """
        if not ctx.result.stream:
            print(ctx.result.generated_text)
        else:
            print()

    async def on_request_error(self, ctx: OpenAIRequestContext, exc: Exception) -> None:
        """Print an error message for a failed request.

        Args:
            ctx (OpenAIRequestContext): Request context for the failed request.
            exc (Exception): Exception raised by the request flow.
        """
        print(f"Error: {str(exc)}")

    async def on_request_cancel(self, ctx: OpenAIRequestContext) -> None:
        """Print a cancellation message.

        Args:
            ctx (OpenAIRequestContext): Request context for the cancelled
                request.
        """
        print(f"Cancel")

    async def on_chunk(self, ctx: OpenAIRequestContext, chunk_content: str) -> None:
        """Print streamed content as it arrives.

        Args:
            ctx (OpenAIRequestContext): Request context for the active request.
            chunk_content (str): Decoded content emitted by the stream.
        """
        print(chunk_content, end="", flush=True)


async def notify(
    listeners: list[OpenAIBaseListener] | None, method_name: str, *args: Any
) -> None:
    """Invoke a lifecycle callback on each registered listener.

    Args:
        listeners (list[OpenAIBaseListener] | None): Listener instances to
            notify.
        method_name (str): Listener method name to invoke.
        *args (Any): Positional arguments forwarded to the listener method.
    """
    if not listeners:
        return

    for listener in listeners:
        method = getattr(listener, method_name, None)
        if method is None:
            print(f"no method: {method_name}")
            continue

        try:
            await method(*args)
        except Exception as e:
            print(e)
