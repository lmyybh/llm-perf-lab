from typing import List

from .data import OpenAIRequestContext


class OpenAIBaseListener:
    async def on_request_start(self, ctx: OpenAIRequestContext):
        pass

    async def on_request_finish(self, ctx: OpenAIRequestContext):
        pass

    async def on_request_success(self, ctx: OpenAIRequestContext):
        pass

    async def on_request_error(self, ctx: OpenAIRequestContext, exc: Exception):
        pass

    async def on_request_cancel(self, ctx: OpenAIRequestContext):
        pass

    async def on_chunk(self, ctx: OpenAIRequestContext, chunk_content: str):
        pass


class OpenAILogListener(OpenAIBaseListener):
    async def on_request_success(self, ctx: OpenAIRequestContext):
        if not ctx.result.stream:
            print(ctx.result.generated_text)
        else:
            print()

    async def on_request_error(self, ctx: OpenAIRequestContext, exc: Exception):
        print(f"Error: {str(exc)}")

    async def on_request_cancel(self, ctx: OpenAIRequestContext):
        print(f"Cancel")

    async def on_chunk(self, ctx: OpenAIRequestContext, chunk_content: str):
        print(chunk_content, end="", flush=True)


async def notify(listeners: List[OpenAIBaseListener] | None, method_name: str, *args):
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
