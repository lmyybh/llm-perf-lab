"""Shared request, response, and benchmark models for llmperf."""

from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union

from pydantic import BaseModel, Field

ChatTemplateKwargs: TypeAlias = Dict[str, object]
ExtraPayload: TypeAlias = Dict[str, object]


class ChatCompletionMessage(BaseModel):
    """A single chat message in an OpenAI-compatible conversation.

    Attributes:
        role (Literal["system", "user", "assistant", "tool"]): Message role.
        content (str): Message text content.
        reasoning_content (Optional[str]): Optional reasoning text content.
        tool_call_id (Optional[str]): Optional tool call identifier.
        name (Optional[str]): Optional participant name.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    reasoning_content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class Function(BaseModel):
    """A callable tool definition exposed to the model.

    Attributes:
        description (Optional[str]): Human-readable function description.
        name (str): Function name.
        parameters (Optional[object]): JSON schema for function arguments.
        strict (bool): Whether strict schema adherence is required.
    """

    description: Optional[str] = Field(default=None, examples=[None])
    name: str
    parameters: Optional[object] = None
    strict: bool = False


class Tool(BaseModel):
    """A tool entry included in the chat completion payload.

    Attributes:
        type (str): Tool type identifier.
        function (Function): Function metadata for this tool.
    """

    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoiceFuncName(BaseModel):
    """The function name selected in a tool choice.

    Attributes:
        name (Optional[str]): Selected function name.
    """

    name: Optional[str] = None


class ToolChoice(BaseModel):
    """Tool choice configuration for a chat completion request.

    Attributes:
        function (ToolChoiceFuncName): Selected function name payload.
        type (Literal["function"]): Tool choice type.
    """

    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


class FunctionResponse(BaseModel):
    """Function call payload returned by the assistant.

    Attributes:
        name (Optional[str]): Function name emitted by the model.
        arguments (Optional[str | Dict[str, Any]]): Serialized or decoded
            function arguments.
    """

    name: Optional[str] = None
    arguments: Optional[str | Dict[str, Any]] = None


class ToolCall(BaseModel):
    """A single tool call emitted in a chat completion response.

    Attributes:
        id (Optional[str]): Tool call identifier.
        index (Optional[int]): Stream chunk index for partial tool calls.
        type (Literal["function"]): Tool call type.
        function (FunctionResponse): Function payload for this tool call.
    """

    id: Optional[str] = None
    index: Optional[int] = None
    type: Literal["function"] = "function"
    function: FunctionResponse


class ChatCompletionInput(BaseModel):
    """Input payload for a chat completion request.

    Attributes:
        messages (List[ChatCompletionMessage]): Conversation messages.
        tools (Optional[List[Tool]]): Optional tool declarations.
        tool_choice (Union[ToolChoice, Literal["auto", "required", "none"]]):
            Tool selection strategy.
    """

    messages: List[ChatCompletionMessage]
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )


class SamplingParams(BaseModel):
    """Sampling configuration for text generation.

    Attributes:
        max_completion_tokens (Optional[int]): Maximum number of completion
            tokens to generate.
        temperature (float): Sampling temperature.
        presence_penalty (float): Penalty applied to new token presence.
        frequency_penalty (float): Penalty applied to repeated token frequency.
        repetition_penalty (Optional[float]): Optional repetition penalty.
        ignore_eos (bool): Whether to ignore the EOS token.
        seed (Optional[int]): Optional seed for deterministic sampling.
    """

    max_completion_tokens: Optional[int] = None
    temperature: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: Optional[float] = None

    ignore_eos: bool = False
    seed: Optional[int] = None


class ChatCompletionOutput(BaseModel):
    """Output payload for a chat completion response.

    Attributes:
        role (Optional[Literal["system", "user", "assistant", "tool"]]):
            Role of the returned message.
        content (Optional[str]): Final assistant text content.
        reasoning_content (Optional[str]): Optional reasoning text content.
        tool_calls (Optional[List[ToolCall]]): Optional tool calls emitted by
            the assistant.
    """

    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = Field(default=None, examples=[None])


class LLMRequest(BaseModel):
    """Internal request model used by llmperf backends.

    Attributes:
        input (ChatCompletionInput): Chat completion input payload.
        sampling_params (SamplingParams): Sampling configuration.
        model (Optional[str]): Optional model name.
        stream (bool): Whether to request a streaming response.
        rid (Optional[str]): Optional request identifier.
        chat_template_kwargs (Optional[ChatTemplateKwargs]): Optional chat
            template options forwarded to the backend.
        extra (ExtraPayload): Backend-specific extra parameters.
    """

    input: ChatCompletionInput
    sampling_params: SamplingParams = Field(default_factory=SamplingParams)
    model: Optional[str] = None
    stream: bool = True
    rid: Optional[str] = None
    chat_template_kwargs: Optional[ChatTemplateKwargs] = None
    extra: ExtraPayload = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Internal response model produced by llmperf backends.

    Attributes:
        status_code (int): HTTP status code.
        error (Optional[str]): Optional error message.
        finish_reason (Optional[str]): Optional completion finish reason.
        model (Optional[str]): Model name returned by the backend.
        stream (bool): Whether the response was streamed.
        output (ChatCompletionOutput): Aggregated completion output.
        extra (ExtraPayload): Backend-specific extra fields.
        prompt_tokens (int): Prompt token count reported by the backend.
        completion_tokens (int): Completion token count or streamed chunk count.
        start_time (float): Request start time.
        finish_time (float): Request finish time.
        latency (float): End-to-end latency in seconds.
        ttft (float): Time to first token in seconds.
        tpot (float): Time per output token in seconds.
        itls (list[float]): Inter-token latencies in seconds.
    """

    status_code: int = 0
    error: Optional[str] = None
    finish_reason: Optional[str] = None

    model: Optional[str] = None
    stream: bool = True

    output: ChatCompletionOutput = Field(default_factory=ChatCompletionOutput)
    extra: ExtraPayload = Field(default_factory=dict)

    prompt_tokens: int = 0
    completion_tokens: int = 0

    start_time: float = 0.0
    finish_time: float = 0.0
    latency: float = 0.0
    ttft: float = 0.0
    tpot: float = 0.0
    itls: list[float] = Field(default_factory=list)
