from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union


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
        ignore_eos (bool): Whether to ignore the EOS token.
        sampling_seed (Optional[int]): Optional seed for deterministic sampling.
    """

    max_completion_tokens: Optional[int] = 128
    temperature: float = 1.0
    ignore_eos: bool = False
    sampling_seed: Optional[int] = None


class ChatCompletionOutput(BaseModel):
    """Output payload for a chat completion response.

    Attributes:
        message (ChatCompletionMessage): Final assistant message.
        rid (Optional[str]): Optional request identifier.
    """

    message: ChatCompletionMessage
    rid: Optional[str] = None


class LLMRequest(BaseModel):
    """Internal request model used by llmperf backends.

    Attributes:
        input (ChatCompletionInput): Chat completion input payload.
        sampling_params (SamplingParams): Sampling configuration.
        model (Optional[str]): Optional model name.
        stream (bool): Whether to request a streaming response.
        rid (Optional[str]): Optional request identifier.
        extra (Dict[str, object]): Backend-specific extra parameters.
    """

    input: ChatCompletionInput
    sampling_params: SamplingParams
    model: Optional[str] = None
    stream: bool = True
    rid: Optional[str] = None
    extra: Dict[str, object] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Internal response model produced by llmperf backends.

    Attributes:
        status_code (int): HTTP status code.
        error (Optional[str]): Optional error message.
        finish_reason (Optional[str]): Optional completion finish reason.
        model (Optional[str]): Model name returned by the backend.
        stream (bool): Whether the response was streamed.
        output (Optional[ChatCompletionOutput]): Optional completion output.
        extra (Dict[str, object]): Backend-specific extra fields.
        start_time (float): Request start time.
        finish_time (float): Request finish time.
        latency (float): End-to-end latency in seconds.
        ttft (float): Time to first token in seconds.
        tpot (float): Time per output token in seconds.
        itls (list[float]): Inter-token latencies in seconds.
    """

    status_code: int = 200
    error: Optional[str] = None
    finish_reason: Optional[str] = None

    model: Optional[str] = None
    stream: bool = True

    output: Optional[ChatCompletionOutput] = None
    extra: Dict[str, object] = Field(default_factory=dict)

    start_time: float = 0.0
    finish_time: float = 0.0
    latency: float = 0.0
    ttft: float = 0.0
    tpot: float = 0.0
    itls: list[float] = Field(default_factory=list)
