from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Union


class ChatCompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    reasoning_content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class Function(BaseModel):
    description: Optional[str] = Field(default=None, examples=[None])
    name: str
    parameters: Optional[object] = None
    strict: bool = False


class Tool(BaseModel):
    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoiceFuncName(BaseModel):
    name: Optional[str] = None


class ToolChoice(BaseModel):
    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


class ChatCompletionInput(BaseModel):
    messages: List[ChatCompletionMessage]
    tools: Optional[List[Tool]] = Field(default=None, examples=[None])
    tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]] = Field(
        default="auto", examples=["none"]
    )


class SamplingParams(BaseModel):
    max_new_tokens: int = 128
    temperature: float = 1.0
    ignore_eos: bool = False
    sampling_seed: Optional[int] = None


class ChatCompletionOutput(BaseModel):
    message: ChatCompletionMessage
    rid: Optional[str] = None


class LLMRequest(BaseModel):
    input: ChatCompletionInput
    sampling_params: SamplingParams
    model: Optional[str] = None
    stream: bool = True
    rid: Optional[str] = None
    extra: Dict[str, object] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    status_code: int = 200
    error: Optional[str] = None
    finish_reason: Optional[str] = None

    model: Optional[str] = None
    stream: bool = True

    output: ChatCompletionOutput
    extra: Dict[str, object] = Field(default_factory=dict)

    start_time: float = 0.0
    finish_time: float = 0.0
    latency: float = 0.0
    ttft: float = 0.0
    tpot: float = 0.0
    itls: list[float] = Field(default_factory=list)
