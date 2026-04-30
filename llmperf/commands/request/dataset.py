"""Dataset-backed input selection for the request command."""

from dataclasses import dataclass
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from llmperf.commands.bench.args import DatasetMode
from llmperf.common import ChatCompletionInput
from llmperf.datasets import FileDataset, RandomDataset
from llmperf.errors import ConfigError, ValidationError


@dataclass(frozen=True)
class DatasetSelection:
    """A dataset sample selected for one request."""

    input: ChatCompletionInput
    prompt_tokens: int
    description: str


def _token_range(target_input_tokens: int, tolerance: int) -> tuple[int, int]:
    lower_bound = max(1, target_input_tokens - tolerance)
    upper_bound = target_input_tokens + tolerance
    return lower_bound, upper_bound


def _has_non_empty_tools(input: ChatCompletionInput) -> bool:
    return input.tools is not None and len(input.tools) > 0


def _is_tool_eligible(input: ChatCompletionInput, with_tools: bool) -> bool:
    has_tools = _has_non_empty_tools(input)
    return has_tools if with_tools else not has_tools


def _estimate_prompt_tokens(
    input: ChatCompletionInput, tokenizer: PreTrainedTokenizerBase
) -> int:
    try:
        return input.estimate_prompt_tokens_length(tokenizer)
    except:
        return -1


def _closest_prompt_tokens(
    current: int | None, candidate: int, target_input_tokens: int
) -> int:
    if current is None:
        return candidate

    current_distance = abs(current - target_input_tokens)
    candidate_distance = abs(candidate - target_input_tokens)
    if candidate_distance < current_distance:
        return candidate

    return current


def _finalize_input(
    input: ChatCompletionInput, with_tools: bool
) -> ChatCompletionInput:
    if with_tools:
        input.tool_choice = "auto"
    return input


def _build_description(
    mode: DatasetMode,
    prompt_tokens: int,
    target_input_tokens: int,
    tolerance: int,
    with_tools: bool,
    source: str | None = None,
) -> str:
    source_part = f" source={source}" if source is not None else ""
    return (
        f"Selected dataset sample: mode={mode.value}{source_part} "
        f"prompt_tokens={prompt_tokens} target={target_input_tokens} "
        f"tolerance={tolerance} with_tools={str(with_tools).lower()}"
    )


def select_file_dataset_input(
    dataset_file: Path,
    dataset_mode: DatasetMode,
    target_input_tokens: int,
    tolerance: int,
    with_tools: bool,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetSelection:
    """Select the first file-backed dataset sample within the target range."""
    lower_bound, upper_bound = _token_range(target_input_tokens, tolerance)
    scanned = 0
    eligible = 0
    closest: int | None = None

    dataset = FileDataset(file=dataset_file, mode=dataset_mode.value)
    for request in dataset.iter_requests():
        scanned += 1
        if not isinstance(request.input, ChatCompletionInput):
            continue

        input = request.input
        if not _is_tool_eligible(input, with_tools):
            continue

        eligible += 1
        input = _finalize_input(input, with_tools)
        prompt_tokens = _estimate_prompt_tokens(input, tokenizer)
        closest = _closest_prompt_tokens(closest, prompt_tokens, target_input_tokens)
        if lower_bound <= prompt_tokens <= upper_bound:
            return DatasetSelection(
                input=input,
                prompt_tokens=prompt_tokens,
                description=_build_description(
                    mode=dataset_mode,
                    prompt_tokens=prompt_tokens,
                    target_input_tokens=target_input_tokens,
                    tolerance=tolerance,
                    with_tools=with_tools,
                ),
            )

    raise ConfigError(
        "no dataset sample matched target range "
        f"[{lower_bound}, {upper_bound}] with with_tools={str(with_tools).lower()}; "
        f"scanned={scanned}, eligible={eligible}, closest_prompt_tokens={closest}"
    )


def select_random_dataset_input(
    target_input_tokens: int,
    tolerance: int,
    seed: int | None,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetSelection:
    """Generate one random dataset sample and validate its prompt length."""
    lower_bound, upper_bound = _token_range(target_input_tokens, tolerance)
    dataset = RandomDataset(
        tokenizer=tokenizer,
        num_requests=1,
        min_input_tokens=target_input_tokens,
        max_input_tokens=target_input_tokens,
        min_output_tokens=1,
        max_output_tokens=1,
        seed=seed if seed is not None else 0,
    )

    request = next(iter(dataset.iter_requests()))
    if not isinstance(request.input, ChatCompletionInput):
        raise ConfigError("random dataset produced a non-chat request")

    prompt_tokens = request.extra.get("prompt_tokens")
    if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
        prompt_tokens = _estimate_prompt_tokens(request.input, tokenizer)

    if not lower_bound <= prompt_tokens <= upper_bound:
        raise ConfigError(
            "random dataset prompt token count is outside target range "
            f"[{lower_bound}, {upper_bound}]; target={target_input_tokens}, "
            f"tolerance={tolerance}, actual_prompt_tokens={prompt_tokens}"
        )

    return DatasetSelection(
        input=request.input,
        prompt_tokens=prompt_tokens,
        description=_build_description(
            mode=DatasetMode.random,
            prompt_tokens=prompt_tokens,
            target_input_tokens=target_input_tokens,
            tolerance=tolerance,
            with_tools=False,
            source="random",
        ),
    )


def select_dataset_input(
    dataset_file: Path | None,
    dataset_mode: DatasetMode,
    target_input_tokens: int,
    tolerance: int,
    with_tools: bool,
    seed: int | None,
    tokenizer: PreTrainedTokenizerBase,
) -> DatasetSelection:
    """Select or generate one chat input from a request dataset mode."""
    if dataset_mode == DatasetMode.random:
        return select_random_dataset_input(
            target_input_tokens=target_input_tokens,
            tolerance=tolerance,
            seed=seed,
            tokenizer=tokenizer,
        )

    if dataset_file is None:
        raise ConfigError("file-backed dataset modes require --dataset-file")

    return select_file_dataset_input(
        dataset_file=dataset_file,
        dataset_mode=dataset_mode,
        target_input_tokens=target_input_tokens,
        tolerance=tolerance,
        with_tools=with_tools,
        tokenizer=tokenizer,
    )
