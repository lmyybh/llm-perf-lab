"""Argument models for the benchmark command."""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import typer
from pydantic import BaseModel, Field


class DatasetMode(str, Enum):
    """Supported dataset modes for the benchmark command."""

    openai = "openai-jsonl"
    zss = "zss-jsonl"
    random = "random"


class BenchCommonArgs(BaseModel):
    """Common arguments for benchmark execution.

    Attributes:
        url (str): Target endpoint URL.
        num_requests (Optional[int]): Optional request count limit.
        qps (Optional[float]): Optional request rate cap.
        max_concurrency (Optional[int]): Optional in-flight request cap.
        timeout (float): Request timeout in seconds.
        model (Optional[str]): Model name sent in benchmark requests.
        tokenizer_path (Optional[Path]): Optional local or remote tokenizer path
            used by the random dataset.
        temperature (Optional[float]): Optional temperature override.
        max_completion_tokens (Optional[int]): Optional max token override.
        ignore_eos (Optional[bool]): Optional EOS override.
        enable_thinking (Optional[bool]): Optional thinking override.
    """

    url: str

    num_requests: Optional[int] = Field(default=None, gt=0)
    qps: Optional[float] = Field(default=None, gt=0)
    max_concurrency: Optional[int] = Field(default=None, gt=0)
    timeout: float = Field(default=300.0, gt=0)

    model: Optional[str] = None
    tokenizer_path: Optional[Path] = None
    temperature: Optional[float] = None
    max_completion_tokens: Optional[int] = None
    ignore_eos: Optional[bool] = None
    enable_thinking: Optional[bool] = None


class FileDatasetArgs(BaseModel):
    """Arguments for file-backed benchmark datasets.

    Attributes:
        mode (Literal["openai-jsonl", "zss-jsonl"]): Dataset file mode.
        file (Path): Dataset file path.
    """

    mode: Literal["openai-jsonl", "zss-jsonl"]
    file: Path


class RandomDatasetArgs(BaseModel):
    """Arguments for randomly generated benchmark datasets.

    Attributes:
        mode (Literal["random"]): Random dataset mode discriminator.
        seed (int): Random seed for request generation.
        min_input_tokens (int): Minimum input token count per request.
        max_input_tokens (int): Maximum input token count per request.
        min_output_tokens (int): Minimum output token target per request.
        max_output_tokens (int): Maximum output token target per request.
    """

    mode: Literal["random"]
    seed: int = 0
    min_input_tokens: int = Field(default=64, gt=0)
    max_input_tokens: int = Field(default=256, gt=0)
    min_output_tokens: int = Field(default=64, gt=0)
    max_output_tokens: int = Field(default=256, gt=0)


DatasetArgs = FileDatasetArgs | RandomDatasetArgs


class BenchCommandArgs(BaseModel):
    """Arguments required to run the benchmark command.

    Attributes:
        common (BenchCommonArgs): Benchmark runtime configuration.
        dataset (DatasetArgs): Dataset-specific configuration.
    """

    common: BenchCommonArgs
    dataset: DatasetArgs


def build_bench_dataset_args(
    mode: DatasetMode,
    file: Optional[Path],
    seed: int = 0,
    min_input_tokens: int = 64,
    max_input_tokens: int = 256,
    min_output_tokens: int = 64,
    max_output_tokens: int = 256,
) -> DatasetArgs:
    """Build dataset arguments for the selected benchmark mode.

    Args:
        mode (DatasetMode): Dataset mode selected from the CLI.
        file (Optional[Path]): Optional dataset file path.
        seed (int): Random seed for random mode generation.
        min_input_tokens (int): Minimum input tokens for random mode.
        max_input_tokens (int): Maximum input tokens for random mode.
        min_output_tokens (int): Minimum output tokens for random mode.
        max_output_tokens (int): Maximum output tokens for random mode.

    Returns:
        DatasetArgs: Dataset configuration matching the selected mode.

    Raises:
        typer.BadParameter: Raised when required file input is missing.
    """
    if mode in {DatasetMode.openai, DatasetMode.zss}:
        if file is None or not file.is_file():
            raise typer.BadParameter(
                "file-backed dataset modes require a valid --file path"
            )

        return FileDatasetArgs(mode=mode.value, file=file)

    if mode is DatasetMode.random:
        return RandomDatasetArgs(
            mode=mode.value,
            seed=seed,
            min_input_tokens=min_input_tokens,
            max_input_tokens=max_input_tokens,
            min_output_tokens=min_output_tokens,
            max_output_tokens=max_output_tokens,
        )

    raise typer.BadParameter(f"unsupported dataset mode: {mode}")
