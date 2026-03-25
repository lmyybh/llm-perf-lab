"""File-backed dataset implementation."""

import typer
from pathlib import Path
from typing import Iterable, Optional

from llmperf.common import LLMRequest
from llmperf.datasets.base import DatasetReader
from llmperf.datasets.jsonl import (
    JsonlDatasetReader,
    JsonlRecordParser,
    build_jsonl_parsers,
)


def select_jsonl_parser(mode: str) -> JsonlRecordParser:
    """Select the JSONL parser registered for a dataset mode.

    Args:
        mode (str): Dataset mode name.

    Returns:
        JsonlRecordParser: Parser matching ``mode``.

    Raises:
        typer.BadParameter: Raised when the mode is unsupported.
    """
    parsers = build_jsonl_parsers()

    for parser in parsers:
        if parser.name == mode:
            return parser

    raise typer.BadParameter(f"unsupported dataset mode: {mode}")


def select_reader(file: Path, mode: str) -> DatasetReader:
    """Select the dataset reader for a file and mode combination.

    Args:
        file (Path): Dataset file path.
        mode (str): Dataset mode name used by parser selection.

    Returns:
        DatasetReader: Reader capable of loading the requested file.

    Raises:
        typer.BadParameter: Raised when the file type is unsupported.
    """
    if file.suffix == ".jsonl":
        return JsonlDatasetReader(parser=select_jsonl_parser(mode=mode))

    raise typer.BadParameter(f"unsupported dataset file: {file}")


class FileDataset:
    """Dataset adapter that reads requests from a local file.

    Attributes:
        name (str): Dataset name used for identification.
        file (Path): Path to the dataset file.
        mode (str): Dataset mode name used by the reader registry.
        num_requests (Optional[int]): Optional maximum number of requests.
    """

    name: str = "file"

    def __init__(
        self, file: Path, mode: str, num_requests: Optional[int] = None
    ) -> None:
        """Initialize a file-backed dataset.

        Args:
            file (Path): Path to the dataset file.
            mode (str): Dataset mode name used by the reader registry.
            num_requests (Optional[int]): Optional maximum number of requests.
        """
        self.file = file
        self.mode = mode
        self.num_requests = num_requests

    def iter_requests(self) -> Iterable[LLMRequest]:
        """Yield parsed requests from the dataset file.

        Returns:
            Iterable[LLMRequest]: Requests parsed from the dataset file.
        """
        reader = select_reader(self.file, self.mode)

        return reader.iter_requests(self.file, self.num_requests)
