"""Registry helpers for selecting dataset readers and parsers."""

import typer
from pathlib import Path

from llmperf.datasets.base import DatasetReader
from llmperf.datasets.jsonl import (
    JsonlDatasetReader,
    JsonlRecordParser,
    build_jsonl_parsers,
)


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
