"""JSONL dataset readers and record parsers."""

import json
import typer
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Mapping, NoReturn, Optional, Protocol, TypeAlias

from pydantic import ValidationError

from llmperf.common import LLMRequest

RecordType: TypeAlias = Mapping[str, object]
RequestPayload: TypeAlias = Mapping[str, object]


class JsonlRecordParser(Protocol):
    """Protocol implemented by mode-specific JSONL record parsers.

    Attributes:
        name (str): Parser name used for CLI selection.
    """

    name: str

    def parse_record(self, record: RecordType, line_number: int) -> LLMRequest:
        """Parse one JSONL record into an internal request model.

        Args:
            record (RecordType): Decoded JSONL record.
            line_number (int): One-based source line number.

        Returns:
            LLMRequest: Request validated from the source record.
        """
        ...


class BaseJsonlParser(ABC):
    """Base class for parsers that convert JSONL records into requests.

    Attributes:
        name (str): Parser name used for CLI selection.
    """

    name: str

    @abstractmethod
    def build_payload(self, record: RecordType) -> RequestPayload:
        """Build the raw request payload for model validation.

        Args:
            record (RecordType): Decoded JSONL record.

        Returns:
            RequestPayload: Payload accepted by ``LLMRequest.model_validate``.
        """
        raise NotImplementedError

    def parse_record(self, record: RecordType, line_number: int) -> LLMRequest:
        """Parse one JSONL record into an internal request model.

        Args:
            record (RecordType): Decoded JSONL record.
            line_number (int): One-based source line number.

        Returns:
            LLMRequest: Validated request model.

        Raises:
            typer.BadParameter: Raised when the record is structurally invalid.
        """
        try:
            data = self.build_payload(record=record)
        except KeyError as exc:
            self._raise_record_error(
                line_number=line_number,
                message=f"missing required field: {str(exc).strip("'")}",
                exc=exc,
            )
        except TypeError as exc:
            self._raise_record_error(
                line_number=line_number,
                message="record has an invalid object structure",
                exc=exc,
            )

        try:
            return LLMRequest.model_validate(data)
        except ValidationError as exc:
            self._raise_record_error(
                line_number=line_number,
                message="invalid data",
                exc=exc,
            )

    def _raise_record_error(
        self, line_number: int, message: str, exc: Exception
    ) -> NoReturn:
        """Raise a normalized CLI validation error for one record.

        Args:
            line_number (int): One-based source line number.
            message (str): Human-readable validation error.
            exc (Exception): Original exception to chain.

        Raises:
            typer.BadParameter: Always raised with record context.
        """
        raise typer.BadParameter(
            f"invalid {self.name} record at line {line_number}: {message}"
        ) from exc


class ZssJsonlParser(BaseJsonlParser):
    """Parser for the legacy ZSS JSONL request mode."""

    name: str = "zss-jsonl"

    def build_payload(self, record: RecordType) -> RequestPayload:
        """Build an ``LLMRequest`` payload from a ZSS JSONL record.

        Args:
            record (RecordType): Decoded JSONL record.

        Returns:
            RequestPayload: Payload ready for ``LLMRequest`` validation.
        """
        return {
            "input": {
                "messages": record["messages"],
                "tools": record.get("tools"),
            },
            "sampling_params": {
                "max_completion_tokens": record["max_tokens"],
                "temperature": record["temperature"],
                "seed": record["seed"],
                "frequency_penalty": record["frequency_penalty"],
                "repetition_penalty": record["repetition_penalty"],
                "presence_penalty": record["presence_penalty"],
            },
            "model": record["model"],
            "stream": record["stream"],
            # "rid": record["rid"],
            "chat_template_kwargs": record["chat_template_kwargs"],
        }


class OpenAIJsonlParser(BaseJsonlParser):
    """Parser for OpenAI-style conversation JSONL files."""

    name: str = "openai-jsonl"

    def build_payload(self, record: RecordType) -> RequestPayload:
        """Build an ``LLMRequest`` payload from an OpenAI JSONL record.

        Args:
            record (RecordType): Decoded JSONL record.

        Returns:
            RequestPayload: Payload ready for ``LLMRequest`` validation.
        """
        return {
            "input": {
                "messages": record["conversations"],
            },
        }


def build_jsonl_parsers() -> List[JsonlRecordParser]:
    """Build all supported JSONL record parsers.

    Returns:
        List[JsonlRecordParser]: Parsers keyed by their ``name`` attribute.
    """
    return [ZssJsonlParser(), OpenAIJsonlParser()]


class JsonlDatasetReader:
    """Dataset reader for JSONL files.

    Attributes:
        name (str): Reader name used for identification.
    """

    name: str = "jsonl"

    def __init__(self, parser: JsonlRecordParser) -> None:
        """Initialize the JSONL dataset reader.

        Args:
            parser (JsonlRecordParser): Mode-specific record parser.
        """
        self.parser = parser

    def supports(self, file: Path) -> bool:
        """Check whether the file uses the JSONL extension.

        Args:
            file (Path): Dataset file path.

        Returns:
            bool: ``True`` when the file has a ``.jsonl`` suffix.
        """
        return file.suffix == ".jsonl"

    def iter_requests(
        self, file: Path, num_requests: Optional[int] = None
    ) -> Iterable[LLMRequest]:
        """Yield requests parsed from a JSONL file.

        Args:
            file (Path): Dataset file path.
            num_requests (Optional[int]): Optional maximum number of requests.

        Returns:
            Iterable[LLMRequest]: Requests parsed from the JSONL file.

        Raises:
            typer.BadParameter: Raised when any line contains invalid JSON.
        """
        with file.open() as f:
            count = 0
            for line_number, line in enumerate(f, start=1):
                if num_requests is not None and count >= num_requests:
                    break

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise typer.BadParameter(
                        f"invalid JSON at line {line_number}: {exc.msg}"
                    ) from exc

                yield self.parser.parse_record(record=record, line_number=line_number)
                count += 1
