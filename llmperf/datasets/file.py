"""File-backed dataset implementation."""

from pathlib import Path
from typing import Iterable, Optional

from llmperf.core import LLMRequest
from llmperf.datasets.registry import select_reader


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
