"""Dataset protocols used by benchmark input loaders."""

from pathlib import Path
from typing import Iterable, Optional, Protocol

from llmperf.core.models import LLMRequest


class Dataset(Protocol):
    """Protocol implemented by benchmark datasets.

    Attributes:
        name (str): Dataset name used for identification.
    """

    name: str

    def iter_requests(self) -> Iterable[LLMRequest]:
        """Yield benchmark requests from the dataset source.

        Returns:
            Iterable[LLMRequest]: Requests exposed by the dataset.
        """
        ...


class DatasetReader(Protocol):
    """Protocol implemented by dataset file readers.

    Attributes:
        name (str): Reader name used for identification.
    """

    name: str

    def supports(self, file: Path) -> bool:
        """Check whether the reader supports a dataset file.

        Args:
            file (Path): Dataset file path.

        Returns:
            bool: ``True`` when the file can be read by this reader.
        """
        ...

    def iter_requests(
        self, file: Path, num_requests: Optional[int] = None
    ) -> Iterable[LLMRequest]:
        """Yield benchmark requests from a dataset file.

        Args:
            file (Path): Dataset file path.
            num_requests (Optional[int]): Optional maximum number of requests.

        Returns:
            Iterable[LLMRequest]: Requests parsed from the file.
        """
        ...
