from abc import ABC, abstractmethod
from typing import List, Optional, Callable
from models.dataset import DatasetMetadata

class BaseProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider name (e.g., 'huggingface', 'openml')."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        """Search the provider for datasets matching the query."""
        pass

    @abstractmethod
    def download(
        self,
        dataset: DatasetMetadata,
        dest_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event = None
    ) -> bool:
        """Download the dataset files to dest_dir."""
        pass
