import os
from typing import List, Optional, Callable
from providers.base import BaseProvider
from models.dataset import DatasetMetadata
from networking.client import NetworkClient
from app_logging.logger import AprilLogger
from config.manager import ConfigManager

class KaggleProvider(BaseProvider):
    def __init__(self, client: Optional[NetworkClient] = None, config_manager: Optional[ConfigManager] = None):
        self.client = client or NetworkClient()
        self.config_manager = config_manager or ConfigManager()
        self.logger = AprilLogger.get_logger()
        self.api = None
        self._init_api()

    @property
    def name(self) -> str:
        return "kaggle"

    def _init_api(self):
        """Initialize Kaggle API client with credentials from config or env."""
        username = self.config_manager.get("providers.kaggle.username")
        key = self.config_manager.get("providers.kaggle.key")

        if username and key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key

        kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
        has_credentials = (
            "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
        ) or os.path.exists(kaggle_json)

        if not has_credentials:
            self.logger.info("Kaggle credentials not found in config or ~/.kaggle/kaggle.json. Kaggle provider disabled.")
            return

        try:

            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            self.api = api
            self.logger.info("Successfully authenticated with Kaggle API.")
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Kaggle: {e}")
            self.api = None

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        if not self.api:
            self.logger.debug("Kaggle provider is inactive (no credentials). Skipping search.")
            return []

        self.logger.info(f"Kaggle Search: '{query}'")
        results = []
        try:

            datasets = self.api.dataset_list(search=query)
            for ds in datasets[:limit]:

                ref = ds.ref
                name = ds.title
                size_str = getattr(ds, "size", "0 B")
                downloads = getattr(ds, "downloadCount", 0)

                size_bytes = self._parse_size_str(size_str)

                popularity = min(100.0, float(downloads) / 1000.0 + 10.0) if downloads else 10.0

                license_name = getattr(ds, "licenseName", "Unknown")

                metadata = DatasetMetadata(
                    id=f"kaggle/{ref}",
                    name=name,
                    provider=self.name,
                    task="Kaggle Dataset",
                    rows=None,
                    columns=None,
                    size_bytes=size_bytes,
                    license=license_name,
                    last_updated=str(getattr(ds, "lastUpdated", "Unknown")),
                    downloads=downloads,
                    popularity=popularity,
                    quality_score=getattr(ds, "usabilityRating", 0.0) * 10.0,
                    description=getattr(ds, "description", f"Kaggle dataset: {ref}"),
                    data_url=ref,
                    source_url=f"https://www.kaggle.com/datasets/{ref}",
                    tags=["kaggle"]
                )
                results.append(metadata)
        except Exception as e:
            self.logger.error(f"Error searching Kaggle datasets: {e}")

        return results

    def _parse_size_str(self, size_str: str) -> Optional[int]:
        try:
            parts = size_str.strip().split()
            if not parts:
                return None
            num = float(parts[0])
            if len(parts) == 1:
                return int(num)
            unit = parts[1].upper()
            multipliers = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
            return int(num * multipliers.get(unit, 1))
        except Exception:
            return None

    def download(
        self,
        dataset: DatasetMetadata,
        dest_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event = None
    ) -> bool:
        if not self.api:
            self.logger.error("Cannot download from Kaggle: Provider is inactive (no credentials).")
            return False

        ref = dataset.data_url or dataset.id.replace("kaggle/", "", 1)
        self.logger.info(f"Downloading Kaggle dataset {ref} to {dest_dir}")

        try:
            os.makedirs(dest_dir, exist_ok=True)

            self.api.dataset_download_files(dataset=ref, path=dest_dir, unzip=True, quiet=False)
            return True
        except Exception as e:
            self.logger.error(f"Error downloading Kaggle dataset: {e}")
            return False
