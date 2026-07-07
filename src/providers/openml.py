import os
from typing import List, Optional, Callable, Dict, Any
from providers.base import BaseProvider
from models.dataset import DatasetMetadata
from networking.client import NetworkClient
from app_logging.logger import AprilLogger

class OpenMLProvider(BaseProvider):
    def __init__(self, client: Optional[NetworkClient] = None):
        self.client = client or NetworkClient()
        self.logger = AprilLogger.get_logger()

    @property
    def name(self) -> str:
        return "openml"

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        self.logger.info(f"OpenML Search: '{query}'")

        clean_query = query.strip().replace(" ", "_")
        url = f"https://www.openml.org/api/v1/json/data/list/data_name/{clean_query}"

        response = self.client.request("GET", url)
        if not response:
            return []

        results = []
        try:
            data = response.json()
            if "data" in data and "dataset" in data["data"]:
                datasets = data["data"]["dataset"]

                for ds in datasets[:limit]:
                    metadata = self._parse_dataset(ds)
                    results.append(metadata)
        except Exception as e:
            self.logger.error(f"Error parsing OpenML search results: {e}")

        return results

    def _parse_dataset(self, ds: Dict[str, Any]) -> DatasetMetadata:
        did = ds.get("did")
        name = ds.get("name", "Unknown")
        file_id = ds.get("file_id")
        file_format = ds.get("format", "ARFF").upper()

        qualities = ds.get("quality", [])
        rows = None
        cols = None
        num_classes = None

        for q in qualities:
            q_name = q.get("name")
            q_val = q.get("value")
            if not q_val:
                continue
            try:
                if q_name == "NumberOfInstances":
                    rows = int(float(q_val))
                elif q_name == "NumberOfFeatures":
                    cols = int(float(q_val))
                elif q_name == "NumberOfClasses":
                    num_classes = int(float(q_val))
            except ValueError:
                pass

        task = "Tabular"
        if num_classes is not None:
            if num_classes > 2:
                task = "Multi-class Classification"
            elif num_classes == 2:
                task = "Binary Classification"
            elif num_classes == 0 or num_classes == 1:
                task = "Regression / Clustering"

        popularity = 50.0
        if did is not None:
            if int(did) < 100:
                popularity = 90.0
            elif int(did) < 1000:
                popularity = 75.0
            elif int(did) < 10000:
                popularity = 60.0

        quality_score = 60.0
        if rows and cols:
            quality_score += 20.0
        if num_classes is not None:
            quality_score += 10.0

        data_url = f"https://www.openml.org/data/v1/download/{file_id}" if file_id else None
        source_url = f"https://www.openml.org/d/{did}"

        description = f"OpenML standard dataset '{name}' (version {ds.get('version', 1)}). "
        if rows and cols:
            description += f"Contains {rows} instances and {cols} features. "
        description += f"Format: {file_format}."

        return DatasetMetadata(
            id=f"openml/{did}",
            name=name,
            provider=self.name,
            task=task,
            rows=rows,
            columns=cols,
            size_bytes=None,
            license="Public / CC Attribution",
            last_updated="Unknown",
            downloads=None,
            popularity=popularity,
            quality_score=quality_score,
            description=description,
            data_url=data_url,
            source_url=source_url,
            tags=[file_format, task],
            raw_metadata=ds
        )

    def download(
        self,
        dataset: DatasetMetadata,
        dest_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event = None
    ) -> bool:
        data_url = dataset.data_url
        if not data_url:
            self.logger.error(f"No download URL available for OpenML dataset {dataset.id}")
            return False

        ext = "arff"
        for tag in dataset.tags:
            if tag.lower() in ["arff", "csv", "parquet"]:
                ext = tag.lower()
                break

        dest_file = os.path.join(dest_dir, f"{dataset.name}.{ext}")
        self.logger.info(f"Downloading OpenML dataset to {dest_file}")

        from downloader.manager import DownloadManager
        dl = DownloadManager()
        return dl.download_file(
            url=data_url,
            dest_path=dest_file,
            cancel_event=cancel_event,
            progress_callback=progress_callback
        )

        return True
