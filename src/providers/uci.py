import os
import concurrent.futures
from typing import List, Optional, Callable, Dict, Any
from providers.base import BaseProvider
from models.dataset import DatasetMetadata
from networking.client import NetworkClient
from app_logging.logger import AprilLogger

POPULAR_UCI_DATASETS = {
    "iris": 53,
    "adult": 2,
    "wine": 186,
    "heart disease": 45,
    "breast cancer": 17,
    "car evaluation": 19,
    "diabetes": 34,
    "mushroom": 73,
    "abalone": 1,
    "bank marketing": 222,
    "dry bean": 602,
    "rice": 545,
    "spambase": 94,
    "boston housing": 52,
    "sonar": 151,
    "auto mpg": 9,
    "yeast": 189,
    "glass": 42,
    "student performance": 320,
    "forest fires": 162,
    "credit approval": 27,
    "statlog german credit": 144,
    "dermatology": 33,
}

class UCIProvider(BaseProvider):
    def __init__(self, client: Optional[NetworkClient] = None):
        self.client = client or NetworkClient()
        self.logger = AprilLogger.get_logger()

    @property
    def name(self) -> str:
        return "uci"

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        self.logger.info(f"UCI Search: '{query}'")
        query_lower = query.lower()

        matched_ids = []
        for name, uci_id in POPULAR_UCI_DATASETS.items():
            if query_lower in name or name in query_lower:
                matched_ids.append(uci_id)

        if not matched_ids and query.isdigit():
            matched_ids.append(int(query))

        matched_ids = matched_ids[:limit]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, max(1, len(matched_ids)))) as executor:
            future_to_id = {executor.submit(self.fetch_metadata, uid): uid for uid in matched_ids}
            for future in concurrent.futures.as_completed(future_to_id):
                meta = future.result()
                if meta:
                    results.append(meta)

        return results

    def fetch_metadata(self, uci_id: int) -> Optional[DatasetMetadata]:
        url = f"https://archive.ics.uci.edu/api/dataset?id={uci_id}"
        response = self.client.request("GET", url)
        if not response:
            return None

        try:
            res_json = response.json()
            if res_json.get("status") != 200:
                return None
            data = res_json.get("data", {})
            return self._parse_dataset(data)
        except Exception as e:
            self.logger.error(f"Error fetching/parsing UCI dataset ID {uci_id}: {e}")
            return None

    def _parse_dataset(self, data: Dict[str, Any]) -> DatasetMetadata:
        uci_id = data.get("uci_id")
        name = data.get("name", "Unknown")
        tasks = data.get("tasks", [])
        task = tasks[0] if tasks else "Tabular"

        rows = data.get("num_instances")
        cols = data.get("num_features")

        popularity = 85.0 if name.lower() in POPULAR_UCI_DATASETS else 60.0

        quality_score = 70.0
        if data.get("variables"):
            quality_score += 15.0
        if data.get("has_missing_values") != "unknown":
            quality_score += 15.0

        description = data.get("abstract", "") or f"UCI Machine Learning Repository dataset: {name}."
        data_url = data.get("data_url")
        source_url = data.get("repository_url") or f"https://archive.ics.uci.edu/dataset/{uci_id}"

        tags = [task]
        if data.get("area"):
            tags.append(data.get("area"))
        for char in data.get("characteristics", []):
            tags.append(char)

        return DatasetMetadata(
            id=f"uci/{uci_id}",
            name=name,
            provider=self.name,
            task=task,
            rows=rows,
            columns=cols,
            size_bytes=None,
            license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
            last_updated=data.get("last_updated", "Unknown"),
            downloads=None,
            popularity=popularity,
            quality_score=quality_score,
            description=description,
            data_url=data_url,
            source_url=source_url,
            tags=tags,
            raw_metadata=data
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

            uci_id = dataset.id.split("/")[-1]
            data_url = f"https://archive.ics.uci.edu/static/public/{uci_id}/data.csv"

        dest_file = os.path.join(dest_dir, f"{dataset.name}.csv")
        self.logger.info(f"Downloading UCI dataset from {data_url} to {dest_file}")

        from downloader.manager import DownloadManager
        dl = DownloadManager()
        return dl.download_file(
            url=data_url,
            dest_path=dest_file,
            cancel_event=cancel_event,
            progress_callback=progress_callback
        )

        return True
