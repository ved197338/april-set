import math
import os
from typing import List, Optional, Callable, Dict, Any
from providers.base import BaseProvider
from models.dataset import DatasetMetadata
from networking.client import NetworkClient
from app_logging.logger import AprilLogger

class HuggingFaceProvider(BaseProvider):
    def __init__(self, client: Optional[NetworkClient] = None):
        self.client = client or NetworkClient()
        self.logger = AprilLogger.get_logger()

    @property
    def name(self) -> str:
        return "huggingface"

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        self.logger.info(f"Hugging Face Search: '{query}'")
        url = "https://huggingface.co/api/datasets"
        params = {"search": query, "limit": limit}

        response = self.client.request("GET", url, params=params)
        if not response:
            return []

        results = []
        try:
            datasets_json = response.json()
            for ds in datasets_json:
                metadata = self._parse_dataset(ds)
                results.append(metadata)
        except Exception as e:
            self.logger.error(f"Error parsing Hugging Face search results: {e}")

        return results

    def _parse_dataset(self, ds: Dict[str, Any]) -> DatasetMetadata:
        ds_id = ds.get("id", "")
        name = ds_id.split("/")[-1] if "/" in ds_id else ds_id

        tags = ds.get("tags", [])
        task = "Unknown"
        license_name = "Unknown"
        file_format = "Unknown"

        for tag in tags:
            if tag.startswith("task_categories:"):
                task = tag.replace("task_categories:", "")
            elif tag.startswith("license:"):
                license_name = tag.replace("license:", "")
            elif tag.startswith("format:"):
                file_format = tag.replace("format:", "")

        downloads = ds.get("downloads", 0)
        likes = ds.get("likes", 0)

        popularity = min(100.0, math.log1p(downloads) * 5 + likes * 0.5)

        quality_score = 30.0
        if ds.get("description"):
            quality_score += 20.0
        if license_name != "Unknown":
            quality_score += 20.0
        if task != "Unknown":
            quality_score += 15.0
        if len(tags) > 5:
            quality_score += 15.0

        source_url = f"https://huggingface.co/datasets/{ds_id}"

        return DatasetMetadata(
            id=f"hf/{ds_id}",
            name=name,
            provider=self.name,
            task=task,
            rows=None,
            columns=None,
            size_bytes=None,
            license=license_name,
            last_updated=ds.get("lastModified", "Unknown"),
            downloads=downloads,
            popularity=popularity,
            quality_score=quality_score,
            description=ds.get("description", ""),
            source_url=source_url,
            tags=tags,
            raw_metadata=ds
        )

    def _list_files_recursive(self, dataset_id: str, path: str = "") -> List[Dict[str, Any]]:
        """List files in the Hugging Face repository tree recursively."""
        url = f"https://huggingface.co/api/datasets/{dataset_id}/tree/main/{path}".rstrip("/")
        response = self.client.request("GET", url)
        if not response:
            return []

        files = []
        try:
            items = response.json()
            for item in items:
                if item.get("type") == "file":
                    files.append(item)
                elif item.get("type") == "directory":
                    dir_path = item.get("path", "")
                    files.extend(self._list_files_recursive(dataset_id, dir_path))
        except Exception as e:
            self.logger.warning(f"Error listing path {path} in HF repo {dataset_id}: {e}")

        return files

    def download(
        self,
        dataset: DatasetMetadata,
        dest_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event = None
    ) -> bool:

        dataset_id = dataset.id.replace("hf/", "", 1)
        self.logger.info(f"Downloading Hugging Face dataset: {dataset_id}")

        all_files = self._list_files_recursive(dataset_id)
        if not all_files:
            self.logger.error(f"No files found in Hugging Face dataset {dataset_id}")
            return False

        data_files = [
            f for f in all_files 
            if not f["path"].startswith(".") and not f["path"].lower().endswith((".md", ".txt", ".gitattributes"))
        ]

        if not data_files:
            data_files = all_files

        total_size = sum(f.get("size", 0) for f in data_files)
        downloaded_bytes = 0

        os.makedirs(dest_dir, exist_ok=True)

        for file_info in data_files:
            if cancel_event and cancel_event.is_set():
                self.logger.info("Download cancelled.")
                return False

            file_path = file_info["path"]
            file_size = file_info.get("size", 0)

            file_dest = os.path.join(dest_dir, file_path)
            os.makedirs(os.path.dirname(file_dest), exist_ok=True)

            resolve_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{file_path}"

            from downloader.manager import DownloadManager
            dl = DownloadManager()

            def make_cb(start_bytes):
                return lambda d, t: progress_callback(start_bytes + d, total_size) if progress_callback else None

            success = dl.download_file(
                url=resolve_url,
                dest_path=file_dest,
                cancel_event=cancel_event,
                progress_callback=make_cb(downloaded_bytes)
            )
            if success:
                downloaded_bytes += file_size

        return True
