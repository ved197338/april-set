import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from models.dataset import DatasetMetadata
from app_logging.logger import AprilLogger
from config.manager import ConfigManager

class CacheStore:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = AprilLogger.get_logger()
        self.metadata_dir = self.config_manager.metadata_dir
        self.datasets_dir = self.config_manager.datasets_dir

        self.bookmarks_file = self.metadata_dir / "bookmarks.json"
        self.history_file = self.metadata_dir / "history.json"
        self.downloads_file = self.metadata_dir / "downloads.json"

        self._init_files()

    def _init_files(self):
        for f in [self.bookmarks_file, self.history_file, self.downloads_file]:
            if not f.exists():
                with open(f, "w") as fh:
                    json.dump([], fh)

    def _read_json(self, filepath: Path) -> List[Any]:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_json(self, filepath: Path, data: List[Any]):
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write to {filepath.name}: {e}")

    def add_bookmark(self, ds: DatasetMetadata):
        bookmarks = self._read_json(self.bookmarks_file)

        if any(b["id"] == ds.id for b in bookmarks):
            return
        bookmarks.append(ds.to_dict())
        self._write_json(self.bookmarks_file, bookmarks)
        self.logger.info(f"Bookmarked dataset: {ds.id}")

    def remove_bookmark(self, dataset_id: str):
        bookmarks = self._read_json(self.bookmarks_file)
        bookmarks = [b for b in bookmarks if b["id"] != dataset_id]
        self._write_json(self.bookmarks_file, bookmarks)
        self.logger.info(f"Removed bookmark: {dataset_id}")

    def get_bookmarks(self) -> List[Dict[str, Any]]:
        return self._read_json(self.bookmarks_file)

    def add_search_history(self, query: str):
        history = self._read_json(self.history_file)

        history = [h for h in history if h.lower() != query.lower()]
        history.insert(0, query)

        history = history[:50]
        self._write_json(self.history_file, history)

    def get_search_history(self) -> List[str]:
        return self._read_json(self.history_file)

    def track_download(self, ds: DatasetMetadata, local_path: str):
        downloads = self._read_json(self.downloads_file)

        downloads = [d for d in downloads if d["id"] != ds.id]

        download_entry = ds.to_dict()
        download_entry["local_path"] = local_path
        download_entry["downloaded_at"] = os.path.getmtime(local_path) if os.path.exists(local_path) else 0.0

        downloads.insert(0, download_entry)
        self._write_json(self.downloads_file, downloads)
        self._enforce_size_limit()

    def get_downloads(self) -> List[Dict[str, Any]]:
        return self._read_json(self.downloads_file)

    def get_dataset_dir(self, ds_id: str) -> Path:
        """Get standard folder name for local cache extraction."""

        safe_id = ds_id.replace("/", "_")
        return self.datasets_dir / safe_id

    def _enforce_size_limit(self):
        """Delete oldest downloaded datasets if overall cache size exceeds configured maximum size."""
        max_gb = self.config_manager.get("cache.max_size_gb", 10.0)
        max_bytes = max_gb * (1024**3)

        downloads = self._read_json(self.downloads_file)

        total_size = 0
        valid_downloads = []

        for d in downloads:
            path = d.get("local_path", "")
            if path and os.path.exists(path):

                size = self._get_path_size(path)
                d["actual_size"] = size
                total_size += size
                valid_downloads.append(d)
            else:

                pass

        if total_size <= max_bytes:
            return

        self.logger.warning(f"Cache size ({total_size / 1024**2:.1f} MB) exceeds maximum limit ({max_gb} GB). Cleaning up oldest datasets...")

        valid_downloads.sort(key=lambda x: x.get("downloaded_at", 0))

        for d in valid_downloads:
            if total_size <= max_bytes:
                break
            path = d.get("local_path", "")
            size = d.get("actual_size", 0)
            try:
                if os.path.isdir(path):
                    import shutil
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                total_size -= size
                self.logger.info(f"Removed cached dataset from {path}")

                downloads = [dl for dl in downloads if dl["id"] != d["id"]]
            except Exception as e:
                self.logger.error(f"Failed to delete cached dataset {d['id']}: {e}")

        self._write_json(self.downloads_file, downloads)

    def _get_path_size(self, path: str) -> int:
        if os.path.isfile(path):
            return os.path.getsize(path)
        total = 0
        try:
            for root, dirs, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
        except Exception:
            pass
        return total
