import os
import time
import hashlib
from typing import Optional, Callable
import requests
from models.dataset import DatasetMetadata
from app_logging.logger import AprilLogger
from config.manager import ConfigManager

class DownloadManager:
    def __init__(self, config_manager: Optional[ConfigManager] = None, max_retries: int = 5, backoff_factor: float = 2.0):
        self.config_manager = config_manager or ConfigManager()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = AprilLogger.get_logger()

    def download_file(
        self,
        url: str,
        dest_path: str,
        headers: Optional[dict] = None,
        sha256_checksum: Optional[str] = None,
        cancel_event = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """Download a single file supporting resume, retry, cancellation, and checksum verification."""
        if headers is None:
            headers = {}

        if "User-Agent" not in headers:
            headers["User-Agent"] = "AprilSet/1.0.0 (Dataset downloader)"

        part_path = dest_path + ".part"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        downloaded_bytes = 0
        if os.path.exists(part_path):
            downloaded_bytes = os.path.getsize(part_path)
            self.logger.info(f"Resuming download from byte {downloaded_bytes} for {dest_path}")

        retries = 0
        while retries < self.max_retries:
            if cancel_event and cancel_event.is_set():
                self.logger.info("Download cancelled.")
                return False

            try:

                req_headers = headers.copy()
                if downloaded_bytes > 0:
                    req_headers["Range"] = f"bytes={downloaded_bytes}-"

                response = requests.get(url, headers=req_headers, stream=True, timeout=10)

                if response.status_code == 416:

                    self.logger.warning("Range not satisfiable. Resetting download.")
                    if os.path.exists(part_path):
                        os.remove(part_path)
                    downloaded_bytes = 0
                    continue

                if downloaded_bytes > 0 and response.status_code != 206:
                    self.logger.warning("Server does not support resuming. Restarting download.")
                    if os.path.exists(part_path):
                        os.remove(part_path)
                    downloaded_bytes = 0

                total_size = int(response.headers.get("content-length", 0)) + downloaded_bytes

                mode = "ab" if downloaded_bytes > 0 else "wb"
                with open(part_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=65536):
                        if cancel_event and cancel_event.is_set():
                            return False
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                            if progress_callback:
                                progress_callback(downloaded_bytes, total_size)

                if os.path.exists(part_path):
                    os.rename(part_path, dest_path)

                if sha256_checksum:
                    self.logger.info("Verifying SHA256 checksum...")
                    if not self._verify_checksum(dest_path, sha256_checksum):
                        self.logger.error("Checksum verification failed! Deleting file.")
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        return False
                    self.logger.info("Checksum verified successfully.")

                return True

            except (requests.exceptions.RequestException, IOError) as e:
                retries += 1
                self.logger.warning(f"Download error: {e}. Retrying {retries}/{self.max_retries}...")
                time.sleep(self.backoff_factor ** retries)

        self.logger.error(f"Failed to download {url} after {self.max_retries} retries.")
        return False

    def _verify_checksum(self, filepath: str, expected_sha256: str) -> bool:
        sha256 = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while chunk := f.read(65536):
                    sha256.update(chunk)
            return sha256.hexdigest().lower() == expected_sha256.lower()
        except Exception as e:
            self.logger.error(f"Error calculating checksum: {e}")
            return False
