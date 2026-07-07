import os
from typing import List, Optional, Callable, Dict, Any
from providers.base import BaseProvider
from models.dataset import DatasetMetadata
from networking.client import NetworkClient
from app_logging.logger import AprilLogger

class GitHubProvider(BaseProvider):
    def __init__(self, client: Optional[NetworkClient] = None):
        self.client = client or NetworkClient()
        self.logger = AprilLogger.get_logger()

    @property
    def name(self) -> str:
        return "github"

    def search(self, query: str, limit: int = 10) -> List[DatasetMetadata]:
        self.logger.info(f"GitHub Search: '{query}'")

        url = "https://api.github.com/search/repositories"
        params = {"q": f"{query} dataset in:name,description,readme", "sort": "stars", "order": "desc", "per_page": limit}

        headers = {
            "Accept": "application/vnd.github+json"
        }

        response = self.client.request("GET", url, params=params, headers=headers)
        if not response:
            return []

        results = []
        try:
            data = response.json()
            items = data.get("items", [])
            for item in items:
                metadata = self._parse_dataset(item)
                results.append(metadata)
        except Exception as e:
            self.logger.error(f"Error parsing GitHub search results: {e}")

        return results

    def _parse_dataset(self, repo: Dict[str, Any]) -> DatasetMetadata:
        full_name = repo.get("full_name", "")
        name = repo.get("name", "")
        stars = repo.get("stargazers_count", 0)

        popularity = min(100.0, stars / 100.0 + 10.0) if stars > 0 else 10.0

        quality_score = 40.0
        if repo.get("description"):
            quality_score += 20.0
        if repo.get("has_wiki"):
            quality_score += 10.0
        if repo.get("license"):
            quality_score += 20.0

        lic = repo.get("license", {})
        license_name = lic.get("name", "Unknown") if lic else "Unknown"

        owner = repo.get("owner", {}).get("login", "")
        default_branch = repo.get("default_branch", "main")

        data_url = f"https://github.com/{full_name}/archive/refs/heads/{default_branch}.zip"
        source_url = repo.get("html_url", f"https://github.com/{full_name}")

        description = repo.get("description", "") or f"GitHub dataset repository by {owner}."

        return DatasetMetadata(
            id=f"github/{full_name}",
            name=name,
            provider=self.name,
            task="General / Git Repo",
            rows=None,
            columns=None,
            size_bytes=repo.get("size", 0) * 1024,
            license=license_name,
            last_updated=repo.get("updated_at", "Unknown"),
            downloads=stars,
            popularity=popularity,
            quality_score=quality_score,
            description=description,
            data_url=data_url,
            source_url=source_url,
            tags=["github-repository", "dataset"],
            raw_metadata=repo
        )

    def download(
        self,
        dataset: DatasetMetadata,
        dest_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        cancel_event = None
    ) -> bool:
        full_name = dataset.id.replace("github/", "", 1)

        url = f"https://api.github.com/repos/{full_name}/contents"
        headers = {"Accept": "application/vnd.github+json"}

        response = self.client.request("GET", url, headers=headers)
        target_files = []
        if response:
            try:
                contents = response.json()
                if isinstance(contents, list):
                    for item in contents:
                        if item.get("type") == "file":
                            name = item.get("name", "").lower()
                            if name.endswith((".csv", ".parquet", ".json", ".tsv")):
                                target_files.append(item)
            except Exception:
                pass

        os.makedirs(dest_dir, exist_ok=True)

        if target_files:

            self.logger.info(f"Found {len(target_files)} dataset file(s) in GitHub repo. Downloading...")
            total_size = sum(f.get("size", 0) for f in target_files)
            downloaded_bytes = 0

            from downloader.manager import DownloadManager
            dl = DownloadManager()

            for file_info in target_files:
                if cancel_event and cancel_event.is_set():
                    return False

                file_name = file_info.get("name")
                download_url = file_info.get("download_url")
                if not download_url:
                    continue

                dest_file = os.path.join(dest_dir, file_name)

                def make_cb(start_bytes):
                    return lambda d, t: progress_callback(start_bytes + d, total_size) if progress_callback else None

                success = dl.download_file(
                    url=download_url,
                    dest_path=dest_file,
                    cancel_event=cancel_event,
                    progress_callback=make_cb(downloaded_bytes)
                )
                if success:
                    downloaded_bytes += file_info.get("size", 0)
            return True
        else:

            dest_file = os.path.join(dest_dir, f"{dataset.name}.zip")
            self.logger.info(f"No standalone dataset files found. Downloading entire repo ZIP from {dataset.data_url} to {dest_file}")

            from downloader.manager import DownloadManager
            dl = DownloadManager()
            return dl.download_file(
                url=dataset.data_url,
                dest_path=dest_file,
                cancel_event=cancel_event,
                progress_callback=progress_callback
            )
