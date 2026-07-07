import socket
import time
import requests
from typing import Dict, Any, Optional
from app_logging.logger import AprilLogger

class NetworkClient:
    def __init__(self, timeout_seconds: int = 10, max_retries: int = 3):
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.logger = AprilLogger.get_logger()

    @staticmethod
    def is_connected() -> bool:
        """Check for active internet connection."""
        try:

            socket.setdefaulttimeout(3.0)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
            return True
        except socket.error:
            try:

                requests.head("https://www.google.com", timeout=3.0)
                return True
            except Exception:
                return False

    def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        stream: bool = False
    ) -> Optional[requests.Response]:
        """Perform HTTP request with retries and timeout logic."""
        if headers is None:
            headers = {}

        if "User-Agent" not in headers:
            headers["User-Agent"] = "AprilSet/1.0.0 (Dataset search engine)"

        retries = 0
        while retries < self.max_retries:
            try:
                self.logger.debug(f"HTTP {method} {url} (Attempt {retries + 1}/{self.max_retries})")
                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    json=json_data,
                    data=data,
                    timeout=self.timeout,
                    stream=stream
                )
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as e:

                if response is not None and response.status_code in [404, 412]:
                    self.logger.debug(f"{response.status_code} Not Found/No Match for {url}")
                else:
                    self.logger.warning(f"HTTP error for {url}: {e}")

                if response is not None and response.status_code in [429, 500, 502, 503, 504]:
                    retries += 1
                    time.sleep(2.0 ** retries)
                else:
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request exception for {url}: {e}")
                retries += 1
                time.sleep(2.0 ** retries)

        if retries >= self.max_retries:
            self.logger.error(f"Failed to fetch {url} after {self.max_retries} attempts.")
        return None
