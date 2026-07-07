from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class DatasetMetadata:
    id: str
    name: str
    provider: str
    task: str = "Unknown"
    rows: Optional[int] = None
    columns: Optional[int] = None
    size_bytes: Optional[int] = None
    license: str = "Unknown"
    last_updated: str = "Unknown"
    downloads: Optional[int] = None
    popularity: float = 0.0
    quality_score: float = 0.0
    description: str = ""
    data_url: Optional[str] = None
    source_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_size(self) -> str:
        if self.size_bytes is None or self.size_bytes < 0:
            return "Unknown"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if self.size_bytes < 1024.0:
                return f"{self.size_bytes:.1f} {unit}"
            self.size_bytes /= 1024.0
        return f"{self.size_bytes:.1f} PB"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "task": self.task,
            "rows": self.rows,
            "columns": self.columns,
            "size_bytes": self.size_bytes,
            "license": self.license,
            "last_updated": self.last_updated,
            "downloads": self.downloads,
            "popularity": self.popularity,
            "quality_score": self.quality_score,
            "description": self.description,
            "data_url": self.data_url,
            "source_url": self.source_url,
            "tags": self.tags
        }

@dataclass
class SearchResult:
    dataset: DatasetMetadata
    score: float
