import concurrent.futures
import re
from typing import List, Dict, Any, Optional
from models.dataset import DatasetMetadata, SearchResult
from providers.base import BaseProvider
from providers.huggingface import HuggingFaceProvider
from providers.openml import OpenMLProvider
from providers.uci import UCIProvider
from providers.github import GitHubProvider
from providers.kaggle import KaggleProvider
from config.manager import ConfigManager
from app_logging.logger import AprilLogger

class SearchEngine:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = AprilLogger.get_logger()
        self._init_providers()

    def _init_providers(self):
        enabled_names = self.config_manager.get("providers.enabled", [])
        self.providers: List[BaseProvider] = []

        provider_classes = {
            "huggingface": HuggingFaceProvider,
            "openml": OpenMLProvider,
            "uci": UCIProvider,
            "github": GitHubProvider,
            "kaggle": KaggleProvider
        }

        for name in enabled_names:
            if name in provider_classes:
                try:

                    if name == "kaggle":
                        prov = KaggleProvider(config_manager=self.config_manager)
                    else:
                        prov = provider_classes[name]()
                    self.providers.append(prov)
                except Exception as e:
                    self.logger.error(f"Failed to initialize provider '{name}': {e}")

    def search(self, query: str, limit_per_provider: int = 15) -> List[SearchResult]:
        self.logger.info(f"Orchestrating search for query: '{query}'")

        raw_results: List[DatasetMetadata] = []
        timeout = self.config_manager.get("search.timeout_seconds", 10)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.providers) or 1) as executor:
            future_to_provider = {
                executor.submit(prov.search, query, limit_per_provider): prov 
                for prov in self.providers
            }

            for future in concurrent.futures.as_completed(future_to_provider):
                prov = future_to_provider[future]
                try:
                    res = future.result(timeout=timeout)
                    if res:
                        raw_results.extend(res)
                        self.logger.info(f"Provider '{prov.name}' returned {len(res)} results.")
                except concurrent.futures.TimeoutError:
                    self.logger.warning(f"Provider '{prov.name}' search timed out after {timeout}s.")
                except Exception as e:
                    self.logger.error(f"Provider '{prov.name}' search failed: {e}")

        ranked_results = self._rank_and_deduplicate(query, raw_results)
        return ranked_results

    def _rank_and_deduplicate(self, query: str, datasets: List[DatasetMetadata]) -> List[SearchResult]:
        deduped: Dict[str, DatasetMetadata] = {}

        for ds in datasets:

            norm_name = re.sub(r"[^a-z0-9]", "", ds.name.lower())

            if norm_name not in deduped:
                deduped[norm_name] = ds
            else:

                existing = deduped[norm_name]
                if (ds.popularity + ds.quality_score) > (existing.popularity + existing.quality_score):
                    deduped[norm_name] = ds

        results: List[SearchResult] = []
        for ds in deduped.values():
            score = self._calculate_match_score(query, ds)
            results.append(SearchResult(dataset=ds, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _calculate_match_score(self, query: str, ds: DatasetMetadata) -> float:

        query_words = set(query.lower().split())
        if not query_words:
            return 0.0

        name_lower = ds.name.lower()
        desc_lower = ds.description.lower()
        id_lower = ds.id.lower()

        exact_bonus = 0.0
        if query.lower() == name_lower or query.lower() == id_lower:
            exact_bonus = 50.0
        elif query.lower() in name_lower or query.lower() in id_lower:
            exact_bonus = 30.0

        title_matches = sum(1 for w in query_words if w in name_lower or w in id_lower)
        title_ratio = title_matches / len(query_words)

        desc_matches = sum(1 for w in query_words if w in desc_lower)
        desc_ratio = desc_matches / len(query_words)

        if title_matches == 0 and desc_matches == 0 and exact_bonus == 0.0:
            return 0.0

        relevance = (title_ratio * 40.0) + (desc_ratio * 10.0) + exact_bonus

        popularity_weight = ds.popularity * 0.2

        quality_weight = ds.quality_score * 0.1

        final_score = relevance + popularity_weight + quality_weight
        return min(100.0, final_score)
