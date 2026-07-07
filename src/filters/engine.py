import re
from typing import List, Dict, Any, Optional
from models.dataset import DatasetMetadata, SearchResult

class FilterEngine:
    def __init__(self):
        pass

    def filter_results(
        self,
        results: List[SearchResult],
        task: Optional[str] = None,
        license_type: Optional[str] = None,
        rows_cond: Optional[str] = None,
        cols_cond: Optional[str] = None,
        size_cond: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[SearchResult]:
        filtered = []
        for r in results:
            ds = r.dataset
            if not self._match_task(ds, task):
                continue
            if not self._match_license(ds, license_type):
                continue
            if not self._match_comparison(ds.rows, rows_cond):
                continue
            if not self._match_comparison(ds.columns, cols_cond):
                continue
            if not self._match_size(ds, size_cond):
                continue
            if not self._match_tag(ds, tag):
                continue
            filtered.append(r)
        return filtered

    def _match_task(self, ds: DatasetMetadata, task: Optional[str]) -> bool:
        if not task:
            return True
        return task.lower() in ds.task.lower() or any(task.lower() in t.lower() for t in ds.tags)

    def _match_license(self, ds: DatasetMetadata, license_type: Optional[str]) -> bool:
        if not license_type:
            return True
        if ds.license == "Unknown":
            return False
        return license_type.lower() in ds.license.lower()

    def _match_comparison(self, value: Optional[int], cond: Optional[str]) -> bool:
        if not cond:
            return True
        if value is None:
            return False

        match = re.match(r"^([<>=]+)\s*(\d+)$", cond.strip())
        if not match:
            return True

        op, threshold_str = match.groups()
        threshold = int(threshold_str)

        if op == ">":
            return value > threshold
        elif op == "<":
            return value < threshold
        elif op == ">=":
            return value >= threshold
        elif op == "<=":
            return value <= threshold
        elif op == "==":
            return value == threshold
        return True

    def _match_size(self, ds: DatasetMetadata, size_cond: Optional[str]) -> bool:
        if not size_cond:
            return True
        if ds.size_bytes is None:
            return False

        match = re.match(r"^([<>=]+)\s*(\d+)\s*(kb|mb|gb|tb|b)?$", size_cond.strip().lower())
        if not match:
            return True

        op, num_str, unit = match.groups()
        num = float(num_str)

        multipliers = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
        unit_mult = multipliers.get(unit or "b", 1)
        threshold_bytes = num * unit_mult

        if op == ">":
            return ds.size_bytes > threshold_bytes
        elif op == "<":
            return ds.size_bytes < threshold_bytes
        elif op == ">=":
            return ds.size_bytes >= threshold_bytes
        elif op == "<=":
            return ds.size_bytes <= threshold_bytes
        return True

    def _match_tag(self, ds: DatasetMetadata, tag: Optional[str]) -> bool:
        if not tag:
            return True
        return any(tag.lower() in t.lower() for t in ds.tags)

    def parse_natural_language(self, nl_query: str) -> Dict[str, Any]:
        """Parse natural language search string into filter conditions using regex."""
        conditions = {}
        nl_query_lower = nl_query.lower()

        if "classification" in nl_query_lower:
            conditions["task"] = "classification"
        elif "regression" in nl_query_lower:
            conditions["task"] = "regression"
        elif "nlp" in nl_query_lower or "text" in nl_query_lower:
            conditions["task"] = "nlp"
        elif "image" in nl_query_lower or "vision" in nl_query_lower:
            conditions["task"] = "vision"

        size_match = re.search(r"(smaller than|less than|<)\s*(\d+)\s*(kb|mb|gb|tb)", nl_query_lower)
        if size_match:
            _, val, unit = size_match.groups()
            conditions["size_cond"] = f"<{val}{unit.upper()}"
        else:
            size_match_gt = re.search(r"(larger than|greater than|>)\s*(\d+)\s*(kb|mb|gb|tb)", nl_query_lower)
            if size_match_gt:
                _, val, unit = size_match_gt.groups()
                conditions["size_cond"] = f">{val}{unit.upper()}"

        row_match = re.search(r"(more than|greater than|>)\s*(\d+)\s*(rows|instances|samples)", nl_query_lower)
        if row_match:
            _, val, _ = row_match.groups()
            conditions["rows_cond"] = f">{val}"
        else:
            row_match_lt = re.search(r"(less than|fewer than|<)\s*(\d+)\s*(rows|instances|samples)", nl_query_lower)
            if row_match_lt:
                _, val, _ = row_match_lt.groups()
                conditions["rows_cond"] = f"<{val}"

        license_patterns = [r"mit", r"apache", r"gpl", r"creative commons", r"cc0", r"bsd"]
        for pat in license_patterns:
            if re.search(r"\b" + pat + r"\b", nl_query_lower):
                conditions["license_type"] = pat
                break

        return conditions
