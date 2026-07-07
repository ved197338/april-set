import os
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class DatasetInspector:
    def __init__(self, filepath: str):
        self.filepath = filepath
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

    def _read_arff(self) -> pd.DataFrame:
        attributes = []
        data_started = False
        data_lines = []

        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_strip = line.strip()
                if not line_strip or line_strip.startswith("%"):
                    continue

                if not data_started:
                    line_lower = line_strip.lower()
                    if line_lower.startswith("@attribute"):
                        parts = line_strip.split(None, 2)
                        if len(parts) >= 2:
                            attr_name = parts[1].strip("'\"")
                            attributes.append(attr_name)
                    elif line_lower.startswith("@data"):
                        data_started = True
                else:
                    data_lines.append(line_strip)

        import io
        csv_data = "\n".join(data_lines)
        return pd.read_csv(io.StringIO(csv_data), names=attributes, header=None)

    def detect_delimiter_and_encoding(self) -> Tuple[str, str]:
        """Detect encoding and CSV delimiter of the file."""

        encoding = "utf-8"
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                f.read(4096)
        except UnicodeDecodeError:
            encoding = "latin-1"

        delimiter = ","
        if self.filepath.lower().endswith((".csv", ".tsv", ".txt")):
            try:
                with open(self.filepath, "r", encoding=encoding) as f:
                    first_lines = [f.readline() for _ in range(5)]

                counts = {",": 0, ";": 0, "\t": 0, "|": 0}
                for line in first_lines:
                    for char in counts:
                        counts[char] += line.count(char)

                best_delim = max(counts, key=counts.get)
                if counts[best_delim] > 0:
                    delimiter = best_delim
            except Exception:
                pass

        return delimiter, encoding

    def inspect(self) -> Dict[str, Any]:
        """Run deep statistical profile of the dataset."""
        ext = os.path.splitext(self.filepath)[1].lower()
        delimiter, encoding = self.detect_delimiter_and_encoding()

        try:
            if ext == ".parquet":
                df = pd.read_parquet(self.filepath)
            elif ext == ".arff":
                df = self._read_arff()
            elif ext in [".json", ".jsonl"]:
                df = pd.read_json(self.filepath, lines=ext == ".jsonl")
            elif ext in [".tsv", ".txt"]:
                df = pd.read_csv(self.filepath, sep="\t", encoding=encoding)
            else:
                df = pd.read_csv(self.filepath, sep=delimiter, encoding=encoding)
        except Exception as e:
            return {"error": f"Failed to load dataset: {str(e)}"}

        total_rows = len(df)
        total_cols = len(df.columns)
        if total_rows == 0:
            return {"error": "Dataset is empty."}

        memory_usage_bytes = df.memory_usage(deep=True).sum()
        memory_usage_mb = memory_usage_bytes / (1024 ** 2)
        duplicate_rows = int(df.duplicated().sum())
        duplicate_percentage = (duplicate_rows / total_rows) * 100

        columns_info = []
        total_missing = 0

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            total_missing += missing_count
            missing_pct = (missing_count / total_rows) * 100
            dtype = str(df[col].dtype)
            cardinality = int(df[col].nunique())

            columns_info.append({
                "name": col,
                "type": dtype,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "cardinality": cardinality
            })

        overall_missing_pct = (total_missing / (total_rows * total_cols)) * 100

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        summary_numerical = {}
        outliers_info = {}
        for col in numerical_cols:
            desc = df[col].describe()
            summary_numerical[col] = {
                "mean": float(desc.get("mean", 0)),
                "std": float(desc.get("std", 0)),
                "min": float(desc.get("min", 0)),
                "25%": float(desc.get("25%", 0)),
                "50%": float(desc.get("50%", 0)),
                "75%": float(desc.get("75%", 0)),
                "max": float(desc.get("max", 0)),
            }

            q1 = desc.get("25%", 0)
            q3 = desc.get("75%", 0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outliers_count = len(outliers)
            outliers_info[col] = {
                "count": outliers_count,
                "percentage": (outliers_count / total_rows) * 100
            }

        summary_categorical = {}
        for col in categorical_cols:
            desc = df[col].describe()
            summary_categorical[col] = {
                "count": int(desc.get("count", 0)),
                "unique": int(desc.get("unique", 0)),
                "top": str(desc.get("top", "N/A")),
                "freq": int(desc.get("freq", 0)),
            }

        delimiter_name = "Comma"
        if delimiter == "\t":
            delimiter_name = "Tab"
        elif delimiter == ";":
            delimiter_name = "Semicolon"
        elif delimiter == "|":
            delimiter_name = "Pipe"

        correlations = {}
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().fillna(0)

            corr_pairs = []
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    val = corr_matrix.iloc[i, j]
                    corr_pairs.append((numerical_cols[i], numerical_cols[j], float(val)))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            correlations = {
                "matrix": corr_matrix.to_dict(),
                "top_correlated_pairs": corr_pairs[:10]
            }

        target_candidates = []
        target_patterns = [r"class", r"target", r"label", r"^y$", r"predict", r"outcome", r"species", r"diagnos"]

        for col in df.columns:
            score = 0.0
            name_lower = col.lower()

            if any(re.search(pat, name_lower) for pat in target_patterns):
                score += 50.0

            cardinality = df[col].nunique()

            if cardinality > 1 and cardinality <= min(15, total_rows * 0.1):
                score += 30.0

            if col == df.columns[-1]:
                score += 20.0

            if score > 20.0:
                target_type = "Classification" if cardinality <= 20 else "Regression / High Cardinality"
                target_candidates.append({
                    "column": col,
                    "score": score,
                    "inferred_task": target_type,
                    "cardinality": cardinality
                })

        target_candidates.sort(key=lambda x: x["score"], reverse=True)

        class_balance = {}
        if target_candidates:
            primary_target = target_candidates[0]["column"]
            if df[primary_target].nunique() <= 30:
                counts = df[primary_target].value_counts()
                class_balance = {
                    "column": primary_target,
                    "distribution": {str(k): {"count": int(v), "percentage": (v / total_rows) * 100} for k, v in counts.items()}
                }

        return {
            "filepath": self.filepath,
            "filename": os.path.basename(self.filepath),
            "rows": total_rows,
            "columns": total_cols,
            "memory_usage_mb": memory_usage_mb,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": duplicate_percentage,
            "encoding": encoding,
            "delimiter": delimiter_name,
            "columns_info": columns_info,
            "overall_missing_pct": overall_missing_pct,
            "total_missing": total_missing,
            "numerical_summary": summary_numerical,
            "categorical_summary": summary_categorical,
            "outliers": outliers_info,
            "correlations": correlations,
            "target_candidates": target_candidates[:3],
            "class_balance": class_balance
        }
