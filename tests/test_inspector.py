import os
import tempfile
import pandas as pd
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analytics.inspector import DatasetInspector

def test_inspect_csv():

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"

        df = pd.DataFrame({
            "age": [20, 21, 22, 23, 100],
            "salary": [50000, 55000, 60000, 65000, 70000],
            "label": ["no", "no", "yes", "yes", "yes"],
            "ignored_id": [1, 2, 3, 4, 5]
        })
        df.to_csv(csv_file, index=False)

        inspector = DatasetInspector(str(csv_file))
        report = inspector.inspect()

        assert report["rows"] == 5
        assert report["columns"] == 4
        assert report["delimiter"] == "Comma"

        assert report["outliers"]["age"]["count"] > 0

        target_cands = [c["column"] for c in report["target_candidates"]]
        assert "label" in target_cands

        assert report["class_balance"]["column"] == "label"
        assert report["class_balance"]["distribution"]["no"]["count"] == 2
        assert report["class_balance"]["distribution"]["yes"]["count"] == 3

def test_inspect_arff():

    with tempfile.TemporaryDirectory() as tmpdir:
        arff_file = Path(tmpdir) / "test.arff"

        arff_content = """
% Comments
@relation test_rel
@attribute 'feat1' real
@attribute feat2 real
@attribute class {yes, no}
@data
1.5,10.0,yes
2.5,20.0,no
3.5,30.0,yes
"""
        with open(arff_file, "w") as f:
            f.write(arff_content.strip())

        inspector = DatasetInspector(str(arff_file))
        report = inspector.inspect()

        assert report["rows"] == 3
        assert report["columns"] == 3

        cols = [c["name"] for c in report["columns_info"]]
        assert cols == ["feat1", "feat2", "class"]
