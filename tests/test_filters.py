import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from filters.engine import FilterEngine
from models.dataset import DatasetMetadata, SearchResult

@pytest.fixture
def sample_results():
    ds1 = DatasetMetadata(
        id="uci/1", name="iris", provider="uci", task="Classification", 
        rows=150, columns=4, size_bytes=5000, license="cc-by", 
        last_updated="2020-01-01", downloads=1000, popularity=80.0, 
        quality_score=90.0, description="Iris dataset", source_url="", tags=["classification"]
    )
    ds2 = DatasetMetadata(
        id="hf/2", name="mnist", provider="huggingface", task="computer_vision", 
        rows=70000, columns=784, size_bytes=15000000, license="mit", 
        last_updated="2021-01-01", downloads=50000, popularity=95.0, 
        quality_score=95.0, description="MNIST dataset", source_url="", tags=["vision"]
    )
    ds3 = DatasetMetadata(
        id="openml/3", name="boston", provider="openml", task="Regression", 
        rows=506, columns=13, size_bytes=50000, license="gpl", 
        last_updated="2019-01-01", downloads=500, popularity=60.0, 
        quality_score=75.0, description="Boston housing", source_url="", tags=["regression"]
    )

    return [
        SearchResult(dataset=ds1, score=1.0),
        SearchResult(dataset=ds2, score=1.0),
        SearchResult(dataset=ds3, score=1.0)
    ]

def test_filter_by_task(sample_results):
    fe = FilterEngine()

    filtered = fe.filter_results(sample_results, task="classification")
    assert len(filtered) == 1
    assert filtered[0].dataset.name == "iris"

    filtered_reg = fe.filter_results(sample_results, task="regression")
    assert len(filtered_reg) == 1
    assert filtered_reg[0].dataset.name == "boston"

def test_filter_by_size(sample_results):
    fe = FilterEngine()

    filtered = fe.filter_results(sample_results, size_cond="<1MB")
    assert len(filtered) == 2
    assert "iris" in [f.dataset.name for f in filtered]
    assert "boston" in [f.dataset.name for f in filtered]
    assert "mnist" not in [f.dataset.name for f in filtered]

def test_filter_by_rows(sample_results):
    fe = FilterEngine()

    filtered = fe.filter_results(sample_results, rows_cond=">1000")
    assert len(filtered) == 1
    assert filtered[0].dataset.name == "mnist"

def test_natural_language_parsing():
    fe = FilterEngine()

    conds = fe.parse_natural_language("classification datasets smaller than 10MB")
    assert conds["task"] == "classification"
    assert conds["size_cond"] == "<10MB"

    conds2 = fe.parse_natural_language("datasets with more than 50000 rows")
    assert conds2["rows_cond"] == ">50000"
