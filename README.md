# APRIL SET (Search Engine for Training)

A terminal search engine, statistics profiler, and code exporter for machine learning datasets.

## System Architecture

APRIL SET is structured as a CLI utility written in Python. It contains three main modules:
1. Search Engine: Queries multiple remote dataset providers concurrently.
2. Downloader: Manages chunked and resumable dataset retrieval.
3. Inspector: Profiles tabular files locally (CSV, TSV, Parquet, ARFF).

```
                  ┌─────────────────────────────────────────┐
                  │          CLI Entry / REPL Mode          │
                  └────────────────────┬────────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│     Search Engine     │  │  Downloader Manager   │  │   Dataset Inspector   │
│ (Concurrent Providers)│  │ (Resumable, Chunked)  │  │ (Stat Profiler/ARFF)  │
└───────────┬───────────┘  └───────────────────────┘  └───────────────────────┘
            │
            ├───────────────┬───────────────┬───────────────┐
            ▼               ▼               ▼               ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │ HuggingFace │ │   OpenML    │ │    UCI      │ │   Kaggle    │
     └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

## Dataset Search and Ranking Mechanics

### 1. API Integrations
The tool interfaces with several machine learning data repositories:
* Hugging Face: Uses the Web API to fetch dataset details.
* OpenML: Queries the OpenML JSON API for metadata.
* UCI Machine Learning Repository: Downloads dataset index lists and archive files.
* Kaggle: Invokes the Kaggle API to find and retrieve hosted datasets.
* GitHub: Searches repository indexes for public files.

### 2. Relevance Scoring and Ranking
Search results are sorted using a custom scorer:
* Exact Match: A bonus of 50.0 is applied if the query matches the dataset name or ID exactly.
* Substring Match: A bonus of 30.0 is applied if the query is a substring of the name or ID.
* Term Frequency: Calculated based on token overlap in the name (40% weight) and description (10% weight).
* Threshold Filter: Datasets with zero word overlap are discarded.
* Quality and Popularity: Up to 30.0 additional points are based on repository stars, downloads, and metadata completeness.

## Configuration and Environment Variables

### 1. Configuration Options (~/.config/april-set/config.yaml)

| Key | Description | Default |
|-----|-------------|---------|
| cache.max_size_gb | Maximum local dataset storage size | 10.0 |
| ai.default_provider | Active LLM assistant provider (gemini, openai, ollama) | "ollama" |
| ai.ollama_url | Connection URL for Ollama local service | "http://localhost:11434" |
| search.max_results | Maximum results returned per search provider | 20 |
| providers.enabled | Active search providers | ["huggingface", "openml", "uci", "github", "kaggle"] |

### 2. Environment Variables
* APRIL_SET_REPL: Disables the initial ASCII art banner inside the interactive loop.
* KAGGLE_USERNAME & KAGGLE_KEY: Credentials used for the Kaggle API.

## Installation and Setup

1. Add aliases to ~/.bashrc or ~/.bash_aliases:
   ```bash
   alias april-set='/path/to/april-set/bin/set'
   alias aset='/path/to/april-set/bin/set'
   ```

2. Reload shell configuration:
   ```bash
   source ~/.bashrc
   ```

## CLI Commands Reference

### 1. aset search [query]
Search for datasets. If no query is provided, it starts an interactive prompt.
```bash
aset search "diabetes" --limit 5
```
After listing, enter the index number to download immediately.

### 2. aset download [dataset_id]
Download a dataset using its ID. Supports resuming.
```bash
aset download openml/37
```

### 3. aset inspect [file]
Profile local tabular files (CSV, TSV, Parquet, ARFF).
```bash
aset inspect ~/.cache/april-set/datasets/openml_37/diabetes.arff
```

### 4. aset ai [dataset_id/file] [question]
Query LLM assistant for preprocessing and model recommendations.
```bash
aset ai openml/37 "What preprocessing steps are recommended?"
```

### 5. aset export [dataset_id/file] --framework [name]
Generate code loaders for PyTorch, TensorFlow, Scikit-learn, XGBoost, LightGBM, CatBoost, DuckDB, R, Julia, GitHub actions, or AWS S3.
```bash
aset export openml/37 --framework pytorch
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```
