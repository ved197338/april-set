from typing import Dict, Any, Optional

class ExportGenerator:
    def __init__(self):
        pass

    def generate_snippet(self, framework: str, dataset_name: str, file_path: str, target_col: Optional[str] = "target") -> str:
        """Generate code snippet for loading dataset in a specific framework."""
        fw = framework.lower().strip()

        if fw == "pytorch":
            return self._pytorch_snippet(dataset_name, file_path, target_col)
        elif fw == "tensorflow":
            return self._tensorflow_snippet(dataset_name, file_path, target_col)
        elif fw in ["scikit-learn", "sklearn"]:
            return self._sklearn_snippet(dataset_name, file_path, target_col)
        elif fw == "xgboost":
            return self._xgboost_snippet(dataset_name, file_path, target_col)
        elif fw in ["lightgbm", "lgb"]:
            return self._lightgbm_snippet(dataset_name, file_path, target_col)
        elif fw in ["catboost", "cb"]:
            return self._catboost_snippet(dataset_name, file_path, target_col)
        elif fw == "duckdb":
            return self._duckdb_snippet(file_path)
        elif fw == "parquet":
            return self._parquet_snippet(file_path)
        elif fw == "r":
            return self._r_snippet(file_path)
        elif fw == "julia":
            return self._julia_snippet(file_path)
        elif fw == "github":
            return self._github_snippet(dataset_name, file_path)
        elif fw in ["aws", "s3"]:
            return self._aws_snippet(dataset_name, file_path)
        else:
            return self._pandas_snippet(file_path)

    def _pytorch_snippet(self, dataset_name: str, file_path: str, target_col: str) -> str:
        return f"""# PyTorch DataLoader for {dataset_name}
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class {dataset_name.replace(" ", "").replace("-", "")}Dataset(Dataset):
    def __init__(self, filepath, target_col):
        # Load data (handles CSV/TSV/Parquet)
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        # Fill missing values
        df = df.fillna(df.median(numeric_only=True))

        # Split features and target
        if target_col in df.columns:
            self.y = torch.tensor(df[target_col].values, dtype=torch.float32)
            self.X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
        else:
            self.y = None
            self.X = df.select_dtypes(include=['number'])

        # Scale features
        scaler = StandardScaler()
        self.X = torch.tensor(scaler.fit_transform(self.X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# Usage
dataset = {dataset_name.replace(" ", "").replace("-", "")}Dataset(
    filepath="{file_path}", 
    target_col="{target_col}"
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate
for batch_X, batch_y in dataloader:
    print(batch_X.shape, batch_y.shape)
    break
"""

    def _tensorflow_snippet(self, dataset_name: str, file_path: str, target_col: str) -> str:
        return f"""# TensorFlow Dataset Loader for {dataset_name}
import tensorflow as tf
import pandas as pd

# Load CSV
def load_tf_dataset(filepath, target_col, batch_size=32):
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    df = df.fillna(df.median(numeric_only=True))

    if target_col in df.columns:
        target = df.pop(target_col)
        # Select numeric features for simplicity
        features = df.select_dtypes(include=['number'])
        ds = tf.data.Dataset.from_tensor_slices((dict(features), target))
    else:
        features = df.select_dtypes(include=['number'])
        ds = tf.data.Dataset.from_tensor_slices(dict(features))

    return ds.shuffle(1000).batch(batch_size)

# Usage
dataset = load_tf_dataset(
    filepath="{file_path}", 
    target_col="{target_col}"
)

for features, labels in dataset.take(1):
    print("Features:", list(features.keys()))
    print("Labels:", labels.numpy())
"""

    def _sklearn_snippet(self, dataset_name: str, file_path: str, target_col: str) -> str:
        return f"""# Scikit-learn Boilerplate for {dataset_name}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # or Regressor

# Load data
df = pd.read_parquet("{file_path}") if "{file_path}".endswith(".parquet") else pd.read_csv("{file_path}")

# Preprocess
df = df.fillna(df.median(numeric_only=True))
X = df.drop(columns=["{target_col}"]).select_dtypes(include=["number"])
y = df["{target_col}"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
print("Model Score:", model.score(X_test_scaled, y_test))
"""

    def _xgboost_snippet(self, dataset_name: str, file_path: str, target_col: str) -> str:
        return f"""# XGBoost Loading and Training for {dataset_name}
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_parquet("{file_path}") if "{file_path}".endswith(".parquet") else pd.read_csv("{file_path}")
X = df.drop(columns=["{target_col}"]).select_dtypes(include=["number"])
y = df["{target_col}"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Train
params = {{
    'max_depth': 6,
    'eta': 0.3,
    'objective': 'binary:logistic' if y.nunique() <= 2 else 'multi:softprob',
    'eval_metric': 'logloss'
}}
bst = xgb.train(params, dtrain, num_boost_round=10)
print(bst.predict(dtest))
"""

    def _lightgbm_snippet(self, dataset_name: str, file_path: str, target_col: str) -> str:
        return f"""# LightGBM Loading and Training for {dataset_name}
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_parquet("{file_path}") if "{file_path}".endswith(".parquet") else pd.read_csv("{file_path}")
X = df.drop(columns=["{target_col}"])
y = df["{target_col}"]

# Categorical column handling
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Train
params = {{
    'objective': 'binary' if y.nunique() <= 2 else 'multiclass',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31
}}
gbm = lgb.train(params, train_data, num_boost_round=20, valid_sets=[test_data])
"""

    def _catboost_snippet(self, dataset_name: str, file_path: str, target_col: str) -> str:
        return f"""# CatBoost Loading and Training for {dataset_name}
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_parquet("{file_path}") if "{file_path}".endswith(".parquet") else pd.read_csv("{file_path}")
X = df.drop(columns=["{target_col}"]).fillna("missing")
y = df["{target_col}"]

cat_features = list(X.select_dtypes(include=['object', 'category']).columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = CatBoostClassifier(iterations=50, learning_rate=0.1, depth=6)
model.fit(X_train, y_train, cat_features=cat_features, verbose=10)
print("Accuracy:", model.score(X_test, y_test))
"""

    def _duckdb_snippet(self, file_path: str) -> str:
        return f"""# DuckDB Querying for Tabular Dataset
import duckdb

# Query CSV or Parquet directly with SQL
con = duckdb.connect(database=':memory:')

# Perform aggregations without loading full file into memory
query = "SELECT * FROM '{file_path}' LIMIT 5"
result = con.execute(query).df()
print(result)
"""

    def _parquet_snippet(self, file_path: str) -> str:
        return f"""# Fast Parquet Loading
import pandas as pd

# Load only necessary columns to optimize memory
columns_to_load = None # Define list of strings to load subset e.g. ['col1', 'col2']
df = pd.read_parquet("{file_path}", columns=columns_to_load)
print(df.info())
"""

    def _r_snippet(self, file_path: str) -> str:
        return f"""# R Script for loading Dataset
library(tidyverse)

# Load CSV or Parquet
{"df <- read_parquet('" + file_path + "')" if file_path.endswith('.parquet') else "df <- read_csv('" + file_path + "')"}
summary(df)
"""

    def _julia_snippet(self, file_path: str) -> str:
        return f"""# Julia Script for loading Dataset
using CSV, DataFrames

{"df = DataFrame(CSV.File(\"" + file_path + "\"))" if not file_path.endswith('.parquet') else "using ParquetFiles; df = DataFrame(load(\"" + file_path + "\"))"}
first(df, 5)
"""

    def _pandas_snippet(self, file_path: str) -> str:
        return f"""# Python Pandas Loading
import pandas as pd

df = pd.read_csv("{file_path}")
print(df.head())
"""

    def _github_snippet(self, dataset_name: str, file_path: str) -> str:
        import os
        filename = os.path.basename(file_path)
        return f"""# shell commands & workflow to upload {dataset_name} to GitHub

# Option 1: Commit and push directly (best for small datasets)
git init
git add "{file_path}"
git commit -m "Add dataset: {dataset_name}"
git remote add origin https://github.com/your-username/your-repo.git
git branch -M main
git push -u origin main

# Option 2: Upload as a GitHub Release Asset using GitHub CLI (for larger files)
gh release create v1.0.0 "{file_path}" --title "Dataset Release v1.0.0" --notes "Uploaded via april-set"
"""

    def _aws_snippet(self, dataset_name: str, file_path: str) -> str:
        import os
        filename = os.path.basename(file_path)
        return f"""# Upload {dataset_name} to Amazon S3

# Option 1: AWS CLI command
aws s3 cp "{file_path}" s3://your-bucket-name/datasets/{filename}

# Option 2: Python Boto3 script
import boto3
from botocore.exceptions import NoCredentialsError

def upload_to_s3(local_file, bucket, s3_file):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

# Usage
upload_to_s3(
    local_file="{file_path}",
    bucket="your-bucket-name",
    s3_file="datasets/{filename}"
)
"""
