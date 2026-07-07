import os
import yaml
from pathlib import Path
from typing import Dict, Any, List

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "april-set" / "config.yaml"

DEFAULT_CONFIG = {
    "cache": {
        "datasets_dir": str(Path.home() / ".cache" / "april-set" / "datasets"),
        "metadata_dir": str(Path.home() / ".cache" / "april-set" / "metadata"),
        "max_size_gb": 10.0,
    },
    "providers": {
        "enabled": ["huggingface", "openml", "uci", "github", "kaggle"],
        "kaggle": {
            "username": "",
            "key": ""
        }
    },
    "ai": {
        "default_provider": "ollama",
        "gemini_api_key": "",
        "openai_api_key": "",
        "anthropic_api_key": "",
        "ollama_url": "http://localhost:11434",
        "model_names": {
            "ollama": "llama3",
            "gemini": "gemini-1.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20240620"
        }
    },
    "search": {
        "max_results": 20,
        "timeout_seconds": 10
    }
}

class ConfigManager:
    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            self._save_default_config()
            return DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f) or {}

            return self._merge_dicts(DEFAULT_CONFIG, config)
        except Exception:
            return DEFAULT_CONFIG.copy()

    def _save_default_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config_path, "w") as f:
                yaml.safe_dump(DEFAULT_CONFIG, f, default_flow_style=False)
        except Exception:
            pass

    def save(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)

    def _merge_dicts(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        merged = default.copy()
        for k, v in user.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = self._merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    def get(self, path: str, default: Any = None) -> Any:
        parts = path.split(".")
        val = self.config
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return default
        return val

    def set(self, path: str, value: Any):
        parts = path.split(".")
        val = self.config
        for part in parts[:-1]:
            if part not in val or not isinstance(val[part], dict):
                val[part] = {}
            val = val[part]
        val[parts[-1]] = value
        self.save()

    @property
    def datasets_dir(self) -> Path:
        p = Path(self.get("cache.datasets_dir"))
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def metadata_dir(self) -> Path:
        p = Path(self.get("cache.metadata_dir"))
        p.mkdir(parents=True, exist_ok=True)
        return p
