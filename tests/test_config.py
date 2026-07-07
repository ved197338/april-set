import os
import shutil
import tempfile
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.manager import ConfigManager

def test_default_config():

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        config_path = tmp_path / "config.yaml"
        cm = ConfigManager(config_path=config_path)

        assert cm.get("cache.max_size_gb") == 10.0
        assert cm.get("ai.default_provider") == "ollama"
        assert "uci" in cm.get("providers.enabled")

        assert cm.metadata_dir.exists()
        assert cm.datasets_dir.exists()

def test_set_and_get():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config_path = tmp_path / "config.yaml"
        cm = ConfigManager(config_path=config_path)

        cm.set("cache.max_size_gb", 5.5)
        cm.set("providers.kaggle.username", "testuser")

        assert cm.get("cache.max_size_gb") == 5.5
        assert cm.get("providers.kaggle.username") == "testuser"

        cm2 = ConfigManager(config_path=config_path)
        assert cm2.get("cache.max_size_gb") == 5.5
        assert cm2.get("providers.kaggle.username") == "testuser"
