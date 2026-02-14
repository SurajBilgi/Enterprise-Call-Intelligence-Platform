"""
Configuration Loader
====================

Centralized configuration management with environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


class Config:
    """
    Singleton configuration manager.

    Loads configuration from:
    1. YAML file (default.yaml)
    2. Environment variables (override)
    """

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self._load_config()

    def _load_config(self):
        """Load configuration from file and environment"""
        # Load environment variables
        load_dotenv()

        # Load YAML config
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Override with environment variables
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Example overrides
        if os.getenv("OPENAI_API_KEY"):
            self._config.setdefault("llm", {})
            self._config["llm"]["api_key"] = os.getenv("OPENAI_API_KEY")

        if os.getenv("APP_ENV"):
            self._config.setdefault("app", {})
            self._config["app"]["environment"] = os.getenv("APP_ENV")

        if os.getenv("LOG_LEVEL"):
            self._config.setdefault("observability", {})
            self._config["observability"]["log_level"] = os.getenv("LOG_LEVEL")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Example:
            config.get('services.preprocessing.importance_thresholds.high_priority')
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration"""
        return self._config

    @property
    def storage_paths(self) -> Dict[str, Path]:
        """Get all storage paths as Path objects"""
        base_path = Path(__file__).parent.parent
        storage = self.get("storage", {})

        return {
            "raw_transcripts": base_path
            / storage.get("raw_transcripts_path", "storage/raw_transcripts"),
            "structured_db": base_path
            / storage.get("structured_db_path", "storage/structured/calls.db"),
            "vector_store": base_path
            / storage.get("vector_store_path", "storage/vectors"),
            "cache": base_path / storage.get("cache_path", "storage/cache"),
            "keyword_index": base_path
            / storage.get("keyword_index_path", "storage/search/keyword_index.pkl"),
        }

    def ensure_directories(self):
        """Create storage directories if they don't exist"""
        for path in self.storage_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
