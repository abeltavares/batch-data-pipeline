"""Configuration management for the pipeline."""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Pipeline configuration singleton."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """Load and validate configuration."""
        config_path = Path("/opt/airflow/config/pipeline_config.yaml")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            cls._config = yaml.safe_load(f)
        
        cls._config = cls._replace_env_vars(cls._config)
        
        # Validate required keys
        required_keys = [
            ('s3', 'endpoint'),
            ('s3', 'access_key'),
            ('s3', 'secret_key'),
            ('storage', 'bronze_bucket'),
            ('storage', 'silver_bucket'),
            ('storage', 'gold_bucket'),
        ]
        
        for keys in required_keys:
            if cls.get(*keys) is None:
                raise ValueError(f"Missing required config: {'.'.join(keys)}")
    
    @classmethod
    def _replace_env_vars(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively replace ${VAR} with environment variables."""
        if isinstance(config, dict):
            return {k: cls._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            var_name = config[2:-1]
            return os.getenv(var_name, config)
        return config
    
    @classmethod
    def get(cls, *keys, default=None):
        """Get configuration value by nested keys."""
        value = cls._config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value


# Convenience function
def get_config(*keys, default=None):
    """Get configuration value."""
    return Config().get(*keys, default=default)
