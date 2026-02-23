"""Configuration management for Price Intelligence service."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for LLM models."""

    base_model: str = os.getenv("BASE_MODEL", "meta-llama/Llama-3.2-3B")
    finetuned_model: str = os.getenv("FINETUNED_MODEL", "")
    model_revision: Optional[str] = os.getenv("MODEL_REVISION", None)
    cache_dir: str = os.getenv("HF_HUB_CACHE", "/cache")
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    quantization_enabled: bool = os.getenv("QUANTIZATION_ENABLED", "true").lower() == "true"
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "5"))


@dataclass
class ServiceConfig:
    """Configuration for services."""

    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)
    modal_enabled: bool = os.getenv("MODAL_ENABLED", "false").lower() == "true"
    gpu_type: str = os.getenv("GPU_TYPE", "T4")
    timeout_seconds: int = int(os.getenv("TIMEOUT_SECONDS", "1800"))
    min_containers: int = int(os.getenv("MIN_CONTAINERS", "0"))


@dataclass
class AppConfig:
    """Main application configuration."""

    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    model: ModelConfig = None
    service: ServiceConfig = None

    def __post_init__(self):
        """Initialize nested configs if not provided."""
        if self.model is None:
            self.model = ModelConfig()
        if self.service is None:
            self.service = ServiceConfig()


def get_config() -> AppConfig:
    """Get global application configuration."""
    return AppConfig()
