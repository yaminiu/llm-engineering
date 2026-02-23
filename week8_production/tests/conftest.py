"""Pytest configuration and fixtures."""

import pytest
from price_intelligence.config import ModelConfig, ServiceConfig, AppConfig
from price_intelligence.utils import setup_logging


@pytest.fixture
def test_model_config():
    """Create test model configuration."""
    return ModelConfig(
        base_model="gpt2",  # Smaller model for testing
        finetuned_model="test/model",
        max_new_tokens=5,
    )


@pytest.fixture
def test_app_config(test_model_config):
    """Create test app configuration."""
    return AppConfig(
        debug=True,
        log_level="DEBUG",
        model=test_model_config,
        service=ServiceConfig(modal_enabled=False),
    )


@pytest.fixture
def test_logger():
    """Create test logger."""
    return setup_logging("DEBUG", "test")
