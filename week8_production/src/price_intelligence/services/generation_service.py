"""Text generation service."""

import logging
from typing import Optional

from ..config import ModelConfig, get_config
from ..models import LlamaModel
from ..utils import validate_string_input

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for text generation using Llama model."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize generation service.
        
        Args:
            config: Model configuration (uses default if not provided)
        """
        if config is None:
            config = get_config().model
        
        self.config = config
        self.model = LlamaModel(config)
        logger.info("Generation service initialized")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text based on prompt.
        
        Args:
            prompt: Input prompt for text generation
            
        Returns:
            str: Generated text
            
        Raises:
            ValidationError: If input is invalid
            ModelInferenceError: If generation fails
        """
        prompt = validate_string_input(prompt, field_name="prompt")
        
        logger.info(f"Generating text for prompt: {prompt[:50]}...")
        result = self.model.generate(prompt)
        logger.info(f"Generated {len(result)} characters")
        
        return result
    
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            list[str]: List of generated texts
        """
        if not isinstance(prompts, list):
            raise TypeError("prompts must be a list")
        
        results = []
        for i, prompt in enumerate(prompts, 1):
            try:
                result = self.generate(prompt)
                results.append(result)
                logger.info(f"[{i}/{len(prompts)}] Generated text")
            except Exception as e:
                logger.error(f"Failed to generate for prompt {i}: {str(e)}")
                results.append("")  # Return empty string for failed generations
        
        return results
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        self.model.cleanup()
        logger.info("Generation service cleaned up")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
