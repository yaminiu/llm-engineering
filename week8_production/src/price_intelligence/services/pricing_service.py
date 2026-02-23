"""Pricing service for price estimation."""

import logging
from typing import Optional

from ..config import ModelConfig, get_config
from ..exceptions import ModelLoadError
from ..models import PricerModel
from ..utils import validate_string_input

logger = logging.getLogger(__name__)


class PricingService:
    """Service for estimating product prices using fine-tuned Llama model."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize pricing service.
        
        Args:
            config: Model configuration (uses default if not provided)
            
        Raises:
            ModelLoadError: If configuration is invalid
        """
        if config is None:
            config = get_config().model
        
        self.config = config
        self.model = PricerModel(config)
        logger.info("Pricing service initialized")
    
    def estimate_price(self, product_description: str) -> float:
        """
        Estimate price for a product.
        
        Args:
            product_description: Description of the product
            
        Returns:
            float: Estimated price in dollars
            
        Raises:
            ValidationError: If input is invalid
            ModelInferenceError: If inference fails
        """
        product_description = validate_string_input(
            product_description,
            field_name="product_description"
        )
        
        logger.info(f"Estimating price for: {product_description[:100]}...")
        price = self.model.estimate_price(product_description)
        logger.info(f"Price estimate: ${price:.2f}")
        
        return price
    
    def estimate_prices_batch(self, descriptions: list[str]) -> list[float]:
        """
        Estimate prices for multiple products.
        
        Args:
            descriptions: List of product descriptions
            
        Returns:
            list[float]: List of estimated prices
        """
        if not isinstance(descriptions, list):
            raise TypeError("descriptions must be a list")
        
        prices = []
        for i, desc in enumerate(descriptions, 1):
            try:
                price = self.estimate_price(desc)
                prices.append(price)
                logger.info(f"[{i}/{len(descriptions)}] Estimated: ${price:.2f}")
            except Exception as e:
                logger.error(f"Failed to estimate price for item {i}: {str(e)}")
                prices.append(0.0)  # Return 0 for failed estimates
        
        return prices
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        self.model.cleanup()
        logger.info("Pricing service cleaned up")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup()
