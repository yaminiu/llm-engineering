# API Reference

Complete API documentation for Price Intelligence package.

## Table of Contents

1. [PricingService](#pricingservice)
2. [GenerationService](#generationservice)
3. [Models](#models)
4. [Configuration](#configuration)
5. [Exceptions](#exceptions)
6. [Utilities](#utilities)

## PricingService

### Overview

Service for estimating product prices using fine-tuned Llama model.

### Class Definition

```python
class PricingService:
    def __init__(self, config: Optional[ModelConfig] = None)
    def estimate_price(self, product_description: str) -> float
    def estimate_prices_batch(self, descriptions: list[str]) -> list[float]
    def cleanup(self) -> None
```

### Methods

#### `__init__(config: Optional[ModelConfig] = None)`

Initialize the pricing service.

**Parameters:**
- `config` (ModelConfig, optional): Model configuration. Uses default if not provided.

**Raises:**
- `ModelLoadError`: If configuration is invalid or model loading fails.

**Example:**
```python
from price_intelligence import PricingService
from price_intelligence.config import ModelConfig

# Default config
service = PricingService()

# Custom config
config = ModelConfig(
    base_model="meta-llama/Llama-3.2-3B",
    finetuned_model="my-org/my-model",
    max_new_tokens=10,
)
service = PricingService(config)
```

#### `estimate_price(product_description: str) -> float`

Estimate price for a single product.

**Parameters:**
- `product_description` (str): Description of the product. Must be non-empty and ≤10,000 characters.

**Returns:**
- `float`: Estimated price in dollars.

**Raises:**
- `ValidationError`: If input is invalid.
- `ModelInferenceError`: If inference fails.
- `PriceParsingError`: If price cannot be extracted from output.

**Example:**
```python
service = PricingService()
price = service.estimate_price("iPhone 15 Pro Max 256GB")
print(f"Estimated price: ${price:.2f}")  # Output: Estimated price: $999.99
```

#### `estimate_prices_batch(descriptions: list[str]) -> list[float]`

Estimate prices for multiple products.

**Parameters:**
- `descriptions` (list[str]): List of product descriptions.

**Returns:**
- `list[float]`: List of estimated prices. Returns 0.0 for failed estimates.

**Raises:**
- `TypeError`: If descriptions is not a list.

**Example:**
```python
descriptions = [
    "MacBook Pro 16-inch",
    "iPad Air 11-inch",
    "Apple Watch Series 9",
]
prices = service.estimate_prices_batch(descriptions)
for desc, price in zip(descriptions, prices):
    print(f"{desc}: ${price:.2f}")
```

#### `cleanup() -> None`

Clean up model resources (GPU memory, etc.).

**Example:**
```python
service = PricingService()
try:
    price = service.estimate_price("Product description")
finally:
    service.cleanup()  # Always cleanup
```

---

## GenerationService

### Overview

Service for general-purpose text generation using Llama model.

### Class Definition

```python
class GenerationService:
    def __init__(self, config: Optional[ModelConfig] = None)
    def generate(self, prompt: str) -> str
    def generate_batch(self, prompts: list[str]) -> list[str]
    def cleanup(self) -> None
```

### Methods

#### `__init__(config: Optional[ModelConfig] = None)`

Initialize the generation service.

**Parameters:**
- `config` (ModelConfig, optional): Model configuration. Uses default if not provided.

**Example:**
```python
from price_intelligence import GenerationService

service = GenerationService()
```

#### `generate(prompt: str) -> str`

Generate text based on input prompt.

**Parameters:**
- `prompt` (str): Input prompt for generation. Must be non-empty and ≤10,000 characters.

**Returns:**
- `str`: Generated text.

**Raises:**
- `ValidationError`: If input is invalid.
- `ModelInferenceError`: If generation fails.

**Example:**
```python
service = GenerationService()
prompt = "Describe the features of a flagship smartphone:"
text = service.generate(prompt)
print(text)
```

#### `generate_batch(prompts: list[str]) -> list[str]`

Generate text for multiple prompts.

**Parameters:**
- `prompts` (list[str]): List of input prompts.

**Returns:**
- `list[str]`: List of generated texts. Returns empty string for failed generations.

**Example:**
```python
prompts = [
    "Describe a smartphone",
    "Explain AI technology",
    "Write product review",
]
results = service.generate_batch(prompts)
```

#### `cleanup() -> None`

Clean up model resources.

---

## Models

### BaseModel

Abstract base class for all models.

```python
class BaseModel(ABC):
    def load(self) -> None
    def infer(self, prompt: str) -> str
    def ensure_loaded(self) -> None
    def cleanup(self) -> None
    @property
    def is_loaded(self) -> bool
```

### CachedModel

Model with inference caching support.

```python
class CachedModel(BaseModel):
    def infer_with_cache(self, prompt: str) -> str
    def clear_cache(self) -> None
```

### PricerModel

Fine-tuned pricing model.

```python
class PricerModel(CachedModel):
    def __init__(self, config: ModelConfig, question: str = None, prefix: str = None)
    def load(self) -> None
    def infer(self, product_description: str) -> str
    def estimate_price(self, product_description: str) -> float
```

### LlamaModel

General-purpose Llama text generation.

```python
class LlamaModel(CachedModel):
    def load(self) -> None
    def infer(self, prompt: str) -> str
    def generate(self, prompt: str) -> str
```

---

## Configuration

### ModelConfig

```python
@dataclass
class ModelConfig:
    base_model: str  # Default: "meta-llama/Llama-3.2-3B"
    finetuned_model: str  # Required for PricerModel
    model_revision: Optional[str]  # Default: None
    cache_dir: str  # Default: "/cache"
    device_map: str  # Default: "auto"
    quantization_enabled: bool  # Default: True
    max_new_tokens: int  # Default: 5
```

### ServiceConfig

```python
@dataclass
class ServiceConfig:
    hf_token: Optional[str]
    modal_enabled: bool  # Default: False
    gpu_type: str  # Default: "T4"
    timeout_seconds: int  # Default: 1800
    min_containers: int  # Default: 0
```

### AppConfig

```python
@dataclass
class AppConfig:
    debug: bool  # Default: False
    log_level: str  # Default: "INFO"
    model: ModelConfig
    service: ServiceConfig
```

### Creating Custom Config

```python
from price_intelligence.config import ModelConfig, AppConfig, ServiceConfig

config = AppConfig(
    debug=False,
    log_level="INFO",
    model=ModelConfig(
        base_model="meta-llama/Llama-3.2-3B",
        finetuned_model="my-org/my-model",
        quantization_enabled=True,
    ),
    service=ServiceConfig(
        modal_enabled=False,
        gpu_type="A100",
    ),
)
```

---

## Exceptions

### Exception Hierarchy

```
PriceIntelligenceError (base)
├── ModelLoadError
├── ModelInferenceError
├── PriceParsingError
├── ValidationError
└── ConfigurationError
```

### Exception Classes

#### `PriceIntelligenceError`

Base exception for all Price Intelligence errors.

```python
try:
    service.estimate_price("product")
except PriceIntelligenceError as e:
    print(f"Price Intelligence error: {e}")
```

#### `ModelLoadError`

Raised when model loading fails.

```python
from price_intelligence.exceptions import ModelLoadError

try:
    service = PricingService()
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
    # Check configuration, HF_TOKEN, etc.
```

#### `ModelInferenceError`

Raised when model inference fails.

```python
from price_intelligence.exceptions import ModelInferenceError

try:
    price = service.estimate_price(description)
except ModelInferenceError as e:
    print(f"Inference failed: {e}")
```

#### `PriceParsingError`

Raised when price cannot be extracted from model output.

```python
from price_intelligence.exceptions import PriceParsingError

try:
    price = service.estimate_price(description)
except PriceParsingError as e:
    print(f"Failed to parse price: {e}")
```

#### `ValidationError`

Raised when input validation fails.

```python
from price_intelligence.exceptions import ValidationError

try:
    # Empty string fails validation
    price = service.estimate_price("")
except ValidationError as e:
    print(f"Invalid input: {e}")
```

#### `ConfigurationError`

Raised when configuration is invalid.

---

## Utilities

### Logging

```python
from price_intelligence.utils import setup_logging, LogContext

# Setup logging
logger = setup_logging(level="INFO", name="my_app")
logger.info("Application started")

# Temporary level change
logger = setup_logging("INFO")
with LogContext(logger, "DEBUG"):
    logger.debug("Debug message")  # Visible
logger.debug("Debug message")  # Not visible (back to INFO)
```

### Validation

```python
from price_intelligence.utils import (
    validate_string_input,
    validate_numeric_input,
)
from price_intelligence.exceptions import ValidationError

# String validation
try:
    text = validate_string_input("  hello  ", min_length=1, max_length=1000)
    assert text == "hello"  # Trimmed
except ValidationError as e:
    print(f"Validation failed: {e}")

# Numeric validation
try:
    value = validate_numeric_input(42, min_val=0, max_val=100)
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Parsing

```python
from price_intelligence.utils import extract_price, extract_text_content

# Extract price
text = "The item costs Price is $49.99 today"
price = extract_price(text)  # Returns 49.99

# Extract text content
text = "PREFIX important content SUFFIX"
content = extract_text_content(text, start_marker="PREFIX ", end_marker=" SUFFIX")
# Returns "important content"
```

---

## Complete Example

```python
from price_intelligence import PricingService, GenerationService
from price_intelligence.config import ModelConfig
from price_intelligence.exceptions import PriceIntelligenceError
from price_intelligence.utils import setup_logging

# Setup logging
logger = setup_logging("INFO")

# Create config
config = ModelConfig(
    base_model="meta-llama/Llama-3.2-3B",
    finetuned_model="my-org/price-model",
    quantization_enabled=True,
)

try:
    # Estimate prices
    pricing_service = PricingService(config)
    products = [
        "iPhone 15 Pro Max 256GB",
        "MacBook Pro 16-inch",
        "iPad Air 11-inch",
    ]
    
    prices = pricing_service.estimate_prices_batch(products)
    
    for product, price in zip(products, prices):
        logger.info(f"{product}: ${price:.2f}")
    
    # Generate reviews
    generation_service = GenerationService(config)
    prompts = ["Write a review for the iPhone 15:", "Describe the MacBook Pro:"]
    reviews = generation_service.generate_batch(prompts)
    
    for prompt, review in zip(prompts, reviews):
        logger.info(f"{prompt}\n{review}\n")

except PriceIntelligenceError as e:
    logger.error(f"Error: {e}")

finally:
    pricing_service.cleanup()
    generation_service.cleanup()
    logger.info("Cleanup complete")
```

---

For more information, see the [README](../README.md) and [Deployment Guide](./DEPLOYMENT.md).
