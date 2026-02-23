# Price Intelligence - Production-Ready LLM Pricing Service

A production-grade Python package for price estimation using fine-tuned Llama models and advanced ML techniques. Built with comprehensive error handling, logging, configuration management, and testing.

## Features

- **Fine-tuned Price Estimation**: Specialized Llama-based model for accurate price predictions
- **Text Generation**: General-purpose text generation with Llama
- **Production-Ready**: Comprehensive logging, error handling, and configuration management
- **Modular Architecture**: Clean separation of concerns with models, services, and utilities
- **Type-Safe**: Full type hints throughout the codebase
- **Well-Tested**: Unit and integration tests with pytest
- **Modal Support**: Easy deployment to Modal serverless platform
- **Configurable**: Environment-based configuration for multiple deployments

## Installation

### Prerequisites
- Python 3.10+
- GPU with 8GB+ VRAM (recommended)
- HuggingFace account for model access

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/price-intelligence.git
cd price-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### With Optional Features

```bash
# For UI with Gradio
pip install -e ".[ui]"

# For Modal deployment
pip install -e ".[modal]"

# For vector database
pip install -e ".[chromadb]"

# All optional features
pip install -e ".[dev,ui,modal,chromadb]"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Model Configuration
BASE_MODEL=meta-llama/Llama-3.2-3B
FINETUNED_MODEL=your-username/your-model-name
HF_TOKEN=your_huggingface_token

# Service Configuration
DEBUG=false
LOG_LEVEL=INFO
QUANTIZATION_ENABLED=true
GPU_TYPE=T4
```

## Usage

### Basic Price Estimation

```python
from price_intelligence import PricingService

# Initialize service
service = PricingService()

# Estimate price for a product
description = "Apple MacBook Pro 16-inch with M2 Max, 32GB RAM, 512GB SSD"
price = service.estimate_price(description)
print(f"Estimated price: ${price:.2f}")

# Cleanup
service.cleanup()
```

### Batch Processing

```python
from price_intelligence import PricingService

service = PricingService()

descriptions = [
    "iPhone 15 Pro Max 256GB",
    "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro",
]

prices = service.estimate_prices_batch(descriptions)
for desc, price in zip(descriptions, prices):
    print(f"{desc}: ${price:.2f}")

service.cleanup()
```

### Text Generation

```python
from price_intelligence import GenerationService

service = GenerationService()

prompt = "Write a product review for a smartphone:"
generated_text = service.generate(prompt)
print(generated_text)

service.cleanup()
```

### Custom Configuration

```python
from price_intelligence.config import ModelConfig
from price_intelligence import PricingService

# Create custom config
config = ModelConfig(
    base_model="meta-llama/Llama-3.2-3B",
    finetuned_model="my-org/my-price-model",
    max_new_tokens=10,
    quantization_enabled=True,
)

# Initialize with custom config
service = PricingService(config)
price = service.estimate_price("Your product description")
```

## API Documentation

### PricingService

```python
class PricingService:
    def estimate_price(product_description: str) -> float:
        """Estimate price for a single product."""
    
    def estimate_prices_batch(descriptions: list[str]) -> list[float]:
        """Estimate prices for multiple products."""
    
    def cleanup() -> None:
        """Clean up model resources."""
```

### GenerationService

```python
class GenerationService:
    def generate(prompt: str) -> str:
        """Generate text based on prompt."""
    
    def generate_batch(prompts: list[str]) -> list[str]:
        """Generate text for multiple prompts."""
    
    def cleanup() -> None:
        """Clean up model resources."""
```

### Exceptions

- `ValidationError`: Input validation failed
- `ModelLoadError`: Model loading failed
- `ModelInferenceError`: Model inference failed
- `PriceParsingError`: Price parsing from output failed
- `ConfigurationError`: Configuration is invalid

## Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=price_intelligence

# Run specific test file
pytest tests/unit/test_parsing.py

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
│   ├── test_validation.py
│   └── test_parsing.py
├── integration/    # Integration tests
├── fixtures/       # Test fixtures and data
└── conftest.py     # Pytest configuration
```

## Deployment

### Local Development

```bash
# Run with debugging
export DEBUG=true
export LOG_LEVEL=DEBUG
python -c "from price_intelligence import PricingService; ..."
```

### Docker Deployment

```bash
# Build image
docker build -t price-intelligence:latest .

# Run container
docker run --gpus all -e HF_TOKEN=$HF_TOKEN price-intelligence:latest
```

### Modal Deployment

See `DEPLOYMENT.md` for detailed Modal deployment instructions.

## Architecture

### Module Structure

```
price_intelligence/
├── models/          # ML model implementations
│   ├── base.py      # Base model classes
│   ├── pricer.py    # Fine-tuned pricing model
│   └── llama.py     # Llama text generation
├── services/        # Service layer
│   ├── pricing_service.py
│   └── generation_service.py
├── utils/           # Utility functions
│   ├── logging.py   # Logging setup
│   ├── parsing.py   # Output parsing
│   └── validation.py # Input validation
├── config.py        # Configuration management
└── exceptions.py    # Custom exceptions
```

### Design Patterns

1. **Service Layer Pattern**: Services provide clean APIs for model usage
2. **Configuration Management**: Environment-based config with dataclasses
3. **Error Handling**: Custom exceptions for specific error cases
4. **Logging**: Structured logging throughout
5. **Type Safety**: Full type hints for better IDE support

## Logging

The package includes comprehensive logging:

```python
from price_intelligence.utils import setup_logging

logger = setup_logging(level="INFO", name="my_app")
logger.info("Starting price estimation...")
```

Log format:
```
[2024-01-15 10:30:45] [my_app] [INFO] Starting price estimation...
```

## Performance Optimization

### Quantization

4-bit quantization is enabled by default to reduce memory usage:

```env
QUANTIZATION_ENABLED=true
```

### Caching

Models cache inference results:

```python
service = PricingService()

# First call loads and caches
price1 = service.estimate_price(desc)

# Second call with same description uses cache
price2 = service.estimate_price(desc)
```

## Troubleshooting

### Out of Memory

If you encounter OOM errors:

1. Enable quantization: `QUANTIZATION_ENABLED=true`
2. Reduce batch size
3. Use smaller GPU (`GPU_TYPE=T4` instead of `A100`)

### Model Loading Issues

```python
from price_intelligence.exceptions import ModelLoadError

try:
    service = PricingService()
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
    # Check HF_TOKEN and FINETUNED_MODEL env vars
```

### Import Errors

Make sure to install in editable mode:
```bash
pip install -e .
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Commit: `git commit -am 'Add feature'`
6. Push: `git push origin feature/name`
7. Submit a Pull Request

## Code Quality

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/

# All checks
black src/ tests/ && ruff check src/ tests/ && mypy src/
```

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or contributions, please create an issue on GitHub.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{price_intelligence_2024,
  title={Price Intelligence: Production-Ready LLM Pricing Service},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/price-intelligence}
}
```

## References

- [Llama Models Documentation](https://huggingface.co/meta-llama)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Library](https://huggingface.co/transformers/)
- [Modal Documentation](https://modal.com/docs)
