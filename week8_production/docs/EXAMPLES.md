# Usage Examples

Practical examples of using the Price Intelligence package.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Batch Processing](#batch-processing)
3. [Error Handling](#error-handling)
4. [Advanced Configuration](#advanced-configuration)
5. [Integration Examples](#integration-examples)

## Basic Usage

### Simple Price Estimation

```python
from price_intelligence import PricingService

# Initialize service
service = PricingService()

# Estimate price
description = "Apple MacBook Pro 16-inch M2 Max, 32GB RAM, 512GB SSD"
price = service.estimate_price(description)

print(f"Product: {description}")
print(f"Estimated Price: ${price:.2f}")

# Always cleanup
service.cleanup()
```

Output:
```
Product: Apple MacBook Pro 16-inch M2 Max, 32GB RAM, 512GB SSD
Estimated Price: $2499.00
```

### Text Generation

```python
from price_intelligence import GenerationService

service = GenerationService()

prompt = "Write a product review for a high-end smartphone in 100 words:"
review = service.generate(prompt)

print(review)
service.cleanup()
```

## Batch Processing

### Process Multiple Products

```python
from price_intelligence import PricingService

service = PricingService()

products = {
    "phone": "iPhone 15 Pro Max 256GB",
    "laptop": "MacBook Pro 16-inch M2 Max",
    "tablet": "iPad Air 11-inch M2",
    "watch": "Apple Watch Series 9 45mm",
}

descriptions = list(products.values())
prices = service.estimate_prices_batch(descriptions)

print("Price Estimates:")
print("-" * 50)
for product_name, price in zip(products.keys(), prices):
    print(f"{product_name:15} ${price:>8.2f}")

service.cleanup()
```

Output:
```
Price Estimates:
--------------------------------------------------
phone                $1199.00
laptop               $2499.00
tablet                $599.00
watch                 $399.00
```

### Parallel Processing with ThreadPoolExecutor

```python
from price_intelligence import PricingService
from concurrent.futures import ThreadPoolExecutor, as_completed

descriptions = [
    "iPhone 15 Pro", "Samsung Galaxy S24 Ultra",
    "Google Pixel 8 Pro", "OnePlus 12",
]

def estimate_with_service(description):
    service = PricingService()
    try:
        return description, service.estimate_price(description)
    finally:
        service.cleanup()

results = {}
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(estimate_with_service, desc): desc
        for desc in descriptions
    }
    
    for future in as_completed(futures):
        desc, price = future.result()
        results[desc] = price
        print(f"✓ {desc}: ${price:.2f}")

print("\nFinal Results:")
for desc, price in results.items():
    print(f"{desc:30} ${price:>8.2f}")
```

## Error Handling

### Comprehensive Error Handling

```python
from price_intelligence import PricingService
from price_intelligence.exceptions import (
    PriceIntelligenceError,
    ValidationError,
    ModelInferenceError,
    PriceParsingError,
    ModelLoadError,
)
import logging

logger = logging.getLogger(__name__)

try:
    service = PricingService()
    
    # Test various error conditions
    test_cases = [
        ("", "Empty description"),
        ("x" * 20000, "Description too long"),
        ("Valid iPhone 15", "Valid input"),
    ]
    
    for description, test_name in test_cases:
        try:
            logger.info(f"Testing: {test_name}")
            price = service.estimate_price(description)
            logger.info(f"Result: ${price:.2f}")
            
        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            continue
            
        except PriceParsingError as e:
            logger.warning(f"Failed to parse price: {e}")
            continue
            
        except ModelInferenceError as e:
            logger.error(f"Model inference failed: {e}")
            continue

except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
    logger.error("Check HF_TOKEN and FINETUNED_MODEL environment variables")
    
except PriceIntelligenceError as e:
    logger.error(f"Unexpected error: {e}")
    
finally:
    try:
        service.cleanup()
    except:
        pass
```

### Retry Logic

```python
from price_intelligence import PricingService
from price_intelligence.exceptions import ModelInferenceError
import time

def estimate_with_retry(description, max_retries=3, delay=1):
    """Estimate price with retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            service = PricingService()
            price = service.estimate_price(description)
            service.cleanup()
            return price
            
        except ModelInferenceError as e:
            if attempt == max_retries:
                raise
            print(f"Attempt {attempt} failed, retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    
    return None

# Usage
price = estimate_with_retry("iPhone 15 Pro Max 256GB", max_retries=3)
print(f"Final price: ${price:.2f}")
```

## Advanced Configuration

### Custom Model Configuration

```python
from price_intelligence.config import ModelConfig, ServiceConfig, AppConfig
from price_intelligence import PricingService
from price_intelligence.utils import setup_logging

# Setup logging
logger = setup_logging("DEBUG", "price_intelligence")

# Create custom configuration
config = AppConfig(
    debug=False,
    log_level="INFO",
    model=ModelConfig(
        base_model="meta-llama/Llama-3.2-3B",
        finetuned_model="my-org/price-estimator-v2",
        model_revision="main",
        cache_dir="/home/user/.cache/huggingface",
        device_map="auto",
        quantization_enabled=True,
        max_new_tokens=10,
    ),
    service=ServiceConfig(
        modal_enabled=False,
        gpu_type="A100",
        timeout_seconds=3600,
    ),
)

# Initialize service with custom config
service = PricingService(config.model)

# Use service
price = service.estimate_price("High-end gaming laptop")
print(f"Estimated price: ${price:.2f}")

service.cleanup()
```

### Environment-Based Configuration

```python
import os
from price_intelligence.config import get_config
from price_intelligence import PricingService

# Configuration is loaded from environment variables (.env file)
# BASE_MODEL=meta-llama/Llama-3.2-3B
# FINETUNED_MODEL=my-org/price-model
# HF_TOKEN=...
# DEBUG=false
# LOG_LEVEL=INFO

config = get_config()
service = PricingService(config.model)

# Use configuration
print(f"Using model: {config.model.base_model}")
print(f"Quantization: {config.model.quantization_enabled}")
print(f"Debug mode: {config.debug}")

service.cleanup()
```

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from price_intelligence import PricingService
from price_intelligence.exceptions import PriceIntelligenceError
import logging

app = Flask(__name__)
service = None
logger = logging.getLogger(__name__)

@app.before_request
def init_service():
    global service
    if service is None:
        service = PricingService()

@app.route('/estimate', methods=['POST'])
def estimate_price():
    """Estimate price for product description."""
    try:
        data = request.json
        description = data.get('description')
        
        if not description:
            return jsonify({'error': 'description required'}), 400
        
        price = service.estimate_price(description)
        return jsonify({
            'description': description,
            'estimated_price': round(price, 2),
        })
        
    except PriceIntelligenceError as e:
        logger.error(f"Error estimating price: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-estimate', methods=['POST'])
def batch_estimate():
    """Estimate prices for multiple products."""
    try:
        data = request.json
        descriptions = data.get('descriptions', [])
        
        if not descriptions:
            return jsonify({'error': 'descriptions required'}), 400
        
        prices = service.estimate_prices_batch(descriptions)
        
        return jsonify({
            'results': [
                {'description': desc, 'price': price}
                for desc, price in zip(descriptions, prices)
            ]
        })
        
    except PriceIntelligenceError as e:
        logger.error(f"Error estimating prices: {e}")
        return jsonify({'error': str(e)}), 500

@app.teardown_appcontext
def cleanup(error=None):
    global service
    if service:
        service.cleanup()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Usage:
```bash
# Single estimate
curl -X POST http://localhost:5000/estimate \
  -H "Content-Type: application/json" \
  -d '{"description": "iPhone 15 Pro Max 256GB"}'

# Response:
# {"description": "iPhone 15 Pro Max 256GB", "estimated_price": 1199.0}

# Batch estimate
curl -X POST http://localhost:5000/batch-estimate \
  -H "Content-Type: application/json" \
  -d '{
    "descriptions": [
      "iPhone 15 Pro",
      "MacBook Pro 16-inch",
      "iPad Air 11-inch"
    ]
  }'
```

### FastAPI with Async

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from price_intelligence import PricingService
from price_intelligence.exceptions import PriceIntelligenceError
import asyncio

app = FastAPI(title="Price Intelligence API", version="1.0.0")

class PriceEstimateRequest(BaseModel):
    description: str

class PriceEstimateResponse(BaseModel):
    description: str
    estimated_price: float

@app.post("/estimate", response_model=PriceEstimateResponse)
async def estimate_price(request: PriceEstimateRequest):
    """Estimate price for product description."""
    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        service = PricingService()
        price = await loop.run_in_executor(
            None,
            service.estimate_price,
            request.description
        )
        service.cleanup()
        
        return PriceEstimateResponse(
            description=request.description,
            estimated_price=round(price, 2),
        )
    except PriceIntelligenceError as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### CLI Tool

```python
#!/usr/bin/env python
"""Command-line tool for price estimation."""

import click
from price_intelligence import PricingService
from price_intelligence.exceptions import PriceIntelligenceError
from price_intelligence.utils import setup_logging

logger = setup_logging("INFO")

@click.group()
def cli():
    """Price Intelligence CLI"""
    pass

@cli.command()
@click.argument('description')
def estimate(description):
    """Estimate price for a product."""
    try:
        service = PricingService()
        price = service.estimate_price(description)
        click.echo(f"✓ Estimated price: ${price:.2f}")
        service.cleanup()
    except PriceIntelligenceError as e:
        click.echo(f"✗ Error: {e}", err=True)

@cli.command()
@click.argument('file', type=click.File('r'))
def batch(file):
    """Estimate prices from file (one product per line)."""
    try:
        service = PricingService()
        descriptions = [line.strip() for line in file if line.strip()]
        
        click.echo(f"Processing {len(descriptions)} products...")
        prices = service.estimate_prices_batch(descriptions)
        
        for desc, price in zip(descriptions, prices):
            click.echo(f"{desc:50} ${price:>8.2f}")
        
        service.cleanup()
    except PriceIntelligenceError as e:
        click.echo(f"✗ Error: {e}", err=True)

@cli.command()
@click.option('--count', default=5, help='Number of products')
def demo(count):
    """Run demo with sample products."""
    sample_products = [
        "iPhone 15 Pro Max 256GB",
        "MacBook Pro 16-inch M3 Max",
        "iPad Air 11-inch M2",
        "Apple Watch Series 9 45mm",
        "AirPods Pro 2nd Gen",
    ]
    
    try:
        service = PricingService()
        products = sample_products[:count]
        click.echo("Running price estimation demo...\n")
        
        for product in products:
            price = service.estimate_price(product)
            click.echo(f"✓ {product:40} ${price:>8.2f}")
        
        service.cleanup()
    except PriceIntelligenceError as e:
        click.echo(f"✗ Error: {e}", err=True)

if __name__ == '__main__':
    cli()
```

Usage:
```bash
# Single estimate
python cli.py estimate "iPhone 15 Pro"

# Batch from file
python cli.py batch products.txt

# Demo
python cli.py demo --count 3
```

---

For more information, see the [API Reference](./API.md) and [README](../README.md).
