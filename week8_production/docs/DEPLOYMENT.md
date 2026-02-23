# Deployment Guide

This guide covers deploying Price Intelligence service to different environments.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker](#docker)
3. [Modal](#modal)
4. [Kubernetes](#kubernetes)
5. [Production Checklist](#production-checklist)

## Local Development

### Setup

```bash
# Clone and setup
git clone <repo>
cd price-intelligence
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Create .env file
cp .env.example .env
# Edit .env with your credentials
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=price_intelligence

# Watch mode (install pytest-watch)
ptw
```

### Development Server

```bash
# Debug mode
DEBUG=true LOG_LEVEL=DEBUG python your_app.py

# Profile code
python -m cProfile -o profile.stats your_app.py
```

## Docker

### Build Docker Image

```bash
# Build
docker build -t price-intelligence:latest .

# Build with buildkit (faster)
docker buildx build -t price-intelligence:latest .
```

### Run Container

```bash
# CPU only
docker run -it price-intelligence:latest

# With GPU
docker run --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e DEBUG=false \
  price-intelligence:latest

# Mount local code
docker run --gpus all \
  -v $(pwd)/src:/app/src \
  -e HF_TOKEN=$HF_TOKEN \
  price-intelligence:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  price-intelligence:
    build: .
    image: price-intelligence:latest
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - DEBUG=false
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "7860:7860"  # Gradio UI
    volumes:
      - cache:/cache
      - logs:/app/logs

volumes:
  cache:
  logs:
```

## Modal

### Setup Modal

```bash
# Install Modal
pip install modal

# Authenticate
modal token new

# Verify
modal volume create hf-hub-cache
```

### Deploy to Modal

```bash
# Deploy function
modal run pricer_service.py

# Deploy as web endpoint
modal serve pricer_service.py

# Deploy with custom config
modal run pricer_service.py \
  --env-secret huggingface-secret \
  --gpu T4 \
  --timeout 1800
```

### Modal Secrets

```bash
# Create HuggingFace secret
modal secret create huggingface-secret \
  --field token=$HF_TOKEN

# Create multiple secrets
modal secret create huggingface-secret \
  --field token=$HF_TOKEN \
  --field username=$HF_USERNAME
```

### Example Modal Deployment

```python
import modal
from modal import Image, Secret, Volume

app = modal.App("price-intelligence")

image = Image.debian_slim().pip_install(
    "torch", "transformers", "peft", "bitsandbytes", "accelerate"
)

volumes = {"/cache": Volume.from_name("hf-hub-cache", create_if_missing=True)}

@app.function(
    image=image,
    secrets=[Secret.from_name("huggingface-secret")],
    gpu="T4",
    timeout=1800,
    volumes=volumes,
)
def estimate_price(description: str) -> float:
    from price_intelligence import PricingService
    service = PricingService()
    return service.estimate_price(description)

@app.function(
    image=image,
    secrets=[Secret.from_name("huggingface-secret")],
    gpu="T4",
    timeout=1800,
)
@modal.web_endpoint()
def web_endpoint(description: str = "MacBook Pro") -> dict:
    try:
        from price_intelligence import PricingService
        service = PricingService()
        price = service.estimate_price(description)
        return {"description": description, "price": price}
    except Exception as e:
        return {"error": str(e)}
```

## Kubernetes

### Docker Image for K8s

```bash
docker build -t gcr.io/my-project/price-intelligence:v1.0.0 .
docker push gcr.io/my-project/price-intelligence:v1.0.0
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: price-intelligence
  labels:
    app: price-intelligence
spec:
  replicas: 2
  selector:
    matchLabels:
      app: price-intelligence
  template:
    metadata:
      labels:
        app: price-intelligence
    spec:
      containers:
      - name: price-intelligence
        image: gcr.io/my-project/price-intelligence:v1.0.0
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-credentials
              key: token
        - name: LOG_LEVEL
          value: "INFO"
        - name: DEBUG
          value: "false"
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          limits:
            memory: "16Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: cache
          mountPath: /cache
      volumes:
      - name: cache
        emptyDir:
          sizeLimit: 50Gi
      nodeSelector:
        accelerator: nvidia-tesla-a100  # Or T4, V100, etc.
```

### Service for K8s

```yaml
apiVersion: v1
kind: Service
metadata:
  name: price-intelligence-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: price-intelligence
```

## Production Checklist

- [ ] **Security**
  - [ ] Remove debug mode (`DEBUG=false`)
  - [ ] Rotate API tokens regularly
  - [ ] Use secrets management (K8s Secrets, HashiCorp Vault)
  - [ ] Enable TLS/HTTPS
  - [ ] Implement rate limiting
  - [ ] Add authentication/authorization

- [ ] **Monitoring**
  - [ ] Setup logging aggregation (ELK, Datadog, Splunk)
  - [ ] Add metrics collection (Prometheus)
  - [ ] Configure alerting (PagerDuty, Slack)
  - [ ] Monitor GPU usage
  - [ ] Track inference latency

- [ ] **Performance**
  - [ ] Enable quantization (`QUANTIZATION_ENABLED=true`)
  - [ ] Configure caching appropriately
  - [ ] Load test under expected load
  - [ ] Profile for bottlenecks
  - [ ] Optimize batch sizes

- [ ] **Reliability**
  - [ ] Setup health checks
  - [ ] Configure auto-scaling
  - [ ] Implement circuit breakers
  - [ ] Add retry logic with exponential backoff
  - [ ] Plan disaster recovery

- [ ] **Documentation**
  - [ ] Document deployment process
  - [ ] Create runbooks for common issues
  - [ ] Document configuration parameters
  - [ ] Create SOP for updates/rollbacks

- [ ] **Testing**
  - [ ] Run full test suite
  - [ ] Perform load testing
  - [ ] Test failure scenarios
  - [ ] Verify rollback procedures

## Monitoring & Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, start_http_server

# Setup metrics
price_estimates = Counter('price_estimates_total', 'Total price estimates')
estimation_time = Histogram('price_estimation_seconds', 'Estimation time')

# Start metrics server
start_http_server(8000)

# Record metrics
price_estimates.inc()
with estimation_time.time():
    price = service.estimate_price(description)
```

### Structured Logging

```python
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.getLogger().addHandler(handler)
```

## Scaling

### Horizontal Scaling (Multiple Instances)

Use load balancers (NGINX, HAProxy) to distribute requests across instances.

### Vertical Scaling (Larger GPU)

Upgrade from T4 → V100 → A100 for better performance.

### Model Caching

```python
# Implement Redis caching for inference results
import redis

cache = redis.Redis(host='localhost', port=6379)

def estimate_price_cached(description):
    cache_key = f"price:{description}"
    cached = cache.get(cache_key)
    if cached:
        return float(cached)
    
    price = service.estimate_price(description)
    cache.setex(cache_key, 3600, price)  # Cache for 1 hour
    return price
```

## Troubleshooting

### High Memory Usage

- Enable quantization
- Reduce batch size
- Use smaller model variant

### Slow Inference

- Check GPU utilization: `nvidia-smi`
- Profile code: `python -m cProfile`
- Optimize batch processing

### Model Loading Fails

- Verify HF_TOKEN is set
- Check model repository is public/accessible
- Increase timeout value

For more support, see the main README and API documentation.
