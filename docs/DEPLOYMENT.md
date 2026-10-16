# Deployment Guide

## Overview

Deploy ArtDiffuse-RL for inference in various environments.

## Local Deployment

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CPU or GPU

### Basic Inference

```python
import torch
from src.model import UNet, DDPM

# Load model
model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))
ddpm.eval()

# Generate images
with torch.no_grad():
    samples = ddpm.sample((1, 3, 64, 64), torch.device('cpu'))
```

### Batch Inference

```python
# Generate multiple images
prompts = ["a cat", "a dog", "a bird"]
batch_size = len(prompts)

with torch.no_grad():
    samples = ddpm.sample((batch_size, 3, 64, 64), torch.device('cuda'))
```

## Web API

### Flask

```python
from flask import Flask, request, jsonify
import torch
from src.model import UNet, DDPM

app = Flask(__name__)

# Load model
model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))
ddpm.eval()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    num_images = data.get('num_images', 1)
    
    with torch.no_grad():
        samples = ddpm.sample((num_images, 3, 64, 64), torch.device('cpu'))
    
    # Convert to base64 or save to file
    return jsonify({'status': 'success', 'num_images': num_images})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from src.model import UNet, DDPM

app = FastAPI()

# Load model
model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))
ddpm.eval()

class GenerateRequest(BaseModel):
    prompt: str
    num_images: int = 1

@app.post('/generate')
def generate(request: GenerateRequest):
    with torch.no_grad():
        samples = ddpm.sample((request.num_images, 3, 64, 64), torch.device('cpu'))
    return {'status': 'success', 'num_images': request.num_images}
```

## Docker

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 5000

CMD ["python", "-m", "src.serve"]
```

### Build and Run

```bash
docker build -t artdiffuse-rl .
docker run -p 5000:5000 artdiffuse-rl
```

## Cloud Deployment

### AWS

1. **EC2**: Deploy on a GPU instance
2. **SageMaker**: Use SageMaker endpoints
3. **Lambda**: For serverless inference (CPU only)

### Google Cloud

1. **Compute Engine**: Deploy on a GPU instance
2. **AI Platform**: Use AI Platform endpoints
3. **Cloud Run**: For containerized deployment

### Azure

1. **Virtual Machines**: Deploy on a GPU instance
2. **Container Instances**: For containerized deployment
3. **Kubernetes Service**: For scalable deployment

## Performance Optimization

### ONNX Export

```python
import torch
from src.model import UNet, DDPM

model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))
ddpm.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 64, 64)
torch.onnx.export(ddpm, dummy_input, 'model.onnx')
```

### TensorRT

```python
import torch
from src.model import UNet, DDPM

# Load model
model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))
ddpm.eval()

# Convert to TensorRT
# Requires TensorRT installation
```

### Quantization

```python
import torch
from src.model import UNet, DDPM

# Load model
model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))
ddpm.eval()

# Quantize
ddpm_quantized = torch.quantization.quantize_dynamic(
    ddpm, {torch.nn.Linear}, dtype=torch.qint8
)
```

## Monitoring

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log generation
logger.info(f'Generated {num_images} images')
```

### Metrics

```python
from prometheus_client import Counter, Histogram

generation_count = Counter('generation_count', 'Number of images generated')
generation_time = Histogram('generation_time', 'Time to generate images')

@generation_time.time()
def generate():
    generation_count.inc()
    # Generate images
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use CPU instead of GPU
- Use quantization

### Slow Inference

- Use GPU
- Use ONNX or TensorRT
- Reduce image size

### Poor Quality

- Use more sampling steps
- Use a larger model
- Fine-tune on specific data
