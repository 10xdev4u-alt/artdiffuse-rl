# Examples

## Overview

This document provides examples of using ArtDiffuse-RL for various tasks.

## Basic Usage

### Generate Images

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
    samples = ddpm.sample((16, 3, 64, 64), torch.device('cuda'))

# Save images
from torchvision.utils import save_image
save_image(samples, 'samples.png', nrow=4, normalize=True)
```

### Generate Single Image

```python
# Generate a single image
with torch.no_grad():
    image = ddpm.sample((1, 3, 64, 64), torch.device('cuda'))

# Save image
save_image(image, 'image.png', normalize=True)
```

## Advanced Usage

### Custom Sampling

```python
# Custom number of steps
def sample_with_steps(ddpm, shape, device, steps=50):
    x = torch.randn(shape, device=device)
    for t in reversed(range(0, ddpm.timesteps, ddpm.timesteps // steps)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
        x = ddpm.p_sample(x, t_batch)
    return x

samples = sample_with_steps(ddpm, (16, 3, 64, 64), torch.device('cuda'), steps=50)
```

### Batch Generation

```python
# Generate images in batches
prompts = ["a cat", "a dog", "a bird"]
all_samples = []

for prompt in prompts:
    with torch.no_grad():
        samples = ddpm.sample((1, 3, 64, 64), torch.device('cuda'))
    all_samples.append(samples)

all_samples = torch.cat(all_samples, dim=0)
```

### Conditional Generation

```python
# Generate images conditioned on text
def generate_with_prompt(ddpm, prompt, device):
    # Encode prompt (placeholder)
    prompt_embedding = torch.randn(1, 77, 768).to(device)
    
    # Generate
    with torch.no_grad():
        samples = ddpm.sample((1, 3, 64, 64), device)
    
    return samples

samples = generate_with_prompt(ddpm, "a beautiful sunset", torch.device('cuda'))
```

## Training Examples

### Train from Scratch

```python
from src.train import build_model, train, load_config
from src.dataset import MidjourneyDataset
from torch.utils.data import DataLoader

# Load config
config = load_config('configs/base.yaml')

# Create dataset
dataset = MidjourneyDataset('./data', image_size=64, max_samples=50000)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

# Build model
device = torch.device('cuda')
model = build_model(config).to(device)

# Train
model = train(model, dataloader, config['training'], device, 'models')
```

### Resume Training

```python
# Load checkpoint
checkpoint = torch.load('models/checkpoint_10000.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_step = checkpoint['step']

# Continue training
model = train(model, dataloader, config['training'], device, 'models')
```

### RL Fine-tuning

```python
from src.rl import rl_finetune

# Load base model
model = build_model(config)
model.load_state_dict(torch.load('models/base.pt'))

# Fine-tune with RL
model = rl_finetune(model, dataloader, config['rl'], device, 'models/rl')
```

## Evaluation Examples

### Compute FID

```python
from src.evaluate import compute_fid

# Load real and generated images
real_images = torch.load('data/real.pt')
fake_images = torch.load('data/fake.pt')

# Compute FID
fid = compute_fid(real_images, fake_images)
print(f'FID: {fid:.2f}')
```

### Compute CLIP Score

```python
from src.evaluate import compute_clip_score

# Load images and captions
images = torch.load('data/images.pt')
captions = ["a cat", "a dog", "a bird"]

# Compute CLIP score
clip_score = compute_clip_score(images, captions)
print(f'CLIP Score: {clip_score:.4f}')
```

### Compute Aesthetic Score

```python
from src.evaluate import compute_aesthetic_score

# Load images
images = torch.load('data/images.pt')

# Compute aesthetic score
aesthetic_score = compute_aesthetic_score(images)
print(f'Aesthetic Score: {aesthetic_score:.2f}')
```

## Visualization Examples

### Plot Training Loss

```python
import matplotlib.pyplot as plt

# Load loss history
losses = [1.5, 1.2, 0.9, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.18]

# Plot
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')
```

### Visualize Samples

```python
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Generate samples
samples = ddpm.sample((16, 3, 64, 64), device)

# Create grid
grid = vutils.make_grid(samples, nrow=4, normalize=True)

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis('off')
plt.savefig('samples.png')
```

### Compare Models

```python
# Generate samples from different models
samples_model1 = model1.sample((16, 3, 64, 64), device)
samples_model2 = model2.sample((16, 3, 64, 64), device)

# Create comparison grid
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    axes[0, i].imshow(samples_model1[i].permute(1, 2, 0).cpu())
    axes[0, i].axis('off')
    axes[1, i].imshow(samples_model2[i].permute(1, 2, 0).cpu())
    axes[1, i].axis('off')
plt.savefig('comparison.png')
```

## Web API Example

### Flask API

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
    num_images = data.get('num_images', 1)
    
    with torch.no_grad():
        samples = ddpm.sample((num_images, 3, 64, 64), torch.device('cpu'))
    
    # Convert to base64
    from torchvision.utils import save_image
    import base64
    from io import BytesIO
    
    buffer = BytesIO()
    save_image(samples, buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return jsonify({'image': img_str})

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
    num_images: int = 1

@app.post('/generate')
def generate(request: GenerateRequest):
    with torch.no_grad():
        samples = ddpm.sample((request.num_images, 3, 64, 64), torch.device('cpu'))
    return {'status': 'success', 'num_images': request.num_images}
```

## Jupyter Notebook Example

```python
# In a Jupyter notebook

import torch
from src.model import UNet, DDPM
from IPython.display import display

# Load model
model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))
ddpm.eval()

# Generate and display
with torch.no_grad():
    samples = ddpm.sample((1, 3, 64, 64), torch.device('cuda'))

# Display
from torchvision.utils import make_grid
grid = make_grid(samples, normalize=True)
display(grid.permute(1, 2, 0).cpu())
```

## Tips

1. **Use GPU for faster inference**: `device = torch.device('cuda')`
2. **Batch generation**: Generate multiple images at once
3. **Save memory**: Use `torch.no_grad()` during inference
4. **Visualize results**: Use `make_grid` for easy visualization
5. **Experiment**: Try different sampling steps and parameters
