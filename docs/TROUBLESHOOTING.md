# Troubleshooting Guide

## Common Issues

### Training Issues

#### Out of Memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce batch size
2. Enable gradient accumulation
3. Use mixed precision
4. Use gradient checkpointing

```python
# Reduce batch size
config['training']['batch_size'] = 128

# Enable gradient accumulation
config['training']['grad_accum'] = 4

# Use mixed precision
config['training']['amp'] = True
```

#### Slow Training

**Symptoms**:
- Training is much slower than expected
- GPU utilization is low

**Solutions**:
1. Check data loading
2. Use mixed precision
3. Increase batch size
4. Use multiple GPUs

```python
# Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
```

#### Loss Not Decreasing

**Symptoms**:
- Loss stays constant or increases
- Model doesn't learn

**Solutions**:
1. Check learning rate
2. Check data
3. Check model architecture
4. Train longer

```python
# Adjust learning rate
config['training']['lr'] = 5e-5

# Train longer
config['training']['max_steps'] = 50000
```

### Evaluation Issues

#### High FID Score

**Symptoms**:
- FID > 30
- Poor image quality

**Solutions**:
1. Train longer
2. Use better data
3. Adjust hyperparameters
4. Use RL fine-tuning

#### Low CLIP Score

**Symptoms**:
- CLIP score < 0.25
- Poor text-image alignment

**Solutions**:
1. Use better text encoder
2. Improve training data
3. Fine-tune on specific data

### Inference Issues

#### Slow Inference

**Symptoms**:
- Takes too long to generate images

**Solutions**:
1. Use GPU instead of CPU
2. Export to ONNX
3. Use TensorRT
4. Reduce image size

```python
# Use GPU
device = torch.device('cuda')
samples = ddpm.sample((1, 3, 64, 64), device)
```

#### Poor Quality

**Symptoms**:
- Generated images are blurry or distorted

**Solutions**:
1. Use more sampling steps
2. Use a larger model
3. Fine-tune on specific data
4. Adjust sampling parameters

### Data Issues

#### Dataset Not Loading

**Symptoms**:
- Dataset returns 0 samples
- Errors when loading data

**Solutions**:
1. Check data path
2. Check data format
3. Check file permissions

```python
# Check data path
import os
print(os.path.exists('./data'))

# Check data format
import json
with open('./data/file.json') as f:
    data = json.load(f)
    print(type(data))
```

#### Corrupted Images

**Symptoms**:
- Some images fail to load
- Errors during training

**Solutions**:
1. Add error handling
2. Skip corrupted images
3. Clean dataset

```python
try:
    image = Image.open(path)
except Exception as e:
    print(f"Skipping corrupted image: {path}")
    continue
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Model Summary

```python
from torchsummary import summary
summary(model, (3, 64, 64))
```

### Check Gradients

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

### Check Activations

```python
def check_activations(model, x):
    activations = {}
    def hook(name):
        def fn(module, input, output):
            activations[name] = output.detach()
        return fn
    
    for name, module in model.named_modules():
        module.register_forward_hook(hook(name))
    
    model(x)
    return activations
```

## Getting Help

### GitHub Issues

If you can't find a solution here, please open an issue on GitHub:

1. Go to [Issues](https://github.com/10xdev4u-alt/artdiffuse-rl/issues)
2. Click "New Issue"
3. Use the bug report template
4. Provide as much detail as possible

### Information to Include

- Operating system
- Python version
- PyTorch version
- GPU model
- Error message
- Steps to reproduce
- Minimal example

### Community

- GitHub Discussions
- Stack Overflow (tag: artdiffuse-rl)
- Reddit (r/MachineLearning)

## FAQ

### Q: How do I resume training from a checkpoint?

```python
checkpoint = torch.load('models/checkpoint_10000.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

### Q: How do I use a different dataset?

Modify the `MidjourneyDataset` class to load your data.

### Q: How do I change the model size?

Adjust the `ch` and `ch_mult` parameters in the config.

### Q: How do I export the model?

```python
torch.onnx.export(model, dummy_input, 'model.onnx')
```

### Q: How do I contribute?

See the [Contributing Guide](CONTRIBUTING.md).
