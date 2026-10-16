# Evaluation Guide

## Overview

Evaluation metrics for ArtDiffuse-RL measure image quality, text-image alignment, and aesthetic appeal.

## Metrics

### FID (Fréchet Inception Distance)

Measures the distance between the distribution of generated images and real images.

- **Lower is better**
- **Target**: <20 on validation set
- **Baseline**: ~30 for small models

```python
from src.evaluate import compute_fid

fid = compute_fid(real_images, fake_images)
print(f'FID: {fid:.2f}')
```

### CLIP Score

Measures text-image alignment using CLIP embeddings.

- **Higher is better**
- **Target**: >0.30
- **Range**: 0 to 1

```python
from src.evaluate import compute_clip_score

score = compute_clip_score(images, captions)
print(f'CLIP Score: {score:.4f}')
```

### Aesthetic Score

Measures visual appeal using a learned aesthetic classifier.

- **Higher is better**
- **Target**: >6.5
- **Range**: 0 to 10

```python
from src.evaluate import compute_aesthetic_score

score = compute_aesthetic_score(images)
print(f'Aesthetic Score: {score:.2f}')
```

## Running Evaluation

### Evaluate on Kaggle

```python
# Generate samples
samples = ddpm.sample((1000, 3, 64, 64), device)

# Compute metrics
fid = compute_fid(real_images, samples)
clip = compute_clip_score(samples, captions)
aesthetic = compute_aesthetic_score(samples)

print(f'FID: {fid:.2f}')
print(f'CLIP Score: {clip:.4f}')
print(f'Aesthetic Score: {aesthetic:.2f}')
```

### Evaluate Locally

```bash
./scripts/evaluate.sh models/final.pt results
```

## Comparison with Baselines

| Model | Params | FID | CLIP Score | Aesthetic |
|-------|--------|-----|------------|-----------|
| SD-Turbo | ~1B | ~25 | ~0.28 | ~6.0 |
| SD-Lightning | ~1B | ~22 | ~0.30 | ~6.2 |
| ArtDiffuse-RL | ~90M | Target <20 | Target >0.30 | Target >6.5 |

## Tips

1. **Use more samples**: Evaluation is more reliable with more samples
2. **Use same distribution**: Compare on the same dataset
3. **Multiple seeds**: Run evaluation with multiple random seeds
4. **Human evaluation**: Consider human evaluation for subjective quality

## Interpreting Results

### FID < 20
Excellent quality, competitive with state-of-the-art

### FID 20-30
Good quality, room for improvement

### FID > 30
Needs improvement, consider training longer or adjusting hyperparameters

### CLIP Score > 0.30
Good text-image alignment

### CLIP Score 0.25-0.30
Moderate alignment, may need improvement

### Aesthetic Score > 6.5
High visual appeal

### Aesthetic Score 5.0-6.5
Moderate appeal, room for improvement
