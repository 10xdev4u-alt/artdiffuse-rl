# ArtDiffuse-RL Architecture

## Overview

ArtDiffuse-RL is a small latent diffusion model (~90M parameters) trained from scratch with RL fine-tuning for text-to-image generation.

## Model Architecture

### U-Net Denoising Network

The core of the model is a U-Net with the following components:

- **Input**: 64x64 RGB images
- **Base channels**: 128
- **Channel multipliers**: (1, 2, 4, 4)
- **Residual blocks per level**: 2
- **Attention**: Self-attention at the bottleneck
- **Time embedding**: Sinusoidal embedding with MLP

### DDPM (Denoising Diffusion Probabilistic Model)

- **Timesteps**: 1000
- **Beta schedule**: Linear (1e-4 to 2e-2)
- **Training objective**: MSE denoising loss
- **Sampling**: Reverse diffusion process

## Training Pipeline

### Phase 1: Base Training
- Train the diffusion model from scratch
- Mixed precision (AMP) training
- EMA weight averaging
- Cosine learning rate schedule
- Batch size: 256
- Max steps: 30,000

### Phase 2: RL Fine-tuning
- Direct Preference Optimization (DPO)
- Reward model: Aesthetic classifier
- Preference pairs: Generated images ranked by reward
- Fine-tuning steps: 10,000

## Evaluation Metrics

- **FID** (Fréchet Inception Distance): Measures image quality
- **CLIP Score**: Measures text-image alignment
- **Aesthetic Score**: Measures visual appeal

## Project Structure

```
src/
  model.py      # U-Net + DDPM architecture
  train.py      # Training pipeline
  rl.py         # RL fine-tuning (DPO)
  dataset.py    # Dataset loader
  evaluate.py   # Evaluation metrics
configs/
  base.yaml     # Training configuration
scripts/
  download_data.sh
  train.sh
  evaluate.sh
notebooks/
  train_kaggle.ipynb  # Kaggle training notebook
tests/
  test_model.py       # Unit tests
```

## Key Design Decisions

1. **Small model size**: ~90M parameters for efficiency
2. **Training from scratch**: No pre-trained weights
3. **RL fine-tuning**: DPO for quality improvement
4. **Mixed precision**: Faster training on T4 GPUs
5. **EMA**: Better generalization

## Performance

- **Model size**: 346.5 MB
- **Training time**: ~8-12 hours on 2x T4
- **Inference time**: ~2-5s per image on CPU
- **Target FID**: <20 on validation set
