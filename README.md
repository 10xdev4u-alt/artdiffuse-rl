# ArtDiffuse-RL

A small latent diffusion model trained from scratch with RL fine-tuning for text-to-image generation.

## Overview

ArtDiffuse-RL is a ~100M parameter latent diffusion model that competes with models 10x its size through:
- Training from scratch on curated art datasets
- RL fine-tuning (DPO/PPO) for quality and text-image alignment
- Modern architecture: efficient U-Net, latent diffusion, mixed precision

## Project Structure

```
src/           # Source code
  model.py     # Diffusion model architecture
  train.py     # Training pipeline
  rl.py        # RL fine-tuning (DPO/PPO)
  evaluate.py  # Evaluation metrics
configs/       # Training configs
scripts/       # Utility scripts
notebooks/     # Kaggle notebooks
data/          # Dataset (gitignored)
models/        # Model checkpoints (gitignored)
results/       # Evaluation results (gitignored)
tests/         # Unit tests
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.train --config configs/base.yaml
```

## License

MIT
