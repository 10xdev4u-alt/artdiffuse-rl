# RL Fine-tuning Guide

## Overview

RL fine-tuning uses Direct Preference Optimization (DPO) to improve the quality of the diffusion model by learning from preference pairs.

## How It Works

1. **Generate Pairs**: Generate multiple images from the same prompt
2. **Rank by Reward**: Score each image using a reward model (aesthetic classifier)
3. **Create Preferences**: Select the best and worst images as chosen/rejected pairs
4. **Optimize**: Train the model to prefer chosen images over rejected ones

## DPO Loss

The DPO loss is:

```
L_DPO = -log(sigmoid(beta * (log_p(chosen) - log_p(rejected))))
```

Where:
- `log_p(chosen)`: Log-likelihood of the chosen image under the model
- `log_p(rejected)`: Log-likelihood of the rejected image under the model
- `beta`: Temperature parameter (default: 0.1)

## Reward Model

The reward model is a simple aesthetic classifier that scores images based on:
- Color harmony
- Composition
- Visual appeal

## Training Configuration

```yaml
rl:
  rl_epochs: 10
  rl_lr: 0.00001
  dpo_beta: 0.1
  rl_save_every: 1000
```

## Usage

### Fine-tune on Kaggle

1. Train the base model first
2. Load the base model checkpoint
3. Run RL fine-tuning
4. Save the fine-tuned model

### Fine-tune Locally

```python
from src.rl import rl_finetune
from src.train import build_model, load_config

config = load_config('configs/base.yaml')
model = build_model(config)
# Load base model weights
model.load_state_dict(torch.load('models/base.pt'))
# Fine-tune
rl_finetune(model, dataloader, config['rl'], device, 'models/rl')
```

## Expected Results

- **Quality improvement**: 10-20% better FID
- **Aesthetic score**: 15-25% improvement
- **Training time**: ~2-4 hours on 2x T4

## Tips

1. **Start with a good base model**: RL fine-tuning works best with a well-trained base model
2. **Use a good reward model**: The quality of the reward model directly affects the results
3. **Monitor overfitting**: RL fine-tuning can overfit to the reward model
4. **Use a small learning rate**: RL fine-tuning is sensitive to the learning rate
