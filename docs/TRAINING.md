# Training Guide

## Quick Start

### 1. Download Data

```bash
./scripts/download_data.sh ./data
```

### 2. Train Model

```bash
./scripts/train.sh configs/base.yaml models
```

### 3. Evaluate Model

```bash
./scripts/evaluate.sh models/final.pt results
```

## Kaggle Training

The project includes a self-contained Kaggle notebook for training on 2x T4 GPUs.

### Steps

1. Go to [Kaggle](https://www.kaggle.com)
2. Create a new notebook
3. Upload `notebooks/train_kaggle.ipynb`
4. Add the Midjourney dataset: `succinctlyai/midjourney-texttoimage`
5. Enable GPU (2x T4)
6. Run the notebook

### Training Configuration

- **Batch size**: 256
- **Max steps**: 30,000
- **Learning rate**: 1e-4
- **EMA decay**: 0.9999
- **Mixed precision**: Enabled

### Expected Results

- **Training time**: ~8-12 hours
- **Final FID**: <30 (target <20)
- **Model size**: 346.5 MB

## Local Training

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (optional)

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python -m src.train --config configs/base.yaml --output models
```

## Monitoring

### Weights & Biases

Add W&B tracking to monitor training:

```python
import wandb
wandb.init(project="artdiffuse-rl")
```

### TensorBoard

```bash
tensorboard --logdir models/
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Enable gradient accumulation
- Use mixed precision

### Slow Training

- Enable mixed precision
- Increase batch size
- Use more GPUs

### Poor Quality

- Train longer
- Adjust learning rate
- Use larger model
