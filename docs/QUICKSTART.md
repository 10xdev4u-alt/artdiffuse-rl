# Quick Start

Get started with ArtDiffuse-RL in 5 minutes!

## 1. Clone the Repository

```bash
git clone https://github.com/10xdev4u-alt/artdiffuse-rl.git
cd artdiffuse-rl
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Download Data

```bash
./scripts/download_data.sh ./data
```

## 4. Train the Model

```bash
./scripts/train.sh configs/base.yaml models
```

## 5. Generate Images

```python
import torch
from src.model import UNet, DDPM

# Load model
model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
ddpm = DDPM(model, timesteps=1000)
ddpm.load_state_dict(torch.load('models/final.pt'))

# Generate images
samples = ddpm.sample((16, 3, 64, 64), torch.device('cuda'))
```

## Kaggle Quick Start

1. Go to [Kaggle](https://www.kaggle.com)
2. Create a new notebook
3. Upload `notebooks/train_kaggle.ipynb`
4. Add the Midjourney dataset
5. Enable GPU and run!

## Next Steps

- Read the [Training Guide](TRAINING.md)
- Read the [Evaluation Guide](EVALUATION.md)
- Read the [RL Fine-tuning Guide](RL_FINETUNING.md)
- Check the [FAQ](FAQ.md)

## Need Help?

- Open an issue on GitHub
- Check the [Contributing Guide](CONTRIBUTING.md)
