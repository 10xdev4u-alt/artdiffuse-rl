# Frequently Asked Questions

## General

### What is ArtDiffuse-RL?

ArtDiffuse-RL is a small latent diffusion model (~90M parameters) trained from scratch with RL fine-tuning for text-to-image generation.

### Why small model?

Small models are:
- Faster to train
- Easier to deploy
- More efficient for inference
- Accessible to more users

### How does it compare to Stable Diffusion?

ArtDiffuse-RL is ~10x smaller than Stable Diffusion but aims to be competitive through:
- Training from scratch on curated data
- RL fine-tuning for quality improvement
- Modern architecture design

## Training

### How long does training take?

- **Base training**: ~8-12 hours on 2x T4
- **RL fine-tuning**: ~2-4 hours on 2x T4
- **Total**: ~10-16 hours

### What dataset is used?

The Midjourney text-to-image dataset from Kaggle, containing ~3400 high-quality images with prompts.

### Can I train on my own data?

Yes! Modify the `MidjourneyDataset` class to load your own data.

## RL Fine-tuning

### What is DPO?

Direct Preference Optimization (DPO) is a method for fine-tuning models using preference pairs instead of explicit rewards.

### Why DPO instead of PPO?

DPO is:
- Simpler to implement
- More stable to train
- Less computationally expensive
- Easier to tune

### What is the reward model?

A simple aesthetic classifier that scores images based on visual appeal.

## Evaluation

### What is FID?

Fréchet Inception Distance measures the distance between the distribution of generated images and real images. Lower is better.

### What is a good FID score?

- **<20**: Excellent
- **20-30**: Good
- **>30**: Needs improvement

### How do I evaluate my model?

```bash
./scripts/evaluate.sh models/final.pt results
```

## Deployment

### Can I run this on CPU?

Yes! The model is small enough to run on CPU, though it will be slower (~2-5s per image).

### Can I export to ONNX?

Yes, use the `torch.onnx.export` function.

### Can I use this in a web app?

Yes, wrap the model in a Flask or FastAPI server.

## Contributing

### How do I contribute?

See the [Contributing Guide](CONTRIBUTING.md).

### How do I report a bug?

Use the bug report template on GitHub.

### How do I request a feature?

Use the feature request template on GitHub.

## License

### What is the license?

MIT License - free for personal and commercial use.

### Can I use this commercially?

Yes, the MIT License allows commercial use.
