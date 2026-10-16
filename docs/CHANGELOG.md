# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-17

### Added
- Initial release of ArtDiffuse-RL
- U-Net denoising network with 90.8M parameters
- DDPM training pipeline with mixed precision
- RL fine-tuning with DPO
- Midjourney dataset loader
- Kaggle training notebook
- Comprehensive documentation
- Unit tests for model architecture
- CI/CD pipeline with GitHub Actions

### Features
- Train small diffusion model from scratch
- RL fine-tuning for quality improvement
- Evaluation metrics (FID, CLIP score, aesthetic score)
- Local and Kaggle training support
- Comprehensive documentation

### Technical Details
- Model size: 346.5 MB
- Training time: ~8-12 hours on 2x T4
- Inference time: ~2-5s per image on CPU
- Target FID: <20 on validation set

## [Unreleased]

### Planned
- [ ] ONNX export
- [ ] Web demo
- [ ] More evaluation metrics
- [ ] Hyperparameter optimization
- [ ] Distributed training support
