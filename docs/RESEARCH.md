# Research Notes

## Overview

This document tracks research findings and design decisions for ArtDiffuse-RL.

## Key Papers

### Diffusion Models
- **DDPM** (Ho et al., 2020): Denoising Diffusion Probabilistic Models
- **DDIM** (Song et al., 2020): Denoising Diffusion Implicit Models
- **Stable DiffModel** (Rombach et al., 2022): High-Resolution Image Synthesis with Latent Diffusion Models

### RL Fine-tuning
- **DPO** (Rafailov et al., 2023): Direct Preference Optimization
- **PPO** (Schulman et al., 2017): Proximal Policy Optimization
- **RLHF** (Christiano et al., 2017): Deep Reinforcement Learning from Human Preferences

### Text-to-Image
- **DALL-E** (Ramesh et al., 2021): Zero-Shot Text-to-Image Generation
- **Imagen** (Saharia et al., 2022): Photorealistic Text-to-Image Diffusion Models
- **Midjourney** (2022): AI Art Generation

## Design Decisions

### Why U-Net?

U-Net is the standard architecture for diffusion models because:
- Skip connections preserve spatial details
- Multi-scale processing captures both global and local features
- Proven track record in image generation

### Why DDPM?

DDPM is chosen because:
- Simple and stable training
- Good sample quality
- Well-understood theory

### Why DPO for RL Fine-tuning?

DPO is chosen because:
- Simpler than PPO
- More stable training
- No need for a separate reward model
- Direct optimization of preferences

### Why Small Model?

A small model (~90M params) is chosen because:
- Faster training
- Easier to deploy
- More efficient inference
- Accessible to more users

## Future Research Directions

### Architecture Improvements
- [ ] Attention mechanisms
- [ ] Efficient U-Net variants
- [ ] Latent diffusion

### Training Improvements
- [ ] Curriculum learning
- [ ] Data augmentation
- [ ] Multi-task learning

### RL Fine-tuning Improvements
- [ ] Better reward models
- [ ] Human-in-the-loop training
- [ ] Multi-objective optimization

### Evaluation Improvements
- [ ] Better metrics
- [ ] Human evaluation
- [ ] A/B testing

## Open Questions

1. **How to improve FID without increasing model size?**
   - Better training data
   - Better architecture
   - Better training techniques

2. **How to improve text-image alignment?**
   - Better text encoder
   - Better conditioning
   - Better training data

3. **How to make RL fine-tuning more stable?**
   - Better reward model
   - Better optimization
   - Better regularization

4. **How to reduce training time?**
   - Distributed training
   - Mixed precision
   - Gradient accumulation

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.
2. Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. ICLR.
3. Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. CVPR.
4. Rafailov, R., et al. (2023). Direct preference optimization. NeurIPS.
5. Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv.
