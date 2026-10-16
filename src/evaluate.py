"""Evaluation metrics for ArtDiffuse-RL.

Computes FID, CLIP score, and aesthetic score for generated images.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """Compute Fréchet Inception Distance.

    Args:
        real_images: Real images tensor (N, C, H, W)
        fake_images: Generated images tensor (N, C, H, W)

    Returns:
        FID score (lower is better)
    """
    # Simplified FID using feature statistics
    # In practice, use a pre-trained Inception network
    real_mean = real_images.mean(dim=(0, 2, 3))
    fake_mean = fake_images.mean(dim=(0, 2, 3))
    real_std = real_images.std(dim=(0, 2, 3))
    fake_std = fake_images.std(dim=(0, 2, 3))

    fid = ((real_mean - fake_mean) ** 2).sum() + ((real_std - fake_std) ** 2).sum()
    return fid.item()


def compute_clip_score(images: torch.Tensor, captions: list) -> float:
    """Compute CLIP score for text-image alignment.

    Args:
        images: Generated images tensor (N, C, H, W)
        captions: List of text captions

    Returns:
        CLIP score (higher is better)
    """
    # Placeholder: use a simple heuristic
    # In practice, use a pre-trained CLIP model
    return 0.0


def compute_aesthetic_score(images: torch.Tensor) -> float:
    """Compute aesthetic score.

    Args:
        images: Generated images tensor (N, C, H, W)

    Returns:
        Aesthetic score (higher is better)
    """
    # Placeholder: use color variance as a proxy
    # In practice, use a learned aesthetic classifier
    return images.std().item()


def save_samples(images: torch.Tensor, output_dir: str, n: int = 16):
    """Save generated samples as a grid."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from torchvision.utils import save_image
    save_image(images[:n], output_dir / "samples.png", nrow=4, normalize=True)
    print(f"Saved samples to {output_dir / 'samples.png'}")
