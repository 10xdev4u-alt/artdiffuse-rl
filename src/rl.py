"""RL fine-tuning for ArtDiffuse-RL using Direct Preference Optimization (DPO).

Generates image pairs, ranks them with a reward model (aesthetic classifier),
and fine-tunes the diffusion model to prefer higher-quality outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from model import DDPM


class AestheticReward(nn.Module):
    """Simple aesthetic reward model based on CLIP embeddings.

    Uses a pre-trained CLIP model to score images based on aesthetic quality.
    Can be replaced with a learned reward model.
    """

    def __init__(self, clip_model=None):
        super().__init__()
        if clip_model is None:
            # Placeholder: use a simple CNN-based scorer
            self.scorer = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 1),
            )
        else:
            self.scorer = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.scorer(images).squeeze(-1)


def generate_pairs(
    model: DDPM,
    prompts: list,
    device: torch.device,
    n_pairs: int = 100,
    samples_per_prompt: int = 2,
) -> tuple:
    """Generate image pairs for DPO training.

    Returns:
        chosen: Higher-reward images
        rejected: Lower-reward images
    """
    model.eval()
    chosen = []
    rejected = []

    with torch.no_grad():
        for i in range(n_pairs):
            prompt = prompts[i % len(prompts)]
            shape = (samples_per_prompt, 3, 64, 64)
            samples = model.sample(shape, device)

            # Score samples
            reward_model = AestheticReward().to(device)
            scores = reward_model(samples)

            # Select best and worst
            best_idx = scores.argmax()
            worst_idx = scores.argmin()
            chosen.append(samples[best_idx])
            rejected.append(samples[worst_idx])

    return torch.stack(chosen), torch.stack(rejected)


def dpo_loss(
    model: DDPM,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization loss.

    Args:
        model: Diffusion model
        chosen: Preferred images
        rejected: Non-preferred images
        beta: Temperature parameter

    Returns:
        DPO loss
    """
    # Compute log-likelihood of chosen and rejected under the model
    # For diffusion, we use the denoising loss as a proxy
    t_chosen = torch.randint(0, model.timesteps, (chosen.shape[0],), device=chosen.device)
    t_rejected = torch.randint(0, model.timesteps, (rejected.shape[0],), device=rejected.device)

    noise_chosen = torch.randn_like(chosen)
    noise_rejected = torch.randn_like(rejected)

    xt_chosen = model.q_sample(chosen, t_chosen, noise_chosen)
    xt_rejected = model.q_sample(rejected, t_rejected, noise_rejected)

    eps_chosen = model.model(xt_chosen, t_chosen)
    eps_rejected = model.model(xt_rejected, t_rejected)

    ll_chosen = -F.mse_loss(eps_chosen, noise_chosen, reduction="none").mean(dim=(1, 2, 3))
    ll_rejected = -F.mse_loss(eps_rejected, noise_rejected, reduction="none").mean(dim=(1, 2, 3))

    # DPO loss
    loss = -F.logsigmoid(beta * (ll_chosen - ll_rejected)).mean()
    return loss


def rl_finetune(
    model: DDPM,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: str,
):
    """RL fine-tuning loop using DPO."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lr = config.get("rl_lr", 1e-5)
    beta = config.get("dpo_beta", 0.1)
    epochs = config.get("rl_epochs", 10)
    save_every = config.get("rl_save_every", 1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"RL Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch["image"].to(device)

            # Generate pairs
            chosen, rejected = generate_pairs(
                model, ["art"] * images.shape[0], device, n_pairs=images.shape[0]
            )

            # DPO loss
            loss = dpo_loss(model, chosen, rejected, beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            pbar.set_postfix(loss=loss.item(), step=global_step)

            if global_step % save_every == 0:
                ckpt_path = output_dir / f"rl_checkpoint_{global_step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                    },
                    ckpt_path,
                )
                print(f"Saved RL checkpoint to {ckpt_path}")

    final_path = output_dir / "rl_final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": global_step,
        },
        final_path,
    )
    print(f"RL fine-tuning complete. Final model saved to {final_path}")
    return model
