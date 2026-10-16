"""Training pipeline for ArtDiffuse-RL.

Supports DDPM training with mixed precision, gradient accumulation,
cosine LR schedule, and EMA weight averaging.
"""

import os
import math
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm

from model import UNet, DDPM


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow


def cosine_lr(step: int, total_steps: int, warmup: int = 1000, base_lr: float = 1e-4) -> float:
    """Cosine learning rate with warmup."""
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(
    model: DDPM,
    dataloader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: str,
):
    """Main training loop."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # EMA
    ema = EMA(model, decay=config.get("ema_decay", 0.9999))

    # Mixed precision
    scaler = GradScaler()
    amp = config.get("amp", True)

    # Training params
    epochs = config.get("epochs", 100)
    grad_accum = config.get("grad_accum", 1)
    save_every = config.get("save_every", 5000)
    log_every = config.get("log_every", 100)
    max_steps = config.get("max_steps", None)

    global_step = 0
    total_steps = epochs * len(dataloader) if max_steps is None else max_steps

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            if "caption" in batch:
                captions = batch["caption"]
            else:
                captions = None

            # Update LR
            lr_now = cosine_lr(global_step, total_steps, base_lr=lr)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            # Forward with AMP
            with autocast(enabled=amp):
                loss = model.compute_loss(images)
                loss = loss / grad_accum

            # Backward
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                global_step += 1

            # Logging
            if global_step % log_every == 0:
                pbar.set_postfix(loss=loss.item() * grad_accum, lr=f'{lr_now:.2e}', step=global_step)

            # Save checkpoint
            if global_step % save_every == 0:
                ckpt_path = output_dir / f"checkpoint_{global_step}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "config": config,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

            if max_steps and global_step >= max_steps:
                break

        if max_steps and global_step >= max_steps:
            break

    # Final save
    final_path = output_dir / "final.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": global_step,
            "config": config,
        },
        final_path,
    )
    print(f"Training complete. Final model saved to {final_path}")
    return model


def load_config(config_path: str) -> dict:
    """Load training config from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> DDPM:
    """Build model from config."""
    model_cfg = config.get("model", {})
    unet = UNet(
        in_ch=model_cfg.get("in_ch", 3),
        out_ch=model_cfg.get("out_ch", 3),
        ch=model_cfg.get("ch", 128),
        ch_mult=tuple(model_cfg.get("ch_mult", [1, 2, 4, 4])),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
        dropout=model_cfg.get("dropout", 0.1),
    )
    ddpm = DDPM(
        unet,
        timesteps=model_cfg.get("timesteps", 1000),
        beta_start=model_cfg.get("beta_start", 1e-4),
        beta_end=model_cfg.get("beta_end", 2e-2),
    )
    return ddpm
