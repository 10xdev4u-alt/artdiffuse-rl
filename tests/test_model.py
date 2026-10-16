"""Tests for the diffusion model architecture."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
from model import UNet, DDPM, SinusoidalTimeEmbedding, ResBlock, AttentionBlock


def test_sinusoidal_time_embedding():
    """Test time embedding produces correct shape."""
    emb = SinusoidalTimeEmbedding(dim=128)
    t = torch.tensor([0, 100, 500, 999])
    out = emb(t)
    assert out.shape == (4, 128)
    assert not torch.isnan(out).any()


def test_res_block():
    """Test residual block with time conditioning."""
    block = ResBlock(in_ch=64, out_ch=128, time_dim=512)
    x = torch.randn(2, 64, 32, 32)
    t_emb = torch.randn(2, 512)
    out = block(x, t_emb)
    assert out.shape == (2, 128, 32, 32)


def test_attention_block():
    """Test attention block."""
    attn = AttentionBlock(ch=128)
    x = torch.randn(2, 128, 16, 16)
    out = attn(x)
    assert out.shape == (2, 128, 16, 16)


def test_unet_forward():
    """Test U-Net forward pass."""
    model = UNet(ch=64, ch_mult=(1, 2, 4), num_res_blocks=1)
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    assert out.shape == (2, 3, 64, 64)
    assert not torch.isnan(out).any()


def test_ddpm_q_sample():
    """Test forward diffusion process."""
    unet = UNet(ch=64, ch_mult=(1, 2), num_res_blocks=1)
    ddpm = DDPM(unet, timesteps=100)
    x0 = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0, 50])
    noise = torch.randn_like(x0)
    xt = ddpm.q_sample(x0, t, noise)
    assert xt.shape == x0.shape


def test_ddpm_compute_loss():
    """Test DDPM loss computation."""
    unet = UNet(ch=64, ch_mult=(1, 2), num_res_blocks=1)
    ddpm = DDPM(unet, timesteps=100)
    x0 = torch.randn(4, 3, 32, 32)
    loss = ddpm.compute_loss(x0)
    assert loss.item() > 0
    assert not torch.isnan(loss).any()


def test_model_parameter_count():
    """Test model has reasonable parameter count."""
    model = UNet(ch=128, ch_mult=(1, 2, 4, 4), num_res_blocks=2)
    total_params = sum(p.numel() for p in model.parameters())
    # Should be around 50-100M params
    assert 10_000_000 < total_params < 200_000_000
    print(f"Model parameters: {total_params:,}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
