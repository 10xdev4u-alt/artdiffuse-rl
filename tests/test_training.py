"""Tests for training pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
from model import UNet, DDPM
from train import EMA, cosine_lr


def test_ema():
    """Test EMA weight averaging."""
    model = UNet(ch=32, ch_mult=(1, 2), num_res_blocks=1)
    ema = EMA(model, decay=0.99)
    
    # Update EMA
    ema.update(model)
    
    # Check shadow exists
    assert len(ema.shadow) > 0
    print("EMA test passed")


def test_cosine_lr():
    """Test cosine learning rate schedule."""
    # Test warmup
    lr = cosine_lr(0, 1000, warmup=100, base_lr=1e-4)
    assert lr == 0
    
    # Test peak
    lr = cosine_lr(100, 1000, warmup=100, base_lr=1e-4)
    assert abs(lr - 1e-4) < 1e-6
    
    # Test decay
    lr = cosine_lr(1000, 1000, warmup=100, base_lr=1e-4)
    assert lr < 1e-4
    
    print("Cosine LR test passed")


def test_training_step():
    """Test single training step."""
    model = UNet(ch=32, ch_mult=(1, 2), num_res_blocks=1)
    ddpm = DDPM(model, timesteps=100)
    
    # Create dummy data
    x = torch.randn(4, 3, 32, 32)
    
    # Forward pass
    loss = ddpm.compute_loss(x)
    assert loss.item() > 0
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for param in model.parameters():
        if param.grad is not None:
            assert param.grad.norm() > 0
            break
    
    print("Training step test passed")


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    model = UNet(ch=32, ch_mult=(1, 2), num_res_blocks=1)
    ddpm = DDPM(model, timesteps=100)
    
    # Save checkpoint
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save({'model': ddpm.state_dict(), 'step': 100}, f.name)
        checkpoint_path = f.name
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint['step'] == 100
    
    # Clean up
    import os
    os.unlink(checkpoint_path)
    
    print("Checkpoint test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
