"""Tests for evaluation metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import pytest
from evaluate import compute_fid, compute_aesthetic_score, save_samples


def test_compute_fid():
    """Test FID computation."""
    real = torch.randn(10, 3, 64, 64)
    fake = torch.randn(10, 3, 64, 64)
    fid = compute_fid(real, fake)
    assert isinstance(fid, float)
    print(f"FID test passed: {fid:.4f}")


def test_compute_aesthetic_score():
    """Test aesthetic score computation."""
    images = torch.randn(10, 3, 64, 64)
    score = compute_aesthetic_score(images)
    assert isinstance(score, float)
    print(f"Aesthetic score test passed: {score:.4f}")


def test_save_samples():
    """Test sample saving."""
    import tempfile
    images = torch.randn(4, 3, 64, 64)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_samples(images, tmpdir, n=4)
        # Check that either samples.png or individual samples exist
        has_samples = (Path(tmpdir) / "samples.png").exists() or (Path(tmpdir) / "sample_0.png").exists()
        assert has_samples
    print("Save samples test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
