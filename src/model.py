"""Diffusion model architecture for ArtDiffuse-RL.

A compact U-Net based denoising network for DDPM training on 64x64 images.
Designed to be efficient on 2x T4 GPUs while maintaining competitive quality.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for capturing global dependencies."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c), dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    """U-Net denoising network for DDPM.

    Args:
        in_ch: Input channels (3 for RGB)
        out_ch: Output channels (3 for RGB)
        ch: Base channel width
        ch_mult: Channel multipliers at each resolution level
        num_res_blocks: Number of residual blocks per level
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        ch: int = 128,
        ch_mult: tuple = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_dim = ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(ch),
            nn.Linear(ch, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        curr_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(curr_ch, out_ch, self.time_dim, dropout))
                curr_ch = out_ch
            if i < len(ch_mult) - 1:
                self.down_samples.append(
                    nn.Conv2d(curr_ch, curr_ch, 3, stride=2, padding=1)
                )

        # Middle blocks
        self.mid_block1 = ResBlock(curr_ch, curr_ch, self.time_dim, dropout)
        self.mid_attn = AttentionBlock(curr_ch)
        self.mid_block2 = ResBlock(curr_ch, curr_ch, self.time_dim, dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(curr_ch + out_ch, out_ch, self.time_dim, dropout)
                )
                curr_ch = out_ch
            if i > 0:
                self.up_samples.append(nn.Conv2d(curr_ch, curr_ch, 3, padding=1))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, curr_ch),
            nn.SiLU(),
            nn.Conv2d(curr_ch, out_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        h = self.conv_in(x)

        # Downsampling
        skips = []
        sample_idx = 0
        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            skips.append(h)
            if (i + 1) % 2 == 0 and sample_idx < len(self.down_samples):
                h = self.down_samples[sample_idx](h)
                sample_idx += 1

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        up_idx = 0
        for i, block in enumerate(self.up_blocks):
            h = torch.cat([h, skips.pop()], dim=1)
            h = block(h, t_emb)
            if (i + 1) % 3 == 0 and up_idx < len(self.up_samples):
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = self.up_samples[up_idx](h)
                up_idx += 1

        return self.conv_out(h)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model.

    Wraps the U-Net with the diffusion process (noise schedule, sampling).
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_omac * noise

    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Single denoising step."""
        eps = self.model(x, t)
        sqrt_recip_ac = self.sqrt_recip_alphas_cumprod[t][:, None, None, None]
        sqrt_recipm1_ac = self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None]
        mean = sqrt_recip_ac * (x - sqrt_recipm1_ac * eps)
        var = self.posterior_variance[t][:, None, None, None]
        noise = torch.randn_like(x)
        nonzero = (t != 0)[:, None, None, None].float()
        return mean + nonzero * torch.sqrt(var) * noise

    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """Generate samples via reverse diffusion."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
        return x

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """MSE denoising loss."""
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        eps_pred = self.model(xt, t)
        return F.mse_loss(eps_pred, noise)
