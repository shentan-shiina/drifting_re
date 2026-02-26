
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
from drifting.models.core import RMSNorm, Attention, SwiGLU, modulate

class DiTBlock(nn.Module):
    """
    DiT Block with adaLN-Zero conditioning.

    Regresses 6 modulation parameters from conditioning:
    - shift_msa, scale_msa, gate_msa (for attention)
    - shift_mlp, scale_mlp, gate_mlp (for MLP)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_qk_norm: bool = True,
    ):
        super().__init__()

        # Pre-norm with RMSNorm (no learned affine - will be modulated)
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, use_qk_norm=use_qk_norm)

        self.norm2 = RMSNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        # Use SwiGLU as per paper
        self.mlp = SwiGLU(dim, mlp_hidden, dim)

        # adaLN-Zero modulation: 6 parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get modulation parameters
        modulation = self.adaLN_modulation(c).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation

        # Self-attention with adaLN + gating
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            rope_cos,
            rope_sin,
        )

        # MLP with adaLN + gating
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class FinalLayer(nn.Module):
    """Final layer with adaLN modulation and linear projection."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels)

        # adaLN modulation: 2 parameters (shift, scale)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x
