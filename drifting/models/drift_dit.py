"""
DiT-style generator for Drifting Models.
Adapted for 32x32 images (MNIST, CIFAR-10) with adaLN-Zero conditioning.

Key differences from standard DiT:
- No timestep input (one-step generator, not diffusion)
- Conditioning = class_embed + alpha_embed + style_embed
- Uses register tokens, RoPE, SwiGLU, RMSNorm, QK-Norm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
from drifting.models.core import RotaryPositionEmbedding
from drifting.models.dit import DiTBlock, FinalLayer

class PatchEmbed(nn.Module):
    """Convert image patches to embeddings using Conv2d."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class LabelEmbedder(nn.Module):
    """Embed class labels with null class for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        # +1 for null/unconditional class
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Randomly drop labels to null class for CFG training."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool()
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.dropout_prob > 0 and train:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class AlphaEmbedder(nn.Module):
    """Embed CFG alpha scale using Fourier features."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def fourier_features(alpha: torch.Tensor, dim: int, max_period: float = 10.0) -> torch.Tensor:
        """Create sinusoidal embeddings for alpha."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=alpha.device) / half
        )
        args = alpha[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        fourier = self.fourier_features(alpha, self.frequency_embedding_size)
        return self.mlp(fourier)


class StyleEmbedder(nn.Module):
    """
    Style embeddings from paper Sec A.2.
    32 random style tokens index into a codebook of 64 learnable embeddings.
    """

    def __init__(self, hidden_size: int, num_tokens: int = 32, codebook_size: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, hidden_size)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random style embeddings."""
        # Random indices for each sample in the batch
        indices = torch.randint(
            0, self.codebook_size, (batch_size, self.num_tokens), device=device
        )
        embeddings = self.codebook(indices)  # (B, num_tokens, D)
        # Sum over tokens
        style = embeddings.sum(dim=1)  # (B, D)
        return style


class DriftDiT(nn.Module):
    """
    DiT-style generator for Drifting Models.

    Input: Gaussian noise epsilon ~ N(0, I), shape (B, C, 32, 32)
    Additional inputs: class label c, CFG scale alpha
    Output: generated image x, same shape as input
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        label_dropout: float = 0.1,
        num_register_tokens: int = 8,
        use_style_embed: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        # Register (in-context) tokens
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(
            torch.randn(1, num_register_tokens, hidden_size) * 0.02
        )

        # RoPE for positional encoding
        head_dim = hidden_size // num_heads
        self.rope = RotaryPositionEmbedding(
            dim=head_dim,
            max_seq_len=self.num_patches + num_register_tokens + 64,
        )

        # Conditioning embeddings
        self.label_embed = LabelEmbedder(num_classes, hidden_size, label_dropout)
        self.alpha_embed = AlphaEmbedder(hidden_size)
        self.use_style_embed = use_style_embed
        if use_style_embed:
            self.style_embed = StyleEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_qk_norm=True,
            )
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with specific strategy for adaLN-Zero."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)

        # Zero-init adaLN modulation layers (critical for training stability)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Zero-init final layer adaLN modulation only
        # NOTE: For drifting models (one-step generator), we keep the final linear
        # layer with small random weights so the model outputs non-zero images initially
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        # Use small initialization for final linear (not zero!)
        nn.init.normal_(self.final_layer.linear.weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens back to image.
        x: (B, N, patch_size^2 * C)
        Returns: (B, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = self.img_size // p

        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        alpha: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input noise, shape (B, C, H, W)
            labels: Class labels, shape (B,)
            alpha: CFG scale, shape (B,)
            force_drop_ids: Force label dropout for specific samples

        Returns:
            Generated images, shape (B, C, H, W)
        """
        B = x.shape[0]
        device = x.device

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add register tokens
        register = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([register, x], dim=1)  # (B, num_reg + N, D)

        # Get RoPE embeddings
        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(x, seq_len)

        # Conditioning
        c = self.label_embed(labels, self.training, force_drop_ids)
        c = c + self.alpha_embed(alpha)
        if self.use_style_embed:
            c = c + self.style_embed(B, device)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c, rope_cos, rope_sin)

        # Remove register tokens
        x = x[:, self.num_register_tokens:, :]

        # Final layer and unpatchify
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        Runs two forward passes: conditional and unconditional.

        Args:
            x: Input noise, shape (B, C, H, W)
            labels: Class labels, shape (B,)
            alpha: CFG scale (scalar)

        Returns:
            Generated images, shape (B, C, H, W)
        """
        B = x.shape[0]
        device = x.device

        # Create alpha tensor
        alpha_tensor = torch.full((B,), alpha, device=device, dtype=x.dtype)

        # Duplicate inputs for conditional and unconditional
        x_combined = torch.cat([x, x], dim=0)
        labels_combined = torch.cat([labels, labels], dim=0)
        alpha_combined = torch.cat([alpha_tensor, alpha_tensor], dim=0)

        # Force unconditional for second half
        force_drop = torch.cat([
            torch.zeros(B, device=device),
            torch.ones(B, device=device),
        ]).bool()

        # Forward pass
        out = self.forward(x_combined, labels_combined, alpha_combined, force_drop)

        # Split and apply CFG
        cond, uncond = out.chunk(2, dim=0)
        return uncond + alpha * (cond - uncond)

def DriftDiT_Tiny(img_size=32, patch_size=4, num_register_tokens=8, in_channels=3, num_classes=10, label_dropout=0.1):
    """DriftDiT-Tiny: depth=6, hidden_dim=256, heads=4 -> ~5M params"""
    return DriftDiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        num_classes=num_classes,
        label_dropout=label_dropout,
        num_register_tokens=num_register_tokens,
    )

def DriftDiT_Small(img_size=32, patch_size=4, num_register_tokens=8, in_channels=3, num_classes=10, label_dropout=0.1):
    """DriftDiT-Small: depth=12, hidden_dim=384, heads=6 -> ~15M params"""
    return DriftDiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=6.0,
        num_classes=num_classes,
        label_dropout=label_dropout,
        num_register_tokens=num_register_tokens,
    )

def DriftDiT_Big(img_size=32, patch_size=4, num_register_tokens=8, in_channels=3, num_classes=10, label_dropout=0.1):
    """DriftDiT-Big: depth=12, hidden_dim=768, heads=12 -> ~40M params"""
    return DriftDiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=6.0,
        num_classes=num_classes,
        label_dropout=label_dropout,
        num_register_tokens=num_register_tokens,
    )

def DriftDiT_Large(img_size=32, patch_size=4, num_register_tokens=16, in_channels=3, num_classes=10, label_dropout=0.1):
    """DriftDiT-Large: depth=24, hidden_dim=1024, heads=16 -> ~80M params"""
    return DriftDiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=6.0,
        num_classes=num_classes,
        label_dropout=label_dropout,
        num_register_tokens=num_register_tokens,
    )

def DriftDiT_XLarge(img_size=32, patch_size=4, num_register_tokens=16, in_channels=3, num_classes=10, label_dropout=0.1):
    """DriftDiT-Small: depth=8, hidden_dim=384, heads=6 -> ~15M params"""
    return DriftDiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=1152,
        depth=24,
        num_heads=16,
        mlp_ratio=6.0,
        num_classes=num_classes,
        label_dropout=label_dropout,
        num_register_tokens=num_register_tokens,
    )

# Model registry
DriftDiT_models = {
    "DriftDiT-Tiny": DriftDiT_Tiny,
    "DriftDiT-Small": DriftDiT_Small,
    "DriftDiT-Big": DriftDiT_Big,
    "DriftDiT-Large": DriftDiT_Large,
    "DriftDiT-XLarge": DriftDiT_XLarge,
}
