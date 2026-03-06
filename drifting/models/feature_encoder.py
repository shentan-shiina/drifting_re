"""
Simple CNN feature encoder for drifting loss.
Multi-scale ResNet-style architecture for MNIST and CIFAR-10.

Phase 1: Can skip feature encoder and compute drifting loss in pixel space
Phase 2: Use this encoder trained with MAE objective for better results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Sequence, Union
import torchvision.models as models
from termcolor import cprint

class PretrainedResNetEncoder(nn.Module):
    """
    Feature encoder using pretrained ResNet.
    Returns multi-scale feature MAPS (not pooled vectors) for per-location loss.

    Following paper Section A.5: compute drifting loss at each scale/location.
    """

    def __init__(
        self,
        pretrained: bool = False,
        ssl_checkpoint_path: Optional[str] = None,
        arch: str = "resnet50",
    ):
        super().__init__()

        if arch == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Unsupported ResNet arch: {arch}")

        if ssl_checkpoint_path is not None:
            ckpt = torch.load(ssl_checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            cleaned = {}
            for k, v in state_dict.items():
                key = k
                for prefix in ("module.", "backbone.", "encoder.", "model."):
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                cleaned[key] = v
            resnet.load_state_dict(cleaned, strict=False)

        # Extract layers (don't include final fc)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale feature maps.

        Returns:
            List of feature maps at different scales, each (B, C, H, W)
        """
        # Resize if needed (CIFAR is 32x32)
        if x.shape[-1] < 64:
            x = F.interpolate(x, size=64, mode='bilinear', align_corners=False)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)   # (B, 64, H/4, W/4)
        f2 = self.layer2(f1)  # (B, 128, H/8, W/8)
        f3 = self.layer3(f2)  # (B, 256, H/16, W/16)
        f4 = self.layer4(f3)  # (B, 512, H/32, W/32)

        return [f1, f2, f3, f4]


class BasicBlock(nn.Module):
    """Basic residual block with GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)

        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(32, out_channels), out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.gelu(out)
        return out


class MultiScaleFeatureEncoder(nn.Module):
    """
    Multi-scale CNN feature encoder.

    4-stage architecture with progressive downsampling:
    - Stage 1: 32x32 features
    - Stage 2: 16x16 features
    - Stage 3: 8x8 features
    - Stage 4: 4x4 features

    Features from all stages are pooled and concatenated for multi-scale representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_width: int = 64,
        blocks_per_stage: Union[int, Sequence[int]] = 2,
        feature_dim: int = 512,
        multi_scale: bool = True,
        input_patch_size: int = 1,
        output_mode: str = "projected",
    ):
        """
        Args:
            in_channels: Number of input channels (1 for MNIST, 3 for CIFAR)
            base_width: Base number of channels (64 for MNIST, 128 for CIFAR)
            blocks_per_stage: Number of residual blocks per stage (int) or per-stage list [s1,s2,s3,s4]
            feature_dim: Output feature dimension
            multi_scale: Whether to use multi-scale features
            input_patch_size: Optional space-to-depth patchification size before encoder
            output_mode: "projected" or "multiscale"
        """
        super().__init__()
        self.multi_scale = multi_scale
        self.output_mode = output_mode
        self.input_patch_size = int(input_patch_size)

        if isinstance(blocks_per_stage, int):
            stage_depths = [blocks_per_stage] * 4
        else:
            stage_depths = list(blocks_per_stage)
            if len(stage_depths) != 4:
                raise ValueError("blocks_per_stage must be int or a sequence of length 4")

        self.stage_depths = stage_depths

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(32, base_width), base_width),
            nn.GELU(),
        )

        # Stage 1: 32x32 -> 32x32
        self.stage1 = self._make_stage(base_width, base_width, stage_depths[0], stride=1)

        # Stage 2: 32x32 -> 16x16
        self.stage2 = self._make_stage(base_width, base_width * 2, stage_depths[1], stride=2)

        # Stage 3: 16x16 -> 8x8
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, stage_depths[2], stride=2)

        # Stage 4: 8x8 -> 4x4
        self.stage4 = self._make_stage(base_width * 4, base_width * 8, stage_depths[3], stride=2)

        # Feature projection
        if multi_scale:
            total_channels = base_width + base_width * 2 + base_width * 4 + base_width * 8
        else:
            total_channels = base_width * 8

        self.proj = nn.Linear(total_channels, feature_dim)

        cprint(f"[Feature Encoder Module] Initialized with parameters:", "yellow")
        cprint(f"  - in_channels: {in_channels}", "yellow")
        cprint(f"  - base_width: {base_width}", "yellow")
        cprint(f"  - stage_depths: {self.stage_depths}", "yellow")
        cprint(f"  - feature_dim: {feature_dim}", "yellow")
        cprint(f"  - multi_scale: {self.multi_scale}", "yellow")
        cprint(f"  - input_patch_size: {self.input_patch_size}", "yellow")
        cprint(f"  - output_mode: {self.output_mode}", "yellow")

    def _space_to_depth(self, x: torch.Tensor) -> torch.Tensor:
        p = self.input_patch_size
        if p <= 1:
            return x
        B, C, H, W = x.shape
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Input size ({H}, {W}) must be divisible by input_patch_size={p}")
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * p * p, H // p, W // p)
        return x

    def extract_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self._space_to_depth(x)
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

    def project_from_multiscale(self, features: List[torch.Tensor]) -> torch.Tensor:
        f1, f2, f3, f4 = features
        if self.multi_scale:
            p1 = F.adaptive_avg_pool2d(f1, 1).flatten(1)
            p2 = F.adaptive_avg_pool2d(f2, 1).flatten(1)
            p3 = F.adaptive_avg_pool2d(f3, 1).flatten(1)
            p4 = F.adaptive_avg_pool2d(f4, 1).flatten(1)
            pooled = torch.cat([p1, p2, p3, p4], dim=1)
        else:
            pooled = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        return self.proj(pooled)

    def forward_projected(self, x: torch.Tensor) -> torch.Tensor:
        return self.project_from_multiscale(self.extract_multiscale(x))

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a stage with multiple residual blocks."""
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            Features, shape (B, feature_dim)
        """
        features = self.extract_multiscale(x)
        if self.output_mode == "multiscale":
            return features
        return self.project_from_multiscale(features)


class MAEUpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, upsample: bool = True):
        super().__init__()
        self.upsample = upsample
        self.norm = nn.GroupNorm(min(32, in_channels + skip_channels), in_channels + skip_channels)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.norm(x)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.gelu(self.gn2(self.conv2(x)))
        return x


class MAEEncoder(nn.Module):
    """
    Masked Autoencoder encoder for self-supervised pre-training.

    Pre-trains the feature encoder using reconstruction of masked patches.
    """

    def __init__(
        self,
        feature_encoder: MultiScaleFeatureEncoder,
        in_channels: int = 3,
        img_size: int = 32,
        input_patch_size: int = 1,
        mask_block_size: int = 2,
        mask_prob: float = 0.5,
    ):
        """
        Args:
            feature_encoder: The feature encoder to pre-train
            in_channels: Number of input channels
            img_size: Input image size
            input_patch_size: Optional patchify input by space-to-depth before encoder
            mask_block_size: Spatial block size for zero masking (Appendix A.3 uses 2)
            mask_prob: Independent mask probability per block (Appendix A.3 uses 0.5)
        """
        super().__init__()
        self.encoder = feature_encoder
        self.in_channels = in_channels
        self.img_size = img_size
        self.input_patch_size = int(input_patch_size)
        self.mask_block_size = int(mask_block_size)
        self.mask_prob = float(mask_prob)

        base_width = self.encoder.stem[0].out_channels
        self.pre_decode = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(32, base_width * 8), base_width * 8),
            nn.GELU(),
        )
        self.up1 = MAEUpBlock(base_width * 8, base_width * 4, base_width * 4, upsample=True)
        self.up2 = MAEUpBlock(base_width * 4, base_width * 2, base_width * 2, upsample=True)
        self.up3 = MAEUpBlock(base_width * 2, base_width, base_width, upsample=True)
        self.up4 = MAEUpBlock(base_width, 0, base_width, upsample=False)

        decoder_channels = in_channels * (self.input_patch_size ** 2)
        self.out_conv = nn.Conv2d(base_width, decoder_channels, kernel_size=1)

    def _space_to_depth(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        if patch_size <= 1:
            return x
        B, C, H, W = x.shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"Input size ({H}, {W}) must be divisible by patch_size={patch_size}")
        x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * patch_size * patch_size, H // patch_size, W // patch_size)
        return x

    def _depth_to_space(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        if patch_size <= 1:
            return x
        B, C, H, W = x.shape
        if C % (patch_size * patch_size) != 0:
            raise ValueError("Channel dimension is not divisible by patch_size^2 for unpatchify")
        out_c = C // (patch_size * patch_size)
        x = x.reshape(B, out_c, patch_size, patch_size, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, out_c, H * patch_size, W * patch_size)
        return x

    def random_masking(self, x: torch.Tensor) -> tuple:
        """
        Apply random 2x2-style masking by zeroing spatial blocks.

        Returns:
            x_masked: Masked image
            mask: Binary mask at feature-map resolution (1 = masked, 0 = visible)
        """
        B, C, H, W = x.shape
        p = self.mask_block_size
        if H % p != 0 or W % p != 0:
            raise ValueError(f"Feature map size ({H}, {W}) must be divisible by mask_block_size={p}")

        mask_blocks = (torch.rand(B, H // p, W // p, device=x.device) < self.mask_prob).float()
        mask = mask_blocks.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)
        mask = mask[:, :H, :W]
        mask = mask.unsqueeze(1)

        x_masked = x * (1 - mask)
        return x_masked, mask

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for MAE pre-training.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            loss: Reconstruction loss on masked patches
            pred: Reconstructed image
            mask: Binary mask
        """
        x_work = self._space_to_depth(x, self.input_patch_size)
        x_masked, mask = self.random_masking(x_work)

        f1, f2, f3, f4 = self.encoder.extract_multiscale(x_masked)
        z = self.pre_decode(f4)
        z = self.up1(z, f3)
        z = self.up2(z, f2)
        z = self.up3(z, f1)
        z = self.up4(z, None)
        pred_work = self.out_conv(z)

        err = (pred_work - x_work) ** 2
        denom = (mask.sum() * x_work.shape[1]).clamp_min(1.0)
        loss = (err * mask).sum() / denom

        pred = self._depth_to_space(pred_work, self.input_patch_size)

        return loss, pred, mask

def create_feature_encoder(
    dataset: str = "cifar10",
    in_channels: int = 3, # <-- Added to handle latent dimensions!
    feature_dim: int = 512,
    base_width: int = 128,
    blocks_per_stage: Union[int, Sequence[int]] = 2,
    multi_scale: bool = True,
    input_patch_size: int = 1,
    output_mode: str = "multiscale",
    use_pretrained: bool = True,
    mae_checkpoint_path: Optional[str] = None, # <-- Added custom checkpoint path
    ssl_checkpoint_path: Optional[str] = None,
    allow_supervised_fallback: bool = False,
):
    """Create a feature encoder for the specified dataset."""

    if mae_checkpoint_path:
        print(f"Loading custom MAE pre-trained encoder from {mae_checkpoint_path}")
        
        encoder = MultiScaleFeatureEncoder(
            in_channels=in_channels,
            base_width=base_width,
            blocks_per_stage=blocks_per_stage,
            feature_dim=feature_dim,
            multi_scale=multi_scale,
            input_patch_size=input_patch_size,
            output_mode=output_mode,
        )
        
        # Extract purely the encoder weights from the Lightning MAE checkpoint
        ckpt = torch.load(mae_checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        encoder_state = {}
        ema_state = {}
        for k, v in state_dict.items():
            if k.startswith("ema_encoder."):
                ema_state[k.replace("ema_encoder.", "")] = v
            elif k.startswith("encoder."):
                encoder_state[k.replace("encoder.", "")] = v
            elif k.startswith("mae.encoder."):
                encoder_state[k.replace("mae.encoder.", "")] = v

        # Prefer EMA weights when available (paper recipe uses EMA during MAE pretrain)
        if ema_state:
            encoder_state.update(ema_state)
                
        encoder.load_state_dict(encoder_state)
        return encoder

    if dataset.lower() == "mnist":
        return MultiScaleFeatureEncoder(
            in_channels=1, base_width=64, blocks_per_stage=blocks_per_stage,
            feature_dim=feature_dim, multi_scale=multi_scale,
            input_patch_size=input_patch_size,
            output_mode=output_mode,
        )
    elif dataset.lower() in ["cifar10", "cifar", "imagenet", "imagenet-tiny", "tiny-imagenet", "tiny_imagenet"]:
        if use_pretrained:
            if ssl_checkpoint_path is None and not allow_supervised_fallback:
                raise ValueError(
                    "Paper-faithful pretrained feature encoders require SSL checkpoints. "
                    "Set ssl_checkpoint_path or pass allow_supervised_fallback=True."
                )
            return PretrainedResNetEncoder(
                pretrained=allow_supervised_fallback and ssl_checkpoint_path is None,
                ssl_checkpoint_path=ssl_checkpoint_path,
                arch="resnet50",
            )
        else:
            return MultiScaleFeatureEncoder(
                in_channels=in_channels, base_width=128, blocks_per_stage=blocks_per_stage,
                feature_dim=feature_dim, multi_scale=multi_scale,
                input_patch_size=input_patch_size,
                output_mode=output_mode,
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def pretrain_mae(
    feature_encoder: MultiScaleFeatureEncoder,
    train_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
) -> MultiScaleFeatureEncoder:
    """
    Pre-train feature encoder using MAE objective.

    Args:
        feature_encoder: The encoder to pre-train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        Pre-trained feature encoder
    """
    in_channels = feature_encoder.stem[0].in_channels

    mae = MAEEncoder(
        feature_encoder,
        in_channels=in_channels,
        img_size=32,
        input_patch_size=1,
        mask_block_size=2,
        mask_prob=0.5,
    ).to(device)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    mae.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            loss, _, _ = mae(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"MAE Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / num_batches:.4f}")

    # Return the pre-trained encoder (without decoder)
    return mae.encoder
