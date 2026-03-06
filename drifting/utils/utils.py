"""
Utility functions for Drifting Models.
Includes EMA, learning rate scheduling, visualization, and checkpointing.
"""

import copy
import math
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any
from drifting.models.drift_dit import DriftDiT_models
import scipy.linalg

def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Numpy implementation of the Frechet Distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (higher = slower update)
        """
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for ema_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )

    def forward(self, *args, **kwargs):
        """Forward pass through the EMA model."""
        return self.shadow(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        """Return EMA model state dict."""
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA model state dict."""
        self.shadow.load_state_dict(state_dict)

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import LRScheduler

class WarmupLRScheduler(LRScheduler):
    """Linear warmup then constant learning rate scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float,
        last_epoch: int = -1
    ):
        self.warmup_steps = max(1, warmup_steps) # Prevent division by zero
        self.target_lr = base_lr
        # The base class automatically handles state_dict, load_state_dict, 
        # and calling step() via tracking `self.last_epoch`.
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute current learning rate natively."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.target_lr * (step / self.warmup_steps)
            return [lr for _ in self.base_lrs]
            
        # Constant LR after warmup
        return [self.target_lr for _ in self.base_lrs]


def save_checkpoint(
    path: str,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupLRScheduler,
    epoch: int,
    step: int,
    config: Dict[str, Any],
):
    """Save training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    ema: Optional[EMA] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[WarmupLRScheduler] = None,
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Path,
    config: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Load model weights from checkpoint, preferring EMA weights."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = checkpoint.get("config", {})

    model_name = ckpt_cfg.get("model", config["model"])
    img_size = ckpt_cfg.get("img_size", config["img_size"])
    patch_size = ckpt_cfg.get("patch_size", config["patch_size"])
    in_channels = ckpt_cfg.get("in_channels", config["in_channels"])
    num_classes = ckpt_cfg.get("num_classes", config["num_classes"])
    label_dropout = ckpt_cfg.get("label_dropout", config.get("label_dropout", 0.0))
    num_register_tokens = int(ckpt_cfg.get("num_register_tokens", config.get("num_register_tokens", 16)))

    model_fn = DriftDiT_models[model_name]
    model = model_fn(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        label_dropout=label_dropout,
        num_register_tokens=num_register_tokens,

    ).to(device)

    # 1. Try to load EMA weights first (injected by our EMACallback)
    if "ema" in checkpoint:
        print("Loading EMA weights from checkpoint...")
        state = checkpoint["ema"]
        
    # 2. Fall back to standard model weights
    else:
        print("Loading standard model weights from checkpoint (EMA not found)...")
        raw_state_dict = checkpoint["state_dict"]
        state = {}
        for k, v in raw_state_dict.items():
            # Strip the 'model.' prefix that Lightning adds to wrapped modules
            if k.startswith("model."):
                new_key = k[len("model."):] 
            else:
                new_key = k
            state[new_key] = v
    model.load_state_dict(state)
    model.eval()
    return model


def make_image_grid(
    images: torch.Tensor,
    nrow: int = 10,
    padding: int = 2,
    normalize: bool = True,
    value_range: tuple = (-1, 1),
) -> np.ndarray:
    """
    Create a grid of images for visualization.

    Args:
        images: Tensor of shape (N, C, H, W)
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize images to [0, 1]
        value_range: Expected value range of input images

    Returns:
        numpy array of shape (H, W, C) or (H, W) for visualization
    """
    if normalize:
        images = images.clone()
        low, high = value_range
        images = (images - low) / (high - low)
        images = images.clamp(0, 1)

    n, c, h, w = images.shape
    ncol = (n + nrow - 1) // nrow

    # Create grid
    grid_h = ncol * h + (ncol + 1) * padding
    grid_w = nrow * w + (nrow + 1) * padding
    grid = torch.ones(c, grid_h, grid_w)

    idx = 0
    for row in range(ncol):
        for col in range(nrow):
            if idx >= n:
                break
            y = padding + row * (h + padding)
            x = padding + col * (w + padding)
            grid[:, y : y + h, x : x + w] = images[idx]
            idx += 1

    # Convert to numpy (H, W, C) or (H, W) for grayscale
    grid = grid.permute(1, 2, 0).numpy()
    if c == 1:
        grid = grid.squeeze(-1)
    return grid


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 10,
    padding: int = 2,
    normalize: bool = True,
    value_range: tuple = (-1, 1),
):
    """Save a grid of images to file."""
    grid = make_image_grid(
        images,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, grid, cmap="gray" if grid.ndim == 2 else None)


def visualize_samples(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    num_classes: int = 10,
    samples_per_class: int = 8,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Visualize samples organized by class.

    Args:
        images: Tensor of shape (N, C, H, W)
        labels: Tensor of shape (N,) with class labels
        num_classes: Number of classes
        samples_per_class: Number of samples to show per class
        save_path: Path to save the figure
        title: Figure title
    """
    if labels is not None:
        # Organize by class
        organized = []
        for c in range(num_classes):
            mask = labels == c
            class_images = images[mask][:samples_per_class]
            if len(class_images) < samples_per_class:
                # Pad with zeros if not enough samples
                pad = torch.zeros(
                    samples_per_class - len(class_images),
                    *images.shape[1:],
                    device=images.device,
                )
                class_images = torch.cat([class_images, pad], dim=0)
            organized.append(class_images)
        images = torch.cat(organized, dim=0)
        nrow = samples_per_class
    else:
        nrow = int(math.ceil(math.sqrt(len(images))))

    grid = make_image_grid(images, nrow=nrow)

    plt.figure(figsize=(12, 12))
    if grid.ndim == 2:
        plt.imshow(grid, cmap="gray")
    else:
        plt.imshow(grid)
    plt.axis("off")
    if title:
        plt.title(title)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


class SampleQueue:
    """
    Queue of cached samples per class for efficient batch sampling.
    Used to avoid needing a specialized data loader (Sec A.8).
    """

    def __init__(
        self,
        num_classes: int,
        queue_size: int = 128,
        sample_shape: tuple = (1, 32, 32),
    ):
        """
        Args:
            num_classes: Number of classes
            queue_size: Number of samples to cache per class
            sample_shape: Shape of each sample (C, H, W)
        """
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.sample_shape = sample_shape

        # Initialize queues
        self.queues = {
            c: torch.zeros(queue_size, *sample_shape)
            for c in range(num_classes)
        }
        self.counts = {c: 0 for c in range(num_classes)}
        self.indices = {c: 0 for c in range(num_classes)}

    def add(self, samples: torch.Tensor, labels: torch.Tensor):
        """Add samples to the queues."""
        samples = samples.detach().cpu()
        labels = labels.detach().cpu()

        for sample, label in zip(samples, labels):
            c = label.item()
            idx = self.indices[c] % self.queue_size
            self.queues[c][idx] = sample
            self.indices[c] += 1
            self.counts[c] = min(self.counts[c] + 1, self.queue_size)

    def sample(self, label: int, n: int, device: torch.device) -> torch.Tensor:
        """
        Sample n examples from the queue for a given class.

        Args:
            label: Class label
            n: Number of samples
            device: Device to put samples on

        Returns:
            Tensor of shape (n, C, H, W)
        """
        count = self.counts[label]
        if count == 0:
            raise ValueError(f"No samples in queue for class {label}")

        indices = torch.randint(0, count, (n,))
        samples = self.queues[label][indices]
        return samples.to(device)

    def is_ready(self, min_samples: int = 32) -> bool:
        """Check if all queues have at least min_samples."""
        return all(self.counts[c] >= min_samples for c in range(self.num_classes))


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@torch.no_grad()
def compute_fid_statistics(
    images: torch.Tensor,
    inception_model: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Compute FID statistics (mean and covariance) for a batch of images.

    Args:
        images: Tensor of shape (N, C, H, W) in range [-1, 1]
        inception_model: InceptionV3 feature extractor
        device: Device to use

    Returns:
        Tuple of (mean, covariance) tensors
    """
    # Resize to 299x299 for Inception
    images = torch.nn.functional.interpolate(
        images, size=(299, 299), mode="bilinear", align_corners=False
    )

    # Normalize to Inception expected range
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]

    # Get features
    inception_model.eval()
    inception_model.to(device)

    features = []
    batch_size = 64
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        feat = inception_model(batch)[0].squeeze(-1).squeeze(-1)
        features.append(feat.cpu())

    features = torch.cat(features, dim=0).numpy()

    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


def compute_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute FID between two sets of statistics.

    Args:
        mu1, sigma1: Statistics of first distribution
        mu2, sigma2: Statistics of second distribution
        eps: Small constant for numerical stability

    Returns:
        FID score
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return float(
        diff.dot(diff)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * tr_covmean
    )