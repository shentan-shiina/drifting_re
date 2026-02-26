"""
Sampling and visualization for Drifting Models.
Hydra-backed entrypoint mirroring the training script style.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from drifting.models.model import DriftDiT_models
from drifting.utils.data_utils import get_dataset
from drifting.utils.utils import save_image_grid, set_seed


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    num_samples: int,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    labels: Optional[torch.Tensor] = None,
    alpha: float = 1.5,
    use_cfg: bool = True,
) -> torch.Tensor:
    """Generate samples using one-step inference."""
    model.eval()

    z = torch.randn(num_samples, in_channels, img_size, img_size, device=device)

    if labels is None:
        labels = torch.randint(0, num_classes, (num_samples,), device=device)
    else:
        labels = labels.to(device)

    if use_cfg:
        x = model.forward_with_cfg(z, labels, alpha=alpha)
    else:
        alpha_tensor = torch.full((num_samples,), alpha, device=device)
        x = model(z, labels, alpha_tensor)

    return x.clamp(-1, 1)


@torch.no_grad()
def generate_class_grid(
    model: nn.Module,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    samples_per_class: int = 10,
    alpha: float = 1.5,
) -> torch.Tensor:
    """Generate a grid of samples with each row being one class."""
    rows = []
    for c in range(num_classes):
        z = torch.randn(samples_per_class, in_channels, img_size, img_size, device=device)
        labels = torch.full((samples_per_class,), c, device=device, dtype=torch.long)
        x = model.forward_with_cfg(z, labels, alpha=alpha)
        rows.append(x)
    return torch.cat(rows, dim=0).clamp(-1, 1)


@torch.no_grad()
def generate_alpha_sweep(
    model: nn.Module,
    in_channels: int,
    img_size: int,
    device: torch.device,
    label: int = 0,
    alphas: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    samples_per_alpha: int = 8,
) -> torch.Tensor:
    """Generate samples across different CFG alpha values for a single class."""
    z = torch.randn(samples_per_alpha, in_channels, img_size, img_size, device=device)
    labels = torch.full((samples_per_alpha,), label, device=device, dtype=torch.long)

    sweep = []
    for alpha in alphas:
        x = model.forward_with_cfg(z, labels, alpha=alpha)
        sweep.append(x)
    return torch.cat(sweep, dim=0).clamp(-1, 1)


def compute_fid_score(
    model: nn.Module,
    real_loader: DataLoader,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    num_samples: int = 10000,
    batch_size: int = 256,
    alpha: float = 1.5,
) -> float:
    """Compute FID score between generated and real images."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        print("torchmetrics not available, skipping FID computation")
        return float("nan")

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    model.eval()

    print("Processing real images for FID...")
    num_real = 0
    for batch in tqdm(real_loader, desc="Real", leave=False):
        x_real = batch[0].to(device)
        current = x_real.shape[0]
        if num_real + current > num_samples:
            x_real = x_real[: num_samples - num_real]
            current = x_real.shape[0]
        x_real = (x_real + 1) / 2
        if in_channels == 1:
            x_real = x_real.repeat(1, 3, 1, 1)
        x_real = torch.nn.functional.interpolate(
            x_real, size=(299, 299), mode="bilinear", align_corners=False
        )
        fid_metric.update(x_real, real=True)
        num_real += current
        if num_real >= num_samples:
            break

    print("Generating samples for FID...")
    num_generated = 0
    while num_generated < num_samples:
        current_batch = min(batch_size, num_samples - num_generated)
        samples = generate_samples(
            model,
            current_batch,
            in_channels,
            img_size,
            num_classes,
            device,
            alpha=alpha,
        )
        samples = (samples + 1) / 2
        if in_channels == 1:
            samples = samples.repeat(1, 3, 1, 1)
        samples = torch.nn.functional.interpolate(
            samples, size=(299, 299), mode="bilinear", align_corners=False
        )
        fid_metric.update(samples, real=False)
        num_generated += current_batch

    return fid_metric.compute().item()


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
    in_channels = ckpt_cfg.get("in_channels", config["in_channels"])
    num_classes = ckpt_cfg.get("num_classes", config["num_classes"])
    label_dropout = ckpt_cfg.get("label_dropout", config.get("label_dropout", 0.0))

    model_fn = DriftDiT_models[model_name]
    model = model_fn(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        label_dropout=label_dropout,
    ).to(device)

    state = checkpoint.get("ema", checkpoint.get("model"))
    model.load_state_dict(state)
    model.eval()
    return model


def sample_and_save(cfg: DictConfig):
    """Load model, generate visualizations, optionally compute FID."""
    set_seed(cfg.samples.seed)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(dataset_cfg, dict):
        raise ValueError("cfg.dataset must be a mapping")
    config: Dict[str, Any] = dataset_cfg
    dataset_name = config["name"]

    if cfg.checkpoint is None:
        raise ValueError("cfg.checkpoint must be provided (path to checkpoint)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data"
    data_root = Path(cfg.data_root) if cfg.data_root else default_data_root
    data_root = Path(to_absolute_path(str(data_root)))

    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory (Hydra run dir): {output_dir}")

    ckpt_path = Path(to_absolute_path(cfg.checkpoint))
    model = load_model_from_checkpoint(ckpt_path, config, device)

    print(f"Generating class grid ({config['num_classes']} classes x {cfg.samples.samples_per_class} samples)...")
    grid = generate_class_grid(
        model,
        config["in_channels"],
        config["img_size"],
        config["num_classes"],
        device,
        samples_per_class=cfg.samples.samples_per_class,
        alpha=cfg.samples.alpha,
    )
    grid_path = output_dir / f"class_grid_alpha{cfg.samples.alpha:.1f}.png"
    save_image_grid(grid, str(grid_path), nrow=cfg.samples.samples_per_class)
    print(f"Saved class grid to {grid_path}")

    print(f"Generating {cfg.samples.num_samples} random samples...")
    samples = generate_samples(
        model,
        cfg.samples.num_samples,
        config["in_channels"],
        config["img_size"],
        config["num_classes"],
        device,
        alpha=cfg.samples.alpha,
    )
    nrow = int(np.ceil(np.sqrt(cfg.samples.num_samples)))
    samples_path = output_dir / f"random_samples_alpha{cfg.samples.alpha:.1f}.png"
    save_image_grid(samples, str(samples_path), nrow=nrow)
    print(f"Saved random samples to {samples_path}")

    print("Generating alpha sweep (first 3 classes)...")
    alphas = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    for label in range(min(3, config["num_classes"])):
        sweep_samples = generate_alpha_sweep(
            model,
            config["in_channels"],
            config["img_size"],
            device,
            label=label,
            alphas=alphas,
            samples_per_alpha=8,
        )
        sweep_path = output_dir / f"alpha_sweep_class{label}.png"
        save_image_grid(sweep_samples, str(sweep_path), nrow=8)
        print(f"Saved alpha sweep for class {label} to {sweep_path}")

    if cfg.samples.compute_fid:
        print("Computing FID score...")
        _, test_dataset = get_dataset(dataset_name, root=str(data_root))
        real_loader = DataLoader(
            test_dataset,
            batch_size=cfg.samples.fid_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        fid = compute_fid_score(
            model,
            real_loader,
            config["in_channels"],
            config["img_size"],
            config["num_classes"],
            device,
            num_samples=cfg.samples.fid_num_samples,
            batch_size=cfg.samples.fid_batch_size,
            alpha=cfg.samples.alpha,
        )
        print(f"FID Score: {fid:.2f}")
        with open(output_dir / "fid_score.txt", "w") as f:
            f.write(f"FID Score (alpha={cfg.samples.alpha}): {fid:.2f}\n")

    print("\nSampling complete!")


@hydra.main(config_path="../config", config_name="sample", version_base="1.3")
def main(cfg: DictConfig):
    sample_and_save(cfg)


if __name__ == "__main__":
    main()
