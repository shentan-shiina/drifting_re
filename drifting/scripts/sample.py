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

from drifting.utils.data_utils import get_dataset
from drifting.utils.utils import save_image_grid, set_seed, load_model_from_checkpoint, calculate_frechet_distance

#######################################################
#                 VAE Helper Function                 #
#######################################################
@torch.no_grad()
def decode_latents(vae: nn.Module, latents: torch.Tensor, max_batch: int = 32) -> torch.Tensor:
    """Safely decode latents in chunks to prevent CUDA OOM."""
    if vae is None:
        return latents
        
    decoded = []
    for i in range(0, latents.size(0), max_batch):
        batch = latents[i : i + max_batch]
        decoded.append(vae.decode(batch))
    return torch.cat(decoded, dim=0)

#######################################################
#                 Generate Samples                    #
#######################################################
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
    vae: Optional[nn.Module] = None,
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

    # Decode latents to RGB if VAE is provided
    x = decode_latents(vae, x)

    return x.clamp(-1, 1)

#######################################################
#               Generate Class Grid                   #
#######################################################
@torch.no_grad()
def generate_class_grid(
    model: nn.Module,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    samples_per_class: int = 10,
    alpha: float = 1.5,
    vae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Generate a grid of samples with each row being one class."""
    rows = []
    for c in range(num_classes):
        z = torch.randn(samples_per_class, in_channels, img_size, img_size, device=device)
        labels = torch.full((samples_per_class,), c, device=device, dtype=torch.long)
        x = model.forward_with_cfg(z, labels, alpha=alpha)
        
        x = decode_latents(vae, x)
        rows.append(x)
        
    return torch.cat(rows, dim=0).clamp(-1, 1)

#######################################################
#           Generate Alpha Sweep Samples              #
#######################################################
@torch.no_grad()
def generate_alpha_sweep(
    model: nn.Module,
    in_channels: int,
    img_size: int,
    device: torch.device,
    label: int = 0,
    alphas: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    samples_per_alpha: int = 8,
    vae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Generate samples across different CFG alpha values for a single class."""
    z = torch.randn(samples_per_alpha, in_channels, img_size, img_size, device=device)
    labels = torch.full((samples_per_alpha,), label, device=device, dtype=torch.long)

    sweep = []
    for alpha in alphas:
        x = model.forward_with_cfg(z, labels, alpha=alpha)
        x = decode_latents(vae, x)
        sweep.append(x)
        
    return torch.cat(sweep, dim=0).clamp(-1, 1)

#######################################################
#                   Compute FID                       #
#######################################################
@torch.no_grad()
def compute_fid_score(
    model: nn.Module,
    stats_path: str,
    in_channels: int,
    img_size: int,
    num_classes: int,
    device: torch.device,
    num_samples: int = 10000,
    batch_size: int = 256,
    alpha: float = 1.5,
    vae: Optional[nn.Module] = None,
) -> float:
    """Compute FID score using pre-computed real statistics."""
    import torch.nn.functional as F
    from torchvision.models.inception import inception_v3, Inception_V3_Weights
    
    ########### Load Precomputed FID Stats ###########
    print(f"Loading precomputed FID stats from {stats_path}...")
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'], f['sigma']

    print("Loading InceptionV3 feature extractor...")
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception.fc = torch.nn.Identity() # Strip classification head
    inception.eval()

    ########### Generate Samples ###########
    print(f"Generating {num_samples} samples for FID...")
    fake_features = []
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
            vae=vae,
        )

        samples = (samples + 1) / 2
        if samples.shape[1] == 1:
            samples = samples.repeat(1, 3, 1, 1)
            
        samples = F.interpolate(samples, size=(299, 299), mode="bilinear", align_corners=False)
        
        ########### Extract Features ###########
        feat = inception(samples)
        fake_features.append(feat.cpu().numpy())
        num_generated += current_batch

    fake_features = np.concatenate(fake_features, axis=0)

    ########### Compute FID ###########
    print("Calculating mean and covariance for generated samples...")
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    print("Computing Frechet Distance...")
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid

#######################################################
#                       Main                          #
#######################################################
def sample_and_save(cfg: DictConfig):
    ########### Initialize ###########
    set_seed(cfg.samples.seed)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(dataset_cfg, dict):
        raise ValueError("cfg.dataset must be a mapping")
    config: Dict[str, Any] = dataset_cfg
    dataset_name = config["name"]

    ########### Load Model & VAE ###########
    if cfg.checkpoint is None:
        raise ValueError("cfg.checkpoint must be provided (path to checkpoint)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize VAE if operating in latent space
    vae_manager = None
    if config.get("use_latent", False):
        from drifting.utils.vae_utils import VAEManager # Assuming you saved it here
        print("Loading VAE Manager for latent decoding...")
        vae_manager = VAEManager().to(device)
        vae_manager.eval()

    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data"
    data_root = Path(cfg.data_root) if cfg.data_root else default_data_root
    data_root = Path(to_absolute_path(str(data_root)))

    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory (Hydra run dir): {output_dir}")

    ckpt_path = Path(to_absolute_path(cfg.checkpoint))
    model = load_model_from_checkpoint(ckpt_path, config, device)

    ########### Generate Samples ###########
    print(f"Generating class grid ({config['num_classes']} classes x {cfg.samples.samples_per_class} samples)...")
    grid = generate_class_grid(
        model,
        config["in_channels"],
        config["img_size"],
        config["num_classes"],
        device,
        samples_per_class=cfg.samples.samples_per_class,
        alpha=cfg.samples.alpha,
        vae=vae_manager,
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
        vae=vae_manager,
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
            vae=vae_manager,
        )
        sweep_path = output_dir / f"alpha_sweep_class{label}.png"
        save_image_grid(sweep_samples, str(sweep_path), nrow=8)
        print(f"Saved alpha sweep for class {label} to {sweep_path}")

    ########### Compute FID Score ###########
    if cfg.samples.compute_fid:
        print("Computing FID score...")

        if getattr(cfg.samples, "fid_stats_path", None):
             stats_path = Path(to_absolute_path(cfg.samples.fid_stats_path))
        else:
            stats_path = output_dir / f"fid_stats/{dataset_cfg['name']}_fid_stats.npz" 

        if not stats_path.exists():
             raise FileNotFoundError(f"Cannot compute FID: Missing stats file at {stats_path}. Run precompute_data.py first.")

        fid = compute_fid_score(
            model,
            stats_path=str(stats_path),
            in_channels=config["in_channels"],
            img_size=config["img_size"],
            num_classes=config["num_classes"],
            device=device,
            num_samples=cfg.samples.fid_num_samples,
            batch_size=cfg.samples.fid_batch_size,
            alpha=cfg.samples.alpha,
            vae=vae_manager,
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