#!/usr/bin/env python3
"""
Script for pre-computing VAE latent datasets and FID statistics.
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import transforms

from drifting.utils.data_utils import get_dataset

#######################################################
#              Compute Latent Dataset                 #
#  (Precompute latents for *any* dataset -> size/8)   #
#######################################################

@torch.no_grad()
def compute_latent_dataset(cfg: DictConfig,  device: torch.device):
    """Encodes images to latents using VAE and saves them to disk."""
    latent_dir = Path(cfg.latent_dir)
    latent_dir.mkdir(parents=True, exist_ok=True)
    
    ########### Load VAE ###########
    print(f"Loading VAE model: {cfg.vae_id}")
    vae = AutoencoderKL.from_pretrained(cfg.vae_id).to(device)
    vae.eval()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    
    ########### Load Dataset ###########
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset, _ = get_dataset(dataset_cfg["name"], root=cfg.data_root, resize=dataset_cfg["img_size"])
    # Drop stochastic augmentations so cached latents are deterministic
    if dataset_cfg.get("precompute_drop_flip", True) and hasattr(dataset, "transform"):
        transform = dataset.transform
        if isinstance(transform, transforms.Compose):
            transform = transforms.Compose([t for t in transform.transforms if not isinstance(t, transforms.RandomHorizontalFlip)])
        dataset.transform = transform
    
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    
    ########### Compute Latents ###########
    print(f"Encoding {len(dataset)} images to latents...")
    idx = 0
    autocast_enabled = device.type == "cuda"
    for batch in tqdm(loader, desc="Encoding Latents"):
        images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        # Fake RGB for grayscale datasets so VAE sees 3 channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            latent_dist = vae.encode(images).latent_dist

        # Apply VAE scaling factor to match downstream usage (Latent Diffusion convention)
        mean = latent_dist.mean * scaling_factor
        std = latent_dist.std * scaling_factor

        # Concat along channel dim to shape (B, 8, H/8, W/8)
        cached_latents = torch.cat([mean, std], dim=1).cpu()
        labels = labels.cpu()

        for i in range(images.size(0)):
            save_path = latent_dir / f"{idx:07d}.pt"
            torch.save({
                "image": cached_latents[i],
                "label": labels[i]
            }, save_path)
            idx += 1
            
    print(f"Successfully saved {idx} latents to {latent_dir}")


#######################################################
#                 Compute FID Stats                   #
#              (Precompute FID Scores)                #
#######################################################
@torch.no_grad()
def compute_fid_stats(cfg: DictConfig, device: torch.device):
    """Computes FID statistics (mu, sigma) of the real dataset."""
    from torchvision.models.inception import inception_v3, Inception_V3_Weights
    import torch.nn.functional as F

    output_dir = Path(cfg.fid_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ########### Load Dataset ###########
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    dataset, test_dataset = get_dataset(dataset_cfg["name"], root=cfg.data_root, resize=dataset_cfg["img_size"])
    # Prefer deterministic (no-flip) pipeline for FID
    fid_dataset = test_dataset if test_dataset is not None else dataset
    if dataset_cfg.get("fid_drop_flip", True) and hasattr(fid_dataset, "transform"):
        transform = fid_dataset.transform
        if isinstance(transform, transforms.Compose):
            transform = transforms.Compose([t for t in transform.transforms if not isinstance(t, transforms.RandomHorizontalFlip)])
        fid_dataset.transform = transform
    
    loader = DataLoader(
        fid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )

    ########### Load InceptionV3 ###########
    print("Loading InceptionV3 for FID stats...")
    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception.fc = torch.nn.Identity() # Remove classification head to get pool3 features
    inception.eval()

    all_features = []

    ########### Extract FID Features ###########
    for batch in tqdm(loader, desc="Computing FID Features"):
        images = batch[0].to(device)
        
        # Inception expects images in [0, 1] range and min size 299x299
        images = (images + 1) / 2 
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        
        features = inception(images)
        all_features.append(features.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    
    ########### Compute Mean and Covariance ###########
    #$$d^2 = ||\mu_1 - \mu_2||^2 + \text{Tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1 \Sigma_2)^{1/2})$$#
    print("Calculating mean and covariance...")
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)
    
    ########### Save ###########
    np.savez(output_dir / f"{dataset_cfg['name']}_fid_stats.npz", mu=mu, sigma=sigma)
    print(f"FID stats saved to {output_dir}/{dataset_cfg['name']}_fid_stats.npz")

@hydra.main(config_path="../config", config_name="precompute", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if cfg.compute_latent:
        compute_latent_dataset(cfg, device)
        
    if cfg.compute_fid:
        compute_fid_stats(cfg, device)

if __name__ == "__main__":
    main()