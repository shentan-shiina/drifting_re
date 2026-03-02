#!/usr/bin/env python3
"""
Pre-train the Multi-Scale Feature Encoder using a Masked Autoencoder (MAE) objective.
Hydra-backed and PyTorch Lightning accelerated.
"""

import os 
import hydra
import torch

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from drifting.models.mae_lightning import MAEPretrainModule
from drifting.utils.data_utils import get_dataset
from drifting.utils.vae_utils import LatentDataset
from lightning.pytorch.loggers import WandbLogger

@hydra.main(config_path="../config", config_name="mae", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    config = OmegaConf.to_container(cfg.dataset, resolve=True)
    
    wandb_logger = WandbLogger(
        project="drift-mae",
        name=f"{config['model']}-{cfg.run_name}"
    )
    
    last_checkpoint_path = os.path.join(cfg.mae_checkpoint_dir, cfg.resume_mae_ckpt)
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg.resume_from_ckpt else None

    output_dir = Path.cwd()
    print(f"Working directory (Hydra run dir): {output_dir}")

    # Data Setup
    use_latent = config.get("use_latent", False)
    
    if use_latent:
        if LatentDataset is None:
            raise ImportError("LatentDataset could not be imported. Check vae_util.py")
        
        latent_dir = Path(cfg.latent_dir)
        print(f"Loading Latent Dataset from {latent_dir}...")
        train_dataset = LatentDataset(root=str(latent_dir), use_flip=config.get("use_flip", True))
        
        # Override channels and size for Latents (SD-VAE = 4 channels, 1/8th size)
        config["in_channels"] = 4
        config["img_size"] = config.get("img_size", 256) // 8
    else:
        print(f"Loading Pixel Dataset: {config['name']}...")
        train_dataset, _ = get_dataset(config["name"], root=cfg.data_root, resize=config["img_size"])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        drop_last=True,
        pin_memory=True
    )


    print(f"Initializing MAEPretrainModule (in_channels={config['in_channels']}, img_size={config['img_size']})")
    model = MAEPretrainModule(config)

    # Callbacks Setup
    callbacks = [
        ModelCheckpoint(
            dirpath="mae_checkpoints",
            every_n_epochs=cfg.save_interval,
            save_last=True,
            filename="mae-{epoch:03d}-{mae_loss:.4f}",
            monitor="mae_loss",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Trainer Setup
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["epochs"],
        callbacks=callbacks,
        accelerator="auto",
        precision="16-mixed",
        log_every_n_steps=cfg.log_interval,
        gradient_clip_val=cfg.get("grad_clip", 1.0),
    )

    # Start Training
    print("Starting MAE Pre-training...")
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=last_checkpoint)


if __name__ == "__main__":
    main()