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
from torch import set_float32_matmul_precision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from drifting.models.mae_lightning import MAEPretrainModule
from drifting.utils.data_utils import get_dataset
from drifting.utils.vae_utils import LatentDataset
from lightning.pytorch.loggers import WandbLogger

set_float32_matmul_precision('high')

@hydra.main(config_path="../config", config_name="mae", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    config = OmegaConf.to_container(cfg.dataset, resolve=True)
    mae_cfg = config["mae"]
    
    wandb_logger = WandbLogger(
        project="drift-mae",
        name=f"{config['name']}-{config['model']}-{cfg.run_name}-L_{config['use_latent']}-FT_{config['mae']['finetune_classifier']}"
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
        if mae_cfg.get("use_rrc_aug", True):
            print("[MAE] Latent pretraining uses cached latents; random-resized-crop before VAE is unavailable in this mode.")
    else:
        print(f"Loading Pixel Dataset: {config['name']}...")
        train_dataset, _ = get_dataset(config["name"], root=cfg.data_root, resize=config["img_size"])
        if mae_cfg.get("use_rrc_aug", True):
            normalize = [0.5] if config["in_channels"] == 1 else [0.5, 0.5, 0.5]
            train_dataset.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=config["img_size"],
                    scale=(0.2, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(normalize, normalize),
            ])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(mae_cfg.get("batch_size", cfg.batch_size)), 
        shuffle=True, 
        num_workers=cfg.num_workers, 
        drop_last=True,
        pin_memory=True
    )


    print(f"Initializing MAEPretrainModule (in_channels={config['in_channels']}, img_size={config['img_size']})")
    model = MAEPretrainModule(config)

    # Callbacks Setup (pretrain)
    pretrain_callbacks = [
        ModelCheckpoint(
            dirpath="mae_checkpoints",
            every_n_epochs=cfg.save_interval,
            save_last=True,
            filename=f"mae-{{epoch:03d}}-{{mae_loss:.4f}}-L_{config['use_latent']}-FT_{config['mae']['finetune_classifier']}",
            monitor="mae_loss",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Trainer Setup
    model.set_training_stage("pretrain")
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=mae_cfg["epochs"],
        callbacks=pretrain_callbacks,
        accelerator="auto",
        precision="bf16-mixed",
        log_every_n_steps=cfg.log_interval,
        gradient_clip_val=cfg.get("grad_clip", 1.0),
        # profiler="simple",
    )

    # Start Training
    print("Starting MAE Pre-training...")
    trainer.fit(model, train_dataloaders=train_loader, ckpt_path=last_checkpoint)

    if mae_cfg.get("finetune_classifier", False):
        finetune_steps = int(mae_cfg.get("finetune_steps", 3000))
        if finetune_steps > 0:
            print(f"Starting MAE classifier fine-tuning for {finetune_steps} steps...")
            model.set_training_stage("finetune")

            finetune_callbacks = [
                ModelCheckpoint(
                    dirpath="mae_finetune_checkpoints",
                    every_n_epochs=cfg.save_interval,
                    save_last=True,
                    filename=f"mae-ft-{{step:06d}}-{{mae_loss:.4f}}",
                    monitor="mae_loss",
                    mode="min",
                    save_top_k=3,
                ),
                LearningRateMonitor(logging_interval='step')
            ]

            finetune_trainer = Trainer(
                logger=wandb_logger,
                max_steps=finetune_steps,
                max_epochs=100,
                callbacks=finetune_callbacks,
                accelerator="auto",
                precision="bf16-mixed",
                log_every_n_steps=cfg.log_interval,
                gradient_clip_val=cfg.get("grad_clip", 1.0),
            )
            finetune_trainer.fit(model, train_dataloaders=train_loader, ckpt_path=None)


if __name__ == "__main__":
    main()