import hydra
import os
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch import set_float32_matmul_precision
from drifting.models.driftdit_lightning import DriftDiTModule
from drifting.utils.trainer_callbacks import EMACallback, SamplingCallback
from drifting.utils.data_utils import get_dataset
from lightning.pytorch.loggers import WandbLogger

from drifting.utils.vae_utils import LatentDataset

# from litlogger import LightningLogger # Save for the future
    # logger = LightningLogger(
    #     name="drift-lightning",
    #     metadata={"model": "DriftDiT"},
    # ) # Save for the future

set_float32_matmul_precision('high')

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    config = OmegaConf.to_container(cfg.dataset, resolve=True)

    wandb_logger = WandbLogger(name="drift-lightning"+f"{config["model"]}-{cfg.run_name}")
    
    last_checkpoint_path = os.path.join(cfg.resume_dir, cfg.resume_ckpt)
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg.resume_from_ckpt else None

    # Data Setup
    if cfg.use_latent:
        config["img_size"] = config.get("img_size", 256) // 8
        config["in_channels"] = 4
        config["mae_checkpoint_path"] = cfg.get("mae_checkpoint_dir", None) + cfg.get("resume_mae_ckpt", None)

        if not os.path.exists(config["mae_checkpoint_path"]):
            raise FileNotFoundError(f"Failed to load MAE checkpoint at {config["mae_checkpoint_path"]}. Run pretrain MAE first.")

        latent_dir = os.path.join(cfg.data_root, config["name"], "latents")
        train_dataset = LatentDataset(root=latent_dir, use_flip=config.get("latent_flip", True))
    else:
        train_dataset, _ = get_dataset(config["name"], root=cfg.data_root, resize=config["img_size"])

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, drop_last=True
    )

    # Model Setup
    model = DriftDiTModule(config)

    # Callbacks Setup
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            every_n_epochs=cfg.save_interval,
            save_last=True,
            monitor=cfg.logger.monitor,
            save_top_k=5,
            filename=f"{config["model"]}-{cfg.run_name}"+"-{epoch:02d}-{step:02d}-{train_loss:0.3f}"
        ),
        EMACallback(decay=config["ema_decay"]),
        SamplingCallback(sample_interval=cfg.sample_interval, config=config)
    ]

    # Trainer Setup
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["epochs"],
        callbacks=callbacks,
        accelerator="auto",
        precision="16-mixed",
        gradient_clip_val=config["grad_clip"],
        log_every_n_steps=cfg.log_step_interval,
    )

    # Start Training
    trainer.fit(model,
                train_dataloaders=train_loader,
                ckpt_path=last_checkpoint)

if __name__ == "__main__":
    main()
