import hydra
import os
from omegaconf import DictConfig, OmegaConf
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch import set_float32_matmul_precision
from drifting.models.dit_lightning import DriftDiTModule
from drifting.utils.trainer_callbacks import EMACallback, SamplingCallback
from drifting.utils.data_utils import get_dataset
from lightning.pytorch.loggers import WandbLogger
# from litlogger import LightningLogger # Save for the future

set_float32_matmul_precision('high')

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    config = OmegaConf.to_container(cfg.dataset, resolve=True)

    wandb_logger = WandbLogger(name="drift-lightning"+f"{config["model"]}-{cfg.run_name}")

    # logger = LightningLogger(
    #     name="drift-lightning",
    #     metadata={"model": "DriftDiT"},
    # ) # Save for the future
    
    last_checkpoint_path = os.path.join(cfg.resume_dir, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if os.path.exists(last_checkpoint_path) and cfg.resume_from_ckpt else None

    # 1. Data Setup (Standard PyTorch DataLoader works perfectly in Lightning)
    train_dataset, _ = get_dataset(config["name"], root=cfg.data_root)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=cfg.num_workers, drop_last=True
    )

    # 2. Model Setup
    model = DriftDiTModule(config)

    # 3. Callbacks Setup
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

    # 4. Trainer Setup
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=config["epochs"],
        callbacks=callbacks,
        accelerator="auto", # Automatically uses CUDA if available
        # precision="16-mixed", # <--- FREE 2x SPEEDUP & VRAM REDUCTION
        log_every_n_steps=cfg.log_step_interval,
    )

    # 5. GO!
    trainer.fit(model,
                train_dataloaders=train_loader,
                ckpt_path=last_checkpoint)

if __name__ == "__main__":
    main()
