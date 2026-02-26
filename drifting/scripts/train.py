"""
Training script for Drifting Models on MNIST and CIFAR-10.
Hydra-backed configuration with a clean entrypoint.
"""

import time
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drifting.models.dit_drift import DriftDiT_models
from drifting.models.feature_encoder import create_feature_encoder
from drifting.utils.train_utils import train_step, fill_queue
from drifting.utils.data_utils import get_dataset
from drifting.utils.utils import (
    EMA,
    WarmupLRScheduler,
    SampleQueue,
    save_checkpoint,
    load_checkpoint,
    save_image_grid,
    count_parameters,
    set_seed,
)
from sample import generate_samples

#######################################################
#                       Main                          #
#######################################################
def train(cfg: DictConfig):
    ########### Initialize ###########
    set_seed(cfg.seed)

    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(dataset_cfg, dict):
        raise ValueError("cfg.dataset must be a mapping")
    config: Dict[str, Any] = dataset_cfg
    dataset_name = config["name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    repo_root = Path(__file__).resolve().parents[2]
    default_data_root = repo_root / "data"
    data_root = Path(cfg.data_root) if cfg.data_root else default_data_root
    data_root = Path(to_absolute_path(str(data_root)))

    output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory (Hydra run dir): {output_dir}")

    ########### Createt DataLoader ###########
    train_dataset, _ = get_dataset(dataset_name, root=str(data_root))
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    ########### Create Model ###########
    model_fn = DriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        label_dropout=config["label_dropout"],
    ).to(device)
    print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    ########### Create Train State ###########
    ema = EMA(model, decay=config["ema_decay"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )

    queue = SampleQueue(
        num_classes=config["num_classes"],
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )

    ########### Create Feature Encoder ###########
    feature_encoder = None
    if config["use_feature_encoder"]:
        print("Creating feature encoder...")
        feature_encoder = create_feature_encoder(
            dataset=dataset_name,
            feature_dim=512,
            multi_scale=True,
            use_pretrained=True,
        ).to(device)
        feature_encoder.eval()
        for param in feature_encoder.parameters():
            param.requires_grad = False

    ########### Training Loop ###########
    start_epoch = 0
    global_step = 0
    if cfg.resume:
        resume_path = Path(to_absolute_path(cfg.resume))
        checkpoint = load_checkpoint(str(resume_path), model, ema, optimizer, scheduler)
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    print(f"\nStarting training for {config['epochs']} epochs...")
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_drift_norm = 0.0
        num_batches = 0

        fill_queue(queue, train_loader, device, min_samples=64)

        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config['epochs']}",
            leave=False,
        )
        
        ########### Batch Queue ###########
        for _, batch in enumerate(progress):
            if isinstance(batch, (list, tuple)):
                x_real, labels_real = batch[0].to(device), batch[1].to(device)
            else:
                x_real = batch.to(device)
                labels_real = torch.zeros(x_real.shape[0], dtype=torch.long, device=device)

            queue.add(x_real.cpu(), labels_real.cpu())

            if not queue.is_ready(config["batch_n_pos"]):
                continue

            ########### Training Step ###########
            info = train_step(
                model,
                optimizer,
                queue,
                config,
                device,
                feature_encoder,
            )

            ema.update(model)
            scheduler.step()

            ########### Metrics ###########
            epoch_loss += info["loss"]
            epoch_drift_norm += info["drift_norm"]
            num_batches += 1
            global_step += 1

            if num_batches > 0:
                progress.set_postfix(
                    loss=f"{info['loss']:.4f}", drift=f"{info['drift_norm']:.4f}", grad=f"{info['grad_norm']:.2f}"
                )

        ########### Print Per Epoch ###########
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % cfg.log_interval == 0:
            lr = scheduler.get_lr()
            print(
                f"Epoch {epoch+1}/{config['epochs']} | "
                f"Step {global_step} | "
                f"Loss: {info['loss']:.4f} | "
                f"Drift: {info['drift_norm']:.4f} | "
                f"Grad: {info['grad_norm']:.4f} | "
                f"LR: {lr:.6f}"
            )

        if (epoch + 1) % cfg.sample_interval == 0:
            sample_path = output_dir / f"samples_epoch{(epoch + 1)}.png"
            samples = generate_samples(
                ema.shadow,
                config["num_classes"] * config["batch_n_pos"],
                config["in_channels"],
                config["img_size"],
                config["num_classes"],
                device,
                labels = None,
                use_cfg = False
            )
            save_image_grid(samples, sample_path, nrow=config["batch_n_pos"])
            print(f"Saved samples to {sample_path}")

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_drift = epoch_drift_norm / max(num_batches, 1)
        print(
            f"\nEpoch {epoch+1} completed in {epoch_time:.1f}s | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Avg Drift Norm: {avg_drift:.4f}\n"
        )

        ########### Save Checkpoint ###########
        if (epoch + 1) % cfg.save_interval == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            save_checkpoint(
                str(ckpt_path),
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        str(final_path),
        model,
        ema,
        optimizer,
        scheduler,
        config["epochs"] - 1,
        global_step,
        config,
    )
    print(f"Training complete! Final checkpoint saved to {final_path}")


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
