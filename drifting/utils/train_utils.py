"""
Training utilities for Drifting models.
Separates per-step logic from the training entrypoint.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drifting.models.drifting import compute_V
from drifting.utils.utils import SampleQueue


def sample_batch(queue: SampleQueue, num_classes: int, n_pos: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of positive examples per class from the queue."""
    x_pos_list = []
    labels_list = []

    for c in range(num_classes):
        x_c = queue.sample(c, n_pos, device)
        x_pos_list.append(x_c)
        labels_list.append(torch.full((n_pos,), c, device=device, dtype=torch.long))

    x_pos = torch.cat(x_pos_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return x_pos, labels

def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    temperatures: list,
    use_pixel_space: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """Compute class-conditional drifting loss with multi-scale features."""
    device = x_gen.device
    num_classes = labels_gen.max().item() + 1

    # Extract features
    if use_pixel_space or feature_encoder is None:
        feat_gen_list = [x_gen.flatten(start_dim=1)]
        feat_pos_list = [x_pos.flatten(start_dim=1)]
    else:
        feat_gen_out = feature_encoder(x_gen)
        with torch.no_grad():
            feat_pos_out = feature_encoder(x_pos)

        # Check if the encoder returned a single flat tensor (MAE Encoder) 
        # or a list of spatial feature maps (ResNet Encoder)
        if isinstance(feat_gen_out, torch.Tensor):
            feat_gen_list = [feat_gen_out]
            feat_pos_list = [feat_pos_out]
        else:
            feat_gen_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_gen_out]
            feat_pos_list = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feat_pos_out]

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    total_drift_norm = 0.0
    num_losses = 0

    for c in range(num_classes):
        mask_gen = labels_gen == c
        mask_pos = labels_pos == c

        if not mask_gen.any() or not mask_pos.any():
            continue

        for feat_gen, feat_pos in zip(feat_gen_list, feat_pos_list):
            feat_gen_c = feat_gen[mask_gen]
            feat_pos_c = feat_pos[mask_pos]
            feat_neg_c = feat_gen_c

            feat_gen_c_norm = F.normalize(feat_gen_c, p=2, dim=1)
            feat_pos_c_norm = F.normalize(feat_pos_c, p=2, dim=1)
            feat_neg_c_norm = F.normalize(feat_neg_c, p=2, dim=1)

            V_total = torch.zeros_like(feat_gen_c_norm)
            for tau in temperatures:
                V_tau = compute_V(
                    feat_gen_c_norm,
                    feat_pos_c_norm,
                    feat_neg_c_norm,
                    tau,
                    mask_self=True,
                )
                v_norm = torch.sqrt(torch.mean(V_tau ** 2) + 1e-8)
                V_total = V_total + V_tau / (v_norm + 1e-8)

            target = (feat_gen_c_norm + V_total).detach()
            loss_scale = F.mse_loss(feat_gen_c_norm, target)

            total_loss = total_loss + loss_scale
            total_drift_norm += (V_total ** 2).mean().item() ** 0.5
            num_losses += 1

    if num_losses == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), {"loss": 0.0, "drift_norm": 0.0}

    loss = total_loss / num_losses
    info = {
        "loss": loss.item(),
        "drift_norm": total_drift_norm / num_losses,
    }

    return loss, info


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    queue: SampleQueue,
    config: dict,
    device: torch.device,
    feature_encoder: Optional[nn.Module] = None,
) -> dict:
    """Single training step following Algorithm 1."""
    model.train()
    num_classes = config["num_classes"]
    n_pos = config["batch_n_pos"]
    n_neg = config["batch_n_neg"]
    alpha_min = config["alpha_min"]
    alpha_max = config["alpha_max"]
    temperatures = config["temperatures"]
    use_pixel = not config["use_feature_encoder"]

    batch_size = num_classes * n_neg
    labels = torch.arange(num_classes, device=device).repeat_interleave(n_neg)
    alpha = torch.empty(batch_size, device=device).uniform_(alpha_min, alpha_max)

    noise = torch.randn(
        batch_size,
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    x_gen = model(noise, labels, alpha)
    x_pos, labels_pos = sample_batch(queue, num_classes, n_pos, device)

    loss, info = compute_drifting_loss(
        x_gen,
        labels,
        x_pos,
        labels_pos,
        feature_encoder,
        temperatures,
        use_pixel_space=use_pixel,
    )

    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
    info["grad_norm"] = grad_norm.item()
    optimizer.step()

    return info


def fill_queue(queue: SampleQueue, dataloader: DataLoader, device: torch.device, min_samples: int = 64):
    """Warm up the per-class queues with real samples."""
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, labels = batch[0], batch[1]
        else:
            x, labels = batch, torch.zeros(batch.shape[0], dtype=torch.long)

        queue.add(x, labels)

        if queue.is_ready(min_samples):
            break
