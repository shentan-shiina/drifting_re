"""
Training utilities for Drifting models.
Separates per-step logic from the training entrypoint.
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drifting.models.drifting import compute_V_multi_temperature
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


def sample_unconditional(queue: SampleQueue, n: int, device: torch.device) -> torch.Tensor:
    """Sample unconditional negatives uniformly across classes from the queue."""
    samples = []
    classes = [c for c in queue.queues.keys() if queue.counts[c] > 0]
    if not classes:
        raise ValueError("Queue is empty; cannot sample unconditional negatives.")
    for _ in range(n):
        c = int(torch.randint(0, len(classes), (1,), device=device).item())
        samples.append(queue.sample(classes[c], 1, device))
    return torch.cat(samples, dim=0)

def _normalize_feature_block(features: List[torch.Tensor]) -> List[torch.Tensor]:
    """Appendix A.6 feature normalization with one scalar scale per feature map.

    For spatial feature maps, tokens from all locations are already concatenated
    before this function is called. We compute one shared scale across those
    locations ("Normalization across spatial locations" in A.6).
    """
    concat = torch.cat(features, dim=0)

    # A.6: one scalar Sj chosen so E||phi(x)-phi(y)|| ~= sqrt(Cj)
    # with stop-grad on the scale statistic.
    with torch.no_grad():
        n = min(concat.shape[0], 256)
        idx = torch.randperm(concat.shape[0], device=concat.device)[:n]
        subset = concat[idx]
        if n > 1:
            dists = torch.cdist(subset, subset, p=2)
            mask = ~torch.eye(n, dtype=torch.bool, device=concat.device)
            avg_dist = dists[mask].mean()
            scale = (concat.shape[1] ** 0.5) / (avg_dist + 1e-8)
        else:
            scale = 1.0

    feat_norm = concat * scale
    splits = []
    start = 0
    for f in features:
        end = start + f.shape[0]
        splits.append(feat_norm[start:end])
        start = end
    return splits


def _prepare_features(
    x: torch.Tensor,
    encoder: Optional[nn.Module],
    use_pixel_space: bool,
    use_spatial_features: bool,
) -> List[torch.Tensor]:
    """Return a list of feature tensors; optionally keep spatial locations (Appendix A.5)."""
    if use_pixel_space or encoder is None:
        feats = x.flatten(start_dim=1).float()
        return [feats]

    feat_out = encoder(x.float())
    feat_list = [feat_out] if isinstance(feat_out, torch.Tensor) else list(feat_out)

    feats: List[torch.Tensor] = []
    for f in feat_list:
        if use_spatial_features:
            # Keep spatial grids or token sequences intact; flatten happens later
            if f.dim() in (3, 4):
                feats.append(f)
            elif f.dim() == 2:
                feats.append(f)
            else:
                feats.append(f.view(f.shape[0], -1))
        else:
            if f.dim() == 4:
                feats.append(F.adaptive_avg_pool2d(f, 1).flatten(1))
            elif f.dim() == 3:
                feats.append(f.mean(dim=1))
            elif f.dim() == 2:
                feats.append(f)
            else:
                feats.append(f.view(f.shape[0], -1))
    return feats


def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    feature_encoder: Optional[nn.Module],
    temperatures: list,
    use_pixel_space: bool = False,
    use_spatial_features: bool = False,
    x_uncond_neg: Optional[torch.Tensor] = None,
    neg_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute class-conditional drifting loss with feature/drift normalization."""
    device = x_gen.device
    num_classes = labels_gen.max().item() + 1

    feat_gen_list = _prepare_features(x_gen, feature_encoder, use_pixel_space, use_spatial_features)
    with torch.no_grad():
        feat_pos_list = _prepare_features(x_pos, feature_encoder, use_pixel_space, use_spatial_features)
        feat_uncond_list = _prepare_features(x_uncond_neg, feature_encoder, use_pixel_space, use_spatial_features) if x_uncond_neg is not None else None

    total_loss = 0.0
    total_drift_norm = 0.0
    num_losses = 0

    for c in range(num_classes):
        mask_gen = labels_gen == c
        mask_pos = labels_pos == c

        if not mask_gen.any() or not mask_pos.any():
            continue

        for feat_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):
            feat_gen_c = feat_gen[mask_gen]
            feat_pos_c = feat_pos[mask_pos]
            if feat_gen_c.numel() == 0 or feat_pos_c.numel() == 0:
                continue

            # Flatten spatial locations to per-token features if applicable
            def _flatten_tokens(feat: torch.Tensor) -> torch.Tensor:
                if feat.dim() == 4:
                    B, C, H, W = feat.shape
                    return feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
                if feat.dim() == 3:
                    B, N, C = feat.shape
                    return feat.reshape(B * N, C)
                return feat

            feat_gen_tokens = _flatten_tokens(feat_gen_c)
            feat_pos_tokens = _flatten_tokens(feat_pos_c)

            # Negatives are the conditional batch itself (Alg. 1) plus optional unconditional
            feat_neg_tokens = feat_gen_tokens
            if feat_uncond_list is not None and neg_weight > 0:
                if feat_idx >= len(feat_uncond_list):
                    continue
                feat_uncond = feat_uncond_list[feat_idx]
                feat_uncond = _flatten_tokens(feat_uncond)
                repeat = max(1, int(round(neg_weight)))
                feat_neg_tokens = torch.cat([feat_neg_tokens, feat_uncond.repeat(repeat, 1)], dim=0)

            norm_gen, norm_pos, norm_neg = _normalize_feature_block([
                feat_gen_tokens,
                feat_pos_tokens,
                feat_neg_tokens,
            ])

            # Drifting field (multi-temperature, already normalized per-temp)
            dim_scale = norm_gen.shape[1] ** 0.5
            temps_scaled = [tau * dim_scale for tau in temperatures]

            # Per-paper: normalize each V_tau, then sum; do not renormalize the sum
            V_total = compute_V_multi_temperature(
                norm_gen,
                norm_pos,
                norm_neg,
                temps_scaled,
                mask_self=True,
                normalize_each=True,
                self_mask_count=norm_gen.shape[0],
            ).float()

            target = (norm_gen + V_total).detach()
            loss_c = F.mse_loss(norm_gen, target)

            total_loss = total_loss + loss_c * norm_gen.shape[0]
            total_drift_norm += torch.mean(V_total ** 2).item() ** 0.5 * norm_gen.shape[0]
            num_losses += norm_gen.shape[0]

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
