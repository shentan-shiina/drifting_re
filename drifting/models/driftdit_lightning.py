import lightning as L
import torch
from drifting.models.drift_dit import DriftDiT_models
from drifting.models.feature_encoder import create_feature_encoder
from drifting.utils.utils import SampleQueue, WarmupLRScheduler
from drifting.utils.train_utils import sample_batch_for_classes, sample_unconditional, compute_drifting_loss

from drifting.utils.vae_utils import VAEManager

from termcolor import cprint

#######################################################
#             DriftDiT LightningModule                #
#  (Manage train loop for DriftDiT in dit_drift.py)   #
#######################################################
class DriftDiTModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        self.use_latent = config.get("use_latent", True)
        mae_config = config.get("mae", {})
        if self.use_latent:
            self.vae_manager = VAEManager(vae_id="stabilityai/sd-vae-ft-mse")

        ########### Model Init ###########
        model_fn = DriftDiT_models[config["model"]]
        self.model = model_fn(
            img_size=config["img_size"],
            patch_size=config.get("patch_size", 4),
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            label_dropout=config["label_dropout"],
            num_register_tokens=config.get("num_register_tokens", 16),
        )
        
        ########### Queue Init ###########
        self.queue = SampleQueue(
            num_classes=config["num_classes"],
            queue_size=config["queue_size"],
            sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
        )
        
        ########### Feature Encoder Init ###########
        self.feature_encoder = None
        if config["use_feature_encoder"]:
            # Check if we should use custom MAE or ImageNet ResNet
            has_mae_ckpt = "mae_checkpoint_path" in config and config["mae_checkpoint_path"] is not None
            
            self.feature_encoder = create_feature_encoder(
                dataset=config["name"],
                in_channels=config["in_channels"],
                feature_dim=mae_config.get("feature_dim", 512),
                multi_scale=mae_config.get("multi_scale", True),
                base_width=mae_config.get("base_width", 256),
                blocks_per_stage=mae_config.get("blocks_per_stage", [3, 4, 6, 3]),
                input_patch_size=mae_config.get("mae_input_patch", 1),
                output_mode="multiscale",
                use_pretrained=not has_mae_ckpt, # Disable default ResNet if using MAE
                mae_checkpoint_path=config.get("mae_checkpoint_path", None)
            )
            
            self.feature_encoder.eval()
            for param in self.feature_encoder.parameters():
                param.requires_grad = False

        cprint(f"[Drift DiT Training] Initialized with settings:", "green")
        cprint(f"[Dataset]:", "green")
        cprint(f"  - name: {self.config['name']}", "green")
        cprint(f"  - use_latent: {self.config['use_latent']}", "green")
        cprint(f"  - use_feature_encoder: {self.config['use_feature_encoder']}", "green")
        cprint(f"  - use_spatial_features: {self.config['use_spatial_features']}", "green")
        cprint(f"  - img_size: {self.config['img_size']}", "green")
        cprint(f"  - patch_size: {self.config.get('patch_size', 4)}", "green")
        cprint(f"[Batch]:", "green")
        cprint(f"  - num_classes: {self.config['num_classes']}", "green")
        cprint(f"  - samples_per_class: {self.config['samples_per_class']}", "green")
        cprint(f"  - batch_n_pos: {self.config['batch_n_pos']}", "green")
        cprint(f"  - batch_n_neg: {self.config['batch_n_neg']}", "green")
        cprint(f"  - uncond_neg_samples: {self.config['uncond_neg_samples']}", "green")
        cprint(f"  - uncond_neg_weight: {self.config['uncond_neg_weight']}", "green")
        cprint(f"[Hyperparameters]:", "green")
        cprint(f"  - model: {self.config['model']}", "green")
        cprint(f"  - epochs: {self.config['epochs']}", "green")
        cprint(f"  - warmup_steps: {self.config['warmup_steps']}", "green")
        cprint(f"  - lr: {self.config['lr']}", "green")
        cprint(f"  - temperatures: {self.config['temperatures']}", "green")
        cprint(f"  - grad_clip: {self.config['grad_clip']}", "green")
        cprint(f"  - weight_decay: {self.config['weight_decay']}", "green")
        cprint(f"  - ema_decay: {self.config['ema_decay']}", "green")
        cprint(f"  - queue_size: {self.config['queue_size']}", "green")
        cprint(f"  - alpha_min: {self.config['alpha_min']}", "green")
        cprint(f"  - alpha_max: {self.config['alpha_max']}", "green")
        cprint(f"  - label_dropout: {self.config['label_dropout']}", "green")
    
    def forward(self, x, labels, alpha, force_drop_ids=None):
        return self.model(x, labels, alpha, force_drop_ids)

    def _sample_alpha(self, n: int, device: torch.device) -> torch.Tensor:
        """CFG alpha sampling: 50% at alpha=1, else power-law."""
        alpha_min = float(self.config["alpha_min"])
        alpha_max = float(self.config["alpha_max"])
        alpha_prob_one = float(self.config.get("alpha_prob_one", 0.5))
        alpha_power = float(self.config.get("alpha_power", 3.0))

        if alpha_max <= alpha_min:
            return torch.full((n,), alpha_min, device=device)

        alpha = torch.empty(n, device=device)
        use_one = torch.rand(n, device=device) < alpha_prob_one

        one_value = 1.0 if (alpha_min <= 1.0 <= alpha_max) else alpha_min
        alpha[use_one] = one_value

        m = int((~use_one).sum().item())
        if m > 0:
            u = torch.rand(m, device=device)
            if abs(alpha_power - 1.0) < 1e-8:
                vals = alpha_min * (alpha_max / alpha_min) ** u
            else:
                p = 1.0 - alpha_power
                a0 = alpha_min ** p
                a1 = alpha_max ** p
                vals = (u * (a1 - a0) + a0) ** (1.0 / p)
            alpha[~use_one] = vals

        return alpha

    def on_train_start(self):
        """Pre-fill the sample queue to avoid wasted early steps."""
        if self.queue.is_ready(self.config["batch_n_pos"]):
            return

        print("\nPre-filling sample queue before training starts...")
        train_loader = self.trainer.train_dataloader

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x_real, labels_real = batch[0], batch[1]
            else:
                x_real = batch
                labels_real = torch.zeros(x_real.shape[0], dtype=torch.long)

            if getattr(self, "use_latent", False):
                x_real = x_real.to(self.device)
                with torch.no_grad():
                    x_real = self.vae_manager.sample_and_normalize(x_real)

            self.queue.add(x_real.cpu(), labels_real.cpu())

            if self.queue.is_ready(self.config["batch_n_pos"]):
                print("Sample queue is fully pre-filled! Training will now begin.")
                break

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x_real, labels_real = batch[0], batch[1]
        else:
            x_real = batch
            labels_real = torch.zeros(x_real.shape[0], dtype=torch.long, device=self.device)

        if self.use_latent:
            x_real = self.vae_manager.sample_and_normalize(x_real)

        self.queue.add(x_real.cpu(), labels_real.cpu())

        if not self.queue.is_ready(self.config["batch_n_pos"]):
            return None

        num_classes = self.config["num_classes"]
        batch_nc = min(int(self.config.get("batch_nc", num_classes)), num_classes)
        n_pos = self.config["batch_n_pos"]
        n_neg = self.config["batch_n_neg"]
        temperatures = self.config["temperatures"]
        use_pixel = not self.config["use_feature_encoder"]
        use_spatial = self.config.get("use_spatial_features", True)
        batch_size = batch_nc * n_neg

        class_perm = torch.randperm(num_classes, device=self.device)
        selected_classes = class_perm[:batch_nc]
        labels = selected_classes.repeat_interleave(n_neg)
        alpha = self._sample_alpha(batch_size, self.device)
        noise = torch.randn(
            batch_size,
            self.config["in_channels"],
            self.config["img_size"],
            self.config["img_size"],
            device=self.device,
        )

        x_gen = self.model(noise, labels, alpha)
        x_pos, labels_pos = sample_batch_for_classes(self.queue, selected_classes, n_pos, self.device)

        x_uncond_neg = None
        if self.config.get("uncond_neg_samples", 0) > 0:
            x_uncond_neg = sample_unconditional(
                self.queue,
                self.config["uncond_neg_samples"],
                self.device,
            )

        loss, info = compute_drifting_loss(
            x_gen,
            labels,
            x_pos,
            labels_pos,
            self.feature_encoder,
            temperatures,
            use_pixel_space=use_pixel,
            use_spatial_features=use_spatial,
            x_uncond_neg=x_uncond_neg,
            neg_weight=self.config.get("uncond_neg_weight", 1.0),
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if "drift_norm" in info:
            self.log("drift_norm", info["drift_norm"], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_train_epoch_end(self):
            """Print the epoch averages calculated by PyTorch Lightning."""
            avg_loss = self.trainer.callback_metrics.get("train_loss_epoch")
            avg_drift = self.trainer.callback_metrics.get("drift_norm_epoch")

            if avg_loss is not None:
                print(f"\nEpoch {self.current_epoch} Avg Train Loss:   {avg_loss.item():.4f}")
                if avg_drift is not None:
                    print(f"Epoch {self.current_epoch} Avg Drift Norm:   {avg_drift.item():.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["lr"],
            betas=(0.9, 0.95),
            weight_decay=self.config["weight_decay"],
        )

        scheduler = WarmupLRScheduler(
            optimizer,
            warmup_steps=self.config["warmup_steps"],
            base_lr=self.config["lr"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
