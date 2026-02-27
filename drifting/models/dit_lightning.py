import lightning as L
import torch
from drifting.models.dit_drift import DriftDiT_models
from drifting.models.feature_encoder import create_feature_encoder
from drifting.utils.utils import SampleQueue, WarmupLRScheduler
from drifting.utils.train_utils import sample_batch, compute_drifting_loss

from drifting.utils.vae_utils import VAEManager

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
        if self.use_latent:
            self.vae_manager = VAEManager(vae_id="stabilityai/sd-vae-ft-mse")

        ########### Model Init ###########
        model_fn = DriftDiT_models[config["model"]]
        self.model = model_fn(
            img_size=config["img_size"],
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            label_dropout=config["label_dropout"],
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
            self.feature_encoder = create_feature_encoder(
                dataset=config["name"],
                feature_dim=512,
                multi_scale=True,
                use_pretrained=True,
            )
            self.feature_encoder.eval()
            for param in self.feature_encoder.parameters():
                param.requires_grad = False

    def forward(self, x, labels, alpha, force_drop_ids=None):
        return self.model(x, labels, alpha, force_drop_ids)

    def training_step(self, batch, batch_idx):
            if isinstance(batch, (list, tuple)):
                x_real, labels_real = batch[0], batch[1]
            else:
                x_real = batch
                labels_real = torch.zeros(x_real.shape[0], dtype=torch.long, device=self.device)

            ########### Preprocess Latent ###########
            if self.use_latent:
                # Sample from the (B, 8, H, W) distribution and normalize
                x_real = self.vae_manager.sample_and_normalize(x_real)

            self.queue.add(x_real.cpu(), labels_real.cpu())

            # Skip step if queue isn't ready
            if not self.queue.is_ready(self.config["batch_n_pos"]):
                return None 
            
            num_classes = self.config["num_classes"]
            n_pos = self.config["batch_n_pos"]
            n_neg = self.config["batch_n_neg"]
            alpha_min = self.config["alpha_min"]
            alpha_max = self.config["alpha_max"]
            temperatures = self.config["temperatures"]
            use_pixel = not self.config["use_feature_encoder"]
            batch_size = num_classes * n_neg

            ########### Generate Samples ###########
            labels = torch.arange(num_classes, device=self.device).repeat_interleave(n_neg)
            alpha = torch.empty(batch_size, device=self.device).uniform_(alpha_min, alpha_max)
            noise = torch.randn(
                batch_size,
                self.config["in_channels"],
                self.config["img_size"],
                self.config["img_size"],
                device=self.device,
            )

            # Forward Pass
            x_gen = self.model(noise, labels, alpha)
            x_pos, labels_pos = sample_batch(self.queue, num_classes, n_pos, self.device)

            ########### Metrics ###########
            loss, info = compute_drifting_loss(
                x_gen,
                labels,
                x_pos,
                labels_pos,
                self.feature_encoder,
                temperatures,
                use_pixel_space=use_pixel,
            )

            self.log("train_loss", loss, prog_bar=True)
            if "drift_norm" in info:
                self.log("drift_norm", info["drift_norm"], prog_bar=True)

            return loss

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
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
