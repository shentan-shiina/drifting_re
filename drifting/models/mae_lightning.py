import lightning as L
import torch
from drifting.models.feature_encoder import MultiScaleFeatureEncoder, MAEEncoder
from drifting.utils.vae_utils import VAEManager

#######################################################
#                MAE LightningModule                  #
#  (Manage train loop for MAE in feature_encoder.py)  #
#######################################################
class MAEPretrainModule(L.LightningModule):
    """Lightning wrapper for MAE Pre-training."""
    
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.mae_config = config.get("mae",None)

        in_channels = config.get("in_channels", 3)
        base_width = self.mae_config.get("base_width", 256)

        ########### Base Encoder ###########
        self.encoder = MultiScaleFeatureEncoder(
            in_channels=in_channels,
            base_width=base_width,
            blocks_per_stage=2,
            feature_dim=self.mae_config.get("feature_dim", 512),
            multi_scale=self.mae_config.get("multi_scale", True),
        )
        
        ########### MAE Wrapper ###########
        self.mae = MAEEncoder(
            feature_encoder=self.encoder,
            in_channels=in_channels,
            img_size=config.get("img_size", 32),
            patch_size=self.mae_config.get("patch_size", 4),
            mask_ratio=self.mae_config.get("mask_ratio", 0.75),
        )

        ########### VAE for latent space training ###########
        self.use_latent = config.get("use_latent", False)
        if self.use_latent:
            self.vae_manager = VAEManager(vae_id="stabilityai/sd-vae-ft-mse")
            self.vae_manager.eval()
            for param in self.vae_manager.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        if self.use_latent:
            # x (B, 8, H, W)
            mean, _ = torch.chunk(x, 2, dim=1)
            x = (mean - self.vae_manager.mean.to(self.device)) / self.vae_manager.std.to(self.device)

        loss, pred, mask = self.mae(x)
        self.log("mae_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.mae.parameters(), 
            lr=self.mae_config.get("lr", 1e-3), 
            weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.mae_config.get("epochs", 50)
        )
        return [optimizer], [scheduler]