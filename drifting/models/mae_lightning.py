import lightning as L
import torch
from drifting.models.feature_encoder import MultiScaleFeatureEncoder, MAEEncoder
from drifting.utils.vae_utils import VAEManager
from torch import nn
import torch.nn.functional as F

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

        ########### Optional classifier fine-tune (Appendix A.6) ###########
        self.finetune_classifier = self.mae_config.get("finetune_classifier", False)
        if self.finetune_classifier:
            num_classes = config.get("num_classes", 10)
            feat_dim = self.mae_config.get("feature_dim", 512)
            self.cls_head = nn.Linear(feat_dim, num_classes)
            self.lambda_max = self.mae_config.get("lambda_max", 0.1)
            self.lambda_warmup = self.mae_config.get("lambda_warmup_steps", 1000)

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
            labels = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            labels = None

        if self.use_latent:
            # x (B, 8, H, W)
            mean, _ = torch.chunk(x, 2, dim=1)
            x = (mean - self.vae_manager.mean.to(self.device)) / self.vae_manager.std.to(self.device)

        loss_recon, pred, mask = self.mae(x)

        if self.finetune_classifier and labels is not None:
            feats = self.encoder(x)
            logits = self.cls_head(feats)
            loss_ce = F.cross_entropy(logits, labels)
            lam = min(self.lambda_max, self.lambda_max * (self.global_step + 1) / max(1, self.lambda_warmup))
            loss = lam * loss_ce + (1 - lam) * loss_recon
            self.log("mae_cls_loss", loss_ce, prog_bar=True)
            self.log("mae_lambda", lam, prog_bar=False)
        else:
            loss = loss_recon

        self.log("mae_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.mae.parameters(), 
            lr=self.mae_config.get("lr", 1e-3), 
            weight_decay=self.mae_config.get("weight_decay", 0.05)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.mae_config.get("epochs", 50)
        )
        return [optimizer], [scheduler]