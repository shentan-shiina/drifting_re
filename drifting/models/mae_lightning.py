import lightning as L
import torch
import copy
from drifting.models.feature_encoder import MultiScaleFeatureEncoder, MAEEncoder
from drifting.utils.vae_utils import VAEManager
from torch import nn
import torch.nn.functional as F

from termcolor import cprint
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
        self.mae_config = config.get("mae", {})
        self.training_stage = "pretrain"
        self.use_ema = bool(self.mae_config.get("use_ema", True))
        self.ema_decay = float(self.mae_config.get("ema_decay", 0.9995))

        in_channels = config.get("in_channels", 3)
        base_width = self.mae_config.get("base_width", 256)
        img_size = int(config.get("img_size", 32))
        mae_input_patch = int(self.mae_config.get("mae_input_patch", 8 if img_size > 32 else 1))
        encoder_in_channels = in_channels * (mae_input_patch ** 2)

        ########### Base Encoder ###########
        self.encoder = MultiScaleFeatureEncoder(
            in_channels=encoder_in_channels,
            base_width=base_width,
            blocks_per_stage=self.mae_config.get("blocks_per_stage", [3, 4, 6, 3]),
            feature_dim=self.mae_config.get("feature_dim", 512),
            multi_scale=self.mae_config.get("multi_scale", True),
            input_patch_size=1,
            output_mode="projected",
        )
        
        ########### MAE Wrapper ###########
        self.mae = MAEEncoder(
            feature_encoder=self.encoder,
            in_channels=in_channels,
            img_size=img_size,
            input_patch_size=mae_input_patch,
            mask_block_size=self.mae_config.get("mask_block_size", 2),
            mask_prob=self.mae_config.get("mask_prob", 0.5),
        )

        if self.use_ema:
            self.ema_encoder = copy.deepcopy(self.encoder).eval()
            for p in self.ema_encoder.parameters():
                p.requires_grad = False

        ########### Optional classifier fine-tune (Appendix A.6) ###########
        self.finetune_classifier = self.mae_config.get("finetune_classifier", False)
        if self.finetune_classifier:
            num_classes = config.get("num_classes", 10)
            feat_dim = self.mae_config.get("feature_dim", 512)
            self.cls_head = nn.Linear(feat_dim, num_classes)
            self.lambda_max = self.mae_config.get("lambda_max", 0.1)
            self.lambda_warmup = self.mae_config.get("lambda_warmup_steps", 1000)
            self.finetune_steps = self.mae_config.get("finetune_steps", 3000)

        ########### VAE for latent space training ###########
        self.use_latent = config.get("use_latent", False)
        if self.use_latent:
            self.vae_manager = VAEManager(vae_id="stabilityai/sd-vae-ft-mse")
            self.vae_manager.eval()
            for param in self.vae_manager.parameters():
                param.requires_grad = False
    
        cprint(f"[MAE Training] Initialized with settings:", "green")
        cprint(f"[Dataset]:", "green")
        cprint(f"  - name: {self.config['name']}", "green")
        cprint(f"  - use_latent: {self.config['use_latent']}", "green")
        cprint(f"  - use_feature_encoder: {self.config['use_feature_encoder']}", "green")
        cprint(f"  - use_spatial_features: {self.config['use_spatial_features']}", "green")
        cprint(f"  - img_size: {self.config['img_size']}", "green")
        cprint(f"[Hyperparameters]:", "green")
        cprint(f"  - epochs: {self.mae_config['epochs']}", "green")
        cprint(f"  - lr: {self.mae_config.get('lr', 4e-3)}", "green")
        cprint(f"  - batch_size: {self.mae_config.get('batch_size', 'from global')}", "green")
        cprint(f"  - multi_scale: {self.mae_config.get('multi_scale', True)}", "green")
        cprint(f"  - weight_decay: {self.mae_config.get('weight_decay', 0.05)}", "green")
        cprint(f"  - use_ema: {self.use_ema}", "green")
        cprint(f"  - ema_decay: {self.ema_decay}", "green")
        cprint(f"  - finetune_classifier: {self.mae_config.get('finetune_classifier', False)}", "green")
        cprint(f"  - finetune_steps: {self.mae_config.get('finetune_steps', 3000)}", "green")
        cprint(f"  - finetune_lr: {self.mae_config.get('finetune_lr', self.mae_config.get('lr', 4e-3))}", "green")
        cprint(f"  - mae_input_patch: {self.mae_config.get('mae_input_patch', 1)}", "green")
        cprint(f"  - mask_block_size: {self.mae_config.get('mask_block_size', 2)}", "green")
        cprint(f"  - mask_prob: {self.mae_config.get('mask_prob', 0.5)}", "green")
        cprint(f"  - lambda_max: {self.mae_config.get('lambda_max', 0.1)}", "green")
        cprint(f"  - lambda_warmup_steps: {self.mae_config.get('lambda_warmup_steps', 1000)}", "green")
        

    def forward(self, x):
        return self.encoder.forward_projected(x)

    def set_training_stage(self, stage: str):
        if stage not in {"pretrain", "finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")
        self.training_stage = stage

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

        if self.training_stage == "finetune" and self.finetune_classifier and labels is not None:
            feats = self.encoder.forward_projected(x)
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

    @torch.no_grad()
    def _update_ema_encoder(self):
        if not self.use_ema:
            return
        for ema_p, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.training_stage == "pretrain":
            self._update_ema_encoder()

    def configure_optimizers(self):
        if self.training_stage == "finetune":
            lr = self.mae_config.get("finetune_lr", self.mae_config.get("lr", 4e-3))
            t_max = self.mae_config.get("finetune_steps", 3000)
            params = list(self.mae.parameters())
            if self.finetune_classifier:
                params += list(self.cls_head.parameters())
        else:
            lr = self.mae_config.get("lr", 4e-3)
            t_max = self.mae_config.get("epochs", 50)
            params = self.mae.parameters()

        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=self.mae_config.get("weight_decay", 0.05)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(t_max))
        )
        return [optimizer], [scheduler]