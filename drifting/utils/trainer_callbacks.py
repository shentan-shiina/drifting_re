from lightning.pytorch.callbacks import Callback
from drifting.utils.utils import EMA, save_image_grid
from drifting.scripts.sample import generate_samples

class EMACallback(Callback):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema = None

    def setup(self, trainer, pl_module, stage):
        # 1. Initialize EMA on CPU (so checkpoint can load into it)
        if self.ema is None:
            self.ema = EMA(pl_module.model, decay=self.decay)

    def on_train_start(self, trainer, pl_module):
        # 2. NEW: Move EMA shadow to the correct GPU before training begins!
        if self.ema is not None and hasattr(self.ema, "shadow"):
            self.ema.shadow.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update EMA after every optimization step
        if outputs is not None: # Ensures we don't update if queue wasn't ready
            self.ema.update(pl_module.model)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])

class SamplingCallback(Callback):
    def __init__(self, sample_interval, config):
        self.sample_interval = sample_interval
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.sample_interval == 0:
            # Get the EMA model from the EMACallback
            ema_callback = [c for c in trainer.callbacks if isinstance(c, EMACallback)][0]
            ema_model = ema_callback.ema.shadow
            
            # Generate samples
            samples = generate_samples(
                ema_model,
                self.config["num_classes"] * self.config["batch_n_pos"],
                self.config["in_channels"],
                self.config["img_size"],
                self.config["num_classes"],
                pl_module.device,
                labels=None,
                use_cfg=False
            )
            
            sample_path = f"{trainer.default_root_dir}/samples_epoch_{epoch}.png"
            save_image_grid(samples, sample_path, nrow=self.config["batch_n_pos"])