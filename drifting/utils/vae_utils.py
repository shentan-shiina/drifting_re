"""
VAE (SD-VAE) utils for Drifting training.
This code is revised from iMeanFlow <https://github.com/Lyy-iiis/imeanflow>
Handles dataset existence checks and transform construction to avoid
unnecessary downloads during repeated runs.
"""


import os
import torch
import torch.utils.data
from diffusers.models import AutoencoderKL

class VAEManager(torch.nn.Module):
    def __init__(self, vae_id="stabilityai/sd-vae-ft-mse"):
        """
        PyTorch equivalent of the JAX LatentManager.
        Uses standard HF diffusers VAE.
        """
        super().__init__()
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(vae_id,
                                                 cache_dir="/home/ubt/.cache/huggingface/hub")
        self.vae.eval()
        self.vae.requires_grad_(False)

        # Register scaling parameters as buffers so Lightning moves them to the correct device
        # Reshaped from (1, -1, 1, 1) to match PyTorch's (B, C, H, W) format
        self.register_buffer(
            "mean", torch.tensor([0.86488, -0.27787343, 0.21616915, 0.3738409]).view(1, 4, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([4.85503674, 5.31922414, 3.93725398, 3.9870003]).view(1, 4, 1, 1)
        )

    def sample_and_normalize(self, cached_value: torch.Tensor) -> torch.Tensor:
        """Equivalent to JAX cached_encode."""
        # Ensure channel-first format (B, C, H, W)
        if cached_value.shape[-1] == 8:
            cached_value = cached_value.permute(0, 3, 1, 2)
            
        # Split into mean and std
        mean, std = torch.chunk(cached_value, 2, dim=1)
        
        # Sample (Reparameterization trick)
        noise = torch.randn_like(mean)
        latent = mean + std * noise
        
        # Normalize
        latent = (latent - self.mean) / self.std
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Denormalize and decode back to pixel space."""
        # Denormalize
        latents = latents * self.std + self.mean
        # Decode
        return self.vae.decode(latents).sample


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root, use_flip=False):
        self.root = root
        self.file_list = [file for file in os.listdir(self.root) if file.endswith(".pt")]
        self.use_flip = use_flip

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(file_path, weights_only=False)
        return data["image"], data["label"]