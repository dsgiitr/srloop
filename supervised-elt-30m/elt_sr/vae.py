"""
VAE Wrapper for ELT-SR Latent Diffusion
=======================================
Wraps the Hugging Face diffusers AutoencoderKL.
"""

import torch
import torch.nn as nn

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None


class VAEWrapper(nn.Module):
    """Frozen VAE for encoding images to latents and decoding back to images."""
    
    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse", device: str = "cuda"):
        super().__init__()
        if AutoencoderKL is None:
            raise ImportError("Please install diffusers to use the VAE: pip install diffusers transformers")
            
        print(f"Loading VAE from {model_id}...")
        self.vae = AutoencoderKL.from_pretrained(model_id).to(device)
        self.vae.eval()
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # Standard SD scaling factor
        self.scaling_factor = 0.18215
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latents.
        
        Args:
            x: Image tensor [B, 3, H, W] in [0, 1]
            
        Returns:
            Latent tensor [B, 4, H/8, W/8]
        """
        # VAE expects inputs in [-1, 1]
        x = x * 2.0 - 1.0
        
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.sample()
        latents = latents * self.scaling_factor
        return latents
        
    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.
        
        Args:
            latents: Latent tensor [B, 4, H/8, W/8]
            
        Returns:
            Image tensor [B, 3, H, W] in [0, 1]
        """
        latents = latents / self.scaling_factor
        images = self.vae.decode(latents).sample
        
        # Map back to [0, 1]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images
