import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ELTConfig:
    """Configuration for the ELT-SR system."""
    
    # --- Model Architecture ---
    use_vae: bool = True
    img_size: int = 128       # Pixel image size (128 -> 16x16 latents with f=8 VAE)
    latent_size: int = 16     # Latent dimension (img_size // 8)
    patch_size: int = 1       # UPDATED: 16/1 = 16x16 grid = 256 tokens (crucial for spatial fidelity)
    in_channels: int = 4      # VAE latent channels
    cond_channels: int = 4    # VAE latent channels for conditioning
    hidden_dim: int = 512     # UPDATED: Scaled up for facial priors
    num_heads: int = 8        # UPDATED: 64-dim per head
    mlp_dim: int = 2048       # UPDATED: 4x hidden_dim
    num_blocks: int = 6       # N: Number of unique transformer blocks
    max_loops: int = 4        # UPDATED: L_max: Maximum loop count for teacher
    min_loops: int = 1        # L_min: Minimum loop count for student
    
    # --- Diffusion ---
    num_timesteps: int = 1000
    schedule_shift: float = 1.0
    schedule_cosine_s: float = 0.008
    
    # --- Training Optimizations & Control ---
    batch_size: int = 64
    epochs: int = 100         # With 30M params, convergence will be much faster
    lr: float = 1e-4
    weight_decay: float = 0.03 # UPDATED: Added for regularization at 30M scale
    ema_decay: float = 0.999  # UPDATED: Faster tracking for looped gradients
    use_amp: bool = True       # Automatic Mixed Precision for lower VRAM & higher speed
    grad_accum_steps: int = 1  # Gradient accumulation steps
    
    # --- Data Pipeline ---
    scale: int = 8            # UPDATED: 128 / 16 = 8x Super-resolution scale factor
    val_split_size: int = 2000# Validation split size from dataset

    
    def save(self, path: str):
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=4)
            
    @classmethod
    def load(cls, path: str) -> "ELTConfig":
        """Load configuration from a JSON file."""
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found. Using defaults.")
            return cls()
        with open(path, 'r') as f:
            data = json.load(f)
        # Filter out keys that might be from older config versions
        valid_keys = {k for k in cls.__dataclass_fields__.keys()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
