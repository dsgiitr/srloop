"""
Data Pipeline for ELT-SR
==========================
Handles:
  - Loading HQ images
  - Bicubic degradation: HQ → LQ (↓2×) and LQ → I_base (↑2×)
  - Residual computation: R = I_HQ − I_base
  - Data augmentation (horizontal flip, random crop)
  - Train/val split
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Degradation Utilities
# ---------------------------------------------------------------------------

def bicubic_downsample(img: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Bicubic downsample an image tensor.

    Args:
        img: [C, H, W] or [B, C, H, W] image tensor in [0, 1].
        scale: Downsampling factor.

    Returns:
        Downsampled image tensor.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    _, C, H, W = img.shape
    out = F.interpolate(img, size=(H // scale, W // scale), mode="bicubic", align_corners=False)
    out = torch.clamp(out, 0.0, 1.0)

    if squeeze:
        out = out.squeeze(0)
    return out


def bicubic_upsample(img: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Bicubic upsample an image tensor.

    Args:
        img: [C, H, W] or [B, C, H, W] image tensor in [0, 1].
        scale: Upsampling factor.

    Returns:
        Upsampled image tensor.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    _, C, H, W = img.shape
    out = F.interpolate(img, size=(H * scale, W * scale), mode="bicubic", align_corners=False)
    out = torch.clamp(out, 0.0, 1.0)

    if squeeze:
        out = out.squeeze(0)
    return out


def compute_residual(i_hq: torch.Tensor, i_base: torch.Tensor) -> torch.Tensor:
    """Compute the residual R = I_HQ - I_base.

    Args:
        i_hq: [B, C, H, W] or [C, H, W] high-quality image.
        i_base: Same shape, bicubic-upsampled LQ.

    Returns:
        Residual tensor (can be negative).
    """
    return i_hq - i_base


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SRDataset(Dataset):
    """Super-Resolution dataset.

    Loads HQ images from a directory, generates LQ/I_base/Residual pairs on-the-fly.

    Supports:
      - Standard image directories (recursively finds .png, .jpg, .jpeg, .bmp)
      - Random 32×32 crops from larger images
      - Data augmentation (horizontal flip, vertical flip, rotation)

    The pipeline for each sample:
      1. Load HQ image → random crop to `hq_size` × `hq_size`
      2. Bicubic downsample by `scale` → LQ image
      3. Bicubic upsample LQ to HQ size → I_base
      4. R = I_HQ - I_base
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root_dir: str,
        hq_size: int = 32,
        scale: int = 2,
        augment: bool = True,
        max_images: Optional[int] = None,
    ):
        """
        Args:
            root_dir: Path to directory containing HQ images.
            hq_size: Target HQ crop size.
            scale: SR scale factor.
            augment: Whether to apply data augmentation.
            max_images: If set, limit dataset size (for debugging).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.hq_size = hq_size
        self.scale = scale
        self.augment = augment

        # Find all image files
        self.image_paths = self._find_images()
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

        print(f"SRDataset: Found {len(self.image_paths)} images in {root_dir}")

        # Base transform: convert to tensor [0, 1]
        self.to_tensor = transforms.ToTensor()

    def _find_images(self) -> List[Path]:
        """Recursively find all image files in root_dir."""
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(self.root_dir.rglob(f"*{ext}"))
            paths.extend(self.root_dir.rglob(f"*{ext.upper()}"))
        paths = sorted(set(paths))
        return paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """Load and process a single sample.

        Returns:
            Dictionary with:
              - 'i_hq': [3, 32, 32] HQ image in [0, 1].
              - 'i_lq': [3, 16, 16] LQ image in [0, 1].
              - 'i_base': [3, 32, 32] bicubic-upsampled LQ in [0, 1].
              - 'residual': [3, 32, 32] R = I_HQ - I_base.
        """
        # Load image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.to_tensor(img)  # [3, H, W], values in [0, 1]

        # Ensure image is large enough, but in this task we want to 
        # specifically downscale the 128x128 FFHQ images directly to hq_size (32x32)
        _, H, W = img.shape
        if H != self.hq_size or W != self.hq_size:
            img = F.interpolate(
                img.unsqueeze(0), size=(self.hq_size, self.hq_size), mode="bicubic", align_corners=False
            ).squeeze(0)
            img = torch.clamp(img, 0.0, 1.0)

        # The HQ image is the downscaled 32x32 image
        i_hq = img

        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                i_hq = torch.flip(i_hq, dims=[2])
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                i_hq = torch.flip(i_hq, dims=[1])
            # Random 90° rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                i_hq = torch.rot90(i_hq, k, dims=[1, 2])

        # Generate LQ via bicubic downsample
        i_lq = bicubic_downsample(i_hq, self.scale)  # [3, 16, 16]

        # Generate I_base via bicubic upsample
        i_base = bicubic_upsample(i_lq, self.scale)  # [3, 32, 32]

        # Compute residual
        residual = compute_residual(i_hq, i_base)  # [3, 32, 32]

        return {
            "i_hq": i_hq,
            "i_lq": i_lq,
            "i_base": i_base,
            "residual": residual,
        }


# ---------------------------------------------------------------------------
# Synthetic Dataset (for testing without real images)
# ---------------------------------------------------------------------------

class SyntheticSRDataset(Dataset):
    """Synthetic dataset for smoke testing.

    Generates random "images" and processes them through the SR pipeline.
    Useful for verifying the training loop works end-to-end.
    """

    def __init__(self, num_samples: int = 1000, hq_size: int = 32, scale: int = 2):
        self.num_samples = num_samples
        self.hq_size = hq_size
        self.scale = scale

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Generate random HQ image
        i_hq = torch.rand(3, self.hq_size, self.hq_size)

        # Degradation pipeline
        i_lq = bicubic_downsample(i_hq, self.scale)
        i_base = bicubic_upsample(i_lq, self.scale)
        residual = compute_residual(i_hq, i_base)

        return {
            "i_hq": i_hq,
            "i_lq": i_lq,
            "i_base": i_base,
            "residual": residual,
        }


class PreencodedLatentDataset(Dataset):
    """Dataset serving pre-encoded VAE latents stored in memory/file.

    Entire dataset is ~546 MB, which loads into RAM in ~1-2 seconds.
    Completely eliminates VAE encoding and PNG decoding bottlenecks during training.
    """

    def __init__(self, latent_path: str, max_samples: Optional[int] = None):
        super().__init__()
        self.latent_path = Path(latent_path)
        if not self.latent_path.exists():
            raise FileNotFoundError(f"Latent file {latent_path} not found.")

        print(f"Loading pre-encoded latents from {latent_path}...")
        data = torch.load(self.latent_path, map_location="cpu")
        self.z_hq = data["z_hq"]
        self.z_base = data["z_base"]

        if max_samples is not None and max_samples < len(self.z_hq):
            self.z_hq = self.z_hq[:max_samples]
            self.z_base = self.z_base[:max_samples]

        print(f"PreencodedLatentDataset loaded: {len(self.z_hq)} samples into RAM.")

    def __len__(self) -> int:
        return len(self.z_hq)

    def __getitem__(self, idx: int) -> dict:
        return {
            "z_hq": self.z_hq[idx],
            "z_base": self.z_base[idx],
        }


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    latent_file: Optional[str] = None,
    hq_size: int = 128,
    scale: int = 8,
    batch_size: int = 64,
    num_workers: int = 4,
    max_train_images: Optional[int] = None,
    use_synthetic: bool = False,
    val_split_size: int = 2000,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders.

    Supports:
      - Direct pre-encoded latents (.pt file) for zero-latency training
      - On-the-fly image loading from directory
      - Synthetic testing mode
    """
    if latent_file is not None and os.path.exists(latent_file):
        full_dataset = PreencodedLatentDataset(latent_file, max_samples=max_train_images)
        if val_split_size > 0 and len(full_dataset) > val_split_size:
            val_len = val_split_size
            train_len = len(full_dataset) - val_len
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"Split pre-encoded latents into {train_len} train and {val_len} validation samples.")
        else:
            train_dataset = full_dataset
            val_dataset = None
    elif use_synthetic:
        num_syn = max_train_images if max_train_images is not None else 10000
        train_dataset = SyntheticSRDataset(num_samples=num_syn, hq_size=hq_size, scale=scale)
        val_dataset = SyntheticSRDataset(num_samples=min(num_syn, 500), hq_size=hq_size, scale=scale)
    else:
        full_dataset = SRDataset(
            train_dir, hq_size=hq_size, scale=scale,
            augment=True, max_images=max_train_images,
        )
        if val_dir is not None:
            train_dataset = full_dataset
            val_dataset = SRDataset(
                val_dir, hq_size=hq_size, scale=scale,
                augment=False,
            )
        elif val_split_size > 0 and len(full_dataset) > val_split_size:
            val_len = val_split_size
            train_len = len(full_dataset) - val_len
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_len, val_len],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"Split dataset into {train_len} train and {val_len} validation samples.")
        else:
            train_dataset = full_dataset
            val_dataset = None

    # Pre-encoded latents already live in RAM, so num_workers=0 or 2 is optimal
    nw = 0 if isinstance(train_dataset, (PreencodedLatentDataset, torch.utils.data.Subset)) and latent_file else num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )

    return train_loader, val_loader


