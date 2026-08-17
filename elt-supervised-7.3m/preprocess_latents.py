"""
Pre-encode FFHQ images into VAE latents for zero-latency training.
=================================================================
This runs the VAE encoder ONCE on all 70,000 images and saves the
latent tensors (z_hq, z_base) to a single .pt file.

During training, the dataloader reads directly from RAM — no VAE,
no PNG decoding, no disk I/O bottleneck. Speeds up training ~3-5x.

Usage:
    python preprocess_latents.py --train_dir /path/to/ffhq --output latents_ffhq_128.pt
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from elt_sr.vae import VAEWrapper
from elt_sr.data import SRDataset


def preprocess(train_dir: str, output: str, hq_size: int = 32, scale: int = 2, batch_size: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load VAE
    print("Loading VAE...")
    vae = VAEWrapper(device=device)

    # 2. Load raw image dataset (no augmentation for deterministic encoding)
    dataset = SRDataset(train_dir, hq_size=hq_size, scale=scale, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Encode all images
    all_z_hq = []
    all_z_base = []

    print(f"Encoding {len(dataset)} images through VAE (batch_size={batch_size})...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding latents"):
            i_hq = batch["i_hq"].to(device, non_blocking=True)
            i_base = batch["i_base"].to(device, non_blocking=True)

            z_hq = vae.encode(i_hq)
            z_base = vae.encode(i_base)

            all_z_hq.append(z_hq.cpu())
            all_z_base.append(z_base.cpu())

    # 4. Concatenate and save
    z_hq_tensor = torch.cat(all_z_hq, dim=0)
    z_base_tensor = torch.cat(all_z_base, dim=0)

    print(f"z_hq shape: {z_hq_tensor.shape}")    # [70000, 4, 4, 4]
    print(f"z_base shape: {z_base_tensor.shape}")  # [70000, 4, 4, 4]
    print(f"Total size: {(z_hq_tensor.nbytes + z_base_tensor.nbytes) / 1e6:.1f} MB")

    torch.save({"z_hq": z_hq_tensor, "z_base": z_base_tensor}, output)
    print(f"Saved pre-encoded latents to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode FFHQ images into VAE latents")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to FFHQ images")
    parser.add_argument("--output", type=str, default="latents_ffhq_128.pt", help="Output .pt file")
    parser.add_argument("--hq_size", type=int, default=32)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    preprocess(args.train_dir, args.output, args.hq_size, args.scale, args.batch_size)
