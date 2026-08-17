"""
Pre-encode Image Dataset to VAE Latents
========================================
Converts a folder of 128x128 HQ images into pre-computed VAE latents (z_hq and z_base).
Saves a single .pt file (~546 MB for 70k images) that loads directly into RAM for 18x faster training.

Usage:
    python prepare_latents.py --img_dir thumbnails128x128 --output_file latents_ffhq_128.pt
"""

import os
import torch
import argparse
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


from elt_sr.data import bicubic_downsample, bicubic_upsample
from elt_sr.vae import VAEWrapper


def prepare_latents(
    img_dir: str = "thumbnails128x128",
    output_file: str = "latents_ffhq_128.pt",
    scale: int = 8,
    batch_size: int = 128,
    max_images: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    img_path = Path(img_dir)
    if not img_path.exists():
        raise FileNotFoundError(f"Image directory {img_dir} not found.")

    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = []
    for ext in extensions:
        image_files.extend(list(img_path.glob(f"*{ext}")))
        image_files.extend(list(img_path.glob(f"*{ext.upper()}")))
    image_files = sorted(set(image_files))

    if max_images is not None:
        image_files = image_files[:max_images]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {img_dir}")

    print(f"Found {len(image_files)} images in {img_dir}.")
    print(f"Initializing VAE on {device}...")
    vae = VAEWrapper(device=device)

    to_tensor = transforms.ToTensor()
    z_hq_list = []
    z_base_list = []

    print(f"Pre-encoding dataset into latents (batch size {batch_size})...")
    
    num_batches = (len(image_files) + batch_size - 1) // batch_size
    for i in tqdm(range(num_batches), desc="Encoding Latents"):
        batch_files = image_files[i * batch_size : (i + 1) * batch_size]
        
        imgs = []
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            imgs.append(to_tensor(img))
            
        i_hq = torch.stack(imgs, dim=0).to(device)  # [B, 3, 128, 128]

        # Generate bicubic degraded base image
        i_lq = bicubic_downsample(i_hq, scale=scale)  # [B, 3, 16, 16]
        i_base = bicubic_upsample(i_lq, scale=scale)  # [B, 3, 128, 128]

        use_amp = (device == "cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            z_hq = vae.encode(i_hq).cpu()      # [B, 4, 16, 16] float32
            z_base = vae.encode(i_base).cpu()  # [B, 4, 16, 16] float32

        z_hq_list.append(z_hq)
        z_base_list.append(z_base)

    z_hq_all = torch.cat(z_hq_list, dim=0)    # [N, 4, 16, 16]
    z_base_all = torch.cat(z_base_list, dim=0)  # [N, 4, 16, 16]

    print(f"Encoded shapes: z_hq={z_hq_all.shape}, z_base={z_base_all.shape}")
    print(f"Saving pre-computed latents to {output_file}...")
    
    torch.save({
        "z_hq": z_hq_all,
        "z_base": z_base_all,
        "scale": scale,
        "num_samples": len(image_files),
    }, output_file)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Successfully saved {output_file} ({file_size_mb:.2f} MB)!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-encode dataset to VAE latents")
    parser.add_argument("--img_dir", type=str, default="thumbnails128x128")
    parser.add_argument("--output_file", type=str, default="latents_ffhq_128.pt")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    prepare_latents(
        img_dir=args.img_dir,
        output_file=args.output_file,
        scale=args.scale,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )

