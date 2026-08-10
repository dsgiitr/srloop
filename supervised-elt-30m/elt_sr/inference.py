"""
Inference Script for ELT-SR
===========================
Runs reverse diffusion to restore LQ images.
Supports Any-Time inference (L=1, L=2, L=3).
"""

import os
import torch
import torchvision
from pathlib import Path
from PIL import Image

from elt_sr.model import create_elt_sr
from elt_sr.diffusion import (
    shifted_cosine_schedule,
    ddpm_sample_loop,
    ddim_sample_loop,
    reconstruct_hq,
)
from elt_sr.data import bicubic_downsample, bicubic_upsample
from elt_sr.vae import VAEWrapper
from torchvision import transforms


def load_model(checkpoint_path: str, config=None, device: str = "cuda") -> torch.nn.Module:
    """Load model from checkpoint (prefer EMA weights if available)."""
    model = create_elt_sr(config).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "ema_state_dict" in ckpt:
            print("Using EMA weights.")
            model.load_state_dict(ckpt["ema_state_dict"])
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
            
    model.eval()
    return model


@torch.no_grad()
def infer(
    image_path: str,
    config_path: str = None,
    checkpoint_path: str = None,
    output_dir: str = "results",
    num_loops: int = 3,
    sampler: str = "ddim",
    ddim_steps: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Run inference on a single image.
    
    Args:
        image_path: Path to the input image. (Can be HQ or LQ).
                    If it's > 16x16, we'll downsample it to simulate LQ.
        checkpoint_path: Path to trained checkpoint.
        output_dir: Where to save results.
        num_loops: How many ELT loops to use (1, 2, or 3).
        sampler: 'ddpm' or 'ddim'.
        ddim_steps: Steps for DDIM sampler.
        device: Device to run on.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device)
    
    from elt_sr.config import ELTConfig
    config = ELTConfig.load(config_path) if config_path else ELTConfig()
    
    vae = None
    if getattr(config, "use_vae", False):
        print("Initializing VAE...")
        vae = VAEWrapper(device=device)

    # 1. Load model and schedule
    model = load_model(checkpoint_path, config=config, device=device)
    schedule = shifted_cosine_schedule(
        num_timesteps=config.num_timesteps,
        shift=config.schedule_shift,
        cosine_s=config.schedule_cosine_s
    )
    
    # 2. Prepare Input
    img = Image.open(image_path).convert("RGB")
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    _, _, H, W = img_t.shape
    
    if H == config.img_size and W == config.img_size:
        # Assume it's an HQ image, simulate degradation
        print(f"Input is {config.img_size}x{config.img_size}. Simulating LQ degradation...")
        i_hq = img_t
        i_lq = bicubic_downsample(i_hq, scale=config.scale)
    elif H == config.img_size // config.scale and W == config.img_size // config.scale:
        # It's already LQ
        i_hq = None
        i_lq = img_t
    else:
        # Resize to img_size and simulate degradation
        print(f"Input is {H}x{W}. Resizing to {config.img_size}x{config.img_size} and simulating degradation...")
        i_hq = torch.nn.functional.interpolate(img_t, size=(config.img_size, config.img_size), mode='bicubic', align_corners=False)
        i_hq = torch.clamp(i_hq, 0, 1)
        i_lq = bicubic_downsample(i_hq, scale=config.scale)

    # Base image for residual diffusion (or latent conditioning)
    i_base = bicubic_upsample(i_lq, scale=config.scale)
    
    if vae is not None:
        with torch.no_grad():
            cond = vae.encode(i_base)
    else:
        cond = i_base
    
    # 3. Sample
    print(f"Running inference with L={num_loops} using {sampler.upper()}...")
    if sampler.lower() == "ddim":
        pred = ddim_sample_loop(
            model=model,
            i_base=cond,
            schedule=schedule,
            ddim_steps=ddim_steps,
            num_loops=num_loops,
            device=device
        )
    else:
        pred = ddpm_sample_loop(
            model=model,
            i_base=cond,
            schedule=schedule,
            num_loops=num_loops,
            device=device
        )
        
    # 4. Reconstruct HQ
    if vae is not None:
        print("Decoding latents with VAE...")
        i_hq_pred = vae.decode(pred)
    else:
        i_hq_pred = reconstruct_hq(i_base, pred)
    
    # 5. Save results
    base_name = Path(image_path).stem
    out_path = Path(output_dir) / f"{base_name}_L{num_loops}_{sampler}_pred.png"
    
    # Create a grid: [LQ (upsampled to show blockiness), Base (Bicubic), Pred, HQ (if avail)]
    i_lq_vis = torch.nn.functional.interpolate(i_lq, size=(config.img_size, config.img_size), mode='nearest')
    
    vis_list = [i_lq_vis, i_base, i_hq_pred]
    if i_hq is not None:
        vis_list.append(i_hq)
        
    grid = torchvision.utils.make_grid(torch.cat(vis_list, dim=0), nrow=len(vis_list), padding=2)
    torchvision.utils.save_image(grid, out_path)
    
    print(f"Saved result grid to {out_path}")
    print("Grid order: [Nearest LQ] | [Bicubic I_base] | [Predicted I_HQ] | [Ground Truth HQ]")


if __name__ == "__main__":
    # Create a dummy image and run smoke test
    print("Running inference smoke test...")
    from elt_sr.config import ELTConfig
    config = ELTConfig()
    dummy_img = torch.rand(3, config.img_size, config.img_size)
    dummy_path = "dummy_test.png"
    torchvision.utils.save_image(dummy_img, dummy_path)
    
    try:
        infer(
            image_path=dummy_path,
            output_dir="test_results",
            num_loops=3,
            sampler="ddim",
            ddim_steps=5,  # Few steps for quick test
            device="cpu"   # Ensure it runs anywhere
        )
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
