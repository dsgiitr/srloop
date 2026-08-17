"""
Training Script for ELT-SR
==========================
Main training loop implementing:
  - Intra-Loop Self Distillation (ILSD)
  - Stochastic Student Sampling (S³)
  - λ curriculum
  - EMA
  - Automatic Mixed Precision (AMP) & Gradient Accumulation
  - Visual Checkpoint Sampling
  - Hopper H100 Accelerations (torch.compile, bfloat16, TF32)
"""

import os
import shutil
import torch
import torch.nn.functional as F
from typing import Optional
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from elt_sr.model import create_elt_sr
from elt_sr.diffusion import (
    shifted_cosine_schedule,
    q_sample,
    compute_ilsd_loss,
    get_lambda,
    ddpm_sample_loop,
    ddim_sample_loop,
)
from elt_sr.data import create_dataloaders
from elt_sr.ema import EMA
from elt_sr.vae import VAEWrapper
from accelerate import Accelerator


def create_arrow_tensor(height: int, width: int = 40, bg_color=(30, 30, 30), arrow_color=(220, 220, 220)) -> torch.Tensor:
    """Generate an RGB tensor [3, H, W] containing a right-pointing arrow."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Calculate arrow dimensions
    cy = height // 2
    arrow_len = int(width * 0.6)
    start_x = int(width * 0.2)
    end_x = start_x + arrow_len
    head_size = max(4, int(min(height * 0.15, width * 0.2)))

    # Draw arrow shaft
    shaft_width = max(2, height // 32)
    draw.line([(start_x, cy), (end_x, cy)], fill=arrow_color, width=shaft_width)

    # Draw arrow head
    arrow_head = [
        (end_x, cy),
        (end_x - head_size, cy - head_size),
        (end_x - head_size, cy + head_size)
    ]
    draw.polygon(arrow_head, fill=arrow_color)

    to_tensor = transforms.ToTensor()
    return to_tensor(img)


def find_latest_checkpoint(output_dir: str) -> Optional[Path]:
    """Find the checkpoint file with the highest epoch number in output_dir."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    ckpt_files = list(output_path.glob("elt_sr_ep*.pt"))
    if not ckpt_files:
        return None
    
    def extract_epoch(path: Path) -> int:
        try:
            stem = path.stem
            return int(stem.replace("elt_sr_ep", ""))
        except ValueError:
            return -1

    ckpt_files.sort(key=extract_epoch, reverse=True)
    latest = ckpt_files[0]
    return latest if extract_epoch(latest) >= 0 else None


def train(
    config_path: str = None,
    train_dir: str = "",
    val_dir: str = None,
    output_dir: str = "checkpoints",
    use_synthetic: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    resume_from: Optional[str] = None,
    auto_resume: bool = True,
    **kwargs
):
    """Main training loop."""
    from elt_sr.config import ELTConfig
    
    config = ELTConfig.load(config_path) if config_path else ELTConfig()
    # Override config with kwargs if provided
    for k, v in kwargs.items():
        if hasattr(config, k) and v is not None:
            setattr(config, k, v)
            
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = Path(output_dir) / "samples"
    os.makedirs(samples_dir, exist_ok=True)

    use_bfloat16 = kwargs.get("bfloat16", True) and torch.cuda.is_bf16_supported()
    accelerator = Accelerator(mixed_precision="bf16" if use_bfloat16 else "fp16")
    device = accelerator.device

    if accelerator.is_main_process:
        config.save(os.path.join(output_dir, "config.json"))

    # 0. H100 / CUDA Speed Optimizations
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Setup VAE (if configured)
    vae = None
    if getattr(config, "use_vae", False):
        print("Initializing VAE...")
        vae = VAEWrapper(device=device)

    # 2. Setup Model & Optimizers
    raw_model = create_elt_sr(config)
    ema = EMA(raw_model, decay=config.ema_decay).to(device)
    optimizer = AdamW(raw_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Enable PyTorch 2.0 torch.compile for H100 kernel fusion (2x-3x speedup!)
    model = raw_model
    use_compile = kwargs.get("compile", True) and hasattr(torch, "compile") and device.type == "cuda"
    if use_compile:
        print("Compiling model with torch.compile() for Hopper H100 kernel fusion...")
        try:
            model = torch.compile(raw_model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile failed, falling back to standard execution: {e}")
            


    # 3. Checkpoint Resumption (Restore state if available)
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    
    ckpt_path_to_load = None
    if resume_from and os.path.exists(resume_from):
        ckpt_path_to_load = Path(resume_from)
    elif auto_resume:
        # Prefer the rolling latest checkpoint to avoid losing progress
        latest_ckpt = Path(output_dir) / "elt_sr_latest.pt"
        best_ckpt = Path(output_dir) / "elt_sr_best.pt"
        
        if latest_ckpt.exists():
            ckpt_path_to_load = latest_ckpt
        elif best_ckpt.exists():
            ckpt_path_to_load = best_ckpt
        else:
            fallback_ckpt = find_latest_checkpoint(output_dir)
            if fallback_ckpt:
                ckpt_path_to_load = fallback_ckpt

    if ckpt_path_to_load and os.path.exists(ckpt_path_to_load):
        print(f"Found checkpoint to resume from: {ckpt_path_to_load}")
        ckpt = torch.load(ckpt_path_to_load, map_location=device)
        
        if "model_state_dict" in ckpt:
            m_state = ckpt["model_state_dict"]
            # Clean _orig_mod. prefix if present
            m_state = {k.replace("_orig_mod.", ""): v for k, v in m_state.items()}
            raw_model.load_state_dict(m_state)

        if "ema_state_dict" in ckpt:
            ema_state = ckpt["ema_state_dict"]
            # Clean _orig_mod. prefix if present
            ema_state = {k.replace("_orig_mod.", ""): v for k, v in ema_state.items()}
            ema.load_state_dict(ema_state)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Successfully resumed state: Starting at epoch {start_epoch + 1}/{config.epochs} (global step {global_step}, best loss {best_loss:.4f}).")


    # 4. Setup Diffusion Schedule
    T = config.num_timesteps
    schedule = shifted_cosine_schedule(
        num_timesteps=T, 
        shift=config.schedule_shift,
        cosine_s=config.schedule_cosine_s
    )
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"].to(device)
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    # 5. Setup Data
    latent_file = kwargs.get("latent_file", "latents_ffhq_128.pt")
    if not getattr(config, "use_vae", False) or not os.path.exists(latent_file):
        latent_file = None

    max_train_images = kwargs.get("max_train_images", None)
    train_loader, val_loader = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        latent_file=latent_file,
        hq_size=config.img_size,
        scale=config.scale,
        batch_size=config.batch_size,
        use_synthetic=use_synthetic,
        max_train_images=max_train_images,
        val_split_size=getattr(config, "val_split_size", 2000),
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    grad_accum_steps = getattr(config, "grad_accum_steps", 1)
    total_steps = config.epochs * (len(train_loader) // grad_accum_steps)

    if accelerator.is_main_process: print(f"Starting training on {device}...")
    if accelerator.is_main_process: print(f"Total steps: {total_steps}, Epochs: {config.epochs}, AMP Dtype: {'bfloat16' if use_bfloat16 else 'float16'}")
    if latent_file:
        if accelerator.is_main_process: print(f"Using pre-encoded latents from {latent_file} (Zero VAE latency during training)!")

    # 6. Training Loop
    model.train()
    for epoch in range(start_epoch, config.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", disable=not accelerator.is_main_process)
        for step_idx, batch in enumerate(pbar):
            # Get data with async non_blocking transfers
            if "z_hq" in batch and "z_base" in batch:
                # Pre-encoded latents mode (Fastest)
                target = batch["z_hq"].to(device, non_blocking=True)
                cond = batch["z_base"].to(device, non_blocking=True)
            elif vae is not None:
                # On-the-fly VAE encoding mode
                i_hq = batch["i_hq"].to(device, non_blocking=True)
                i_base = batch["i_base"].to(device, non_blocking=True)
                with torch.no_grad():
                    z_hq = vae.encode(i_hq)
                    z_base = vae.encode(i_base)
                target = z_hq
                cond = z_base
            else:
                target = batch["residual"].to(device, non_blocking=True)
                cond = batch["i_base"].to(device, non_blocking=True)

            B = target.shape[0]

            # Sample random timesteps
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            # Generate noise and forward diffusion (q-sample)
            noise = torch.randn_like(target)
            x_t = q_sample(
                x_0=target,
                t=t,
                noise=noise,
                sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            )

            # S³: Sample random intermediate student loop (L_int)
            l_int = torch.randint(1, config.max_loops, (1,)).item()
            
            # Compute current λ for curriculum
            lam = get_lambda(global_step, max(1, total_steps))

            # Forward pass
            outputs = model(x_t=x_t, i_base=cond, t=t, l_int=l_int)

            eps_teacher = outputs["eps_teacher"]
            eps_student = outputs["eps_student"]

            # Compute ILSD Loss
            loss_dict = compute_ilsd_loss(
                eps_teacher=eps_teacher,
                eps_student=eps_student,
                noise=noise,
                t=t,
                num_timesteps=T,
                lam=lam,
            )
            loss = loss_dict["loss_total"] / grad_accum_steps

            accelerator.backward(loss)
            
            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                ema.update(accelerator.unwrap_model(model))
                global_step += 1

            # Logging
            loss_val = loss.item() * grad_accum_steps
            epoch_loss += loss_val
            
            if step_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "t_loss": f"{loss_dict['loss_gt_teacher'].item():.4f}",
                    "s_loss": f"{loss_dict['loss_gt_student'].item():.4f}",
                    "lam": f"{lam:.2f}",
                })

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config.epochs} | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save best model whenever average loss reaches a new minimum
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_ckpt_path = Path(output_dir) / "elt_sr_best.pt"
            raw_model_state = accelerator.unwrap_model(model).state_dict()
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "model_state_dict": raw_model_state,
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, best_ckpt_path)
            print(f"[*] New best model saved to {best_ckpt_path} with loss: {best_loss:.4f}!")
            # Auto-backup to persistent cache
            cache_dir = "/kaggle/working/elt_cache/checkpoints"
            if os.path.exists("/kaggle/working/elt_cache"):
                os.makedirs(cache_dir, exist_ok=True)
                shutil.copy2(str(best_ckpt_path), f"{cache_dir}/elt_sr_best.pt")

        # Save a rolling "latest" checkpoint every 5 epochs
        if accelerator.is_main_process and (epoch + 1) % 5 == 0:
            latest_ckpt_path = Path(output_dir) / "elt_sr_latest.pt"
            raw_model_state = accelerator.unwrap_model(model).state_dict()
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "model_state_dict": raw_model_state,
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, latest_ckpt_path)
            print(f"[*] Rolling checkpoint saved to {latest_ckpt_path}")
            if os.path.exists("/kaggle/working/elt_cache"):
                cache_dir = "/kaggle/working/elt_cache/checkpoints"
                os.makedirs(cache_dir, exist_ok=True)
                shutil.copy2(str(latest_ckpt_path), f"{cache_dir}/elt_sr_latest.pt")
                
            # Keep a persistent milestone checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                milestone_name = f"elt_sr_ep{epoch + 1}.pt"
                milestone_ckpt_path = Path(output_dir) / milestone_name
                shutil.copy2(str(latest_ckpt_path), str(milestone_ckpt_path))
                print(f"[*] Milestone checkpoint saved to {milestone_ckpt_path}")
                if os.path.exists("/kaggle/working/elt_cache"):
                    shutil.copy2(str(milestone_ckpt_path), f"{cache_dir}/{milestone_name}")

        # Visual sampling on validation/train batch every 10 epochs (main process only)
        if accelerator.is_main_process and ((epoch + 1) % 10 == 0 or epoch == config.epochs - 1) and vae is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.eval()
            eval_loader = val_loader if val_loader is not None else train_loader
            if eval_loader is not None:
                val_batch = next(iter(eval_loader))
                num_samples = 5
                
                with torch.no_grad():
                    if "z_base" in val_batch:
                        # Pre-encoded latents mode
                        z_base_val = val_batch["z_base"][:num_samples].to(device)
                        actual_n = z_base_val.shape[0]
                        val_base = vae.decode(z_base_val)
                    elif "i_base" in val_batch:
                        # Image mode
                        val_base = val_batch["i_base"][:num_samples].to(device)
                        actual_n = val_base.shape[0]
                        z_base_val = vae.encode(val_base)
                    else:
                        actual_n = 0

                    if actual_n > 0:
                        # MUST use EMA model for sampling! Raw weights are too noisy.
                        z_pred_hq = ddim_sample_loop(
                            ema.ema_model, z_base_val, schedule, ddim_steps=50, num_loops=config.max_loops, device=device, verbose=False
                        )
                        img_pred = vae.decode(z_pred_hq)

                        # Create arrow tensors for the batch
                        _, _, H, W = val_base.shape
                        arrow_w = max(24, int(W * 0.35))
                        arrow_tensor = create_arrow_tensor(height=H, width=arrow_w).to(device)
                        arrow_batch = arrow_tensor.unsqueeze(0).repeat(actual_n, 1, 1, 1)

                        # Row: [Low Quality (Bicubic)] | [Arrow] | [Model Generated HQ]
                        row_grid = torch.cat([val_base, arrow_batch, img_pred], dim=-1)
                        
                        save_path = samples_dir / f"ep{epoch+1}_sample.png"
                        save_image(row_grid, save_path, nrow=1, normalize=True)
                        print(f"Saved 5-sample visual grid (LQ -> Arrow -> HQ) to {save_path}")
            unwrapped_model.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ELT-SR")
    parser.add_argument("--train_dir", type=str, default="thumbnails128x128")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--use_synthetic", action="store_true")
    args = parser.parse_args()

    train(
        config_path=args.config,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        use_synthetic=args.use_synthetic,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
