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
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from torchvision.utils import save_image

from elt_sr.model import create_elt_sr
from elt_sr.diffusion import (
    shifted_cosine_schedule,
    q_sample,
    compute_ilsd_loss,
    get_lambda,
    ddpm_sample_loop,
)
from elt_sr.data import create_dataloaders
from elt_sr.ema import EMA
from elt_sr.vae import VAEWrapper


def train(
    config_path: str = None,
    train_dir: str = "",
    val_dir: str = None,
    output_dir: str = "checkpoints",
    use_synthetic: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
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
    config.save(os.path.join(output_dir, "config.json"))
    device = torch.device(device)

    # 1. Setup VAE (if configured)
    vae = None
    if getattr(config, "use_vae", False):
        print("Initializing VAE...")
        vae = VAEWrapper(device=device)

    # 2. Setup Model & Optimizers
    model = create_elt_sr(config).to(device)
    ema = EMA(model, decay=config.ema_decay)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    use_amp = getattr(config, "use_amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    # 3. Setup Diffusion Schedule
    T = config.num_timesteps
    schedule = shifted_cosine_schedule(
        num_timesteps=T, 
        shift=config.schedule_shift,
        cosine_s=config.schedule_cosine_s
    )
    sqrt_alphas_cumprod = schedule["sqrt_alphas_cumprod"].to(device)
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"].to(device)

    # 4. Setup Data
    latent_file = kwargs.get("latent_file", "latents_ffhq_128.pt")
    if not os.path.exists(latent_file):
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

    grad_accum_steps = getattr(config, "grad_accum_steps", 1)
    total_steps = config.epochs * (len(train_loader) // grad_accum_steps)
    global_step = 0

    print(f"Starting training on {device}...")
    print(f"Total steps: {total_steps}, Epochs: {config.epochs}, AMP: {getattr(config, 'use_amp', True)}")
    if latent_file:
        print(f"Using pre-encoded latents from {latent_file} (Zero VAE latency during training)!")

    # 5. Training Loop
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for step_idx, batch in enumerate(pbar):
            # Get data
            if "z_hq" in batch and "z_base" in batch:
                # Pre-encoded latents mode (Fastest)
                target = batch["z_hq"].to(device)
                cond = batch["z_base"].to(device)
            elif vae is not None:
                # On-the-fly VAE encoding mode
                i_hq = batch["i_hq"].to(device)
                i_base = batch["i_base"].to(device)
                with torch.no_grad():
                    z_hq = vae.encode(i_hq)
                    z_base = vae.encode(i_base)
                target = z_hq
                cond = z_base
            else:
                target = batch["residual"].to(device)
                cond = batch["i_base"].to(device)

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
            l_int = torch.randint(1, model.max_loops, (1,)).item()
            
            # Compute current λ for curriculum
            lam = get_lambda(global_step, max(1, total_steps))

            # Mixed precision forward pass
            use_amp = getattr(config, "use_amp", True) and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
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

            # Backward pass with GradScaler
            scaler.scale(loss).backward()
            
            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
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
        
        # Save checkpoint periodically & generate visual samples
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            ckpt_path = Path(output_dir) / f"elt_sr_ep{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            # Visual sampling on validation batch
            if val_loader is not None and vae is not None:
                model.eval()
                val_batch = next(iter(val_loader))
                val_hq = val_batch["i_hq"][:4].to(device)
                val_base = val_batch["i_base"][:4].to(device)
                with torch.no_grad():
                    z_base_val = vae.encode(val_base)
                    z_pred_residual = ddpm_sample_loop(
                        model, z_base_val, schedule, num_loops=config.max_loops, device=device, verbose=False
                    )
                    z_pred_hq = z_base_val + z_pred_residual
                    img_pred = vae.decode(z_pred_hq)
                    
                    # Create comparison grid: [Bicubic | Prediction | GroundTruth]
                    grid = torch.cat([val_base, img_pred, val_hq], dim=-1) # side by side
                    save_path = samples_dir / f"ep{epoch+1}_sample.png"
                    save_image(grid, save_path, nrow=1, normalize=True)
                    print(f"Saved visual sample to {save_path}")
                model.train()


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

