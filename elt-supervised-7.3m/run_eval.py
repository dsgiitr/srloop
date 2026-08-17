import argparse
import os
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path

# Metrics
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# ELT-SR Imports
from elt_sr.config import ELTConfig
from elt_sr.model import create_elt_sr
from elt_sr.vae import VAEWrapper
from elt_sr.diffusion import shifted_cosine_schedule, ddim_sample_loop
from elt_sr.data import SRDataset
from torch.utils.data import DataLoader

class MetricsScorekeeper:
    def __init__(self, device: str = 'cuda', data_range: float = 1.0):
        self.device = torch.device(device)
        self.data_range = data_range
        
        # Initialize standard metrics
        self.psnr = PeakSignalNoiseRatio(data_range=self.data_range).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(self.device)
        
        # Initialize perceptual metric (LPIPS with VGG backbone)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg', 
            normalize=True
        ).to(self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Move to GPU/CPU
        preds = preds.to(self.device)
        target = target.to(self.device)
        
        # Clamp predictions to valid [0, 1] range to avoid math errors
        preds = torch.clamp(preds, 0.0, self.data_range)

        # Accumulate metrics
        self.psnr.update(preds, target)
        self.ssim.update(preds, target)
        self.lpips.update(preds, target)

    def compute(self):
        return {
            "PSNR": self.psnr.compute().item(),
            "SSIM": self.ssim.compute().item(),
            "LPIPS": self.lpips.compute().item()
        }

@torch.no_grad()
def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")
    
    # 1. Load Config & Model
    config = ELTConfig()
    if args.img_size is not None:
        config.img_size = args.img_size
    if args.scale is not None:
        config.scale = args.scale
        
    model = create_elt_sr(config).to(device)
    vae = VAEWrapper(device=device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Use EMA weights for evaluation if available
    if "ema_state_dict" in ckpt:
        m_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["ema_state_dict"].items()}
        print("Loaded EMA model weights.")
    else:
        m_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
        print("Loaded raw model weights (EMA not found).")
        
    model.load_state_dict(m_state)
    model.eval()
    
    # 2. Setup Data
    dataset = SRDataset(
        root_dir=args.val_dir,
        hq_size=config.img_size, # Must be exactly 32 so latents are 4x4
        scale=config.scale,
        augment=False
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 3. Setup Diffusion & Metrics
    schedule = shifted_cosine_schedule(
        num_timesteps=config.num_timesteps, 
        shift=config.schedule_shift, 
        cosine_s=config.schedule_cosine_s
    )
    scorekeeper = MetricsScorekeeper(device=device)
    
    # 4. Evaluation Loop
    total_batches = len(dataloader)
    if args.max_batches:
        total_batches = min(total_batches, args.max_batches)
        
    print(f"Starting evaluation over {total_batches} batches (Batch Size: {args.batch_size})")
    pbar = tqdm(dataloader, total=total_batches, desc="Evaluating")
    
    for i, batch in enumerate(pbar):
        if args.max_batches and i >= args.max_batches:
            break
            
        i_hq = batch["i_hq"].to(device)
        i_base = batch["i_base"].to(device)
        
        # Use Automatic Mixed Precision (FP16) for a massive speedup!
        with torch.autocast("cuda", dtype=torch.float16):
            # Encode LQ image to latents
            z_base = vae.encode(i_base)
            
            # Run diffusion
            z_pred_hq = ddim_sample_loop(
                model=model,
                i_base=z_base,
                schedule=schedule,
                ddim_steps=args.ddim_steps,
                num_loops=config.max_loops,
                device=device,
                verbose=False
            )
            
            # Decode predicted latents back to image space
            img_pred = vae.decode(z_pred_hq)
        
        # Update metrics (Metrics should ideally run in fp32 for stability)
        scorekeeper.update(preds=img_pred, target=i_hq)
        
        # Update progress bar occasionally
        if i % 5 == 0:
            current_metrics = scorekeeper.compute()
            pbar.set_postfix({
                "PSNR": f"{current_metrics['PSNR']:.2f}",
                "SSIM": f"{current_metrics['SSIM']:.3f}"
            })

    # 5. Final Results
    final_scores = scorekeeper.compute()
    print("\n" + "="*40)
    print("FINAL EVALUATION RESULTS")
    print("="*40)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"DDIM Steps: {args.ddim_steps}")
    print(f"PSNR:       {final_scores['PSNR']:.4f} dB (Higher is better)")
    print(f"SSIM:       {final_scores['SSIM']:.4f} (Higher is better)")
    print(f"LPIPS:      {final_scores['LPIPS']:.4f} (Lower is better)")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ELT-SR Model")
    parser.add_argument("--val_dir", type=str, required=True, help="Path to validation images folder (128x128)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/elt_sr_best.pt", help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM sampling steps")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit evaluation to N batches (for quick testing)")
    parser.add_argument("--img_size", type=int, default=None, help="Override target HR pixel size (e.g., 32, 64)")
    parser.add_argument("--scale", type=int, default=None, help="Override super-resolution scale (e.g., 2, 4)")
    
    args = parser.parse_args()
    evaluate_model(args)
