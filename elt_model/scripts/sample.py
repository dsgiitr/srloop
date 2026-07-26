"""
Sampling script for the ELT model.
Demonstrates Any-Time inference capability by varying `custom_num_loops`.
"""

import os
import torch
import torchvision
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elt.model import ElasticLoopedTransformer
from utils.diffusion import make_beta_schedule, GaussianDiffusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="elt_model_final.pt")
    parser.add_argument("--num_loops_infer", type=int, default=10, help="Number of loops to run at inference time (Any-Time).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--class_label", type=int, default=0, help="Class label to generate.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Model (Must match training config)
    model = ElasticLoopedTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=256,
        num_heads=4,
        num_classes=10,
        num_loops=10, # Max loops from training
        num_blocks_per_loop=1
    ).to(device)

    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded checkpoint from {args.ckpt}")
    else:
        print("No checkpoint found. Generating with uninitialized weights.")

    model.eval()

    # 2. Setup Diffusion
    betas = make_beta_schedule("linear", num_timesteps=1000)
    diffusion = GaussianDiffusion(betas)

    # 3. Sampling
    print(f"Sampling {args.batch_size} images with {args.num_loops_infer} loops...")
    y = torch.tensor([args.class_label] * args.batch_size, device=device)
    shape = (args.batch_size, 3, 32, 32)
    
    samples = diffusion.sample_loop(
        model, 
        shape=shape, 
        y=y, 
        custom_num_loops=args.num_loops_infer,
        device=device
    )

    # 4. Save Image Grid
    # Samples are in [-1, 1], normalize to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    os.makedirs("samples", exist_ok=True)
    out_path = f"samples/generated_loops_{args.num_loops_infer}.png"
    torchvision.utils.save_image(samples, out_path, nrow=4)
    print(f"Saved generated samples to {out_path}")

if __name__ == "__main__":
    main()
