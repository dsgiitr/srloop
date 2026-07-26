"""
Training script for the ELT model.
Optimized for multi-GPU (DataParallel) and Mixed Precision (Tensor Cores) for maximum speed.
"""

import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
import torchvision
from tqdm import tqdm
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elt.model import ElasticLoopedTransformer
from elt.distillation import ILSDLoss
from utils.diffusion import make_beta_schedule, GaussianDiffusion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    # Batch size 1024: CIFAR-10 has 50k images -> ~49 steps/epoch -> fast epochs
    parser.add_argument("--batch_size", type=int, default=1024) 
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_loops", type=int, default=10)
    parser.add_argument("--distillation_weight", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint and samples every N epochs")
    parser.add_argument("--resume_ckpt", type=str, default="elt_model_latest.pt", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Enable cudnn benchmark for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup Data: Download and load CIFAR-10 from Hugging Face
    print("Loading CIFAR-10 dataset from Hugging Face...")
    hf_dataset = load_dataset("cifar10", split="train")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    def hf_transform(examples):
        # Apply torchvision transforms to each PIL Image in the batch
        examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["img"]]
        return examples
    
    hf_dataset.set_transform(hf_transform)

    def custom_collate(batch):
        # The training loop expects tuples of (image, label)
        x = torch.stack([item["pixel_values"] for item in batch])
        y = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return x, y

    dataloader = DataLoader(hf_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)

    # 2. Setup Model (~5M params)
    model = ElasticLoopedTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        hidden_size=384,       # Mid-range: gives ~5M params
        num_heads=6,           # 384 / 6 = 64 per head (standard)
        num_classes=10,
        num_loops=args.num_loops,
        num_blocks_per_loop=2  # 2 blocks x 10 loops = 20 effective layers
    ).to(device)
    
    # Calculate and print parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params / 1e6:.2f} Million")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel!")
        model = torch.nn.DataParallel(model)

    # 3. Setup Diffusion
    betas = make_beta_schedule("linear", num_timesteps=1000)
    diffusion = GaussianDiffusion(betas)

    # 4. Setup Loss and Optimizer
    ilsd_loss_fn = ILSDLoss(distillation_weight=args.distillation_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    # 5. Resume from Checkpoint if exists
    start_epoch = 0
    if os.path.exists(args.resume_ckpt):
        print(f"Found checkpoint at '{args.resume_ckpt}'. Resuming training...")
        checkpoint = torch.load(args.resume_ckpt, map_location=device)
        
        # Unwrap DataParallel if necessary for loading
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    # 6. Training Loop
    print("Starting training...")
    model.train()
    for epoch in range(start_epoch, args.epochs):
        
        # Use tqdm for a beautiful progress bar!
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device).long()
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise=noise)

            optimizer.zero_grad()
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loop_preds = model(x_t, t, y, return_all_loops=True)
                loss, loss_dict = ilsd_loss_fn(loop_preds, noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update the progress bar with the current loss
            pbar.set_postfix({
                'Task Loss': f"{loss_dict['task_loss']:.4f}", 
                'Distill Loss': f"{loss_dict['distill_loss']:.4f}"
            })

        # Save Checkpoint & Generate Samples every N epochs
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            # 1. Save Weights
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            torch.save(checkpoint, args.resume_ckpt)
            print(f"--> Saved checkpoint at Epoch {epoch + 1} to {args.resume_ckpt}")
            
            # 2. Generate 20 Sample Images
            print(f"--> Generating 20 sample images for Epoch {epoch + 1}...")
            model.eval()
            with torch.no_grad():
                # Generate random labels for the 20 images
                sample_y = torch.randint(0, 10, (20,), device=device)
                
                # Start from pure noise and let the model generate images
                samples = diffusion.sample_loop(
                    model, shape=(20, 3, 32, 32), y=sample_y, 
                    custom_num_loops=args.num_loops,
                    device=device
                )
                
                # Un-normalize from [-1, 1] to [0, 1] for saving
                samples = (samples + 1) / 2
                samples = samples.clamp(0, 1)
                
                # Save the images in a grid
                os.makedirs("samples", exist_ok=True)
                sample_path = f"samples/epoch_{epoch+1}.png"
                torchvision.utils.save_image(samples, sample_path, nrow=5)
                print(f"--> Saved samples to {sample_path}")
                
            model.train() # Set back to training mode for the next epoch

    print("Training complete.")

if __name__ == "__main__":
    main()
