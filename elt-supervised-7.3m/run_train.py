"""
Launcher script for ELT-SR 30M model training.
==============================================
Usage:
    python run_train.py --train_dir thumbnails128x128 --batch_size 32 --epochs 100
"""

import argparse
import json
from elt_sr.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ELT-SR (30M Parameters)")
    parser.add_argument("--train_dir", type=str, default="thumbnails128x128", help="Path to HQ images folder")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to val images folder (optional)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config JSON (optional)")
    parser.add_argument("--img_size", type=int, default=None, help="Override target HR pixel size (e.g., 32, 64)")
    parser.add_argument("--scale", type=int, default=None, help="Override super-resolution scale (e.g., 2, 4)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--latent_file", type=str, default="latents_ffhq_128.pt", help="Path to pre-encoded latents .pt file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic dataset for testing")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable PyTorch 2.0 torch.compile for H100 kernel fusion")
    parser.add_argument("--bfloat16", action="store_true", default=True, help="Use bfloat16 mixed precision on H100")
    parser.add_argument("--max_train_images", type=int, default=None, help="Limit number of training images")
    parser.add_argument("--gpu_config", type=str, default=None, help="Path to GPU configuration JSON file")

    args = parser.parse_args()

    gpu_kwargs = {}
    if args.gpu_config:
        try:
            with open(args.gpu_config, 'r') as f:
                gpu_kwargs = json.load(f)
            print(f"Loaded GPU config from {args.gpu_config}: {gpu_kwargs}")
        except Exception as e:
            print(f"Failed to load gpu_config: {e}")

    train(
        config_path=args.config,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        latent_file=args.latent_file,
        output_dir=args.output_dir,
        use_synthetic=args.use_synthetic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        scale=args.scale,
        bfloat16=gpu_kwargs.get("use_bfloat16", args.bfloat16),
        compile=gpu_kwargs.get("use_compile", args.compile),
        max_train_images=args.max_train_images,
        **gpu_kwargs
    )
