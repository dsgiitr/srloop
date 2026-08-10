"""
Launcher script for ELT-SR 30M model training.
==============================================
Usage:
    python run_train.py --train_dir thumbnails128x128 --batch_size 32 --epochs 100
"""

import argparse
from elt_sr.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ELT-SR (30M Parameters)")
    parser.add_argument("--train_dir", type=str, default="thumbnails128x128", help="Path to HQ images folder")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to val images folder (optional)")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config JSON (optional)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--latent_file", type=str, default="latents_ffhq_128.pt", help="Path to pre-encoded latents .pt file")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic dataset for testing")

    args = parser.parse_args()

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
    )

