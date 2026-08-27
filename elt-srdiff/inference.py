import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from datasets import load_dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from config import SRDiffELTConfig
from model import ELTSR
from diffusion import DiffusionSchedule
from data import HFCelebASRDataset
from ema import EMA

def evaluate_model(checkpoint_path, num_batches=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting L=1 Inference on {device} ---")

    # 1. Initialize Architecture
    config = SRDiffELTConfig()
    base_model = ELTSR(config).to(device)
    ema = EMA(base_model, decay=config.ema_decay).to(device)

    # 2. Load Weights from Local Working Directory
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "ema_state_dict" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        model = ema.ema_model
        print("Successfully loaded EMA weights (Highest Quality).")
    else:
        base_model.load_state_dict(checkpoint["model_state_dict"])
        model = base_model
        print("Loaded standard model weights.")
    
    model.eval()
    schedule = DiffusionSchedule(timesteps=config.num_timesteps)

    # 3. Prepare Dataset (Test Split)
    print("Fetching CelebA validation set...")
    raw_dataset = load_dataset("nielsr/CelebA-faces", split="train")
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    val_subset = split_dataset["test"].select(range(5000))
    val_data = HFCelebASRDataset(val_subset, img_size=config.img_size)

    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)

    # 4. Initialize Metrics (Strictly [0, 1] range)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0

    print(f"\nEvaluating {num_batches} batches for L=1 Ablation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, total=num_batches)):
            if i >= num_batches:
                break

            v_lq = batch["i_lq"].to(device)
            v_base = batch["i_base"].to(device)
            v_hq = batch["i_hq"].to(device)

            # Iterative Denoising (L=1 naturally triggers a single pass per timestep here)
            pred_res = schedule.sample_sr(model, v_base, v_lq, v_hq.shape, device, num_loops=config.max_loops)
            pred_hq = torch.clamp(v_base + pred_res, -1.0, 1.0)

            # Convert from [-1, 1] to [0, 1] to satisfy torchmetrics
            pred_hq_01 = torch.clamp((pred_hq + 1.0) / 2.0, 0.0, 1.0)
            v_hq_01 = torch.clamp((v_hq + 1.0) / 2.0, 0.0, 1.0)
            v_base_01 = torch.clamp((v_base + 1.0) / 2.0, 0.0, 1.0)

            # Calculate batch metrics
            total_psnr += psnr_metric(pred_hq_01, v_hq_01).item()
            total_ssim += ssim_metric(pred_hq_01, v_hq_01).item()
            total_lpips += lpips_metric(pred_hq_01, v_hq_01).item()

            if i == 0:
                grid = torch.cat([v_base_01, pred_hq_01, v_hq_01], dim=0)
                save_image(grid, "final_L1_eval_grid.png", nrow=16)

    # 5. Final Results
    print("\n=== Final L=1 Evaluation Metrics ===")
    print(f"PSNR:  {total_psnr / num_batches:.4f}")
    print(f"SSIM:  {total_ssim / num_batches:.4f}")
    print(f"LPIPS: {total_lpips / num_batches:.4f}")
    print("Saved 'final_L1_eval_grid.png' to the working directory.")

if __name__ == "__main__":
    # Pulls directly from the working directory where train.py saves it
    MODEL_PATH = "srdiff_elt_epoch_500.pt"
    evaluate_model(MODEL_PATH, num_batches=20)
