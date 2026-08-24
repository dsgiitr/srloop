import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips
from config import SRDiffELTConfig
from data import CIFAR10SRDataset
from model import ELTSR
from diffusion import DiffusionSchedule

@torch.no_grad()
def run_evaluation(checkpoint_path: str, eval_L: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = SRDiffELTConfig()

    model = ELTSR(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt)))
    model.eval()

    schedule = DiffusionSchedule(timesteps=config.num_timesteps)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net="alex").to(device)

    dataset = CIFAR10SRDataset(root=config.data_dir, train=False, download=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    psnr_list, ssim_list, lpips_list = [], [], []

    print(f"\n--- Running Evaluation: Loop Count L={eval_L} ---")
    for batch in tqdm(loader, desc="Evaluating Test Set"):
        i_lq, i_base, i_hq = batch["i_lq"].to(device), batch["i_base"].to(device), batch["i_hq"].to(device)
        pred_res = schedule.sample_sr(model, i_base, i_lq, i_hq.shape, device, num_loops=eval_L)
        sr_01 = (torch.clamp(i_base + pred_res, -1.0, 1.0) + 1.0) / 2.0
        hq_01 = (i_hq + 1.0) / 2.0

        psnr_list.append(psnr_metric(sr_01, hq_01).item())
        ssim_list.append(ssim_metric(sr_01, hq_01).item())
        lpips_list.append(lpips_metric(sr_01 * 2 - 1, i_hq).mean().item())

    print(f" PSNR: {np.mean(psnr_list):.2f} dB | SSIM: {np.mean(ssim_list):.4f} | LPIPS: {np.mean(lpips_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--L", type=int, default=3)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.L)