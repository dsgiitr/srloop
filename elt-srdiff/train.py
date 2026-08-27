import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from config import SRDiffELTConfig
from data import create_dataloaders
from model import ELTSR
from diffusion import DiffusionSchedule
from ema import EMA

def train():
    is_distributed = "RANK" in os.environ
    if is_distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        world_size = 1
        rank = 0

    device = torch.device(f"cuda:{local_rank}")
    config = SRDiffELTConfig()

    train_loader, val_loader, sampler = create_dataloaders(config, rank=rank, world_size=world_size)
    model = ELTSR(config).to(device)
    ema = EMA(model, decay=config.ema_decay).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    schedule = DiffusionSchedule(timesteps=config.num_timesteps)
    model.train()

    if rank == 0:
        print(f"--- Initializing ELT-SRDiff Training (FAST MODE) ---")

    for epoch in range(config.epochs):
        if sampler:
            sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}") if rank == 0 else train_loader

        for batch in pbar:
            optimizer.zero_grad()

            i_base = batch["i_base"].to(device, non_blocking=True)
            i_lq = batch["i_lq"].to(device, non_blocking=True)
            residual = batch["residual"].to(device, non_blocking=True)

            t = torch.randint(0, config.num_timesteps, (residual.shape[0],), device=device).long()
            x_t, noise = schedule.q_sample(residual, t)

            dit = model.module if is_distributed else model
            out = dit(x_t, i_base, i_lq, t, l_int=None)

            loss = F.l1_loss(out["eps_teacher"], noise)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(dit)

            if rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if rank == 0 and (epoch + 1) % config.eval_freq == 0:
            ckpt_path = f"srdiff_elt_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": (model.module if is_distributed else model).state_dict(),
                "ema_state_dict": ema.state_dict(),
            }, ckpt_path)

            ema.ema_model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                v_lq, v_base, v_hq = val_batch["i_lq"][:8].to(device), val_batch["i_base"][:8].to(device), val_batch["i_hq"][:8].to(device)
                pred_res = schedule.sample_sr(ema.ema_model, v_base, v_lq, v_hq.shape, device, num_loops=config.max_loops)
                pred_hq = torch.clamp(v_base + pred_res, -1.0, 1.0)

                grid = torch.cat([(v_base + 1.0) / 2.0, (pred_hq + 1.0) / 2.0, (v_hq + 1.0) / 2.0], dim=0)
                save_image(grid, f"sample_epoch_{epoch+1}.png", nrow=8)
            model.train()

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
