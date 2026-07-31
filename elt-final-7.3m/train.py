import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
import argparse
import logging
import os
from tqdm import tqdm

from ELT.configs.config import get_config
from ELT.models.elt import ELT
from ELT.diffusion import create_diffusion
from diffusers.models import AutoencoderKL

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class CIFAR10Dataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        img = item['img'].convert('RGB')
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
        
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    os.makedirs(args.results_dir, exist_ok=True)
    model_string_name = "ELT-Model"
    existing_dirs = sorted(glob(f"{args.results_dir}/*-{model_string_name}"))
    experiment_dir = None
    resume_checkpoint = None
    
    if existing_dirs:
        latest_dir = existing_dirs[-1]
        if os.path.exists(f"{latest_dir}/elt-best-model.pt"):
            experiment_dir = latest_dir
            resume_checkpoint = f"{latest_dir}/elt-best-model.pt"
            
    if experiment_dir is None:
        experiment_index = len(glob(f"{args.results_dir}/*"))
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory is {experiment_dir}")
    else:
        logger = create_logger(None)

    config = get_config()
    latent_size = config.model.image_size // 8
    model = ELT(config)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    logger.info(f"ELT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    
    start_epoch = 0
    train_steps = 0
    best_loss = float('inf')
    if resume_checkpoint:
        logger.info(f"Resuming from {resume_checkpoint}...")
        checkpoint = torch.load(resume_checkpoint, map_location=f"cuda:{device}", weights_only=False)
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        train_steps = checkpoint.get("train_steps", 0)
        best_loss = checkpoint.get("best_loss", float('inf'))
        logger.info(f"Resumed at epoch {start_epoch} with best_loss {best_loss:.4f}")

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.model.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    hf_dataset = load_dataset("uoft-cs/cifar10", split="train")
    dataset = CIFAR10Dataset(hf_dataset, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    total_steps = args.epochs * len(loader)
    logger.info(f"Training for {args.epochs} epochs (Total Steps: {total_steps})...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        epoch_running_loss = 0
        epoch_steps = 0
        
        if rank == 0:
            loader_iter = tqdm(loader, desc=f"Epoch {epoch}")
        else:
            loader_iter = loader
            
        for x, y in loader_iter:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            
            # Calculate linear decay for ilsd_weight (lambda curriculum)
            current_ilsd_weight = max(0.0, 1.0 - (train_steps / total_steps))

            # ELT specific model kwargs
            model_kwargs = dict(
                y=y, 
                ilsd_weight=current_ilsd_weight,
                return_ilsd_loops=True
            )
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update_ema(ema, model.module)

            epoch_running_loss += loss.item()
            train_steps += 1
            epoch_steps += 1
            
            if rank == 0:
                loader_iter.set_postfix({"loss": f"{loss.item():.4f}"})



        epoch_avg_loss = torch.tensor(epoch_running_loss / epoch_steps, device=device)
        dist.all_reduce(epoch_avg_loss, op=dist.ReduceOp.SUM)
        epoch_avg_loss = epoch_avg_loss.item() / dist.get_world_size()
        
        logger.info(f"Epoch {epoch} finished. Average Loss: {epoch_avg_loss:.4f}")
        
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch,
                    "train_steps": train_steps,
                    "best_loss": best_loss
                }
                best_checkpoint_path = f"{experiment_dir}/elt-best-model.pt"
                torch.save(checkpoint, best_checkpoint_path)
                logger.info(f"Saved best model with epoch loss {best_loss:.4f} to {best_checkpoint_path}")

        if epoch % 10 == 0 and rank == 0:
            logger.info(f"Generating samples at epoch {epoch}...")
            with torch.no_grad():
                sampling_diffusion = create_diffusion("250")
                latent_size = config.model.image_size // 8
                
                # Generate 20 samples (using random labels for variety)
                n = 20
                z = torch.randn(n, 4, latent_size, latent_size, device=device)
                y = torch.randint(0, config.model.num_classes, (n,), device=device)
                
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([config.model.num_classes] * n, device=device)
                y = torch.cat([y, y_null], 0)
                
                model_kwargs = dict(y=y, cfg_scale=4.0, loops=config.sampling.inference_loops)
                
                # Use EMA model for sampling
                samples = sampling_diffusion.p_sample_loop(
                    ema.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
                
                from torchvision.utils import save_image
                sample_path = f"{experiment_dir}/sample_epoch_{epoch}.png"
                save_image(samples, sample_path, nrow=5, normalize=True, value_range=(-1, 1))
                logger.info(f"Saved samples to {sample_path}")

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
