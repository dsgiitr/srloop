import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms

class CIFAR10SRDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root: str, train: bool = True, download: bool = False):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        super().__init__(root, train=train, transform=transform, download=download)

    def __getitem__(self, index: int) -> dict:
        hr_img, _ = super().__getitem__(index)
        lr_img = F.interpolate(hr_img.unsqueeze(0), size=(16, 16), mode="bicubic", align_corners=False).squeeze(0)
        up_lr_img = F.interpolate(lr_img.unsqueeze(0), size=(32, 32), mode="bicubic", align_corners=False).squeeze(0)
        residual = hr_img - up_lr_img

        return {
            "i_hq": hr_img,
            "i_lq": lr_img,
            "i_base": up_lr_img,
            "residual": residual
        }

def create_dataloaders(config, rank: int = 0, world_size: int = 1):
    if world_size > 1:
        if rank == 0:
            torchvision.datasets.CIFAR10(root=config.data_dir, train=True, download=False)
            torchvision.datasets.CIFAR10(root=config.data_dir, train=False, download=False)
        dist.barrier()
    
    train_dataset = CIFAR10SRDataset(root=config.data_dir, train=True, download=False)
    val_dataset = CIFAR10SRDataset(root=config.data_dir, train=False, download=False)

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, sampler