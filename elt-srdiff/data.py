import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from datasets import load_dataset

class HFCelebASRDataset(Dataset):
    def __init__(self, hf_data, img_size=32):
        self.dataset = hf_data
        self.transform_hq = transforms.Compose([
            transforms.CenterCrop(140),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform_lq = transforms.Resize((img_size // 2, img_size // 2), interpolation=transforms.InterpolationMode.BICUBIC)
        self.transform_base = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC)

    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]["image"].convert("RGB")
        i_hq = self.transform_hq(img)
        i_lq = self.transform_lq(i_hq)
        i_base = self.transform_base(i_lq)
        return {"i_hq": i_hq, "i_lq": i_lq, "i_base": i_base, "residual": i_hq - i_base}

def create_dataloaders(config, rank=0, world_size=1):
    raw_dataset = load_dataset("nielsr/CelebA-faces", split="train")
    split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
    
    # 50k subset to match CIFAR-10 baseline timing exactly
    train_subset = split_dataset["train"].select(range(50000))
    val_subset = split_dataset["test"].select(range(5000))
    
    train_data = HFCelebASRDataset(train_subset, img_size=config.img_size)
    val_data = HFCelebASRDataset(val_subset, img_size=config.img_size)
    
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=(train_sampler is None), 
        sampler=train_sampler, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_data, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler
