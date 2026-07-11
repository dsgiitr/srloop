from datasets import load_dataset
from torch.utils.data import Dataset

class HFCifar10(Dataset):
    def __init__(self, split='train', transform=None, cache_dir=None):
        self.ds = load_dataset('uoft-cs/cifar10', split=split, cache_dir=cache_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item['img'].convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item['label']
