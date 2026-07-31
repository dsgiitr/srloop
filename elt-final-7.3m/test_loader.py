import os, sys
sys.path.append(os.path.abspath('dit-model-implementation/dit'))

from configs.config import get_config
from train import center_crop_arr, CIFAR10Dataset
import torch
from datasets import load_dataset
from torchvision import transforms

config = get_config()
print(f"config.model.image_size = {config.model.image_size}")

transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, config.model.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])
hf_dataset = load_dataset('uoft-cs/cifar10', split='train')
dataset = CIFAR10Dataset(hf_dataset, transform=transform)

img, label = dataset[0]
print(f"Dataset image shape: {img.shape}")
