import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from unet import Unet
from mlm_layer import MLMLayer
from maskgit import MaskGIT

def train_maskgit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = Unet(in_channels=4, hidden_dims=[64, 128, 256])
    mlm = MLMLayer(in_channels=64, num_classes=256, color_channels=3)
    maskgit = MaskGIT(base_model=unet, mlm_layer=mlm, img_size=32).to(device)
    
    optimizer = optim.Adam(maskgit.parameters(), lr=1e-3)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long()) 
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    maskgit.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            optimizer.zero_grad()
            loss = maskgit.forward_step(images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    generated_images = maskgit.deploy(num_samples=4, T=10, device=device)

if __name__ == "__main__":
    train_maskgit()