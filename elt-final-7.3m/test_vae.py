import torch
from diffusers.models import AutoencoderKL

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

x = torch.randn(1, 3, 32, 32).to(device)
try:
    with torch.no_grad():
        encoded = vae.encode(x)
        print(f"Encoded shape: {encoded.latent_dist.sample().shape}")
except Exception as e:
    print(f"Error encoding 32x32: {e}")

x = torch.randn(1, 3, 64, 64).to(device)
try:
    with torch.no_grad():
        encoded = vae.encode(x)
        print(f"Encoded shape: {encoded.latent_dist.sample().shape}")
except Exception as e:
    print(f"Error encoding 64x64: {e}")
