import os
import sys
import torch
from diffusers.models import AutoencoderKL

sys.path.append(os.path.abspath('dit-model-implementation/dit'))

from configs.config import get_config
from models.elt import ELT
from diffusion import create_diffusion
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = get_config()

model = ELT(config).to(device)
ema = deepcopy(model).to(device)

checkpoint = torch.load('results_notebook/elt_epoch_0.pt', map_location=device)
ema.load_state_dict(checkpoint['ema'])

vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').to(device)

print("Running sampling logic...")
with torch.no_grad():
    sampling_diffusion = create_diffusion('250')
    latent_size = config.model.image_size // 8
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([config.model.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=4.0, loops=config.sampling.inference_loops)
    samples = sampling_diffusion.p_sample_loop(
        ema.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample
    from torchvision.utils import save_image
    save_image(samples, 'test_sample.png', nrow=4, normalize=True, value_range=(-1, 1))

print("Done! Image generated.")
