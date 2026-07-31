import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from ELT.diffusion import create_diffusion
from ELT.models.elt import ELT
from ELT.configs.config import get_config
from diffusers.models import AutoencoderKL
import argparse

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = get_config()
    latent_size = config.model.image_size // 8
    model = ELT(config).to(device)

    if args.ckpt is not None:
        state_dict = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dict.get("model", state_dict))
    
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = [0, 1, 2, 3, 4, 5, 6, 7]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([config.model.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    
    loops = args.loops if args.loops is not None else config.sampling.inference_loops
    
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, loops=loops)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    samples = vae.decode(samples / 0.18215).sample

    save_image(samples, f"sample_loops_{loops}.png", nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Saved sample_loops_{loops}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to ELT checkpoint.")
    parser.add_argument("--loops", type=int, default=None, help="Number of loops for elastic inference.")
    args = parser.parse_args()
    main(args)
