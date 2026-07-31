import os
import argparse
import torch
import torchvision
from tqdm import tqdm
from ELT.models.elt import ELT
from ELT.configs.config import get_config
from ELT.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import torch_fidelity

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Setup Model
    config = get_config()
    latent_size = config.model.image_size // 8
    model = ELT(config).to(device)
    
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    # The saved checkpoint from train.ipynb is the EMA model's state_dict directly
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # 2. Generate samples
    os.makedirs(args.sample_dir, exist_ok=True)
    
    existing_samples = len(os.listdir(args.sample_dir))
    if existing_samples >= args.num_samples:
        print(f"Found {existing_samples} images in '{args.sample_dir}'. Skipping generation!")
    else:
        print(f"Generating {args.num_samples} samples to {args.sample_dir}...")
        batch_size = args.batch_size
        num_batches = (args.num_samples + batch_size - 1) // batch_size
        
        sample_idx = 0
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Generating Batches"):
                current_batch_size = min(batch_size, args.num_samples - sample_idx)
                z = torch.randn(current_batch_size, 4, latent_size, latent_size, device=device)
                y = torch.randint(0, config.model.num_classes, (current_batch_size,), device=device)
                
                # Setup CFG (Classifier-Free Guidance)
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([config.model.num_classes] * current_batch_size, device=device)
                y_cfg = torch.cat([y, y_null], 0)
                
                loops = args.loops if args.loops is not None else config.sampling.inference_loops
                model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale, loops=loops)
                
                # Sample
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0) # Remove null class samples
                samples = vae.decode(samples / 0.18215).sample
                
                # Save images
                for j in range(current_batch_size):
                    img = samples[j]
                    # Normalize from [-1, 1] to [0, 1] and save
                    torchvision.utils.save_image(img, os.path.join(args.sample_dir, f"{sample_idx:05d}.png"), normalize=True, value_range=(-1, 1))
                    sample_idx += 1

    # 3. Prepare CIFAR-10 reference dataset via Hugging Face for faster downloading
    ref_dir = "ELT/cifar10_ref"
    if not os.path.exists(ref_dir) or len(os.listdir(ref_dir)) < 50000:
        os.makedirs(ref_dir, exist_ok=True)
        print("Downloading CIFAR-10 via Hugging Face for faster speeds...")
        from datasets import load_dataset
        dataset = load_dataset("uoft-cs/cifar10", split="train")
        for i, item in enumerate(tqdm(dataset, desc="Saving Reference Images")):
            item["img"].save(os.path.join(ref_dir, f"{i:05d}.png"))

    # 4. Calculate FID and IS
    print("Calculating FID and IS...")
    metrics = torch_fidelity.calculate_metrics(
        input1=ref_dir,
        input2=args.sample_dir,
        cuda=torch.cuda.is_available(),
        isc=True,
        fid=True,
        verbose=True
    )
    
    print("-" * 50)
    print(f"Inception Score (IS): {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
    print(f"Frechet Inception Distance (FID): {metrics['frechet_inception_distance']:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="ELT/results_notebook/elt-best-model.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--vae", type=str, default="ema")
    parser.add_argument("--sample-dir", type=str, default="ELT/samples_eval_elt", help="Directory to save generated images")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of images to generate (10k for fast eval, 50k for full)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation (lowered to save VRAM)")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--loops", type=int, default=None, help="Number of loops for elastic inference.")
    args = parser.parse_args()
    main(args)
