import torch, argparse, os
from torchvision.utils import save_image
from model import ELTDiT
from diffusion import ELTDiffusion

def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    margs = ck['args']
    model = ELTDiT(dim=margs['dim'], n_layers=margs['n_layers']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    return model, ELTDiffusion(L_max=margs['L_max'])

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--L', type=int, default=4)
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--out', type=str, default='./samples')
    args = p.parse_args()

    device = 'cuda'
    model, diff = load(args.ckpt, device)
    os.makedirs(args.out, exist_ok=True)
    y = torch.randint(0, 10, (args.n,), device=device)
    x = diff.sample(model, args.n, y, args.L, device, steps=args.steps, cfg=args.cfg)
    x = (x+1)/2
    for i in range(args.n):
        save_image(x[i], f'{args.out}/{i}.png')

if __name__ == '__main__':
    main()
