import torch, argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from data import HFCifar10
from sample import load

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--L', type=int, default=4)
    p.add_argument('--n', type=int, default=5000)
    p.add_argument('--bs', type=int, default=100)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cfg', type=float, default=1.5)
    p.add_argument('--data', type=str, default='./data')
    args = p.parse_args()

    device = 'cuda'
    model, diff = load(args.ckpt, device)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    iscore = InceptionScore(normalize=True).to(device)

    ds = HFCifar10(split='train', transform=transforms.ToTensor(), cache_dir=args.data)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True)
    seen = 0
    for x, _ in dl:
        fid.update(x.to(device), real=True)
        seen += x.shape[0]
        if seen >= args.n:
            break

    gen = 0
    while gen < args.n:
        b = min(args.bs, args.n - gen)
        y = torch.randint(0, 10, (b,), device=device)
        x = diff.sample(model, b, y, args.L, device, steps=args.steps, cfg=args.cfg)
        x = (x+1)/2
        fid.update(x, real=False)
        iscore.update(x)
        gen += b

    print(f'L={args.L}  FID={fid.compute().item():.3f}  IS={iscore.compute()}')

if __name__ == '__main__':
    main()
