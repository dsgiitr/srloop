import torch, argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from data import HFCifar10
from model import ELTDiT
from diffusion import ELTDiffusion

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--L_max', type=int, default=4)
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--out', type=str, default='./ckpt.pt')
    args = p.parse_args()

    device = 'cuda'
    tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    ds = HFCifar10(split='train', transform=tf, cache_dir=args.data)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)

    model = ELTDiT(dim=args.dim, n_layers=args.n_layers).to(device)
    diff = ELTDiffusion(L_max=args.L_max, total_steps=args.epochs*len(dl))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    step = 0
    for ep in range(args.epochs):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            if torch.rand(1).item() < 0.1:
                y = torch.full_like(y, 10)
            loss = diff.loss(model, x, y, step)
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 100 == 0:
                print(f'epoch {ep} step {step} loss {loss.item():.4f}')
        torch.save({'model': model.state_dict(), 'args': vars(args)}, args.out)

if __name__ == '__main__':
    main()
