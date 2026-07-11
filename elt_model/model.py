import torch, torch.nn as nn, math

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class AdaLNBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim*6))

    def forward(self, x, c):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1+sc1[:,None]) + s1[:,None]
        h,_ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:,None]*h
        h = self.norm2(x) * (1+sc2[:,None]) + s2[:,None]
        return x + g2[:,None]*self.mlp(h)

class ELTDiT(nn.Module):
    def __init__(self, img_size=32, patch_size=2, in_ch=3, dim=256, n_layers=4, heads=4, n_classes=10):
        super().__init__()
        self.patch_size, self.out_ch, self.img_size = patch_size, in_ch, img_size
        n_patches = (img_size//patch_size)**2
        self.patch_embed = nn.Conv2d(in_ch, dim, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.label_embed = nn.Embedding(n_classes+1, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim*2))
        self.head = nn.Linear(dim, patch_size*patch_size*in_ch)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def unpatchify(self, x):
        p, c, s = self.patch_size, self.out_ch, self.img_size // self.patch_size
        x = x.reshape(x.shape[0], s, s, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], c, s*p, s*p)

    def cond_embed(self, t, y):
        return self.time_mlp(timestep_embedding(t, self.pos_embed.shape[-1])) + self.label_embed(y)

    def final(self, x, c):
        shift, scale = self.adaLN_out(c).chunk(2, dim=-1)
        x = self.norm_out(x)*(1+scale[:,None]) + shift[:,None]
        return self.unpatchify(self.head(x))

    def forward(self, x, t, y, L_max, L_min=1, training=True):
        c = self.cond_embed(t, y)
        x = self.patch_embed(x).flatten(2).transpose(1,2) + self.pos_embed
        L_int = torch.randint(L_min, L_max, (1,)).item() if (training and L_max > L_min) else L_max
        feat_int = None
        for i in range(L_max):
            for blk in self.blocks:
                x = blk(x, c)
            if i+1 == L_int:
                feat_int = x
        out_max = self.final(x, c)
        if training:
            return out_max, self.final(feat_int, c), L_int
        return out_max
