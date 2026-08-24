import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, channels: int = 64, growth_channels: int = 32):
        super().__init__()
        self.rdb1 = DenseBlock(channels, growth_channels)
        self.rdb2 = DenseBlock(channels, growth_channels)
        self.rdb3 = DenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.rdb3(self.rdb2(self.rdb1(x)))) * 0.2 + x

class LREncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 256, num_rrdb: int = 4, channels: int = 64):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, channels, 3, 1, 1)
        self.rrdb_blocks = nn.Sequential(*[RRDB(channels, 32) for _ in range(num_rrdb)])
        self.conv_last = nn.Conv2d(channels, channels, 3, 1, 1)
        self.proj = nn.Conv2d(channels, out_dim, kernel_size=2, stride=2)

    def forward(self, x_lq: torch.Tensor) -> torch.Tensor:
        feat = self.conv_last(self.rrdb_blocks(self.conv_first(x_lq)))
        feat = self.proj(feat) 
        B, C, H, W = feat.shape
        return feat.view(B, C, H * W).transpose(1, 2) 

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    half_dim = embed_dim // 2
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, -1).T
    omega = 1.0 / (10000.0 ** (torch.arange(half_dim // 2, dtype=torch.float64) / (half_dim // 2)))
    out_h = grid[:, 0:1] * omega.unsqueeze(0)
    out_w = grid[:, 1:2] * omega.unsqueeze(0)
    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)
    return torch.cat([emb_h, emb_w], dim=1).float()

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TimestepMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.hidden_dim))

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, self.num_patches, C * p * p)
        return self.proj(x)

def unpatchify(x: torch.Tensor, patch_size: int, out_channels: int) -> torch.Tensor:
    B, N, _ = x.shape
    grid_size = int(math.sqrt(N))
    p = patch_size
    x = x.reshape(B, grid_size, grid_size, out_channels, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, out_channels, grid_size * p, grid_size * p)

class AdaLNModulation(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        params = self.linear(self.silu(c)).unsqueeze(1)
        return params.chunk(6, dim=-1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.adaLN = AdaLNModulation(hidden_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(c)
        h = self.norm1(x) * (1 + gamma1) + beta1
        attn_out, _ = self.attn(h, h, h)
        x = x + alpha1 * attn_out
        h = self.norm2(x) * (1 + gamma2) + beta2
        x = x + alpha2 * self.mlp(h)
        return x

class ELTSR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.max_loops = config.max_loops
        self.grid_size = config.img_size // config.patch_size

        self.input_embed = PatchEmbed(config.img_size, config.patch_size, config.in_channels + config.cond_channels, config.hidden_dim)
        self.lq_embed = LREncoder(in_channels=config.cond_channels, out_dim=config.hidden_dim, num_rrdb=config.num_rrdb, channels=64)

        pos_embed = get_2d_sincos_pos_embed(config.hidden_dim, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))
        self.time_embed = TimestepMLP(config.hidden_dim)

        self.blocks = nn.ModuleList([DiTBlock(config.hidden_dim, config.num_heads, config.mlp_dim) for _ in range(config.num_blocks)])
        self.final_norm = nn.LayerNorm(config.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_dim, 2 * config.hidden_dim))
        self.output_proj = nn.Linear(config.hidden_dim, config.in_channels * config.patch_size * config.patch_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        w = self.input_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.input_embed.proj.bias)

    def _forward_backbone(self, tokens: torch.Tensor, lq_tokens: torch.Tensor, t_emb: torch.Tensor, num_loops: int, save_intermediate: Optional[int] = None):
        x = tokens
        x_intermediate = None
        for loop_idx in range(1, num_loops + 1):
            x = x + lq_tokens
            for block in self.blocks:
                x = block(x, t_emb)
            if save_intermediate is not None and loop_idx == save_intermediate:
                x_intermediate = x
        return x, x_intermediate

    def _predict(self, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.final_adaLN(t_emb).unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        x = self.final_norm(tokens) * (1 + gamma) + beta
        x = self.output_proj(x)
        return unpatchify(x, self.patch_size, self.in_channels)

    def forward(self, x_t: torch.Tensor, i_base: torch.Tensor, i_lq: torch.Tensor, t: torch.Tensor, l_int: Optional[int] = None):
        model_input = torch.cat([x_t, i_base], dim=1)
        tokens = self.input_embed(model_input) + self.pos_embed
        lq_tokens = self.lq_embed(i_lq)
        t_emb = self.time_embed(t)

        x_teacher, x_student = self._forward_backbone(tokens, lq_tokens, t_emb, num_loops=self.max_loops, save_intermediate=l_int)
        eps_teacher = self._predict(x_teacher, t_emb)
        eps_student = self._predict(x_student, t_emb) if x_student is not None else None

        return {"eps_teacher": eps_teacher, "eps_student": eps_student}

    @torch.no_grad()
    def predict(self, x_t: torch.Tensor, i_base: torch.Tensor, i_lq: torch.Tensor, t: torch.Tensor, num_loops: Optional[int] = None) -> torch.Tensor:
        if num_loops is None:
            num_loops = self.max_loops
        tokens = self.input_embed(torch.cat([x_t, i_base], dim=1)) + self.pos_embed
        lq_tokens = self.lq_embed(i_lq)
        t_emb = self.time_embed(t)
        x_final, _ = self._forward_backbone(tokens, lq_tokens, t_emb, num_loops=num_loops)
        return self._predict(x_final, t_emb)