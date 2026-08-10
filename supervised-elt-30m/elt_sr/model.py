"""
ELT-SR Model Architecture
==========================
Elastic Looped Transformer for Single-Image Super-Resolution.

Architecture (from design document):
  - Pixel-space residual diffusion (no VAE)
  - 32×32 HQ, 16×16 LQ, ×2 scale
  - Patch size 4 → 64 tokens (8×8 grid)
  - Hidden dim 256, 4 heads, MLP dim 1024
  - N=6 unique DiT blocks, L_max=3 loops
  - Input concatenation [x_t, I_base] (6-ch) + per-loop LQ additive reinjection
  - AdaLN-Zero conditioned on timestep only (no class labels)
  - ε-prediction on residual R = I_HQ − Upsample(I_LQ)
  - ILSD with sigmoid-weighted MSE
  - ~7.5M trainable parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Positional Embeddings
# ---------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings.

    Args:
        embed_dim: Embedding dimension (must be divisible by 2).
        grid_size: Side length of the 2D grid (total tokens = grid_size²).

    Returns:
        Tensor of shape [grid_size², embed_dim].
    """
    assert embed_dim % 2 == 0, "embed_dim must be divisible by 2 for 2D sin-cos."
    half_dim = embed_dim // 2

    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")  # [H, W] each
    grid = torch.stack(grid, dim=0)  # [2, H, W]
    grid = grid.reshape(2, -1).T  # [H*W, 2]

    omega = torch.arange(half_dim // 2, dtype=torch.float64) / (half_dim // 2)
    omega = 1.0 / (10000.0 ** omega)  # [half_dim//2]

    # For each spatial dim, compute sin/cos embeddings
    out_h = grid[:, 0:1] * omega.unsqueeze(0)  # [N, half_dim//2]
    out_w = grid[:, 1:2] * omega.unsqueeze(0)  # [N, half_dim//2]

    emb_h = torch.cat([torch.sin(out_h), torch.cos(out_h)], dim=1)  # [N, half_dim]
    emb_w = torch.cat([torch.sin(out_w), torch.cos(out_w)], dim=1)  # [N, half_dim]

    pos_embed = torch.cat([emb_h, emb_w], dim=1).float()  # [N, embed_dim]
    return pos_embed


# ---------------------------------------------------------------------------
# Timestep Embedding
# ---------------------------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal timestep embedding (from DDPM / DiT).

    Args:
        t: Tensor of timesteps, shape [B].
        dim: Embedding dimension.
        max_period: Controls the frequency range.

    Returns:
        Tensor of shape [B, dim].
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None, :]  # [B, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepMLP(nn.Module):
    """Timestep → embedding via sinusoidal + 2-layer MLP."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] integer timesteps.
        Returns:
            [B, hidden_dim] embedding.
        """
        t_emb = timestep_embedding(t, self.hidden_dim)  # [B, hidden_dim]
        return self.mlp(t_emb)  # [B, hidden_dim]


# ---------------------------------------------------------------------------
# Patch Embedding & Unpatchify
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Convert image to sequence of patch tokens via linear projection."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}×{W}) doesn't match expected ({self.img_size}×{self.img_size})"

        p = self.patch_size
        # Reshape to patches: [B, C, H/p, p, W/p, p] → [B, N, C*p*p]
        x = x.reshape(B, C, self.grid_size, p, self.grid_size, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H/p, W/p, C, p, p]
        x = x.reshape(B, self.num_patches, C * p * p)  # [B, N, C*p*p]
        return self.proj(x)  # [B, N, embed_dim]


def unpatchify(x: torch.Tensor, patch_size: int, out_channels: int) -> torch.Tensor:
    """Convert patch tokens back to image.

    Args:
        x: [B, N, out_channels * patch_size²]
        patch_size: Patch size.
        out_channels: Number of output channels.

    Returns:
        [B, out_channels, H, W]
    """
    B, N, _ = x.shape
    grid_size = int(math.sqrt(N))
    assert grid_size * grid_size == N

    p = patch_size
    x = x.reshape(B, grid_size, grid_size, out_channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, H/p, p, W/p, p]
    x = x.reshape(B, out_channels, grid_size * p, grid_size * p)
    return x


# ---------------------------------------------------------------------------
# AdaLN-Zero Modulation
# ---------------------------------------------------------------------------

class AdaLNModulation(nn.Module):
    """Produces 6 modulation parameters (γ₁,β₁,α₁, γ₂,β₂,α₂) from timestep embedding.

    Used in AdaLN-Zero DiT blocks. α values are initialized to zero so that
    the block initially acts as an identity (zero-init residual).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_dim, 6 * hidden_dim)
        # Zero-initialize so blocks start as identity
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            c: [B, hidden_dim] conditioning (timestep embedding).
        Returns:
            Tuple of 6 tensors, each [B, 1, hidden_dim] for broadcasting over tokens.
        """
        params = self.linear(self.silu(c))  # [B, 6*hidden_dim]
        params = params.unsqueeze(1)  # [B, 1, 6*hidden_dim]
        return params.chunk(6, dim=-1)  # 6 × [B, 1, hidden_dim]


# ---------------------------------------------------------------------------
# DiT Block (AdaLN-Zero)
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """Single Transformer block with AdaLN-Zero conditioning (DiT-style).

    Architecture:
        LN → AdaLN(γ₁,β₁) → Self-Attention → scale(α₁) → residual
        LN → AdaLN(γ₂,β₂) → MLP → scale(α₂) → residual
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Layer norms (no affine — modulated by AdaLN)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        # Self-attention
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, hidden_dim),
        )

        # AdaLN-Zero modulation
        self.adaLN = AdaLNModulation(hidden_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] token sequence.
            c: [B, D] timestep conditioning embedding.
        Returns:
            [B, N, D] updated token sequence.
        """
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN(c)

        # --- Self-Attention ---
        h = self.norm1(x)
        h = h * (1 + gamma1) + beta1  # AdaLN modulation

        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)  # [B, heads, N, head_dim]
        attn = attn.transpose(1, 2).reshape(B, N, D)
        attn = self.attn_out(attn)

        x = x + alpha1 * attn  # Zero-init residual

        # --- MLP ---
        h = self.norm2(x)
        h = h * (1 + gamma2) + beta2  # AdaLN modulation
        h = self.mlp(h)

        x = x + alpha2 * h  # Zero-init residual

        return x


# ---------------------------------------------------------------------------
# ELT-SR Model
# ---------------------------------------------------------------------------

class ELTSR(nn.Module):
    """Elastic Looped Transformer for Super-Resolution.

    Key design decisions (from research design document):
      1. Pixel-space residual diffusion (no VAE).
      2. Input: [x_t, I_base] concatenated (6 channels).
      3. Patch size 4 on 32×32 → 64 tokens.
      4. N unique DiT blocks looped L times.
      5. Per-loop additive LQ token reinjection.
      6. AdaLN-Zero with timestep conditioning only.
      7. ε-prediction output (3 channels).
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,          # RGB channels of noisy residual x_t
        cond_channels: int = 3,        # RGB channels of upsampled LQ (I_base)
        hidden_dim: int = 256,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        num_blocks: int = 6,           # N: unique transformer blocks
        max_loops: int = 3,            # L_max: maximum loop count
        min_loops: int = 1,            # L_min: minimum student loop count
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.max_loops = max_loops
        self.min_loops = min_loops
        self.grid_size = img_size // patch_size
        self.num_tokens = self.grid_size ** 2

        # --- Patch embeddings ---
        # Main input: [x_t, I_base] concatenated → 6 channels
        self.input_embed = PatchEmbed(
            img_size, patch_size, in_channels + cond_channels, hidden_dim
        )
        # LQ reference tokens (for per-loop reinjection)
        self.lq_embed = PatchEmbed(
            img_size, patch_size, cond_channels, hidden_dim
        )

        # --- Positional embedding (fixed sinusoidal) ---
        pos_embed = get_2d_sincos_pos_embed(hidden_dim, self.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # [1, N, D]

        # --- Timestep embedding ---
        self.time_embed = TimestepMLP(hidden_dim)

        # --- N shared DiT blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_dim)
            for _ in range(num_blocks)
        ])

        # --- Output head ---
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),  # γ, β for final norm
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)

        self.output_proj = nn.Linear(
            hidden_dim, in_channels * patch_size * patch_size
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following DiT conventions."""
        # Initialize patch embedding projections
        w = self.input_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.input_embed.proj.bias)

        w = self.lq_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.lq_embed.proj.bias)

        # Initialize timestep MLP
        for module in self.time_embed.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize transformer blocks
        for block in self.blocks:
            # Attention QKV and output
            nn.init.xavier_uniform_(block.qkv.weight)
            nn.init.zeros_(block.qkv.bias)
            nn.init.xavier_uniform_(block.attn_out.weight)
            nn.init.zeros_(block.attn_out.bias)
            # MLP
            for module in block.mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def _forward_backbone(
        self,
        tokens: torch.Tensor,
        lq_tokens: torch.Tensor,
        t_emb: torch.Tensor,
        num_loops: int,
        save_intermediate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the ELT backbone for a specified number of loops.

        Args:
            tokens: [B, N, D] initial token embeddings.
            lq_tokens: [B, N, D] LQ reference tokens for reinjection.
            t_emb: [B, D] timestep embedding.
            num_loops: Number of loops to execute.
            save_intermediate: If not None, save the state after this loop index (1-indexed).

        Returns:
            (final_tokens, intermediate_tokens) — intermediate is None if save_intermediate is None.
        """
        x = tokens
        x_intermediate = None

        for loop_idx in range(1, num_loops + 1):
            # Per-loop LQ reinjection (additive)
            x = x + lq_tokens

            # Apply N shared blocks
            for block in self.blocks:
                x = block(x, t_emb)

            # Save intermediate state for ILSD
            if save_intermediate is not None and loop_idx == save_intermediate:
                x_intermediate = x  # No clone needed — we continue forward from here

        return x, x_intermediate

    def _predict(self, tokens: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Apply output head to get ε prediction.

        Args:
            tokens: [B, N, D] final token states.
            t_emb: [B, D] timestep embedding.

        Returns:
            [B, C, H, W] predicted noise.
        """
        # Final AdaLN
        gamma_beta = self.final_adaLN(t_emb).unsqueeze(1)  # [B, 1, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        x = self.final_norm(tokens)
        x = x * (1 + gamma) + beta

        # Project to patch pixels
        x = self.output_proj(x)  # [B, N, C*p*p]

        # Unpatchify
        return unpatchify(x, self.patch_size, self.in_channels)  # [B, C, H, W]

    def forward(
        self,
        x_t: torch.Tensor,
        i_base: torch.Tensor,
        t: torch.Tensor,
        l_int: Optional[int] = None,
    ) -> dict:
        """Full forward pass.

        Args:
            x_t: [B, 3, 32, 32] noisy residual at timestep t.
            i_base: [B, 3, 32, 32] bicubic-upsampled LQ image.
            t: [B] integer timesteps.
            l_int: Intermediate loop count for ILSD (student). If None, no student.

        Returns:
            dict with keys:
              - 'eps_teacher': [B, 3, 32, 32] teacher ε prediction (full L_max loops).
              - 'eps_student': [B, 3, 32, 32] student ε prediction (L_int loops), or None.
        """
        B = x_t.shape[0]

        # Concatenate noisy residual with LQ condition
        model_input = torch.cat([x_t, i_base], dim=1)  # [B, 6, 32, 32]

        # Patch embedding
        tokens = self.input_embed(model_input)  # [B, 64, 256]
        tokens = tokens + self.pos_embed        # [B, 64, 256]

        # LQ reference tokens (computed once, reused every loop)
        lq_tokens = self.lq_embed(i_base)       # [B, 64, 256]

        # Timestep embedding
        t_emb = self.time_embed(t)               # [B, 256]

        # ELT backbone
        x_teacher, x_student = self._forward_backbone(
            tokens, lq_tokens, t_emb,
            num_loops=self.max_loops,
            save_intermediate=l_int,
        )

        # Predictions
        eps_teacher = self._predict(x_teacher, t_emb)

        eps_student = None
        if x_student is not None:
            eps_student = self._predict(x_student, t_emb)

        return {
            "eps_teacher": eps_teacher,
            "eps_student": eps_student,
        }

    @torch.no_grad()
    def predict(
        self,
        x_t: torch.Tensor,
        i_base: torch.Tensor,
        t: torch.Tensor,
        num_loops: Optional[int] = None,
    ) -> torch.Tensor:
        """Inference-only forward pass (no ILSD).

        Args:
            x_t: [B, 3, 32, 32] noisy residual.
            i_base: [B, 3, 32, 32] bicubic-upsampled LQ.
            t: [B] timesteps.
            num_loops: Number of loops (default: max_loops). Enables Any-Time inference.

        Returns:
            [B, 3, 32, 32] predicted ε.
        """
        if num_loops is None:
            num_loops = self.max_loops

        model_input = torch.cat([x_t, i_base], dim=1)
        tokens = self.input_embed(model_input) + self.pos_embed
        lq_tokens = self.lq_embed(i_base)
        t_emb = self.time_embed(t)

        x_final, _ = self._forward_backbone(
            tokens, lq_tokens, t_emb,
            num_loops=num_loops,
            save_intermediate=None,
        )
        return self._predict(x_final, t_emb)


# ---------------------------------------------------------------------------
# Model factory & parameter counting
# ---------------------------------------------------------------------------

def create_elt_sr(
    config=None,
    img_size: int = 32,
    patch_size: int = 4,
    hidden_dim: int = 256,
    num_heads: int = 4,
    mlp_dim: int = 1024,
    num_blocks: int = 6,
    max_loops: int = 3,
) -> ELTSR:
    """Create an ELT-SR model with the recommended configuration or from a config object."""
    if config is not None:
        # If using VAE, the model operates in latent space
        actual_size = config.latent_size if getattr(config, "use_vae", False) else config.img_size
        return ELTSR(
            img_size=actual_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            cond_channels=config.cond_channels,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            num_blocks=config.num_blocks,
            max_loops=config.max_loops,
            min_loops=config.min_loops,
        )

    model = ELTSR(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        cond_channels=3,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_blocks=num_blocks,
        max_loops=max_loops,
    )
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters, broken down by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params

    return {
        "total": total,
        "trainable": trainable,
        "breakdown": breakdown,
    }


if __name__ == "__main__":
    # Quick verification
    model = create_elt_sr()
    param_info = count_parameters(model)

    print("=" * 60)
    print("ELT-SR Model Summary")
    print("=" * 60)
    print(f"Total parameters:     {param_info['total']:,}")
    print(f"Trainable parameters: {param_info['trainable']:,}")
    print()
    print("Parameter breakdown:")
    for name, count in param_info["breakdown"].items():
        print(f"  {name:20s}: {count:>10,}")

    print()

    # Test forward pass
    B = 2
    x_t = torch.randn(B, 3, 32, 32)
    i_base = torch.randn(B, 3, 32, 32)
    t = torch.randint(0, 1000, (B,))

    # Training mode (with ILSD student at L_int=2)
    out = model(x_t, i_base, t, l_int=2)
    print(f"eps_teacher shape: {out['eps_teacher'].shape}")
    print(f"eps_student shape: {out['eps_student'].shape}")

    # Inference mode (Any-Time: L=1, L=2, L=3)
    for L in [1, 2, 3]:
        eps = model.predict(x_t, i_base, t, num_loops=L)
        print(f"Inference L={L}: eps shape = {eps.shape}")
