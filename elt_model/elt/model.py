"""
Core ELT Model Architecture.
Implements the Elastic Looped Transformer which loops over shared transformer blocks.
"""

import torch
import torch.nn as nn
import numpy as np

from .embeddings import PatchEmbed, TimestepEmbedder, LabelEmbedder, get_2d_sincos_pos_embed
from .blocks import TransformerBlock, modulate


class FinalLayer(nn.Module):
    """
    The final layer of the ELT/DiT model.
    Applies AdaLN, and a linear projection to map tokens back to patch space.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ElasticLoopedTransformer(nn.Module):
    """
    Elastic Looped Transformer (ELT) Model.
    A recurrent transformer architecture for visual generation.
    """
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=768,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=1000,
        class_dropout_prob=0.1,
        num_loops=10,
        num_blocks_per_loop=1, # Number of distinct blocks inside the loop
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_loops = num_loops
        self.num_blocks_per_loop = num_blocks_per_loop

        # Embeddings
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos pos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # The Elastic Loop: A set of weight-shared transformer blocks
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks_per_loop)
        ])

        # Final projection layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize standard layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize AdaLN modulations to zero (Zero-initialization trick)
        for block in self.shared_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


    def forward(self, x, t, y, return_all_loops=False, custom_num_loops=None):
        """
        Forward pass of ELT.
        return_all_loops: If True, returns a list of outputs for each loop (for ILSD training).
        custom_num_loops: If set, overrides self.num_loops for Any-Time inference.
        """
        # Embeddings
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D) - Conditioning vector

        loops_to_run = custom_num_loops if custom_num_loops is not None else self.num_loops

        loop_outputs = []
        
        # Iterative Loops (Weight-Shared Blocks)
        for loop_idx in range(loops_to_run):
            # Pass through the shared block(s)
            for block in self.shared_blocks:
                x = block(x, c)
            
            # If we need intermediate predictions (e.g. for distillation)
            if return_all_loops:
                loop_outputs.append(x)

        if return_all_loops:
            # We predict noise for each loop's output
            final_outputs = []
            for loop_out in loop_outputs:
                out = self.final_layer(loop_out, c)
                final_outputs.append(self.unpatchify(out))
            return final_outputs
        else:
            # Only predict noise for the final loop's output
            x = self.final_layer(x, c)
            x = self.unpatchify(x)
            return x
