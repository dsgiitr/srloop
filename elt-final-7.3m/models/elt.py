import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from ELT.modules.embeddings import TimestepEmbedder, LabelEmbedder, get_2d_sincos_pos_embed
from ELT.modules.transformer import DiTBlock, FinalLayer

class ELT(nn.Module):
    """
    Elastic Looped Transformer model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config.model
        self.learn_sigma = self.config.learn_sigma
        self.in_channels = self.config.in_channels
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        self.patch_size = self.config.patch_size
        self.num_heads = self.config.num_heads
        self.unique_blocks = self.config.unique_blocks
        self.default_loops = self.config.loops

        self.x_embedder = PatchEmbed(self.config.image_size // 8, self.config.patch_size, self.in_channels, self.config.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.config.hidden_size)
        self.y_embedder = LabelEmbedder(self.config.num_classes, self.config.hidden_size, self.config.class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.config.hidden_size), requires_grad=False)

        # N unique blocks
        self.blocks = nn.ModuleList([
            DiTBlock(self.config.hidden_size, self.config.num_heads, mlp_ratio=self.config.mlp_ratio) for _ in range(self.unique_blocks)
        ])
        
        self.final_layer = FinalLayer(self.config.hidden_size, self.patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, loops=None, return_ilsd_loops=False):
        """
        Forward pass of ELT.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        loops: Number of loop iterations (defaults to self.default_loops)
        return_ilsd_loops: If True, returns a tuple of (pred_int, pred_max) for stochastic ILSD training.
        """
        if loops is None:
            loops = self.default_loops

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        
        if return_ilsd_loops and self.training and loops > 1:
            l_int = torch.randint(1, loops, (1,)).item()
        else:
            l_int = -1
            
        pred_int = None
        
        for l in range(1, loops + 1):
            for block in self.blocks:
                x = block(x, c)
            
            if l == l_int:
                pred_int = self.final_layer(x, c)
                pred_int = self.unpatchify(pred_int)
        
        pred_max = self.final_layer(x, c)
        pred_max = self.unpatchify(pred_max)
        
        if return_ilsd_loops and pred_int is not None:
            return [pred_int, pred_max]
            
        return pred_max

    def forward_with_cfg(self, x, t, y, cfg_scale, loops=None):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        
        model_out = self.forward(combined, t, y, loops=loops, return_ilsd_loops=False)
        
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
