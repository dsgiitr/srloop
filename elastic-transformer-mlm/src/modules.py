import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and projects them into a vector space (tokens).
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # A 2D convolution is a clean way to split an image into patches and project it
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input shape: [Batch, Channels, Height, Width]
        x = self.proj(x) # -> [Batch, Embed_Dim, Grid_H, Grid_W]
        x = x.flatten(2) # -> [Batch, Embed_Dim, Num_Patches]
        x = x.transpose(1, 2) # -> [Batch, Num_Patches, Embed_Dim]
        return x


class ELTBlock(nn.Module):
    """
    The composite block g_Theta containing N unique transformer layers.
    """
    def __init__(self, embed_dim=512, num_heads=8, ff_dim=2048, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        # Build N physically unique transformer encoder layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        # Sequential execution through the N layers inside the block
        for layer in self.layers:
            x = layer(x)
        return x