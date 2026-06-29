import torch
import torch.nn as nn
from src.modules import PatchEmbedding,ELTBlock

class ElasticTransformerMLM(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3,
                 embed_dim=512, num_heads=8, ff_dim=2048,
                 num_layers_N=2, vocabulary_size=1024):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable spatial awareness for the tokens
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # The single shared composite block g_Theta
        self.elt_block = ELTBlock(embed_dim, num_heads, ff_dim, num_layers_N)

        # The shared MLM prediction head
        self.mlm_head = nn.Linear(embed_dim, vocabulary_size)

    def forward(self, x, num_loops_L=1):
        """
        Forward propagation where you can adjust the compute depth dynamically via 'num_loops_L'
        """
        # Step 1: Tokenize image and add positional information
        x = self.patch_embed(x)
        x = x + self.pos_embedding

        # Step 2: Execute the recursive looping loop mechanism: g_Theta^L(x)
        for _ in range(num_loops_L):
            x = self.elt_block(x)

        # Step 3: Project back to vocabulary probabilities
        logits = self.mlm_head(x) # Output shape: [Batch, Num_Patches, Vocabulary_Size]
        return logits
