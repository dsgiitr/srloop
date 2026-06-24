import torch
import torch.nn as nn

class MLMLayer(nn.Module):
    def __init__(self, in_channels, num_classes=256, colour_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.colour_channels = colour_channels
        self.conv_out = nn.Conv2d(in_channels, colour_channels * num_classes, kernel_size=1)
        
    def forward(self, x):
        B, _, H, W = x.shape
        logits = self.conv_out(x)
        logits = logits.view(B, self.colour_channels, self.num_classes, H, W)
        return logits