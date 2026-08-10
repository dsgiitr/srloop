"""
EMA Utility for ELT-SR
======================
Exponential Moving Average for model weights.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        # Create a deepcopy of the model for the EMA weights
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        
        # Ensure EMA model doesn't require gradients
        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters using the current model."""
        for ema_param, param in zip(
            self.ema_model.parameters(), model.parameters()
        ):
            if param.requires_grad:
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        """Return the state dict of the EMA model."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load state dict into the EMA model."""
        self.ema_model.load_state_dict(state_dict)
