from copy import deepcopy
import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False

    def to(self, device):
        self.ema_model.to(device)
        return self

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            if p.requires_grad:
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)