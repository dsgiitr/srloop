"""
Utilities for Gaussian Diffusion process.
Implemented from scratch for ELT training and sampling.
"""

import math
import torch
import numpy as np

def make_beta_schedule(schedule, num_timesteps, linear_start=1e-4, linear_end=2e-2):
    if schedule == "linear":
        betas = torch.linspace(linear_start, linear_end, num_timesteps, dtype=torch.float64)
    elif schedule == "cosine":
        return betas_for_alpha_bar(
            num_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule}")
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float64)

class GaussianDiffusion:
    """
    Utilities for training and sampling Diffusion Models.
    """
    def __init__(self, betas):
        self.betas = betas.float()
        self.num_timesteps = len(betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means one step of diffusion).
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        """
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def p_sample(self, model, x_t, t, y, custom_num_loops=None):
        """
        Sample x_{t-1} from the model at a given timestep.
        """
        # Forward pass (only taking the final prediction)
        model_out = model(x_t, t, y, return_all_loops=False, custom_num_loops=custom_num_loops)
        
        # Split model output if it learns variance
        if model_out.shape[1] == x_t.shape[1] * 2:
            model_eps, model_var_values = torch.split(model_out, x_t.shape[1], dim=1)
            # Use learned variance
            min_log = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
            max_log = self._extract_into_tensor(torch.log(self.betas), t, x_t.shape)
            # model_var_values is roughly [-1, 1] mapped to [min_log, max_log]
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
        else:
            model_eps = model_out
            model_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        # Predict x_start
        sqrt_recip_alphas_cumprod = self._extract_into_tensor(
            torch.sqrt(1.0 / self.alphas_cumprod), t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod = self._extract_into_tensor(
            torch.sqrt(1.0 / self.alphas_cumprod - 1), t, x_t.shape
        )
        pred_xstart = sqrt_recip_alphas_cumprod * x_t - sqrt_recipm1_alphas_cumprod * model_eps
        
        # Calculate mean
        mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        sample = mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample

    @torch.no_grad()
    def sample_loop(self, model, shape, y, custom_num_loops=None, device="cuda"):
        """
        Generate a batch of samples via reverse diffusion.
        """
        b = shape[0]
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, y, custom_num_loops=custom_num_loops)
            
        return img
