"""
Diffusion Utilities for ELT-SR
================================
Implements:
  - Shifted cosine noise schedule
  - Forward diffusion (q-sample on residual)
  - Sigmoid-weighted MSE loss
  - ILSD loss computation
  - λ curriculum (linear 1→0)
  - DDPM reverse sampling
  - DDIM reverse sampling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Noise Schedule
# ---------------------------------------------------------------------------

def shifted_cosine_schedule(
    num_timesteps: int = 1000,
    shift: float = 1.0,
    cosine_s: float = 0.008,
) -> dict:
    """Shifted cosine noise schedule.

    Produces α_bar_t values using a cosine schedule with an optional shift
    to spend more time at intermediate noise levels.

    Based on the schedule used in the ELT paper (reference [34] therein).

    Args:
        num_timesteps: Total number of diffusion steps T.
        shift: Shift parameter. shift=1.0 gives the standard cosine schedule.
               shift>1.0 shifts toward spending more steps at higher noise.
        cosine_s: Small offset to prevent β_t from being too small at t=0.

    Returns:
        Dictionary with precomputed schedule tensors.
    """
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)

    # Standard cosine schedule: α_bar(t) = cos²(π/2 · (t/T + s) / (1 + s))
    f_t = torch.cos(((steps / num_timesteps) + cosine_s) / (1 + cosine_s) * (math.pi / 2)) ** 2
    alphas_cumprod = f_t / f_t[0]

    # Apply shift: α_bar_shifted = α_bar^shift / (α_bar^shift + (1-α_bar)^shift)
    # This redistributes the schedule to spend more time at intermediate levels.
    if shift != 1.0:
        alphas_cumprod = alphas_cumprod ** shift / (
            alphas_cumprod ** shift + (1 - alphas_cumprod) ** shift
        )

    # Clip to prevent numerical issues
    alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0 - 1e-8)

    # Compute betas from α_bar values
    # β_t = 1 - α_bar_t / α_bar_{t-1}
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, min=1e-8, max=0.999)

    alphas = 1.0 - betas
    alphas_cumprod = alphas_cumprod[1:]  # Remove the t=0 entry

    # Precompute useful quantities
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod / alphas) / (1.0 - alphas_cumprod + 1e-8)
    # Clip posterior variance for numerical stability
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
    log_posterior_variance = torch.log(posterior_variance)

    return {
        "betas": betas.float(),
        "alphas": alphas.float(),
        "alphas_cumprod": alphas_cumprod.float(),
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod.float(),
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod.float(),
        "sqrt_recip_alphas": sqrt_recip_alphas.float(),
        "posterior_variance": posterior_variance.float(),
        "log_posterior_variance": log_posterior_variance.float(),
        "num_timesteps": num_timesteps,
    }


# ---------------------------------------------------------------------------
# Forward Diffusion (q-sample)
# ---------------------------------------------------------------------------

def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.

    Args:
        x_0: [B, C, H, W] clean signal (the residual R in our case).
        t: [B] integer timesteps (0-indexed).
        noise: [B, C, H, W] Gaussian noise ε.
        sqrt_alphas_cumprod: [T] precomputed √ᾱ.
        sqrt_one_minus_alphas_cumprod: [T] precomputed √(1-ᾱ).

    Returns:
        [B, C, H, W] noisy signal x_t.
    """
    # Gather schedule values for the batch
    s_ac = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)             # [B, 1, 1, 1]
    s_omac = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    return s_ac * x_0 + s_omac * noise


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def sigmoid_weight(t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    """Sigmoid-based timestep weighting for diffusion loss.

    Assigns higher weight to intermediate timesteps where the model has the
    most to learn, following the ELT paper's approach.

    Args:
        t: [B] integer timesteps.
        num_timesteps: Total number of timesteps T.

    Returns:
        [B] weights.
    """
    # Normalize t to [0, 1]
    t_norm = t.float() / num_timesteps
    # Sigmoid centered at 0.5 with moderate slope
    weight = 1.0 / (1.0 + torch.exp(-12.0 * (t_norm - 0.5)))
    # Scale to [0.5, 1.5] range to avoid vanishing weights
    weight = 0.5 + weight
    return weight


def sigmoid_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    num_timesteps: int,
) -> torch.Tensor:
    """Sigmoid-weighted MSE loss for diffusion training.

    Args:
        prediction: [B, C, H, W] model prediction.
        target: [B, C, H, W] ground truth.
        t: [B] timesteps.
        num_timesteps: Total T.

    Returns:
        Scalar loss.
    """
    # Per-sample MSE
    mse = F.mse_loss(prediction, target, reduction="none")  # [B, C, H, W]
    mse = mse.mean(dim=[1, 2, 3])  # [B]

    # Apply sigmoid weighting
    weights = sigmoid_weight(t, num_timesteps)  # [B]
    weighted_mse = (weights * mse).mean()

    return weighted_mse


def compute_ilsd_loss(
    eps_teacher: torch.Tensor,
    eps_student: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    num_timesteps: int,
    lam: float,
) -> dict:
    """Compute the full ILSD training loss.

    Loss = L_GT(teacher) + λ·L_GT(student) + (1-λ)·L_dist(student, sg(teacher))

    Args:
        eps_teacher: [B, C, H, W] teacher's ε prediction (L_max loops).
        eps_student: [B, C, H, W] student's ε prediction (L_int loops).
        noise: [B, C, H, W] ground-truth noise ε.
        t: [B] timesteps.
        num_timesteps: Total T.
        lam: Current λ value (linearly decayed 1→0 during training).

    Returns:
        Dictionary with loss components and total loss.
    """
    # Teacher GT loss (always computed)
    loss_gt_teacher = sigmoid_weighted_mse(eps_teacher, noise, t, num_timesteps)

    # Student GT loss
    loss_gt_student = sigmoid_weighted_mse(eps_student, noise, t, num_timesteps)

    # Distillation loss: student mimics teacher (stop-gradient on teacher)
    loss_dist = sigmoid_weighted_mse(
        eps_student, eps_teacher.detach(), t, num_timesteps
    )

    # Total ILSD loss
    loss_total = loss_gt_teacher + lam * loss_gt_student + (1 - lam) * loss_dist

    return {
        "loss_total": loss_total,
        "loss_gt_teacher": loss_gt_teacher,
        "loss_gt_student": loss_gt_student,
        "loss_dist": loss_dist,
    }


def get_lambda(step: int, total_steps: int) -> float:
    """Linear λ curriculum: 1 → 0 over training.

    Args:
        step: Current training step.
        total_steps: Total number of training steps.

    Returns:
        Current λ value.
    """
    return max(0.0, 1.0 - step / total_steps)


# ---------------------------------------------------------------------------
# DDPM Reverse Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddpm_sample_step(
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    schedule: dict,
) -> torch.Tensor:
    """Single DDPM reverse sampling step: x_t → x_{t-1}.

    Args:
        model_output: [B, C, H, W] predicted ε from the model.
        x_t: [B, C, H, W] current noisy state.
        t: Current timestep (integer, 0-indexed).
        schedule: Precomputed noise schedule dictionary.

    Returns:
        [B, C, H, W] x_{t-1}.
    """
    betas = schedule["betas"]
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"]
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"]
    posterior_variance = schedule["posterior_variance"]

    # Predict mean: μ = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε̂)
    pred_mean = sqrt_recip_alphas[t] * (
        x_t - betas[t] / sqrt_one_minus_alphas_cumprod[t] * model_output
    )

    if t == 0:
        return pred_mean
    else:
        noise = torch.randn_like(x_t)
        return pred_mean + torch.sqrt(posterior_variance[t]) * noise


@torch.no_grad()
def ddpm_sample_loop(
    model,
    i_base: torch.Tensor,
    schedule: dict,
    num_loops: Optional[int] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> torch.Tensor:
    """Full DDPM reverse sampling loop: x_T → x_0 (predicted residual R̂).

    Args:
        model: ELT-SR model.
        i_base: [B, 3, 32, 32] bicubic-upsampled LQ image.
        schedule: Noise schedule dictionary.
        num_loops: Number of ELT loops for Any-Time inference (default: max).
        device: Device.
        verbose: Print progress.

    Returns:
        [B, 3, 32, 32] predicted clean residual R̂.
    """
    B = i_base.shape[0]
    T = schedule["num_timesteps"]

    # Start from pure noise
    x_t = torch.randn_like(i_base)

    for t_idx in reversed(range(T)):
        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)

        # Model prediction
        eps_pred = model.predict(x_t, i_base, t_batch, num_loops=num_loops)

        # DDPM step
        x_t = ddpm_sample_step(eps_pred, x_t, t_idx, schedule)

        if verbose and (t_idx % 100 == 0 or t_idx < 5):
            print(f"  Sampling step {T - t_idx}/{T}")

    return x_t  # This is R̂


# ---------------------------------------------------------------------------
# DDIM Reverse Sampling (faster)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample_loop(
    model,
    i_base: torch.Tensor,
    schedule: dict,
    ddim_steps: int = 50,
    eta: float = 0.0,
    num_loops: Optional[int] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> torch.Tensor:
    """DDIM reverse sampling loop (deterministic when eta=0).

    Args:
        model: ELT-SR model.
        i_base: [B, 3, 32, 32] bicubic-upsampled LQ image.
        schedule: Noise schedule dictionary.
        ddim_steps: Number of DDIM sampling steps.
        eta: Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM-like).
        num_loops: Number of ELT loops for Any-Time inference.
        device: Device.
        verbose: Print progress.

    Returns:
        [B, 3, 32, 32] predicted clean residual R̂.
    """
    B = i_base.shape[0]
    T = schedule["num_timesteps"]
    alphas_cumprod = schedule["alphas_cumprod"].to(device)

    # Create DDIM timestep subsequence (evenly spaced)
    ddim_timesteps = torch.linspace(0, T - 1, ddim_steps, dtype=torch.long, device=device)
    ddim_timesteps = ddim_timesteps.flip(0)  # Reverse: T-1 → 0

    # Start from pure noise
    x_t = torch.randn_like(i_base)

    for i, t_idx in enumerate(ddim_timesteps):
        t_batch = torch.full((B,), t_idx.item(), device=device, dtype=torch.long)

        # Model prediction
        eps_pred = model.predict(x_t, i_base, t_batch, num_loops=num_loops)

        # Current and next alpha_bar
        alpha_bar_t = alphas_cumprod[t_idx.item()]

        if i + 1 < len(ddim_timesteps):
            alpha_bar_prev = alphas_cumprod[ddim_timesteps[i + 1].item()]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # Predict x_0
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # Direction pointing to x_t
        sigma = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        )
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0)) * eps_pred

        # DDIM step
        noise = torch.randn_like(x_t) if sigma > 0 else torch.zeros_like(x_t)
        x_t = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

        if verbose and (i % max(1, ddim_steps // 10) == 0):
            print(f"  DDIM step {i + 1}/{ddim_steps}")

    return x_t  # This is R̂


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_hq(i_base: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Reconstruct HQ image from base + predicted residual.

    I_HQ = I_base + R̂

    Args:
        i_base: [B, 3, 32, 32] bicubic-upsampled LQ.
        residual: [B, 3, 32, 32] predicted residual R̂.

    Returns:
        [B, 3, 32, 32] reconstructed HQ image, clamped to [0, 1].
    """
    return torch.clamp(i_base + residual, 0.0, 1.0)
