import math
import torch
import torch.nn.functional as F

class DiffusionSchedule:
    def __init__(self, timesteps: int = 1000, s: float = 0.008):
        self.timesteps = timesteps
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / timesteps) + s) / (1 + s) * (math.pi / 2)) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

        self.betas = torch.clamp(betas, 1e-4, 0.02).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x_start)
        s_ac = self.sqrt_alphas_cumprod[t.cpu()].view(-1, 1, 1, 1).to(x_start.device)
        s_omac = self.sqrt_one_minus_alphas_cumprod[t.cpu()].view(-1, 1, 1, 1).to(x_start.device)
        return s_ac * x_start + s_omac * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, i_base: torch.Tensor, i_lq: torch.Tensor, t: torch.Tensor, t_idx: int, num_loops: int):
        betas_t = self.betas[t_idx].to(x_t.device)
        s_omac = self.sqrt_one_minus_alphas_cumprod[t_idx].to(x_t.device)
        s_ra = torch.sqrt(1.0 / self.alphas[t_idx]).to(x_t.device)

        pred_noise = model.predict(x_t, i_base, i_lq, t, num_loops=num_loops)
        model_mean = s_ra * (x_t - (betas_t / s_omac) * pred_noise)

        if t_idx == 0:
            return model_mean
        else:
            var = self.posterior_variance[t_idx].to(x_t.device)
            return model_mean + torch.sqrt(var) * torch.randn_like(x_t)

    @torch.no_grad()
    def sample_sr(self, model, i_base: torch.Tensor, i_lq: torch.Tensor, shape: torch.Size, device: torch.device, num_loops: int = 3):
        x_t = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, i_base, i_lq, t, i, num_loops)
        return x_t

def sigmoid_weight(t: torch.Tensor, num_timesteps: int) -> torch.Tensor:
    t_norm = t.float() / num_timesteps
    weight = 1.0 / (1.0 + torch.exp(-12.0 * (t_norm - 0.5)))
    return 0.5 + weight

def compute_ilsd_loss(eps_teacher: torch.Tensor, eps_student: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, num_timesteps: int, lam: float) -> dict:
    weights = sigmoid_weight(t, num_timesteps).view(-1, 1, 1, 1)
    loss_gt_teacher = (weights * F.l1_loss(eps_teacher, noise, reduction="none")).mean()
    loss_gt_student = (weights * F.l1_loss(eps_student, noise, reduction="none")).mean()
    loss_dist = (weights * F.l1_loss(eps_student, eps_teacher.detach(), reduction="none")).mean()

    return {
        "loss_total": loss_gt_teacher + lam * loss_gt_student + (1.0 - lam) * loss_dist,
        "loss_gt_teacher": loss_gt_teacher,
    }
