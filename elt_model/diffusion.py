import torch, torch.nn.functional as F
from diffusers import DDPMScheduler

class ELTDiffusion:
    def __init__(self, timesteps=1000, L_max=4, L_min=1, lam_start=1.0, lam_end=0.0, total_steps=100000):
        self.sched = DDPMScheduler(num_train_timesteps=timesteps, beta_schedule="squaredcos_cap_v2")
        self.L_max, self.L_min = L_max, L_min
        self.lam_start, self.lam_end, self.total_steps = lam_start, lam_end, total_steps

    def lam(self, step):
        f = min(step/self.total_steps, 1.0)
        return self.lam_start + f*(self.lam_end-self.lam_start)

    def loss(self, model, x0, y, step):
        t = torch.randint(0, self.sched.config.num_train_timesteps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.sched.add_noise(x0, noise, t)
        eps_max, eps_int, _ = model(xt, t, y, self.L_max, self.L_min, training=True)
        l = self.lam(step)
        loss_max = F.mse_loss(eps_max, noise)
        loss_int = F.mse_loss(eps_int, noise)
        loss_dist = F.mse_loss(eps_int, eps_max.detach())
        return loss_max + l*loss_int + (1-l)*loss_dist

    @torch.no_grad()
    def sample(self, model, n, y, L, device, img_size=32, steps=100, cfg=1.0):
        self.sched.set_timesteps(steps)
        x = torch.randn(n, 3, img_size, img_size, device=device)
        y_null = torch.full_like(y, 10)
        for t in self.sched.timesteps:
            tb = torch.full((n,), t, device=device, dtype=torch.long)
            if cfg != 1.0:
                eps_c = model(x, tb, y, L, training=False)
                eps_u = model(x, tb, y_null, L, training=False)
                eps = eps_u + cfg*(eps_c-eps_u)
            else:
                eps = model(x, tb, y, L, training=False)
            x = self.sched.step(eps, t, x).prev_sample
        return x.clamp(-1, 1)
