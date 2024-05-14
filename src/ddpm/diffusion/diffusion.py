from typing import List, Literal, Optional, Union

import torch
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from tqdm import tqdm

from ddpm.training.callbacks import DiffusionCallback
from ddpm.models.unet import Unet


class Diffusion:
    
    def __init__(
        self,
        unet: Unet,
        train_timesteps: int,
        sample_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "cosine"] = "linear",
        device: torch.device = "cpu",
    ):
        self.unet = unet
        self.device = device
        
        self.train_timesteps = train_timesteps
        self.sample_timesteps = sample_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = {
            "linear": "linear",
            "scaled_linear": "scaled_linear",
            "cosine": "squaredcos_cap_v2",
        }[beta_schedule]

        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
        )
        self.ddpm_scheduler.alphas_cumprod = self.ddpm_scheduler.alphas_cumprod.to(device=self.device)
        self.ddpm_scheduler.set_timesteps(num_inference_steps=self.sample_timesteps, device=self.device)
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
        )
        self.ddim_scheduler.alphas_cumprod = self.ddim_scheduler.alphas_cumprod.to(device=self.device)
        self.ddim_scheduler.set_timesteps(num_inference_steps=self.sample_timesteps, device=self.device)

    def _step(
        self,
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        eps: torch.FloatTensor,
        t: torch.LongTensor,
        x: torch.FloatTensor,
        generator: torch.Generator,
    ) -> torch.FloatTensor:
        return scheduler.step(eps, t, x, generator=generator).prev_sample

    def _guidance(
        self,
        x: torch.FloatTensor,
        t: torch.LongTensor,
        conditions: torch.FloatTensor,
        guidance_scale: float,
    ) -> torch.FloatTensor:
        eps_uncond = self.noise_pred(x, t, None)
        eps_cond = self.noise_pred(x, t, conditions)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return eps

    def noise_pred(
        self,
        latents: torch.FloatTensor,
        timesteps: torch.LongTensor,
        conditions: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        return self.unet(latents, timesteps, conditions)

    def add_noise(self, x0: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.LongTensor) -> torch.FloatTensor:
        return self.ddpm_scheduler.add_noise(x0, noise, timesteps)

    @torch.no_grad()
    def sample(
        self,
        xT: torch.FloatTensor,
        conditions: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 0.0,
        deterministic: bool = False,
        generator: Optional[torch.Generator] = None,
        callbacks: Optional[List[DiffusionCallback]] = None,
    ) -> torch.FloatTensor:
        x = xT
        timesteps = self.sample_timesteps if deterministic else self.train_timesteps
        scheduler = self.ddim_scheduler if deterministic else self.ddpm_scheduler
        for t in tqdm(reversed(range(timesteps)), desc="Sampling"):
            t_vector = torch.full((x.shape[0],), t, dtype=torch.long, device=self.device)
            if conditions is not None:
                eps = self._guidance(x, t_vector, conditions, guidance_scale)
            else:
                eps = self.noise_pred(x, t_vector, None)
            x = self._step(scheduler, eps, t, x, generator)
            if callbacks is not None:
                for callback in callbacks:
                    callback(x, t, eps)
        return x
