from typing import List, Literal, Optional, Union

import torch
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler, SchedulerMixin
from tqdm import tqdm

from ddpm.training.callbacks import DiffusionCallback
from ddpm.models.unet import Unet


class Diffusion:
    
    def __init__(
        self,
        unet: Unet,
        train_timesteps: int,
        # TODO: Sample timesteps passed into sample function directly or include a "set_sample_timesteps" function
        sample_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "cosine"] = "linear",
        device: torch.device = "cpu",
    ):
        self.unet = unet
        self.device = device
        
        self.train_timesteps = train_timesteps
        # TODO: Pass in sample timesteps into sample + ddim_inversion methods and not in constructor
        self.sample_timesteps = sample_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = {
            "linear": "linear",
            "scaled_linear": "scaled_linear",
            "cosine": "squaredcos_cap_v2",
        }[beta_schedule]

        # Image generation schedulers
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
        )
        self.ddpm_scheduler.alphas_cumprod = self.ddpm_scheduler.alphas_cumprod.to(device=self.device)
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
        )
        self.ddim_scheduler.alphas_cumprod = self.ddim_scheduler.alphas_cumprod.to(device=self.device)

        # Image inversion scheduler
        self.ddim_inverse_scheduler = DDIMInverseScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
        )
        self.ddim_inverse_scheduler.alphas_cumprod = self.ddim_inverse_scheduler.alphas_cumprod.to(device=self.device)

    def get_timesteps(self, timesteps: Literal["train", "sample"] = "train") -> int:
        if timesteps == "sample":
            return self.sample_timesteps
        elif timesteps == "train":
            return self.train_timesteps
        else:
            raise ValueError(f"`timesteps` must be either 'sample' or 'train' not {timesteps}")

    def get_prediction_scheduler(self, deterministic: bool, timesteps: Literal["train", "sample"]) -> Union[DDPMScheduler, DDIMScheduler]:
        timesteps = self.get_timesteps(timesteps)
        scheduler = self.ddim_scheduler if deterministic else self.ddpm_scheduler
        scheduler.set_timesteps(num_inference_steps=timesteps, device=self.device)
        return scheduler

    def _guidance(
        self,
        x: torch.FloatTensor,
        t: torch.LongTensor,
        null_token: torch.FloatTensor,
        conditions: torch.FloatTensor,
        guidance_scale: float,
    ) -> torch.FloatTensor:
        eps_uncond = self.noise_pred(x, t, null_token)
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
        null_token: Optional[torch.FloatTensor] = None,
        conditions: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 0.0,
        deterministic: bool = False,
        timesteps: Literal["train", "sample"] = "train",
        generator: Optional[torch.Generator] = None,
        callbacks: Optional[List[DiffusionCallback]] = None,
        disable_progress_bar: bool = False,
    ) -> torch.FloatTensor:
        # (batch_size, 1, dims)
        x = xT
        _timesteps = self.get_timesteps(timesteps)
        scheduler = self.get_prediction_scheduler(deterministic, timesteps)
        if callbacks is not None:
            for callback in callbacks:
                callback(x, None, None)
        for t in tqdm(reversed(range(_timesteps)), desc="Sampling", disable=disable_progress_bar):
            t_vector = torch.full((x.shape[0],), t, dtype=torch.long, device=self.device)
            if conditions is not None:
                eps = self._guidance(x, t_vector, null_token, conditions, guidance_scale)
            else:
                eps = self.noise_pred(x, t_vector, None)
            # TODO: Include next_step and prev_step interface
            x = scheduler.step(eps, t, x, generator=generator).prev_sample
            if callbacks is not None:
                for callback in callbacks:
                    callback(x, t_vector, eps)
        return x

    @torch.no_grad()
    def ddim_inversion(
        self,
        x0: torch.FloatTensor,
        conditions: Optional[torch.FloatTensor] = None,
        timesteps: Literal["train", "sample"] = "train",
        callbacks: Optional[List[DiffusionCallback]] = None,
        disable_progress_bar: bool = False,
    ) -> torch.FloatTensor:
        # (batch_size, 1, dims)
        x = x0
        _timesteps = self.get_timesteps(timesteps)
        self.ddim_inverse_scheduler.set_timesteps(num_inference_steps=_timesteps, device=self.device)
        if callbacks is not None:
            for callback in callbacks:
                callback(x, None, None)
        for t in tqdm(range(0, _timesteps), desc="Inversion", disable=disable_progress_bar):
            t_vector = torch.full((x.shape[0],), t, dtype=torch.long, device=self.device)
            eps = self.noise_pred(x, t_vector, conditions)
            x = self.ddim_inverse_scheduler.step(eps, t, x).prev_sample
            if callbacks is not None:
                for callback in callbacks:
                    callback(x, t, eps)
        return x
