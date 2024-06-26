from typing import Any, Dict, List, Literal, Optional, Union

import torch
from diffusers import UNet2DConditionModel, UNet2DModel
from diffusers.schedulers import DDPMScheduler, DDIMScheduler, DDIMInverseScheduler, SchedulerMixin
from tqdm import tqdm

from ddpm.models.one_dimension.unet import Unet
from ddpm.training.callbacks import DiffusionCallback


class Diffusion:
    
    def __init__(
        self,
        unet: Union[Unet, UNet2DConditionModel, UNet2DModel],
        train_timesteps: int,
        # TODO: Sample timesteps passed into sample function directly or include a "set_sample_timesteps" function
        # sample_timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "scaled_linear", "cosine"] = "linear",
        rescale_betas_zero_snr: bool = False,
        device: torch.device = "cpu",
    ):
        self.unet = unet
        # TODO: Create unet interface which can wrap both implementations
        if isinstance(self.unet, Unet):
            self._using_diffusers = "unet"
        elif isinstance(self.unet, UNet2DConditionModel):
            self._using_diffusers = "unet_diffusers_cond"
        elif isinstance(self.unet, UNet2DModel):
            self._using_diffusers = "unet_diffusers"
        else:
            return RuntimeError(f"Unet is type {type(self.unet)}, but must be either Unet or UNet2DConditionModel")
        self.device = device
        
        self.train_timesteps = train_timesteps
        # TODO: Pass in sample timesteps into sample + ddim_inversion methods and not in constructor
        # self.sample_timesteps = sample_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = {
            "linear": "linear",
            "scaled_linear": "scaled_linear",
            "cosine": "squaredcos_cap_v2",
        }[beta_schedule]
        self.rescale_betas_zero_snr = rescale_betas_zero_snr

        # Image generation schedulers
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        self.ddpm_scheduler.alphas_cumprod = self.ddpm_scheduler.alphas_cumprod.to(device=self.device)
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        self.ddim_scheduler.alphas_cumprod = self.ddim_scheduler.alphas_cumprod.to(device=self.device)

        # Image inversion scheduler
        self.ddim_inverse_scheduler = DDIMInverseScheduler(
            num_train_timesteps=self.train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_schedule=self.beta_schedule,
            prediction_type="epsilon",
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        self.ddim_inverse_scheduler.alphas_cumprod = self.ddim_inverse_scheduler.alphas_cumprod.to(device=self.device)

    def get_prediction_scheduler(
        self,
        deterministic: bool,
        timesteps: int,
    ) -> Union[DDPMScheduler, DDIMScheduler]:
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
        cross_attn_kwargs: Dict[str, Any] = {},
    ) -> torch.FloatTensor:
        eps_uncond = self.noise_pred(x, t, null_token, {})
        eps_cond = self.noise_pred(x, t, conditions, cross_attn_kwargs)
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return eps

    def noise_pred(
        self,
        latents: torch.FloatTensor,
        timesteps: torch.LongTensor,
        conditions: Optional[torch.FloatTensor] = None,
        cross_attn_kwargs: Dict[str, Any] = {},
    ) -> torch.FloatTensor:
        # TODO: Wrap up all unets behind an interface
        if self._using_diffusers == "unet_diffusers_cond":
            return self.unet(
                sample=latents,
                timestep=timesteps,
                encoder_hidden_states=conditions,
                cross_attention_kwargs=cross_attn_kwargs,
            ).sample
        elif self._using_diffusers == "unet_diffusers":
            return self.unet(
                sample=latents,
                timestep=timesteps,
                class_labels=conditions,
            ).sample
        else:
            return self.unet(latents, timesteps, conditions, cross_attn_kwargs)

    def add_noise(
        self,
        x0: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.LongTensor,
    ) -> torch.FloatTensor:
        return self.ddpm_scheduler.add_noise(x0, noise, timesteps)

    @torch.no_grad()
    def sample(
        self,
        xT: torch.FloatTensor,
        null_token: Optional[torch.FloatTensor] = None,
        conditions: Optional[torch.FloatTensor] = None,
        guidance_scale: float = 0.0,
        deterministic: bool = False,
        timesteps: int = 50,
        do_cfg: bool = False,
        generator: Optional[torch.Generator] = None,
        callbacks: Optional[List[DiffusionCallback]] = None,
        disable_progress_bar: bool = False,
        scheduler_step_kwargs: Dict[str, Any] = {},
    ) -> torch.FloatTensor:
        x = xT
        scheduler = self.get_prediction_scheduler(deterministic, timesteps)
        if callbacks is not None:
            for callback in callbacks:
                callback(x, None, None)
        for i, t in tqdm(enumerate(scheduler.timesteps), desc="Sampling", disable=disable_progress_bar):
            t_vector = torch.full((x.shape[0],), t, dtype=torch.long, device=self.device)
            if conditions is None:
                # TODO: This should be nullable also
                eps = self.noise_pred(x, t_vector, null_token)
            else:
                if do_cfg:
                    eps = self._guidance(x, t_vector, null_token, conditions, guidance_scale)
                else:   
                    eps = self.noise_pred(x, t_vector, conditions) 
            # TODO: Include next_step and prev_step interface
            x = scheduler.step(eps, t, x, generator=generator, **scheduler_step_kwargs).prev_sample
            if callbacks is not None:
                for callback in callbacks:
                    callback(x, t_vector, eps)
        return x

    @torch.no_grad()
    def ddim_inversion(
        self,
        x0: torch.FloatTensor,
        conditions: Optional[torch.FloatTensor] = None,
        timesteps: int = 50,
        callbacks: Optional[List[DiffusionCallback]] = None,
        use_clipping: bool = True,
        disable_progress_bar: bool = False,
    ) -> torch.FloatTensor:
        x = x0
        self.ddim_scheduler.set_timesteps(
            num_inference_steps=timesteps,
            device=self.device,
        )
        if callbacks is not None:
            for callback in callbacks:
                callback(x, None, None)
        timesteps = reversed(self.ddim_scheduler.timesteps)
        for t in tqdm(timesteps, desc="Inversion", disable=disable_progress_bar):
            t_vector = torch.full((x.shape[0],), t, dtype=torch.long, device=self.device)
            eps = self.noise_pred(x, t_vector, conditions)
            x = self.ddim_scheduler.step(eps, t, x, use_clipped_model_output=use_clipping).prev_sample
            if callbacks is not None:
                for callback in callbacks:
                    callback(x, t, eps)
        return x
