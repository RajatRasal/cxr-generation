from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ddpm.training.callbacks import DiffusionCallback, TrajectoryCallback
from ddpm.training.train import Diffusion


class NullTokenOptimisation:

    def __init__(
        self,
        diffusion: Diffusion,
        null_token: torch.FloatTensor,
        nti_steps: int,
        learning_rate: float,
        guidance_scale: float,
        timesteps: int,
    ):
        self.diffusion = diffusion
        # TODO: Pass in the exact null token with required dimensions
        # then repeat along the 0th dimension batch_size times in fit and generate methods.
        self.null_token = null_token.detach().clone()
        self.nti_steps = nti_steps
        self.learning_rate = learning_rate
        self.guidance_scale = guidance_scale
        self.timesteps = timesteps

    def fit(
        self,
        x: torch.FloatTensor,
        condition: torch.FloatTensor,
        ddim_inversion_callbacks: Optional[List[DiffusionCallback]] = [],
        disable_progress_bar: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        self.x = x.detach().clone().requires_grad_(False)
        self.condition = condition.detach().clone().requires_grad_(False)
        device = self.x.device
        batch_size = x.shape[0]

        # DDIM Inversion
        trajectory_callback = TrajectoryCallback()
        self.diffusion.ddim_inversion(
            x,
            condition,
            self.timesteps,
            callbacks=[trajectory_callback] + ddim_inversion_callbacks,
            disable_progress_bar=disable_progress_bar,
        )
        # From T to 0
        self.latents = trajectory_callback.data[::-1][1:]

        # Variables needed for NTI
        n_latents = len(self.latents)
        # repeats = tuple([batch_size] + [1] * (len(self.null_token.shape) - 1))
        null_token = self.null_token.clone()
        condition = self.condition.clone()
        n_timesteps = self.timesteps
        scheduler = self.diffusion.get_prediction_scheduler(False, self.timesteps)
        latent_cur = self.latents[0].detach().to(device)
        timesteps = scheduler.timesteps
        assert len(timesteps) == len(self.latents) == n_timesteps

        # NTO is performed per timestep
        self.optimised_null_tokens = []
        for i, t in tqdm(enumerate(timesteps), total=n_timesteps, disable=disable_progress_bar):
            # Clone optimised null-token from previous step
            null_token = null_token.detach().clone().requires_grad_(True)

            # NTO target
            latent_prev = self.latents[i].detach().to(device).requires_grad_(False)

            # Initialise optimiser
            lr_scale_factor = 1. - i / n_timesteps
            optimiser = torch.optim.Adam([null_token], lr=self.learning_rate * lr_scale_factor)

            # Null-token optimisation loop
            t_vector = torch.full((batch_size,), t, dtype=torch.long, device=device)
            for _ in range(self.nti_steps):
                # CFG
                eps = self.diffusion._guidance(
                    latent_cur,
                    t_vector,
                    null_token,
                    condition,
                    self.guidance_scale,
                )
                # TODO: Include next_step and prev_step interface
                # TODO: NTO callback to track null_token, losses, timesteps
                latent_prev_pred = scheduler.step(eps, t, latent_cur, generator=generator).prev_sample
                # Compute loss between predicted latent and true latent from DDIM inversion
                loss = F.mse_loss(latent_prev_pred, latent_prev)
                # Optimise null token
                loss.backward(retain_graph=False)
                optimiser.step()
                optimiser.zero_grad()

                if loss < 1e-5:
                    break

            optimised_null_token = null_token.detach().clone()
            self.optimised_null_tokens.append(optimised_null_token)

            with torch.no_grad():
                eps = self.diffusion._guidance(latent_cur, t_vector, optimised_null_token, condition, self.guidance_scale)
                # TODO: Include next_step and prev_step interface
                latent_cur = scheduler.step(eps, t, latent_cur, generator=generator).prev_sample
            
        assert len(self.optimised_null_tokens) == len(self.latents)

    def generate(
        self,
        condition: Optional[torch.FloatTensor] = None,
        swap_fraction: float = 0.25,
        callbacks: Optional[List[DiffusionCallback]] = None,
        disable_progress_bar: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.FloatTensor:
        # Variables needed for guided generation
        device = self.x.device
        scheduler = self.diffusion.get_prediction_scheduler(False, self.timesteps)
        timesteps = scheduler.timesteps
        n_timesteps = len(timesteps)
        assert len(self.optimised_null_tokens) == len(timesteps)
        latent = self.latents[0].to(device)
        if condition is not None:
            # TODO: Remove this nasty hack by passing in the correct conditions
            # TODO: Remove the hack with "corrections" argument when doing 2D NTI
            condition = torch.cat([self.condition, condition])
            latent = latent.repeat((2, 1, 1, 1))

        # Store initial state in callbacks
        if callbacks is not None:
            for callback in callbacks:
                _latent = latent if condition is None else latent.chunk(2)[1]
                callback(_latent, None, None)

        # Guided generation using optimised null-tokens
        for i, (t, null_token) in tqdm(enumerate(zip(timesteps, self.optimised_null_tokens)), total=n_timesteps, disable=disable_progress_bar):
            t_vector = torch.full((self.x.shape[0],), t, dtype=torch.long, device=device)

            if condition is None:
                # Reconstruction
                eps = self.diffusion._guidance(
                    latent,
                    t_vector,
                    null_token,
                    self.condition,
                    self.guidance_scale,
                )
            else:
                # Edit
                if self.diffusion._using_diffusers in ["unet_diffusers_cond", "unet_diffusers"]:
                    cs_kwargs = {
                        "cross_replace": i < int(n_timesteps * swap_fraction),
                        "self_replace": i < int(n_timesteps * swap_fraction),
                        "amplify_indices": {},
                    }
                else:
                    cs_kwargs = {"swap": True} if t > int(n_timesteps * swap_fraction) else {}

                # TODO: Write this repeat statement to be more generic, without the correction condition
                nt = null_token.repeat((2, 1, 1))

                eps = self.diffusion._guidance(
                    latent,
                    t_vector.repeat((2)),
                    nt, 
                    condition,
                    self.guidance_scale,
                    cs_kwargs,
                ).detach()

            latent = scheduler.step(eps, t, latent, generator=generator).prev_sample.detach()

            if callbacks is not None:
                for callback in callbacks:
                    # TODO: Include reconstruction and latent path in callback
                    _latent = latent if condition is None else latent.chunk(2)[1]
                    callback(_latent, t, eps)

        if condition is not None:
            B, C, D1, D2 = latent.shape
            latent = latent.reshape(2, B // 2, C, D1, D2)
            # CrossAttention module assumes original path is 0th and edit is 1st index.
            latent = latent[1]
        
        return latent
