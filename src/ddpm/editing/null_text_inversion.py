from typing import List, Optional

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
    ):
        self.diffusion = diffusion
        self.null_token = null_token.unsqueeze(0)
        self.nti_steps = nti_steps
        self.learning_rate = learning_rate
        self.guidance_scale = guidance_scale

    def fit(
        self,
        x: torch.FloatTensor,
        condition: torch.FloatTensor,
        ddim_inversion_callbacks: Optional[List[DiffusionCallback]] = None,
        disable_progress_bar: bool = True,
    ):
        self.x = x
        self.condition = condition
        device = self.x.device
        batch_size = x.shape[0]

        # DDIM Inversion
        trajectory_callback = TrajectoryCallback()
        self.diffusion.ddim_inversion(
            x,
            condition,
            "sample",
            callbacks=[trajectory_callback] + ddim_inversion_callbacks,
            disable_progress_bar=disable_progress_bar,
        )
        self.latents = trajectory_callback.timesteps[::-1]

        # Variables needed for NTI
        n_latents = len(self.latents)
        null_token = self.null_token.repeat((batch_size, 1, 1)).to(device)
        n_timesteps = self.diffusion.get_timesteps("sample")
        scheduler = self.diffusion.get_prediction_scheduler(False, "sample")
        latent_cur = self.latents[0].detach().to(device)
        condition = self.condition.clone().requires_grad_(False)

        # NTO is performed per timestep
        self.optimised_null_tokens = []
        for t in tqdm(range(0, n_timesteps), disable=disable_progress_bar):
            # Clone optimised null-token from previous step
            null_token = null_token.detach().clone().requires_grad_(True)

            # NTO target
            latent_prev = self.latents[t].detach().to(device).requires_grad_(False)

            # Initialise optimiser
            lr_scale_factor = 1. - (n_timesteps - t) / (n_timesteps * 2)
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
                # TODO: Include generator
                # TODO: Include next_step and prev_step interface
                latent_prev_pred = scheduler.step(eps, t, latent_cur, generator=None).prev_sample
                # Compute loss between predicted latent and true latent from DDIM inversion
                loss = F.l1_loss(latent_prev_pred, latent_prev)
                # Optimise null token
                loss.backward(retain_graph=False)
                optimiser.step()
                optimiser.zero_grad()

            self.optimised_null_tokens.append(null_token.detach().clone())

            with torch.no_grad():
                eps = self.diffusion._guidance(latent_cur, t_vector, null_token, condition, self.guidance_scale)
                # TODO: Include generator
                # TODO: Include next_step and prev_step interface
                latent_cur = scheduler.step(eps, t, latent_cur, generator=None).prev_sample
            
        assert len(self.optimised_null_tokens) == len(self.latents) - 1

    def generate(
        self,
        callbacks: Optional[List[DiffusionCallback]] = None,
        disable_progress_bar: bool = True,
    ) -> torch.FloatTensor:
        # TODO: Include generator for latents
        # Variables needed for guided generation
        n_timesteps = len(self.latents)
        device = self.x.device
        scheduler = self.diffusion.get_prediction_scheduler(False, "sample")
        n_timesteps = self.diffusion.get_timesteps("sample")
        latent = self.latents[0].to(device)
        # Store initial state in callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(latent, None, None)
        # Guided generation using optimised null-tokens
        for t, null_token in tqdm(enumerate(self.optimised_null_tokens), disable=disable_progress_bar):
            t_vector = torch.full((self.x.shape[0],), t, dtype=torch.long, device=device)
            eps = self.diffusion._guidance(
                latent,
                t_vector,
                self.optimised_null_tokens[t],
                self.condition,
                self.guidance_scale,
            )
            latent = scheduler.step(eps, t, latent, generator=None).prev_sample
            if callbacks is not None:
                for callback in callbacks:
                    callback(latent, t, eps)
        return latent
    