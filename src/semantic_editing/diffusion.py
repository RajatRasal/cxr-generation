from abc import ABC, abstractmethod
from typing import List, Literal, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from torch.optim import Adam
from tqdm import tqdm


class StableDiffusionAdapter:

    def __init__(self, model: StableDiffusionPipeline, ddim_steps: int = 50):
        self.model = model
        self.text_encoder = self.model.text_encoder
        self.ddim_steps = ddim_steps
        self.scheduler = DDIMScheduler()
        self.scheduler.set_timesteps(self.ddim_steps)
        self.tokenizer = self.model.tokenizer
        self.vae = self.model.vae
        self.unet = self.model.unet
        self.device = self.model.device

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_input.input_ids.to(self.device)
        text_embeddings = self.text_encoder(input_ids)[0]
        return text_embeddings

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        image = np.array(image)
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
        latent = self.vae.encode(image).latent_dist.mean
        latent = latent * 0.18215
        return latent

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        latent = 1 / 0.18215 * latent.detach()
        image = self.vae.decode(latent).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        return image

    def next_step(
        self,
        noise_pred: torch.FloatTensor,
        timestep: int,
        latent: torch.FloatTensor,
    ) -> torch.FloatTensor:
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_latent = (latent - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        next_latent_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred
        next_latent = alpha_prod_t_next ** 0.5 * next_original_latent + next_latent_direction
        return next_latent

    def prev_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        latent: torch.FloatTensor,
    ) -> torch.FloatTensor:
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_latent = (latent - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_latent_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_latent = alpha_prod_t_prev ** 0.5 * pred_original_latent + pred_latent_direction
        return prev_latent

    def get_noise_pred(
        self,
        latent: torch.Tensor,
        timestep: int,
        embedding: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.unet(latent, timestep, encoder_hidden_states=embedding).sample

    def get_timesteps(self, inversion: bool = False) -> torch.IntTensor:
        timesteps = self.scheduler.timesteps
        return timesteps.flip(dims=(0,)) if inversion else timesteps


@torch.no_grad()
def classifier_free_guidance(
    model: StableDiffusionAdapter,
    latent: torch.FloatTensor,
    prompt: str,
    guidance_scale: float = 7.5,
) -> List[torch.FloatTensor]:
    null_embedding = model.encode_text("")
    prompt_embedding = model.encode_text(prompt)
    latents = [latent.clone()]
    for timestep in model.get_timesteps():
        noise_pred_cond = model.get_noise_pred(latent, timestep, prompt_embedding)
        noise_pred_uncond = model.get_noise_pred(latent, timestep, null_embedding)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = model.prev_step(noise_pred, timestep, latent)
        latents.append(latent.detach())
    return latents


@torch.no_grad()
def ddim_inversion(
    model: StableDiffusionAdapter,
    image: Image.Image,
    prompt: str,
) -> List[torch.FloatTensor]:
    # TODO: Implement this using DDIMInverseScheduler in Diffusers
    latent = model.encode_image(image)
    prompt_embedding = model.encode_text(prompt)
    latents = [latent]
    # From timestep 0 to T
    for timestep in model.get_timesteps(inversion=True):
        noise_pred = model.get_noise_pred(latent, timestep, prompt_embedding)
        latent = model.next_step(noise_pred, timestep, latent)
        latents.append(latent)
    return latents

