import random
import itertools as it
from functools import lru_cache
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.optim import Adam
from tqdm import tqdm

from .ptp_utils import register_attention_control, load_512


class NullTextInversion:

    def __init__(self, model, ddim_steps: int, guidance_scale: int):
        self.model = model
        self.ddim_steps = ddim_steps
        self.guidance_scale = guidance_scale
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.ddim_steps)
        self.prompt = None
        self.context = None

    def prev_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents: torch.FloatTensor, t: int, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.model.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def _embed_text(self, prompt: str) -> torch.tensor:
        tokens = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
            truncation=True,
        )
        embedding = self.model.text_encoder(tokens.input_ids.to(self.model.device))[0]
        return embedding

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent, prompt):
        # uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings = self._embed_text("")
        cond_embeddings = self._embed_text(prompt)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, prompt, rec=True):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent) if rec else None
        ddim_latents = self.ddim_loop(latent, prompt)
        return image_rec, ddim_latents

    def _null_optimization(self, latents, null_embeddings, prompt_embedding, num_inner_steps, lr_scale_factor: float = 1e-2):
        optimised_null_embeddings = []
        latent_cur = latents[-1]
        total_loss = 0
        for i, null_embedding in zip(range(self.ddim_steps), null_embeddings):
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            null_embedding.requires_grad = True
            optimizer = Adam([null_embedding], lr=lr_scale_factor * (1. - i / 100.))
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, prompt_embedding)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, null_embedding)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = F.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            optimised_null_embeddings.append(null_embedding[:1].detach())
            with torch.no_grad():
                context = torch.cat([null_embedding, prompt_embedding])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        mean_loss = total_loss / self.ddim_steps * num_inner_steps
        return optimised_null_embeddings, mean_loss

    # def prompt_optimization(self, latents, num_inner_steps, epsilon):
    #     uncond_embeddings, cond_embeddings = self.context.chunk(2)
    #     uncond_embeddings_list = []
    #     cond_embeddings_list = []
    #     latent_cur = latents[-1]
    #     bar = tqdm(total=num_inner_steps * self.ddim_steps)
    #     for i in range(self.ddim_steps):
    #         cond_embeddings = cond_embeddings.clone().detach()
    #         cond_embeddings.requires_grad = True
    #         optimizer = Adam([cond_embeddings], lr=1e-2 * (1. - i / 100.))
    #         latent_prev = latents[len(latents) - i - 2]
    #         t = self.model.scheduler.timesteps[i]
    #         with torch.no_grad():
    #             noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
    #         for j in range(num_inner_steps):
    #             noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
    #             noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
    #             latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
    #             loss = F.mse_loss(latents_prev_rec, latent_prev)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             loss_item = loss.item()
    #             bar.update()
    #             if loss_item < epsilon + i * 2e-5:
    #                 break
    #         for j in range(j + 1, num_inner_steps):
    #             bar.update()
    #         uncond_embeddings_list.append(uncond_embeddings[:1].detach())
    #         cond_embeddings_list.append(cond_embeddings[:1].detach())
    #         with torch.no_grad():
    #             context = torch.cat([uncond_embeddings, cond_embeddings])
    #             latent_cur = self.get_noise_pred(latent_cur, t, False, context)
    #     bar.close()
    #     return uncond_embeddings_list, cond_embeddings_list
    
    def fit(
        self,
        images: List[Image.Image],
        prompts: List[str],
        max_steps: int,
        num_inner_steps: int = 10,
        seed: int = 45,
        lr_scale_factor: float = 1e-5,
    ):
        null_embedding = self._embed_text("")
        null_embeddings = [null_embedding.clone().detach() for _ in range(self.ddim_steps)]
        random.seed(seed)
        dataset = zip(images, prompts)
        for _, (image, prompt) in tqdm(zip(range(max_steps), it.cycle(dataset)), total=max_steps):
            image_gt = load_512(image)
            prompt_embedding = self._embed_text(prompt)
            _, ddim_latents = self.ddim_inversion(image_gt, prompt, rec=False)
            optimised_null_embeddings, mean_loss = self._null_optimization(
                ddim_latents,
                null_embeddings,
                prompt_embedding,
                num_inner_steps=num_inner_steps,
                lr_scale_factor=lr_scale_factor,
            )
            null_embeddings = [
                embedding.clone().detach()
                for embedding in optimised_null_embeddings
            ]
        return null_embeddings
    
    # def invert(
    #     self,
    #     image_path: str,
    #     prompt: str,
    #     offsets: Tuple[int, int, int, int] = (0, 0, 0, 0),
    #     num_inner_steps: int = 10,
    #     early_stop_epsilon: float = 1e-5,
    #     tune_prompts: bool = False,
    #     verbose = False,
    # ):
    #     # Compute null text value
    #     # Compute embeddings for all strings in the initial dataset
    #     self.init_prompt(prompt)

    #     # Register attention control for prompt-to-prompt
    #     register_attention_control(self.model, None)

        
    #     image_gt = load_512(image_path, *offsets)
    #     if verbose:
    #         print("DDIM inversion...")
    #     image_rec, ddim_latents = self.ddim_inversion(image_gt)
    #     if tune_prompts:
    #         if verbose:
    #             print("Prompt tuning...")
    #         uncond_embeddings, cond_embeddings = self.prompt_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
    #     else:
    #         if verbose:
    #             print("Null-text optimization...")
    #         uncond_embeddings, cond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
    #     return (image_gt, image_rec), ddim_latents, uncond_embeddings, cond_embeddings

