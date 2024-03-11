from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from torch.optim import Adam
from tqdm import tqdm

from semantic_editing.base import CFGOptimisation, NULL_STRING
from semantic_editing.diffusion import StableDiffusionAdapter, classifier_free_guidance, classifier_free_guidance_step, ddim_inversion
from semantic_editing.utils import seed_everything, init_stable_diffusion, plot_image_on_axis


class NullTokenOptimisation(CFGOptimisation):

    def __init__(
        self,
        model: StableDiffusionAdapter,
        guidance_scale: int,
        num_inner_steps: int = 50,
        learning_rate: float = 1e-2,
        image_size: int = 512, 
        epsilon: float = 1e-5,
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.num_inner_steps = num_inner_steps
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.epsilon = epsilon

    def _cfg_with_prompt_noise_pred(
        self,
        latent_cur: torch.FloatTensor,
        timestep: torch.IntTensor,
        null_embedding: torch.FloatTensor,
        noise_pred_cond: torch.FloatTensor,
    ) -> torch.FloatTensor:
        noise_pred_uncond = self.model.get_noise_pred(latent_cur, timestep, null_embedding)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent_prev = self.model.prev_step(noise_pred, timestep, latent_cur)
        return latent_prev

    def fit(self, image: Image.Image, prompt: str):
        # TODO: Move epsilon into init
        image = image.resize((self.image_size, self.image_size))

        self.latents = ddim_inversion(self.model, image, prompt)
        n_latents = len(self.latents)

        null_embeddings = []
        null_embedding = self.model.encode_text(NULL_STRING)
        prompt_embedding = self.model.encode_text(prompt)

        bar = tqdm(total=self.num_inner_steps * self.model.ddim_steps)
        latent_cur = self.latents[-1]
        timesteps = self.model.get_timesteps()
        n_timesteps = len(timesteps)
        for i in range(n_timesteps):
            null_embedding = null_embedding.clone().detach()
            null_embedding.requires_grad = True
            lr_scale_factor = 1. - i / (n_timesteps * 2)
            optimizer = Adam([null_embedding], lr=self.learning_rate * lr_scale_factor)
            latent_prev = self.latents[n_latents - i - 2]
            t = timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.model.get_noise_pred(
                    latent_cur,
                    t,
                    prompt_embedding,
                )
            for j in range(self.num_inner_steps):
                latent_prev_rec = self._cfg_with_prompt_noise_pred(
                    latent_cur,
                    t,
                    null_embedding,
                    noise_pred_cond,
                )
                loss = F.mse_loss(latent_prev_rec, latent_prev)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_item = loss.item()
                bar.update()
                if loss_item < self.epsilon + i * 2e-5:
                    break
            for j in range(j + 1, self.num_inner_steps):
                bar.update()
            null_embeddings.append(null_embedding[:1].detach())
            with torch.no_grad():
                latent_cur = self._cfg_with_prompt_noise_pred(
                    latent_cur,
                    t,
                    null_embedding,
                    noise_pred_cond,
                )
        bar.close()

        self.null_embeddings = null_embeddings

    @torch.no_grad()
    def generate(self, prompt: str, edit_scale: float = 0.8) -> Image.Image:
        if not (hasattr(self, "null_embeddings") and hasattr(self, "latents")):
            raise ValueError(f"Need to fit {self.__class__.__name__} on an image before generating")

        target_prompt_embedding = self.model.encode_text(prompt)
        # TODO: Move this into model adapter
        latent = self.latents[-1].expand(
            1,
            self.model.unet.config.in_channels,
            self.image_size // 8,
            self.image_size // 8
        ).to(self.model.device)

        for i, timestep in enumerate(tqdm(self.model.get_timesteps())):
            latent = classifier_free_guidance_step(
                self.model,
                latent,
                target_prompt_embedding,
                self.null_embeddings[i],
                timestep,
                self.guidance_scale,
            )

        image = self.model.decode_latent(latent)
        return image


def main():
    seed = 88
    seed_everything(seed)

    model = StableDiffusionAdapter(init_stable_diffusion(), 50)

    guidance_scale = 7.5
    size = 512
    epsilon = 1e-5
    image = Image.open("/vol/biomedic3/rrr2417/cxr-generation/notebooks/cat_mirror.jpeg")
    source_prompt = "a cat sitting in front of a mirror"

    nti = NullTokenOptimisation(model, guidance_scale, 10, size)
    nti.fit(image, source_prompt, epsilon)

    recon = nti.generate(source_prompt)

    target_prompt = "a lion sitting in front of a mirror"
    edited = nti.generate(target_prompt)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    plot_image_on_axis(axs[0], image, "Original")
    plot_image_on_axis(axs[1], recon.resize(image.size), "NTI Recon")
    plot_image_on_axis(axs[2], edited.resize(image.size), "NTI Edited")
    fig.savefig("test_nti.pdf", bbox_inches="tight")
