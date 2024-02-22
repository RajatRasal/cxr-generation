from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from torch.optim import Adam
from tqdm import tqdm

from semantic_editing.diffusion import StableDiffusionAdapter, classifier_free_guidance, ddim_inversion
from semantic_editing.utils import seed_everything, init_stable_diffusion, plot_image_on_axis


NULL_STRING = ""


class CFGOptimisation(ABC):

    # @property
    # @abstractmethod
    # def model(self) -> StableDiffusionAdapter:
    #     raise NotImplementedError

    # @property
    # @abstractmethod
    # def guidance_scale(self) -> int:
    #     raise NotImplementedError

    @abstractmethod
    def fit(self, image: Image.Image, prompt: str):
        raise NotImplementedError

    @abstractmethod
    def generate(self, prompt: str) -> Image.Image:
        pass


class NullTokenOptimisation(CFGOptimisation):

    def __init__(
        self,
        model: StableDiffusionAdapter,
        guidance_scale: int,
        num_inner_steps: int = 50,
        learning_rate: float = 1e-2,
        image_size: int = 512, 
    ):
        self.model = model
        # TODO: Create a wrapper for a scheduler which implements the inversion method
        # e.g.
        # DDIMInverseSchedulerAdapter, with method inversion(self, model, image, prompt) 
        # that returns the list of latents given an image and a prompt
        self.guidance_scale = guidance_scale
        self.num_inner_steps = num_inner_steps
        self.learning_rate = learning_rate
        self.image_size = image_size

    def _cfg_with_prompt_noise_pred(self, latent_cur: torch.FloatTensor, timestep: torch.IntTensor, null_embedding: torch.FloatTensor, noise_pred_cond: torch.FloatTensor) -> torch.FloatTensor:
        noise_pred_uncond = self.model.get_noise_pred(latent_cur, timestep, null_embedding)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent_prev = self.model.prev_step(noise_pred, timestep, latent_cur)
        return latent_prev

    def fit(self, image: Image.Image, prompt: str, epsilon: int = 0):
        image = image.resize((self.image_size, self.image_size))

        self.latents = ddim_inversion(self.model, image, prompt)

        null_embeddings = []
        null_embedding = self.model.encode_text(NULL_STRING)
        prompt_embedding = self.model.encode_text(prompt)

        bar = tqdm(total=self.num_inner_steps * self.model.ddim_steps)
        latent_cur = self.latents[-1]
        timesteps = self.model.get_timesteps()
        for i in range(len(timesteps)):
            null_embedding = null_embedding.clone().detach()
            null_embedding.requires_grad = True
            optimizer = Adam([null_embedding], lr=self.learning_rate * (1. - i / 100.))
            latent_prev = self.latents[len(self.latents) - i - 2]
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
                if loss_item < epsilon + i * 2e-5:
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
    def generate(self, prompt: str) -> Image.Image:
        if not (hasattr(self, "null_embeddings") and hasattr(self, "latents")):
            assert ValueError(f"Need to fit {self.__class__.__name__} on an image before generating")

        prompt_embedding = self.model.encode_text(prompt)
        # TODO: Move this into model adapter
        latent = self.latents[-1].expand(
            1,
            self.model.unet.config.in_channels,
            self.image_size // 8,
            self.image_size // 8
        ).to(self.model.device)

        for i, timestep in enumerate(tqdm(self.model.get_timesteps())):
            noise_pred_cond = self.model.get_noise_pred(latent, timestep, prompt_embedding)
            noise_pred_uncond = self.model.get_noise_pred(latent, timestep, self.null_embeddings[i])
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latent = self.model.prev_step(noise_pred, timestep, latent)

        image = self.model.decode_latent(latent)
        return image


class PromptTokenOptimisation(CFGOptimisation):
    
    def __init__(self, model: StableDiffusionAdapter):
        self.model = model

    def fit(self, image: Image.Image, prompt: str):
        raise NotImplementedError

    def generate(self, prompt: str) -> Image.Image:
        raise NotImplementedError


class CFGWithDDIM(CFGOptimisation):

    def __init__(
        self,
        model: StableDiffusionAdapter,
        guidance_scale: int,
        image_size: Optional[int] = None,
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.image_size = image_size

    def fit(self, image: Image.Image, prompt: str):
        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size))
        self.latent_T = ddim_inversion(self.model, image, prompt)[-1]

    def generate(self, prompt: str) -> Image.Image:
        if not hasattr(self, "latent_T"):
            assert ValueError(f"Need to fit {self.__class__.__name__} on an image before generating")

        latents = classifier_free_guidance(
            self.model,
            self.latent_T.clone(),
            prompt,
            self.guidance_scale,
        )
        latent_0 = latents[-1]

        image = self.model.decode_latent(latent_0)
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
