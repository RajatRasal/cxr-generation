from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Type, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from torch.optim import Adam
from tqdm import tqdm

from semantic_editing.attention import AttentionStore, AttendExciteCrossAttnProcessor, prepare_unet


class StableDiffusionAdapter:

    def __init__(self, model: StableDiffusionPipeline, ddim_steps: int = 50):
        self.model = model
        self.text_encoder = self.model.text_encoder
        self.ddim_steps = ddim_steps
        self.scheduler = DDIMScheduler.from_config(self.model.scheduler.config)
        self.scheduler.set_timesteps(self.ddim_steps)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.model.scheduler.config)
        self.inverse_scheduler.set_timesteps(self.ddim_steps)
        self.tokenizer = self.model.tokenizer
        self.vae = self.model.vae
        self.unet = self.model.unet
        self.device = self.model.device
        self._original_embeddings = self.get_embeddings().weight.data.clone()

    @torch.no_grad()
    def tokenise_text(self, prompt: str, string: bool = False) -> Union[torch.IntTensor, List[str]]:
        input_ids = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist()) if string else input_ids

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        text_input_ids = self.tokenise_text(prompt)
        input_ids = text_input_ids.to(self.device)
        text_embeddings = self.text_encoder(input_ids)[0]
        return text_embeddings

    def encode_text_with_grad(self, prompt: str) -> torch.Tensor:
        text_input_ids = self.tokenise_text(prompt)
        input_ids = text_input_ids.to(self.device)
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
        return self.inverse_scheduler.step(noise_pred, timestep, latent).prev_sample

    def prev_step(
        self,
        noise_pred: torch.FloatTensor,
        timestep: int,
        latent: torch.FloatTensor,
    ) -> torch.FloatTensor:
        return self.scheduler.step(noise_pred, timestep, latent).prev_sample

    def get_noise_pred(
        self,
        latent: torch.Tensor,
        timestep: int,
        embedding: torch.FloatTensor,
        zero_grad: bool = False,
    ) -> torch.FloatTensor:
        sample = self.unet(latent, timestep, encoder_hidden_states=embedding).sample
        if zero_grad:
            self.unet.zero_grad()
        return sample

    def get_timesteps(self, inversion: bool = False) -> torch.IntTensor:
        timesteps = self.scheduler.timesteps
        return timesteps.flip(dims=(0,)) if inversion else timesteps

    def get_embeddings(self) -> torch.nn.modules.sparse.Embedding:
        return self.model.text_encoder.get_input_embeddings()

    def get_text_embeddings(self, indices: List[int]) -> torch.nn.parameter.Parameter:
        return self.model.text_encoder.get_input_embeddings().weight[indices]

    def set_text_embeddings(self, text_embeddings: torch.nn.parameter.Parameter, indices: List[int]):
        assert text_embeddings.shape[0] == len(indices)
        self.model.text_encoder.get_input_embeddings().weight[indices] = text_embeddings

    @torch.no_grad()
    def reset_embeddings(self, indices_to_keep: Optional[List[int]] = None):
        if indices_to_keep is None:
            self.get_embeddings().weight[:, :] = self._original_embeddings
        else:
            mask = torch.ones(len(self.model.tokenizer), dtype=bool)
            mask[indices_to_keep] = False
            self.get_embeddings().weight[mask] = self._original_embeddings[mask]

    def register_attention_control(
        self,
        attention_store: AttentionStore,
        attention_processor_type: Type[AttendExciteCrossAttnProcessor],
    ):
        self.attention_store = attention_store
        self.unet = prepare_unet(self.unet)
        attention_processors = {}
        # TODO: Check if we need place_in_unet by looking at the experiments/tests in the DPL code
        # I have a feeling that the AttnProcessor does not need to know the place_in_unet
        # but it is helpful having this for visualisations.
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue
            attention_processors[name] = attention_processor_type(
                attention_store=self.attention_store,
                place_in_unet=place_in_unet,
            )
        self.unet.set_attn_processor(attention_processors)


@torch.no_grad()
def classifier_free_guidance_step(
     model: StableDiffusionAdapter,
     latent: torch.FloatTensor,
     prompt_embedding: torch.FloatTensor,
     null_embedding: torch.FloatTensor,
     timestep: int,
     guidance_scale: float = 7.5,
) -> torch.FloatTensor:
    noise_pred_cond = model.get_noise_pred(latent, timestep, prompt_embedding)
    noise_pred_uncond = model.get_noise_pred(latent, timestep, null_embedding)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    latent = model.prev_step(noise_pred, timestep, latent)
    return latent


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
        latent = classifier_free_guidance_step(
            model,
            latent,
            prompt_embedding,
            null_embedding,
            timestep,
            guidance_scale,
        )
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

