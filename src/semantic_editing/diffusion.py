import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from semantic_editing.attention import AttentionStore, AttnProcessorWithAttentionStore


class PretrainedStableDiffusionAdapter:
    """
    This wrapper assumes that Stable Diffusion is not being changed in any way by any
    editing algorithms.
    """

    def __init__(self, model_name: str, ddim_steps: int = 50, device: Literal["cpu", "cuda"] = "cpu"):
        self.model_name = model_name
        self.ddim_steps = ddim_steps
        self.device = device
        self.model = StableDiffusionPipeline.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = self.model.tokenizer
        self.scheduler = DDIMScheduler.from_config(self.model.scheduler.config)
        self.scheduler.set_timesteps(self.ddim_steps)
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.model.scheduler.config)
        self.inverse_scheduler.set_timesteps(self.ddim_steps)
        self.attention_store = None

    @property
    def vae(self):
        return self.model.vae

    @property
    def unet(self):
        return self.model.unet
    
    @property
    def text_encoder(self) -> CLIPTextModel:
        return self.model.text_encoder

    def reset(self):
        self.model = StableDiffusionPipeline.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = self.model.tokenizer

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
    def add_tokens(self, tokens: List[str], init_words: List[Optional[str]]) -> List[int]:
        # Validation
        if init_words is not None and len(init_words) != len(tokens):
            raise ValueError(f"There must be an init_word for every token `{tokens}`")
        for token in tokens:
            if self.convert_tokens_to_ids([token]) != self.convert_tokens_to_ids([self.tokenizer.eos_token]):
                raise ValueError(f"The token `{token}` already exists")

        # Add token to the tokenizer.
        self.tokenizer.add_tokens(tokens)
        # Get the id of the token just added.
        new_token_ids = self.convert_tokens_to_ids(tokens)
        # Resize the word embeddings matrix to include an embedding for the new token.
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        # Set the new embedding equal to an existing embedding.
        if init_words is not None:
            init_word_ids = self.convert_tokens_to_ids(init_words)
            embeddings = self.get_text_embeddings(init_word_ids)
            self.set_text_embeddings(embeddings, new_token_ids)
        return new_token_ids

    @torch.no_grad()
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

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

    def encode_token_ids_with_grad(self, token_ids: List[int]) -> torch.Tensor:
        input_ids = torch.tensor(token_ids).int().unsqueeze(0).to(self.device)
        text_embeddings = self.text_encoder(input_ids)[0]
        return text_embeddings

    @torch.no_grad()
    def encode_image(self, image: Image.Image, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        image = 2.0 * image - 1.0
        latent = self.vae.encode(image).latent_dist.sample(generator)
        latent = latent * self.vae.config.scaling_factor
        return latent

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        latent = 1 / self.vae.config.scaling_factor * latent
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
        generator: Optional[torch.Generator] = None
    ) -> torch.FloatTensor:
        return self.scheduler.step(noise_pred, timestep, latent, generator=generator).prev_sample

    def get_noise_pred(
        self,
        latent: torch.Tensor,
        timestep: int,
        embedding: torch.FloatTensor,
        zero_grad: bool = False,
        **attn_processor_kwargs,
    ) -> torch.FloatTensor:
        sample = self.unet(
            latent,
            timestep,
            encoder_hidden_states=embedding,
            cross_attention_kwargs=attn_processor_kwargs,
        ).sample
        if zero_grad:
            self.unet.zero_grad()
        return sample

    def get_timesteps(self, inversion: bool = False) -> torch.IntTensor:
        timesteps = self.scheduler.timesteps
        return timesteps.flip(dims=(0,)) if inversion else timesteps

    def get_embeddings(self) -> torch.nn.modules.sparse.Embedding:
        return self.text_encoder.get_input_embeddings()

    def get_text_embeddings(self, indices: List[int]) -> torch.nn.parameter.Parameter:
        return self.get_embeddings().weight[indices]

    def set_text_embeddings(self, text_embeddings: torch.nn.parameter.Parameter, indices: List[int]):
        assert text_embeddings.shape[0] == len(indices)
        self.get_embeddings().weight[indices] = text_embeddings

    @torch.no_grad()
    def set_embeddings(self, embedding: torch.FloatTensor):
        self.get_embeddings().weight[:, :] = embedding[:, :]

    def register_attention_store(
        self,
        attention_store: AttentionStore,
        attention_processor_type: Type[AttnProcessorWithAttentionStore],
    ):
        self.attention_store = attention_store
        attention_processors = {}
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

    def get_attention_store(self) -> AttentionStore:
        return self.attention_store

    def reset_attention_store(self):
        self.attention_store = None

    def save(self, dirname: str):
        os.makedirs(dirname, exist_ok=True)
        model_path = os.path.join(dirname, "model")
        self.model.save_pretrained(model_path)
        tokenizer_path = os.path.join(dirname, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        kwargs = {
            "model_path": model_path,
            "tokenizer_path": tokenizer_path,
            "model_name": self.model_name,
            "ddim_steps": self.ddim_steps,
        }
        with open(os.path.join(dirname, "kwargs"), "wb") as f:
            pickle.dump(kwargs, f)

    @classmethod
    def load(cls, dirname: str, device: Literal["cuda", "cpu"]) -> "PretrainedStableDiffusionAdapter":
        with open(os.path.join(dirname, "kwargs"), "rb") as f:
            kwargs = pickle.load(f)
        tokenizer = CLIPTokenizer.from_pretrained(kwargs["tokenizer_path"])
        sd = StableDiffusionPipeline.from_pretrained(kwargs["model_path"]).to(device)
        del kwargs["tokenizer_path"]
        del kwargs["model_path"]
        model = cls(**kwargs, device=device)
        model.model = sd
        model.tokenizer = tokenizer
        return model


@torch.no_grad()
def classifier_free_guidance_step(
     model: PretrainedStableDiffusionAdapter,
     latent: torch.FloatTensor,
     prompt_embedding: torch.FloatTensor,
     null_embedding: torch.FloatTensor,
     timestep: int,
     guidance_scale: float = 7.5,
     generator: Optional[torch.Generator] = None,
     local: bool = False,
     mask_attn_res: int = 16,
     mask_pool_k: int = 1,
     mask_indices: List[int] = [],
     mask_threshold: float = 0.3,
     cross_attention_kwargs_cond: Dict[str, Any] = {},
     cross_attention_kwargs_uncond: Dict[str, Any] = {},
) -> torch.FloatTensor:
    noise_pred_cond = model.get_noise_pred(
        latent,
        timestep,
        prompt_embedding,
        False,
        **cross_attention_kwargs_cond,
    )
    if local:
        attention_store = model.get_attention_store()
        attn_maps = attention_store.aggregate_attention(
            places_in_unet=["up", "down", "mid"],
            is_cross=True,
            res=mask_attn_res,
            element_name="attn",
        )
    noise_pred_uncond = model.get_noise_pred(
        latent,
        timestep,
        null_embedding,
        False,
        **cross_attention_kwargs_uncond,
    )
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
    latent = model.prev_step(noise_pred, timestep, latent, generator)
    if local:
        # (8 * 2, mask_attn_res, mask_attn_res, len(mask_indices))
        attn_maps = attn_maps[:, :, :, mask_indices]
        # (2, 8, 1, mask_attn_res, mask_attn_res, len(mask_indices))
        attn_maps = attn_maps.reshape(2, -1, 1, mask_attn_res, mask_attn_res, len(mask_indices))
        # (2, 1, mask_attn_res, mask_attn_res)
        mask = attn_maps.sum(-1).mean(1)
        # (2, 1, mask_attn_res, mask_attn_res)
        mask = F.max_pool2d(
            mask,
            kernel_size=(2 * mask_pool_k + 1, 2 * mask_pool_k + 1),
            stride=(1, 1),
            padding=(mask_pool_k, mask_pool_k),
        )
        # (2, 1, 64, 64) <-- 64 in the case of StableDiffusion
        mask = F.interpolate(mask, size=(latent.shape[-1], latent.shape[-1]))
        mask_max = mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask / mask_max
        # (2, 1, 64, 64) in booleans
        mask = mask.gt(mask_threshold)
        mask = mask[:1] + mask
        # (2, 1, 64, 64) in float
        mask = mask.float()
        latent = latent[:1] + mask * (latent - latent[:1])
    return latent


@torch.no_grad()
def classifier_free_guidance(
    model: PretrainedStableDiffusionAdapter,
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
    model: PretrainedStableDiffusionAdapter,
    image: Image.Image,
    prompt: str,
    generator: Optional[torch.Generator] = None,
) -> List[torch.FloatTensor]:
    latent = model.encode_image(image, generator=generator)
    prompt_embedding = model.encode_text(prompt)
    latents = [latent]
    # From timestep 0 to T
    for timestep in model.get_timesteps(inversion=True):
        noise_pred = model.get_noise_pred(latent, timestep, prompt_embedding)
        latent = model.next_step(noise_pred, timestep, latent)
        latents.append(latent)
    return latents


@torch.no_grad()
def ddim_inversion_with_token_ids(
    model: PretrainedStableDiffusionAdapter,
    image: Image.Image,
    prompt: List[int],
    generator: Optional[torch.Generator] = None,
) -> List[torch.FloatTensor]:
    latent = model.encode_image(image, generator=generator)
    prompt_embedding = model.encode_token_ids_with_grad(prompt)
    latents = [latent]
    # From timestep 0 to T
    for timestep in model.get_timesteps(inversion=True):
        noise_pred = model.get_noise_pred(latent, timestep, prompt_embedding)
        latent = model.next_step(noise_pred, timestep, latent)
        latents.append(latent)
    return latents

