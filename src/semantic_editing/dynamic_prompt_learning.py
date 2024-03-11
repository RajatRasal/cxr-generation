from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from torch.optim import Adam
from tqdm import tqdm

from semantic_editing.base import CFGOptimisation, NULL_STRING
from semantic_editing.diffusion import StableDiffusionAdapter, classifier_free_guidance, classifier_free_guidance_step, ddim_inversion
from semantic_editing.utils import seed_everything, init_stable_diffusion, plot_image_on_axis
from semantic_editing.tools import background_mask, find_noun_indices


class DynamicPromptOptimisation(CFGOptimisation):

    def __init__(
        self,
        model: StableDiffusionAdapter,
        guidance_scale: int,
        num_inner_steps_dpl: int = 50,
        num_inner_steps_nti: int = 20,
        learning_rate: int = 1e-2,
        image_size: int = 512,
        epsilon: float = 1e-5,
        attention_resolution: int = 16,
        attention_balancing_coeff: float = 1.0,
        attention_balancing_alpha: float = 25,
        attention_balancing_beta: float = 0.3,
        disjoint_object_coeff: float = 0.1,
        disjoint_object_alpha: float = 25,
        disjoint_object_beta: float = 0.9,
        background_leakage_coeff: float = 0.1,
        background_leakage_alpha: float = 50,
        background_leakage_beta: float = 0.7,
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.num_inner_steps_dpl = num_inner_steps_dpl
        self.num_inner_steps_nti = num_inner_steps_nti
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.epsilon = epsilon

        self.attention_resolution = attention_resolution
        self.attention_balancing_coeff = attention_balancing_coeff
        self.attention_balancing_alpha = attention_balancing_alpha
        self.attention_balancing_beta = attention_balancing_beta
        self.disjoint_object_coeff = disjoint_object_coeff
        self.disjoint_object_alpha = disjoint_object_alpha
        self.disjoint_object_beta = disjoint_object_beta
        self.background_leakage_coeff = background_leakage_coeff
        self.background_leakage_alpha = background_leakage_alpha
        self.background_leakage_beta = background_leakage_beta

    def _loss_attention_balancing(
        self,
        cross_attn_maps: torch.FloatTensor,
        noun_indices: List[int],
    ) -> torch.FloatTensor:
        cross_attn_maps_nouns = cross_attn_maps[:, :, noun_indices]
        cross_attn_maps_nouns_norm = F.softmax(cross_attn_maps_nouns * 100, dim=-1)

        # Equation 8 required cross-attention maps to be passed through
        # Gaussian smoothing filter, as mentioned in Attend-and-Excite paper.
        loss = torch.FloatTensor([float("-inf")]).to(self.model.device)
        for i in range(len(noun_indices)):
            cross_attn_map = cross_attn_maps_nouns_norm[:, :, i].unsqueeze(0)
            cross_attn_map = F.pad(cross_attn_map, pad=(1, 1), mode="reflect")
            cross_attn_map_smooth = F_vision.gaussian_blur(
                cross_attn_map,
                kernel_size=(3, 3),
                sigma=(0.5, 0.5),
            )
            cross_attn_map_smooth = cross_attn_map_smooth.squeeze(0)
            max_attention_per_noun = cross_attn_map_smooth.max()
            # Find the max attention for a noun
            curr_max = max(torch.FloatTensor([0.0]).to(self.model.device), 1 - max_attention_per_noun)
            # Find the max attention across all nouns
            loss = max(curr_max, loss)

        return loss

    def _loss_background_leakage(self, cross_attn_maps, bg_map) -> torch.FloatTensor:
        raise NotImplementedError
        # return torch.Tensor([0.0]).to(self.model.device)

    def _loss_disjoint_object(self, cross_attn_maps, noun_indices) -> torch.FloatTensor:
        raise NotImplementedError
        # return torch.Tensor([0.0]).to(self.model.device)

    def _compute_threshold(self, timestep: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        return (torch.FloatTensor([float("-inf")]).to(self.model.device) for _ in range(3))

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
        # TODO: Include prepare_unet function from UNet2DConditionalModel
        # TODO: Ensure that attention_store is registered
        image = image.resize((self.image_size, self.image_size))

        self.latents = ddim_inversion(self.model, image, prompt)

        null_embedding = self.model.encode_text(NULL_STRING)

        # Compute background mask from attention maps stored during DDIM inversion.
        prompt_embedding = self.model.encode_text(prompt)
        index_noun_pairs = find_noun_indices(self.model, prompt)
        noun_indices = [i for i, _ in index_noun_pairs]
        bg_maps = background_mask(self.model.attention_store, index_noun_pairs)
        self.model.attention_store.reset()

        # Select indices for words in the text encoder that should not be
        # updated during the DPL procedure, i.e. indices that are not present
        # in the tokenised sentence.
        # TODO: Reimplement this so that embeddings in the text_encoder are
        # not updated in place but are updated in a copy of the weights stored
        # within this class.
        encoder_embeddings = self.model.text_encoder.get_input_embeddings().weight.data.clone()
        prompt_token_ids = self.model.tokenise_text(prompt)
        index_no_updates = torch.ones(len(self.model.tokenizer), dtype=bool)
        index_no_updates[prompt_token_ids] = False

        # NTI in outer loop, DPL in inner loop.
        optimised_null_embeddings_list = []
        optimised_prompt_noun_embeddings_list = []

        latent_cur = self.latents[-1]
        timesteps = self.model.get_timesteps()
        n_latents = len(self.latents)
        n_timesteps = len(timesteps)
        for i in range(n_timesteps):
            t = timesteps[i]

            # --------- DPL
            # TODO: Implement thresholding as per equation 10
            thresh_at, thresh_bg, thresh_dj = self._compute_threshold(i)

            # Optimise text tokens w.r.t attention maps
            dpl_optimiser = torch.optim.AdamW(self.model.text_encoder.get_input_embeddings().parameters())
            for _ in range(self.num_inner_steps_dpl):
                # Sample the noise distribution to fill up the attention store
                prompt_embedding = self.model.encode_text_with_grad(prompt)
                self.model.get_noise_pred(
                    latent_cur,
                    t,
                    prompt_embedding,
                    True,
                )

                avg_cross_attn_maps = self.model.attention_store.aggregate_attention(
                    places_in_unet=["up", "down", "mid"],
                    is_cross=True,
                    res=self.attention_resolution,
                    element_name="attn",
                )

                # Equation 8
                if self.attention_balancing_coeff > 0:
                    loss_at = self._loss_attention_balancing(avg_cross_attn_maps, noun_indices)
                else:
                    loss_at = torch.Tensor([0.0]).to(self.model.device)
                # Equation 7
                if self.background_leakage_coeff > 0:
                    loss_bg = self._loss_background_leakage(avg_cross_attn_maps, bg_maps)
                else:
                    loss_bg = torch.Tensor([0.0]).to(self.model.device)
                # Equation 6
                if self.disjoint_object_coeff > 0:
                    loss_dj = self._loss_disjoint_object(avg_cross_attn_maps, noun_indices)
                else:
                    loss_dj = torch.Tensor([0.0]).to(self.model.device)

                # Loss
                loss_dpl = self.attention_balancing_coeff * loss_at + \
                        self.background_leakage_coeff * loss_bg + \
                        self.disjoint_object_coeff * loss_dj

                if (thresh_at > loss_at) and (thresh_bg > loss_bg) and (thresh_dj > loss_dj):
                    break

                loss_dpl.backward(retain_graph=False)
                dpl_optimiser.step()
                dpl_optimiser.zero_grad()

                # During backprop, all embeddings may be updated. We want to reset
                # embeddings that are not in the prompt back to their original values
                with torch.no_grad():
                    self.model.text_encoder.get_input_embeddings().weight[index_no_updates] = encoder_embeddings[index_no_updates]

                self.model.attention_store.reset()
            
            torch.cuda.empty_cache()
            optimised_prompt_noun_embeddings = self.model.text_encoder.get_input_embeddings().weight[~index_no_updates].detach().cpu()
            optimised_prompt_noun_embeddings_list.append(optimised_prompt_noun_embeddings)
            
            # --------- NTI
            # Optimise null tokens for reconstructing latents
            null_embedding = null_embedding.clone().detach()
            null_embedding.requires_grad = True
            lr_scale_factor = 1. - i / (n_timesteps * 2)
            nti_optimiser = Adam([null_embedding], lr=self.learning_rate * lr_scale_factor)
            latent_prev = self.latents[n_latents - i - 2]
            prompt_embedding = self.model.encode_text(prompt)
            with torch.no_grad():
                noise_pred_cond = self.model.get_noise_pred(
                    latent_cur,
                    t,
                    prompt_embedding,
                )
            for _ in range(self.num_inner_steps_nti):
                latent_prev_rec = self._cfg_with_prompt_noise_pred(
                    latent_cur,
                    t,
                    null_embedding,
                    noise_pred_cond,
                )
                loss_nti = F.mse_loss(latent_prev_rec, latent_prev)
                loss_nti.backward(retain_graph=False)
                nti_optimiser.step()
                nti_optimiser.zero_grad()
                loss_item = loss_nti.item()
                if loss_item < self.epsilon + i * 2e-5:
                    break
            optimised_null_embeddings_list.append(null_embedding[:1].detach())
            with torch.no_grad():
                latent_cur = self._cfg_with_prompt_noise_pred(
                    latent_cur,
                    t,
                    null_embedding,
                    noise_pred_cond,
                )

            torch.cuda.empty_cache()
            self.model.attention_store.reset()

        self.null_embeddings = optimised_null_embeddings_list
        self.prompt_noun_embeddings = optimised_prompt_noun_embeddings_list

    def generate(self, prompt: str) -> Image.Image:
        pass

