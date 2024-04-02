import math
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

from semantic_editing.attention import AttentionStoreAccumulate, AttentionStoreTimestep
from semantic_editing.base import CFGOptimisation, NULL_STRING
from semantic_editing.diffusion import StableDiffusionAdapter, classifier_free_guidance, classifier_free_guidance_step, ddim_inversion
from semantic_editing.gaussian_smoothing import GaussianSmoothing
from semantic_editing.utils import seed_everything, init_stable_diffusion, plot_image_on_axis
from semantic_editing.tools import CLUSTERING_ALGORITHM, background_mask, find_noun_indices


class DynamicPromptOptimisation(CFGOptimisation):

    def __init__(
        self,
        model: StableDiffusionAdapter,
        model_ddim: StableDiffusionAdapter,
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
        attention_balancing_smoothing_kernel_sigma: float = 0.5,
        disjoint_object_coeff: float = 0.1,
        disjoint_object_alpha: float = 25,
        disjoint_object_beta: float = 0.9,
        background_leakage_coeff: float = 0.1,
        background_leakage_alpha: float = 50,
        background_leakage_beta: float = 0.7,
        background_clusters_threshold: float = 0.3,
        clustering_algorithm: CLUSTERING_ALGORITHM = "kmeans",
        max_clusters: int = 5,
        clustering_random_state: int = 0,
    ):
        # TODO: Regsiter multiple attention stores for the same model as a dictionary
        self.model = model
        assert isinstance(self.model.attention_store, AttentionStoreTimestep)
        self.model_ddim = model_ddim
        assert isinstance(self.model_ddim.attention_store, AttentionStoreAccumulate)

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
        self.attention_balancing_smoothing_kernel_sigma = attention_balancing_smoothing_kernel_sigma
        self.disjoint_object_coeff = disjoint_object_coeff
        self.disjoint_object_alpha = disjoint_object_alpha
        self.disjoint_object_beta = disjoint_object_beta
        self.background_leakage_coeff = background_leakage_coeff
        self.background_leakage_alpha = background_leakage_alpha
        self.background_leakage_beta = background_leakage_beta

        self.background_clusters_threshold = background_clusters_threshold
        self.clustering_algorithm = clustering_algorithm
        self.max_clusters = max_clusters
        self.clustering_random_state = clustering_random_state

    def _loss_attention_balancing(
        self,
        cross_attn_maps: torch.FloatTensor,
        indices: List[int],
    ) -> torch.FloatTensor:
        # shape = (res, res, 75 - 2 = 73)
        cross_attn_maps_norm = F.softmax(cross_attn_maps[:, :, 1:-1] * 100, dim=-1)

        # Initialise smoothing kernel
        smoothing_kernel = GaussianSmoothing(
            channels=1,
            kernel_size=3,
            sigma=self.attention_balancing_smoothing_kernel_sigma,
            dim=2,
        ).to(self.model.device)

        # Equation 8 required cross-attention maps to be passed through
        # Gaussian smoothing filter, as mentioned in Attend-and-Excite paper.
        max_attn_per_idx = []
        for i in indices:
            # Compute the max activation in cross attn map for each of the chosen indices.
            idx_map = cross_attn_maps_norm[:, :, i - 1]
            padded_map = F.pad(idx_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            smooth_map = smoothing_kernel(padded_map).squeeze(0).squeeze(0)
            max_attn = smooth_map.max()
            max_attn_per_idx.append(max_attn)

        # Compute the loss for each index, as per Equation 8.
        losses = [
            max(torch.Tensor([0.0]).to(self.model.device), 1 - curr_max)
            for curr_max in max_attn_per_idx
        ]
        # Compute the loss across all selected cross attention maps
        loss = max(losses)
        return loss

    def _loss_background_leakage(
        self,
        cross_attn_maps: torch.FloatTensor,
        indices: List[int],
        bg_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        flat_cross_attn_maps = cross_attn_maps[:, :, indices].view(-1, len(indices)).t()
        bg_map = bg_map.flatten().unsqueeze(0)
        return F.cosine_similarity(flat_cross_attn_maps, bg_map).mean()

    def _loss_disjoint_object(
        self,
        cross_attn_maps: torch.FloatTensor,
        indices: List[int],
    ) -> torch.FloatTensor:
        n_indices = len(indices)
        flat_cross_attn_map = cross_attn_maps[:, :, indices].view(-1, n_indices).t()
        cosine_mask = torch.tril(torch.ones((n_indices, n_indices)), diagonal=-1).bool()
        cosine_sim = F.cosine_similarity(flat_cross_attn_map[:, :, None], flat_cross_attn_map.t()[None, :, :])
        cosine_dist = cosine_sim[cosine_mask].mean()
        return cosine_dist

    def _compute_threshold(self, timestep: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        thresh_func = lambda alpha, beta: math.exp(-timestep / alpha) * beta
        thresh_at = thresh_func(
            self.attention_balancing_alpha,
            self.attention_balancing_beta,
        )
        thresh_bg = thresh_func(
            self.background_leakage_alpha,
            self.background_leakage_beta,
        )
        thresh_dj = thresh_func(
            self.disjoint_object_alpha,
            self.disjoint_object_beta,
        )
        return thresh_at, thresh_bg, thresh_dj

    def fit(self, image: Image.Image, prompt: str) -> List[torch.FloatTensor]:
        self.model.attention_store.reset()
        self.model_ddim.attention_store.reset()

        image = image.resize((self.image_size, self.image_size))

        # DDIM inversion to compute latents
        self.latents = ddim_inversion(self.model_ddim, image, prompt)
        n_latents = len(self.latents)

        # Find noun tokens
        index_noun_pairs = find_noun_indices(self.model, prompt)
        noun_indices = [i for i, _ in index_noun_pairs]

        # Compute background mask from attention maps stored during DDIM inversion.
        if self.background_leakage_coeff > 0:
            bg_map = background_mask(
                self.model_ddim.attention_store,
                index_noun_pairs,
                background_threshold=self.background_clusters_threshold,
                algorithm=self.clustering_algorithm,
                n_clusters=self.max_clusters,
                cluster_random_state=self.clustering_random_state,
                upscale_size=self.image_size,
            )
        self.model_ddim.attention_store.reset()

        # Select indices for words in the text encoder that should not be
        # updated during the DPL procedure, i.e. indices that are not present
        # in the tokenised sentence.
        self.noun_token_ids = self.model.tokenise_text(prompt).flatten()[noun_indices].unique(sorted=True)

        # NTI in outer loop, DPL in inner loop.
        dpl_attention_maps = []
        null_embeddings = []
        prompt_noun_embeddings = []

        latent_cur = self.latents[-1]
        timesteps = self.model.get_timesteps()
        n_timesteps = len(timesteps)

        null_embedding = self.model.encode_text(NULL_STRING)

        # TODO: Remove this assertion and put this in the unittests
        assert n_latents == n_timesteps + 1

        for i in range(n_timesteps):
            t = timesteps[i]

            # --------- DPL
            # Equation 10
            thresh_at, thresh_bg, thresh_dj = self._compute_threshold(i)

            # Optimise text tokens w.r.t attention maps
            dpl_optimiser = torch.optim.AdamW(self.model.get_embeddings().parameters())
            for j in range(self.num_inner_steps_dpl):
                self.model.attention_store.reset()
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
                    loss_at = self._loss_attention_balancing(
                        avg_cross_attn_maps,
                        noun_indices,
                    )
                else:
                    loss_at = torch.Tensor([0.0]).to(self.model.device)
                # Equation 7
                if self.background_leakage_coeff > 0:
                    loss_bg = self._loss_background_leakage(
                        avg_cross_attn_maps,
                        noun_indices,
                        bg_map,
                    )
                else:
                    loss_bg = torch.Tensor([0.0]).to(self.model.device)
                # Equation 6
                if self.disjoint_object_coeff > 0:
                    loss_dj = self._loss_disjoint_object(
                        avg_cross_attn_maps,
                        noun_indices,
                    )
                else:
                    loss_dj = torch.Tensor([0.0]).to(self.model.device)

                # Equation 10
                if (thresh_at > loss_at) and (thresh_bg > loss_bg) and (thresh_dj > loss_dj):
                    break

                # Equation 9
                loss_dpl = self.attention_balancing_coeff * loss_at + \
                        self.background_leakage_coeff * loss_bg + \
                        self.disjoint_object_coeff * loss_dj

                loss_dpl.backward(retain_graph=False)
                dpl_optimiser.step()
                dpl_optimiser.zero_grad()

                # During backprop, all embeddings may be updated. We want to reset
                # embeddings that are not in the prompt back to their original values
                self.model.reset_embeddings(self.noun_token_ids)
                
            _prompt_noun_embeddings = self.model.get_text_embeddings(self.noun_token_ids).detach()
            prompt_noun_embeddings.append(_prompt_noun_embeddings)
            
            # --------- NTI
            # Optimise null tokens for reconstructing latents
            null_embedding = null_embedding.clone().detach().requires_grad_(True)
            prompt_embedding = self.model.encode_text(prompt).detach().requires_grad_(False)

            lr_scale_factor = 1. - i / (n_timesteps * 2)
            nti_optimiser = Adam([null_embedding], lr=self.learning_rate * lr_scale_factor)

            latent_prev = self.latents[n_latents - i - 2]

            with torch.no_grad():
                noise_pred_cond = self.model.get_noise_pred(latent_cur, t, prompt_embedding)
            for j in range(self.num_inner_steps_nti):
                # CFG
                noise_pred_uncond = self.model.get_noise_pred(latent_cur, t, null_embedding)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_prev_rec = self.model.prev_step(noise_pred, t, latent_cur)
                # TODO: Include early stopping condition if loss is similar for n steps
                # Compute loss between predicted latent and true latent from DDIM inversion
                loss_nti = F.mse_loss(latent_prev_rec, latent_prev)
                # Optimise null embedding
                loss_nti.backward(retain_graph=False)
                nti_optimiser.step()
                nti_optimiser.zero_grad()
            null_embeddings.append(null_embedding[:1].detach())
            with torch.no_grad():
                noise_pred_uncond = self.model.get_noise_pred(latent_cur, t, null_embedding)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_cur = self.model.prev_step(noise_pred, t, latent_cur)

            # Run a noise prediction to get cross attention maps.
            self.model.attention_store.reset()
            with torch.no_grad():
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
            self.model.attention_store.reset()
            dpl_attention_maps.append(avg_cross_attn_maps.detach().cpu())

        self.null_embeddings = null_embeddings
        self.prompt_noun_embeddings = prompt_noun_embeddings

        self.model.reset_embeddings()

        return dpl_attention_maps

    @torch.no_grad()
    def generate(self, prompt: str) -> Image.Image:
        # TODO: This currently assumes reconstruction. Change this to implement object editing.
        if not (hasattr(self, "null_embeddings") and hasattr(self, "latents") and hasattr(self, "prompt_noun_embeddings") and hasattr(self, "noun_token_ids")):
            raise ValueError(f"Need to fit {self.__class__.__name__} on an image before generating")
        
        self.model.reset_embeddings()
        self.model.attention_store.reset()

        # TODO: Move this into model adapter
        latent = self.latents[-1].expand(
            1,
            self.model.unet.config.in_channels,
            self.image_size // 8,
            self.image_size // 8
        ).to(self.model.device)

        for i, timestep in enumerate(tqdm(self.model.get_timesteps())):
            self.model.set_text_embeddings(self.prompt_noun_embeddings[i], self.noun_token_ids)
            target_prompt_embedding = self.model.encode_text(prompt)
            latent = classifier_free_guidance_step(
                self.model,
                latent,
                target_prompt_embedding,
                self.null_embeddings[i],
                timestep,
                self.guidance_scale,
            )

        self.model.reset_embeddings()
        self.model.attention_store.reset()

        image = self.model.decode_latent(latent)
        return image

