import math
import os
import pickle
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F_vision
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from diffusers.utils import PIL_INTERPOLATION
from torch.optim import Adam
from tqdm import tqdm

from semantic_editing.attention import AttentionStoreAccumulate, AttentionStoreTimestep, AttnProcessorWithAttentionStore, AttentionStoreRefine
from semantic_editing.base import CFGOptimisation, NULL_STRING
from semantic_editing.diffusion import StableDiffusionAdapter, classifier_free_guidance, classifier_free_guidance_step, ddim_inversion, ddim_inversion_with_token_ids
from semantic_editing.gaussian_smoothing import GaussianSmoothing
from semantic_editing.utils import seed_everything, init_stable_diffusion, plot_image_on_axis
from semantic_editing.tools import CLUSTERING_ALGORITHM, background_mask, center_crop, find_tokens_and_noun_indices


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
        clustering_kwargs: Dict[str, Any] = {},
        center_crop: bool = True,
    ):
        self.model = model

        self.guidance_scale = guidance_scale
        self.num_inner_steps_dpl = num_inner_steps_dpl
        self.num_inner_steps_nti = num_inner_steps_nti
        self.learning_rate = learning_rate
        if image_size % 8 != 0:
            raise ValueError(f"image_size {image_size} must be a multiple of 8")
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
        self.clustering_kwargs = clustering_kwargs

        self.center_crop = center_crop

    def _loss_attention_balancing(
        self,
        cross_attn_maps: torch.FloatTensor,
        indices: List[int],
    ) -> torch.FloatTensor:
        # shape = (res, res, 75 - 2 = 73)
        cross_attn_maps_for_text = cross_attn_maps[:, :, 1:-1]
        # NOTE: Do not use the inplace operator as this will directly edit cross_attn_maps 
        # cross_attn_maps_for_text *= 100
        cross_attn_maps_for_text = cross_attn_maps_for_text * 100
        cross_attn_maps_norm = F.softmax(cross_attn_maps_for_text, dim=-1)

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

    def fit(self, image: Image.Image, prompt: str, **kwargs) -> List[torch.FloatTensor]:
        # TODO: Re-initialise the tokenizer and encoder
        self.model.reset()

        self.prompt = prompt

        index_noun_pairs = kwargs.get("index_noun_pairs", None)
        tokens = kwargs.get("tokens", None)
        seed = kwargs.get("seed", 0)
        generator = torch.manual_seed(seed)

        # All nouns are found and optimised if index_noun_pairs and tokens are not provided
        if index_noun_pairs is not None and tokens is not None:
            # TODO: Move separator tokens into StableDiffusionAdapter as properties
            separator = "</w>"
            assert all([not token.endswith(separator) for token in tokens])
            assert tokens[0] != self.model.tokenizer.bos_token
            assert tokens[-1] != self.model.tokenizer.eos_token
            tokens = [self.model.tokenizer.bos_token] + \
                [token + separator for token in tokens] + \
                [self.model.tokenizer.eos_token] * (self.model.tokenizer.model_max_length - len(tokens) - 1)
            assert len(tokens) == self.model.tokenizer.model_max_length
            self.index_noun_pairs = [(i + 1, noun) for i, noun in index_noun_pairs]
        else:
            tokens, self.index_noun_pairs = find_tokens_and_noun_indices(self.model, prompt)

        # Check that each noun is in the prompt 
        assert all([tokens[i][:-4] == noun for i, noun in self.index_noun_pairs])
        self.noun_indices = [i for i, _ in self.index_noun_pairs]
        self.token_ids = self.model.convert_tokens_to_ids(tokens)
        self.tokens = tokens

        # Transform image to a square
        if self.center_crop and image.size[0] != image.size[1]:
            image = center_crop(image)
        else:
            # TODO: Log that cropping was not applied
            pass
        image = image.convert("RGB").resize((self.image_size, self.image_size))

        # Use AttentionStoreAccumulate for background map prediction.
        # This will store attention maps during DDIM Inversion.
        self.model.register_attention_store(
            AttentionStoreAccumulate(),
            AttnProcessorWithAttentionStore,
        )

        # DDIM inversion to compute latents
        # TODO: Unittest to check that ddim_inversion = ddim_inversion_with_token_ids
        # TODO: Unittest tht ddim_inversion latents are the same as reference code
        self.latents = ddim_inversion_with_token_ids(self.model, image, self.token_ids, generator)
        n_latents = len(self.latents)

        # Compute background mask from attention maps stored during DDIM inversion.
        if self.background_leakage_coeff > 0:
            self.bg_map = background_mask(
                self.model.get_attention_store(),
                self.index_noun_pairs,
                background_threshold=self.background_clusters_threshold,
                algorithm=self.clustering_algorithm,
                n_clusters=self.max_clusters,
                upscale_size=self.image_size,
                **self.clustering_kwargs,
            )

        # Select tokens for words in the text encoder that should not be
        # updated during the DPL procedure, i.e. indices that are not present
        # in the tokenised sentence.
        dynamic_tokens = []
        init_words = []
        for i, noun in self.index_noun_pairs:
            dynamic_tokens.append(f"<{noun}>")
            init_words.append(noun)
        self.noun_token_ids = self.model.add_tokens(dynamic_tokens, init_words)
        # Assign new noun tokens in existing tokens list
        for i, (ind, _) in enumerate(self.index_noun_pairs):
            self.token_ids[ind] = self.noun_token_ids[i]

        # NTI in outer loop, DPL in inner loop.
        dpl_attention_maps = []
        null_embeddings = []
        prompt_noun_embeddings = []

        latent_cur = self.latents[-1] * self.model.scheduler.init_noise_sigma
        timesteps = self.model.get_timesteps()
        n_timesteps = len(timesteps)

        # Use AttentionStoreTimestep for DPL + NTI optimisations
        self.model.register_attention_store(
            AttentionStoreTimestep(self.attention_resolution, True),
            AttnProcessorWithAttentionStore,
        )

        # Original embeddings to use for resetting
        original_embeddings = self.model.get_embeddings().weight.data.detach().clone()

        for i in range(n_timesteps):
            t = timesteps[i]

            # --------- DPL
            # Equation 10
            thresh_at, thresh_bg, thresh_dj = self._compute_threshold(i)

            # Optimise text tokens w.r.t attention maps
            dpl_optimiser = torch.optim.AdamW(self.model.get_embeddings().parameters())
            for j in range(self.num_inner_steps_dpl):
                self.model.get_attention_store().reset()
                # Sample the noise distribution to fill up the attention store
                prompt_embedding = self.model.encode_token_ids_with_grad(self.token_ids)

                self.model.get_noise_pred(
                    latent_cur,
                    t,
                    prompt_embedding,
                    True,
                )
                self.model.unet.zero_grad()

                avg_cross_attn_maps = self.model.get_attention_store().aggregate_attention(
                    places_in_unet=["up", "down", "mid"],
                    is_cross=True,
                    res=self.attention_resolution,
                    element_name="attn",
                )

                # Equation 8
                if self.attention_balancing_coeff > 0:
                    loss_at = self._loss_attention_balancing(
                        avg_cross_attn_maps,
                        self.noun_indices,
                    )
                else:
                    loss_at = torch.Tensor([0.0]).to(self.model.device)
                # Equation 7
                if self.background_leakage_coeff > 0:
                    loss_bg = self._loss_background_leakage(
                        avg_cross_attn_maps,
                        self.noun_indices,
                        self.bg_map,
                    )
                else:
                    loss_bg = torch.Tensor([0.0]).to(self.model.device)
                # Equation 6
                if self.disjoint_object_coeff > 0:
                    loss_dj = self._loss_disjoint_object(
                        avg_cross_attn_maps,
                        self.noun_indices,
                    )
                else:
                    loss_dj = torch.Tensor([0.0]).to(self.model.device)

                # Equation 9
                loss_dpl = self.attention_balancing_coeff * loss_at + \
                        self.background_leakage_coeff * loss_bg + \
                        self.disjoint_object_coeff * loss_dj

                # Equation 10
                if (thresh_at > loss_at) and (thresh_bg > loss_bg) and (thresh_dj > loss_dj):
                    torch.cuda.empty_cache()
                    break

                loss_dpl.backward(retain_graph=False)
                dpl_optimiser.step()
                dpl_optimiser.zero_grad()

                # During backprop, all embeddings may be updated. We want to reset
                # embeddings that are not in the prompt back to their original values
                # TODO: I suspect this section is slowing down my implementation a lot.
                # Re-write this the same way as null_attend_textinv_pipeline.py.
                with torch.no_grad():
                    reset_embeddings = original_embeddings.clone()
                    for noun_token_id in self.noun_token_ids:
                        reset_embeddings[noun_token_id] = self.model.get_embeddings().weight.data[noun_token_id]
                    self.model.set_embeddings(reset_embeddings)

            torch.cuda.empty_cache()
            _prompt_noun_embeddings = self.model.get_text_embeddings(self.noun_token_ids).detach().clone()
            prompt_noun_embeddings.append(_prompt_noun_embeddings)
            
            # --------- NTI
            # Optimise null tokens for reconstructing latents
            # TODO: Make it an option whether to reset the null string or use the optimised one
            null_embedding = self.model.encode_text(NULL_STRING).clone().detach().requires_grad_(True)
            prompt_embedding = self.model.encode_token_ids_with_grad(self.token_ids).clone().detach().requires_grad_(False)

            lr_scale_factor = 1. - i / (n_timesteps * 2)
            nti_optimiser = Adam([null_embedding], lr=self.learning_rate * lr_scale_factor)

            latent_prev = self.latents[n_latents - i - 2]

            with torch.no_grad():
                noise_pred_cond = self.model.get_noise_pred(latent_cur, t, prompt_embedding)
            for k in range(self.num_inner_steps_nti):
                # CFG
                noise_pred_uncond = self.model.get_noise_pred(latent_cur, t, null_embedding)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_prev_rec = self.model.prev_step(noise_pred, t, latent_cur, generator)
                # TODO: Include early stopping condition if loss is similar for n steps
                # Compute loss between predicted latent and true latent from DDIM inversion
                loss_nti = F.mse_loss(latent_prev_rec, latent_prev)
                # Optimise null embedding
                loss_nti.backward(retain_graph=False)
                nti_optimiser.step()
                nti_optimiser.zero_grad()

            torch.cuda.empty_cache()

            null_embeddings.append(null_embedding[:1].detach().clone())
            with torch.no_grad():
                noise_pred_uncond = self.model.get_noise_pred(latent_cur, t, null_embedding)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latent_cur = self.model.prev_step(noise_pred, t, latent_cur, None)

            # Run a noise prediction to get cross attention maps.
            self.model.get_attention_store().reset()
            with torch.no_grad():
                self.model.get_noise_pred(
                    latent_cur,
                    t,
                    prompt_embedding,
                    True,
                )
                avg_cross_attn_maps = self.model.get_attention_store().aggregate_attention(
                    places_in_unet=["up", "down", "mid"],
                    is_cross=True,
                    res=self.attention_resolution,
                    element_name="attn",
                )
            self.model.get_attention_store().reset()
            dpl_attention_maps.append(avg_cross_attn_maps.detach().cpu())

            # TODO: Change this to logging
            print(i, j, k, loss_dpl.item(), loss_nti.item())

        self.null_embeddings = null_embeddings
        self.prompt_noun_embeddings = prompt_noun_embeddings

        self.model.set_embeddings(original_embeddings)
        self.model.reset_attention_store()

        return dpl_attention_maps

    def save(self, dirname: str):
        # TODO: The list of properties saved here should be inferred automatically somehow
        # we don't want to miss any incase we add new ones.
        if not (
            hasattr(self, "null_embeddings") and \
            hasattr(self, "latents") and \
            hasattr(self, "prompt_noun_embeddings") and \
            hasattr(self, "noun_token_ids") and \
            hasattr(self, "noun_indices") and \
            hasattr(self, "prompt") and \
            hasattr(self, "index_noun_pairs") and \
            hasattr(self, "token_ids") and \
            hasattr(self, "tokens")
        ):
            raise ValueError(f"Need to fit {self.__class__.__name__} on a prompt-image pair before generating")

        os.makedirs(dirname, exist_ok=True)

        torch.save(self.null_embeddings, os.path.join(dirname, "null_embeddings")) 
        torch.save(self.latents, os.path.join(dirname, "latents"))
        torch.save(self.prompt_noun_embeddings, os.path.join(dirname, "prompt_noun_embeddings"))
        torch.save(self.noun_token_ids, os.path.join(dirname, "noun_token_ids"))
        torch.save(self.noun_indices, os.path.join(dirname, "noun_indices"))
        torch.save(self.prompt, os.path.join(dirname, "prompt"))
        torch.save(self.index_noun_pairs, os.path.join(dirname, "index_noun_pairs"))
        torch.save(self.token_ids, os.path.join(dirname, "token_ids"))
        torch.save(self.tokens, os.path.join(dirname, "tokens"))

        self.model.save(os.path.join(dirname, "model_wrapper"))

        hyperparameters = {
            "guidance_scale": self.guidance_scale,
            "num_inner_steps_dpl": self.num_inner_steps_dpl,
            "num_inner_steps_nti": self.num_inner_steps_nti,
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "epsilon": self.epsilon,
            "attention_resolution": self.attention_resolution,
            "attention_balancing_coeff": self.attention_balancing_coeff,
            "attention_balancing_alpha": self.attention_balancing_alpha,
            "attention_balancing_beta": self.attention_balancing_beta,
            "attention_balancing_smoothing_kernel_sigma": self.attention_balancing_smoothing_kernel_sigma,
            "disjoint_object_coeff": self.disjoint_object_coeff,
            "disjoint_object_alpha": self.disjoint_object_alpha,
            "disjoint_object_beta": self.disjoint_object_beta,
            "background_leakage_coeff": self.background_leakage_coeff,
            "background_leakage_alpha": self.background_leakage_alpha,
            "background_leakage_beta": self.background_leakage_beta,
            "background_clusters_threshold": self.background_clusters_threshold,
            "clustering_algorithm": self.clustering_algorithm,
            "max_clusters": self.max_clusters,
            "clustering_kwargs": self.clustering_kwargs,
            "center_crop": self.center_crop,
        }
        with open(os.path.join(dirname, "hyperparameters.pickle"), "wb") as f:
            pickle.dump(hyperparameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, dirname: str, device: Literal["cuda", "cpu"]) -> "DynamicPromptOptimisation":
        with open(os.path.join(dirname, "hyperparameters.pickle"), "rb") as f:
            hyperparameters = pickle.load(f)
        model = StableDiffusionAdapter.load(os.path.join(dirname, "model_wrapper"), device)

        dpl = cls(model=model, **hyperparameters)

        dpl.null_embeddings = torch.load(os.path.join(dirname, "null_embeddings"))
        dpl.latents = torch.load(os.path.join(dirname, "latents"))
        dpl.prompt_noun_embeddings = torch.load(os.path.join(dirname, "prompt_noun_embeddings"))
        dpl.noun_token_ids = torch.load(os.path.join(dirname, "noun_token_ids"))
        dpl.noun_indices = torch.load(os.path.join(dirname, "noun_indices"))
        dpl.prompt = torch.load(os.path.join(dirname, "prompt"))
        dpl.index_noun_pairs = torch.load(os.path.join(dirname, "index_noun_pairs"))
        dpl.token_ids = torch.load(os.path.join(dirname, "token_ids"))
        dpl.tokens = torch.load(os.path.join(dirname, "tokens"))

        return dpl

    @torch.no_grad()
    def generate(
        self,
        swaps: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, str]] = None,
        cross_replace_steps: Optional[int] = None,
        self_replace_steps: Optional[int] = None,
        seed: int = 0,
        local: bool = False,
        local_mask_threshold: float = 0.3,
    ) -> Image.Image:
        # TODO: This currently assumes reconstruction. Change this to implement object editing.
        if not (
            hasattr(self, "null_embeddings") and \
            hasattr(self, "latents") and \
            hasattr(self, "prompt_noun_embeddings") and \
            hasattr(self, "noun_token_ids") and \
            hasattr(self, "noun_indices") and \
            hasattr(self, "prompt") and \
            hasattr(self, "index_noun_pairs") and \
            hasattr(self, "token_ids") and \
            hasattr(self, "tokens")
        ):
            raise ValueError(f"Need to fit {self.__class__.__name__} on an image before generating")

        if cross_replace_steps is None:
            cross_replace_steps = float("-inf")
        if self_replace_steps is None:
            self_replace_steps = float("-inf")

        if weights is not None or swaps is not None:
            if cross_replace_steps > self.model.ddim_steps:
                raise ValueError(f"cross_replace_steps {cross_replace_steps} must be less than or equal to model.ddim_steps {self.model.ddim_steps}")
            if self_replace_steps > self.model.ddim_steps:
                raise ValueError(f"self_replace_steps {self_replace_steps} must be less than or equal to model.ddim_steps {self.model.ddim_steps}")
        
        # Original embeddings to use for resetting
        original_embeddings = self.model.get_embeddings().weight.data.detach().clone()
        self.model.reset_attention_store()

        # Note: image_size is a multiple of 8
        latent = self.latents[-1].expand(
            1,
            self.model.unet.config.in_channels,
            self.image_size // 8,
            self.image_size // 8,
        ).to(self.model.device)

        edit_token_ids = self.token_ids.copy()

        if swaps is not None:
            for original, replacement in swaps.items():
                # TODO: Assert that replacement does not have <\w> at the end
                orig_token_id = self.model.convert_tokens_to_ids(f"<{original}>")
                replacement_token_id = self.model.convert_tokens_to_ids(f"{replacement}</w>")
                for i, token in enumerate(self.token_ids):
                    if orig_token_id == token:
                        edit_token_ids[i] = replacement_token_id
                        break
                    if i == len(self.token_ids) - 1:
                        raise ValueError(f"Noun `{original}` was not found in the prompt `{self.prompt}` during fitting")

        if weights is not None:
            amplify_indices = {
                index: weights[noun]
                for index, noun in self.index_noun_pairs
                if noun in weights
            }

        refine = weights is not None or swaps is not None
        generator = torch.manual_seed(seed)

        if refine:
            self.model.register_attention_store(
                AttentionStoreRefine(include_store=False),
                AttnProcessorWithAttentionStore,
            )
            latent = latent.repeat(2, 1, 1, 1)
        else:
            pass

        for i, timestep in enumerate(tqdm(self.model.get_timesteps())):
            if (
                self.attention_balancing_coeff > 0 and \
                self.disjoint_object_coeff > 0 and \
                self.background_leakage_coeff > 0
            ):
                self.model.set_text_embeddings(
                    self.prompt_noun_embeddings[i].to(self.model.device),
                    self.noun_token_ids,
                )
            if refine:
                prompt_embedding = self.model.encode_token_ids_with_grad(self.token_ids)
                edit_embedding = self.model.encode_token_ids_with_grad(edit_token_ids)
                embeddings = torch.cat([prompt_embedding, edit_embedding])
                cross_attention_kwargs_cond = {
                    "cross_replace": i < cross_replace_steps,
                    "self_replace": i < self_replace_steps,
                    "amplify_indices": amplify_indices,
                }
                latent = classifier_free_guidance_step(
                    model=self.model,
                    latent=latent,
                    prompt_embedding=embeddings,
                    null_embedding=self.null_embeddings[i].to(self.model.device).repeat(2, 1, 1),
                    timestep=timestep,
                    guidance_scale=self.guidance_scale,
                    generator=generator,
                    local=local,
                    mask_attn_res=self.attention_resolution,
                    mask_indices=self.noun_indices,
                    mask_threshold=local_mask_threshold,
                    cross_attention_kwargs_cond=cross_attention_kwargs_cond,
                ).detach()
            else:
                prompt_embedding = self.model.encode_token_ids_with_grad(self.token_ids)
                latent = classifier_free_guidance_step(
                    self.model,
                    latent,
                    prompt_embedding,
                    self.null_embeddings[i].to(self.model.device),
                    timestep,
                    self.guidance_scale,
                    )

        self.model.set_embeddings(original_embeddings)
        self.model.reset_attention_store()

        latent = latent[1].unsqueeze(0) if refine else latent
        image = self.model.decode_latent(latent)
        return image

