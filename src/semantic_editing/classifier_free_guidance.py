from typing import Dict, Optional

from PIL import Image

from semantic_editing.attention import AttentionStoreAccumulate, AttnProcessorWithAttentionStore
from semantic_editing.base import CFGOptimisation
from semantic_editing.diffusion import PretrainedStableDiffusionAdapter, classifier_free_guidance, ddim_inversion


class CFGWithDDIM(CFGOptimisation):

    def __init__(
        self,
        model: PretrainedStableDiffusionAdapter,
        guidance_scale: int,
        image_size: Optional[int] = None,
        attention_accumulate: bool = False,
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.attention_accumulate = attention_accumulate

    def fit(self, image: Image.Image, prompt: str):
        self.prompt = prompt

        if self.image_size is not None:
            image = image.resize((self.image_size, self.image_size))

        if self.attention_accumulate:
            self.model.register_attention_store(
                AttentionStoreAccumulate(),
                AttnProcessorWithAttentionStore,
            )

        self.latent_T = ddim_inversion(self.model, image, prompt)[-1]

    def generate(
        self,
        swaps: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, str]] = None,
    ) -> Image.Image:
        if not (hasattr(self, "latent_T") and hasattr(self, "prompt")):
            assert ValueError(f"Need to fit {self.__class__.__name__} on an image before generating")

        if weights is not None:
            raise NotImplementedError
        elif swaps is not None:
            raise NotImplementedError
        else:
            edit_prompt = self.prompt

        # if self.attention_accumulate:
        #     self.model.register_attention_store(
        #         AttentionStoreAccumulate(),
        #         AttnProcessorWithAttentionStore,
        #     )

        latents = classifier_free_guidance(
            self.model,
            self.latent_T.clone(),
            edit_prompt,
            self.guidance_scale,
        )
        latent_0 = latents[-1]

        image = self.model.decode_latent(latent_0)
        return image

    def load(cls, dirname: str) -> "CFGWithDDIM":
        raise NotImplementedError

    def save(dirname: str):
        raise NotImplementedError

