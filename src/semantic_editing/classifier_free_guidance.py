from typing import Optional

from PIL import Image

from semantic_editing.base import CFGOptimisation
from semantic_editing.diffusion import StableDiffusionAdapter, classifier_free_guidance, ddim_inversion


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

