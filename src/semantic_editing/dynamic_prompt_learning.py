from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from torch.optim import Adam
from tqdm import tqdm

from semantic_editing.base import CFGOptimisation
from semantic_editing.diffusion import StableDiffusionAdapter, classifier_free_guidance, classifier_free_guidance_step, ddim_inversion
from semantic_editing.utils import seed_everything, init_stable_diffusion, plot_image_on_axis


class DynamicPromptOptimisation(CFGOptimisation):

    def __init__(
        self,
        model: StableDiffusionAdapter,
        guidance_scale: int,
        num_inner_steps: int = 50,
        learning_rate: int = 1e-2,
        image_size: int = 512,
        epsilon: float = 1e-5,
        # attn_res: int = 16,
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.num_inner_steps = num_inner_steps
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.epsilon = epsilon

        # self.attn_res = attn_res
        # self.attention_store = AttentionStore(attn_res=self.attn_res)

    def fit(self, image: Image.Image, prompt: str):
        # TODO: Include prepare_unet function from UNet2DConditionalModel
        image = image.resize((self.image_size, self.image_size))
        self.latents = ddim_inversion(self.model, image, prompt)
