import random
from typing import Literal, Optional

import numpy as np
import torch
from diffusers import StableDiffusionPipeline


def seed_everything(seed: int = 50, deterministic_cuda: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_cuda)


def init_stable_diffusion(
    version: str = "runwayml/stable-diffusion-v1-5",
    torch_dtype: torch.dtype = torch.float32,
    safety_checker: Optional[bool] = None,
    device: Literal["cuda", "cpu"] = "cuda",
) -> StableDiffusionPipeline:
    return StableDiffusionPipeline.from_pretrained(
        version,
        torch_dtype=torch_dtype,
        safety_checker=safety_checker,
    ).to(device)


def plot_image_on_axis(ax, image, title):
    ax.set_title(title)
    ax.imshow(image)
    ax.set_axis_off()
