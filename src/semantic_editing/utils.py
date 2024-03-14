import random
from typing import Literal, Optional, Union

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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


def plot_image_on_axis(
    ax: Axes,
    image: Union[Image.Image, np.ndarray],
    title: str,
    fontsize: int = 10,
):
    ax.set_title(title, fontsize=fontsize)
    ax.imshow(image)
    ax.set_axis_off()


def save_figure(fig: Figure, name: str):
    if not name.endswith(".pdf"):
        raise ValueError("Figure name must end with '.pdf'")
    fig.savefig(name, bbox_inches="tight")
